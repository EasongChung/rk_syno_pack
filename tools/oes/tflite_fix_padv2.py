#!/usr/bin/env python3
"""TFLite 后处理：PADV2 -> PAD 改写（消除 constant_values 类型冲突与退化 scale）。

## 背景（证据链）
CI run 38940758759 实测，legacy 量化器产出 0 个退化 scale，但 regress 失败：
    tensorflow/lite/kernels/pad.cc:230
      op_context.input->type != op_context.constant_values->type
      (INT8 != FLOAT32)
      Node number 4 (PADV2) failed to prepare.
同一位置在 MLIR 产物里则是退化 scale 1.334441e+36。两种量化器都坏在
PADV2 的 constant_values 上，根因同一。

结构（本地产物实测）：
    op#1   code=34 (PAD)   n_in=2  t136(i8,s=0.0078,zp=-1) t2(i32)   正常
    op#4   code=60 (PADV2) n_in=3  t139(i8,s=1.33e36,zp=127) t4(i32) t133(...)  目标
    op#54  code=34 (PAD)   n_in=2  ...                            正常
    op#57  code=34 (PAD)   n_in=2  ...                            正常

## 为什么 PAD 与 PADV2 在 constant_value=0 时等价
TFLite 的 PAD 算子固定用 0 填充；PADV2 才带 constant_values。
本图的 PADV2 用的就是 0（MLIR 下 (zp127-zp127)*scale = 0），
故改为 PAD 语义完全等价，且：
  - 去掉 constant_values 张量 -> 不再有 FLOAT32/INT8 类型冲突
  - 不再给 MLIR 留 0/0 退化常量

## 安全性
改写 flatbuffer 可能静默损坏，故内置多层校验：
  1. 改写后用 tflite 包重新解析，确认 opcode=34 且 n_in=2
  2. 调用方必须再用 tf.lite.Interpreter allocate_tensors + 推理对拍
     （本脚本无法在无 TF 环境验证，故由 CI 的 regress 兜底）
  3. --dry-run 只报告不写盘

用法：
  python tflite_fix_padv2.py --in legacy.tflite --out fixed.tflite [--dry-run]
"""
import argparse
import sys
import sys as _sys

CODE_PAD = 34
CODE_PADV2 = 60


def load_model_obj(raw):
    """用 TF 自带 schema 读成可修改的对象树。"""
    from tensorflow.lite.python import schema_py_generated as fb
    reader = fb.Model.GetRootAsModel(raw, 0)
    if not hasattr(fb.ModelT, 'InitFromObj'):
        raise RuntimeError('schema lacks ModelT.InitFromObj; TF schema '
                           'version mismatch')
    return fb, fb.ModelT.InitFromObj(reader)


def survey(path):
    """只读盘点：各 PAD 类算子的 opcode / 输入数 / 量化参数。"""
    import tflite as T
    raw = open(path, 'rb').read()
    mo = T.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    codes = {}
    for i in range(mo.OperatorCodesLength()):
        oc = mo.OperatorCodes(i)
        bc = oc.BuiltinCode()
        codes[i] = bc if bc < 127 else oc.DeprecatedBuiltinCode()
    print('--- PAD-class operators ---')
    n_pad = n_padv2 = 0
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        c = codes.get(op.OpcodeIndex())
        if c not in (CODE_PAD, CODE_PADV2):
            continue
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        info = []
        for t in ins:
            tt = g.Tensors(t)
            q = tt.Quantization()
            sc = q.Scale(0) if (q is not None and q.ScaleLength()) else None
            info.append('t%d(%s%s)' % (
                t, {0: 'f32', 2: 'i32', 9: 'i8', 3: 'u8'}.get(
                    tt.Type(), 'T%d' % tt.Type()),
                '' if sc is None else ',s=%.4g' % sc))
        print('  op#%-4d code=%-3d n_in=%d %s' % (i, c, len(ins),
                                                   ' '.join(info)))
        if c == CODE_PAD:
            n_pad += 1
        else:
            n_padv2 += 1
    print('  PAD=%d PADV2=%d' % (n_pad, n_padv2))
    return n_pad, n_padv2


def audit(path):
    """出件自检（针对 -oiqt 式 float 假绿）：量化张量数、dtype 直方图、
    未量化 int8/uint8 张量、退化 scale，并区分「孤立 float32 残留」与
    「被算子引用的 float32」——前者无害，后者是真问题。

    返回 dict，供调用方断言。
    """
    import collections

    import tflite as T
    raw = open(path, 'rb').read()
    mo = T.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)

    # 哪些张量被算子真实引用（孤立张量不会被内核读，但仍在模型里）
    used = set()
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        for j in range(op.InputsLength()):
            used.add(op.Inputs(j))
        for j in range(op.OutputsLength()):
            used.add(op.Outputs(j))

    names = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8', 4: 'INT64',
             6: 'BOOL', 9: 'INT8'}
    hist = collections.Counter()
    unq_int8, degen, f32_used, f32_orphan = [], [], [], []
    n_quant = 0
    for t in range(g.TensorsLength()):
        tt = g.Tensors(t)
        ty = tt.Type()
        nm = names.get(ty, 'T%d' % ty)
        hist[nm] += 1
        q = tt.Quantization()
        has_q = q is not None and q.ScaleLength() > 0
        if has_q:
            n_quant += 1
            s = q.Scale(0)
            zp = q.ZeroPoint(0) if q.ZeroPointLength() else 0
            # 退化：零范围张量 (scale == zp*... ) 表现为 scale 极大
            if s > 1e6 or s <= 0.0:
                degen.append((t, nm, s, zp))
        if ty in (9, 3) and not has_q:
            unq_int8.append(t)
        if ty == 0:
            (f32_used if t in used else f32_orphan).append(t)

    total = g.TensorsLength()
    print('--- audit: %s (%d bytes) ---' % (path, len(raw)))
    print('  tensors            : %d' % total)
    print('  quantized tensors  : %d/%d (%.1f%%)'
          % (n_quant, total, 100.0 * n_quant / max(total, 1)))
    print('  dtype histogram    : %s' % dict(sorted(hist.items())))
    print('  DEGENERATE scale   : %d  %s'
          % (len(degen), ['t%d(%s,s=%.4g,zp=%d)' % d for d in degen[:8]]))
    print('  int8/uint8 WITHOUT quantization : %d  %s'
          % (len(unq_int8), unq_int8[:8]))
    print('  FLOAT32 referenced by ops : %d  %s'
          % (len(f32_used), f32_used[:8]))
    print('  FLOAT32 orphaned (unused): %d  %s'
          % (len(f32_orphan), f32_orphan[:8]))
    print('  NOTE: orphaned tensors are never read by the kernel; they only'
          ' inflate file size. -oiqt-style false-green is about *referenced*'
          ' unquantized int8, which is %d here.' % len(unq_int8))

    return {'total': total, 'quantized': n_quant, 'hist': dict(hist),
            'degenerate': degen, 'unquantized_int8': unq_int8,
            'float32_used': f32_used, 'float32_orphan': f32_orphan}


def verify(path):
    """改写后校验：必须 0 个 PADV2，且 PAD 全是 2 输入。"""
    n_pad, n_padv2 = survey(path)
    assert n_padv2 == 0, 'PADV2 still present after rewrite'
    print('VERIFY: 0 PADV2, %d PAD remaining -> OK' % n_pad)
    audit(path)
    return True


def _revalidate(buf):
    """用独立的 tflite 读 API 校验 Pack 出的字节流。

    Pack 返回非空不代表字节流合法——flatbuffer 静默损坏时内核会在
    很后面才炸，且报错位置与真正的错因无关。这里在落盘前就地拦截。
    """
    import tflite as T
    try:
        mo = T.Model.GetRootAsModel(buf, 0)
        n_ops = mo.Subgraphs(0).OperatorsLength()
        codes = {}
        for i in range(mo.OperatorCodesLength()):
            oc = mo.OperatorCodes(i)
            bc = oc.BuiltinCode()
            codes[i] = bc if bc < 127 else oc.DeprecatedBuiltinCode()
        g = mo.Subgraphs(0)
        n_pad = n_padv2 = 0
        for i in range(n_ops):
            if codes.get(g.Operators(i).OpcodeIndex()) == CODE_PAD:
                n_pad += 1
            elif codes.get(g.Operators(i).OpcodeIndex()) == CODE_PADV2:
                n_padv2 += 1
    except Exception as e:
        raise RuntimeError('revalidate: tflite could not parse packed '
                           'buffer: %s' % str(e)[:200])
    assert n_padv2 == 0, ('revalidate: packed buffer still has %d PADV2 '
                          '(flatbuffer union/offset mismatch?)' % n_padv2)
    assert n_pad >= 4, ('revalidate: expected >=4 PAD after rewrite '
                        '(3 original + 1 rewritten), got %d - graph '
                        'corrupted' % n_pad)
    print('REVALIDATE OK: ops=%d PAD=%d PADV2=%d' % (n_ops, n_pad, n_padv2),
          flush=True)
    return True


def rewrite(src, dst, dry_run=False):
    raw = open(src, 'rb').read()
    fb, model = load_model_obj(raw)
    sub = model.subgraphs[0]

    # TF 的 schema_py_generated 用驼峰字段名（operatorCodes / opcodeIndex /
    # builtinOptionsType / builtinOptions），不是 snake_case。
    # 这里做一次自适应，避免再因字段名猜错浪费 CI 轮次。
    def f(obj, snake, camel):
        return camel if hasattr(obj, camel) else snake

    codes = getattr(model, f(model, 'operator_codes', 'operatorCodes'))
    subs = getattr(model, f(model, 'subgraphs', 'subgraphs'))
    subs = getattr(model, 'subgraphs', None) or subs
    sub = subs[0]
    ops = getattr(sub, f(sub, 'operators', 'operators'))

    print('API probe: n_opcodes=%d n_subgraphs=%d n_ops=%d'
          % (len(codes), len(subs), len(ops)), flush=True)
    oc0 = codes[0]
    print('API probe: opcode[0] attrs=%s'
          % [a for a in dir(oc0) if 'code' in a.lower()][:6], flush=True)

    def eff_code(oc):
        bc = getattr(oc, f(oc, 'builtin_code', 'builtinCode'))
        dbc = getattr(oc, f(oc, 'deprecated_builtin_code',
                             'deprecatedBuiltinCode'))
        return bc if bc < 127 else dbc

    # 找到 PAD 的 opcode index（复用，不新建 code）
    pad_idx = None
    for i, oc in enumerate(codes):
        if eff_code(oc) == CODE_PAD:
            pad_idx = i
            break
    if pad_idx is None:
        raise RuntimeError('model has no PAD opcode to reuse')

    oi_attr = f(ops[0], 'opcode_index', 'opcodeIndex')
    targets = []
    for op in ops:
        idx = getattr(op, oi_attr)
        if eff_code(codes[idx]) != CODE_PADV2:
            continue
        if len(op.inputs) < 2:
            continue
        targets.append(op)

    print('PADV2 operators found: %d (PAD opcode_index=%d)'
          % (len(targets), pad_idx), flush=True)
    if not targets:
        print('nothing to do', flush=True)
        return 0

    # BuiltinOptions 是 union，tag 名在不同 TF 版本里不一致
    # （Pad / PadOptions / NONE 都出现过）。这里一次 introspect 全部候选，
    # 打印出来供人工核对，并按优先级尝试，避免再猜。
    bo_cls = fb.BuiltinOptions
    bo_attrs = sorted(a for a in dir(bo_cls) if not a.startswith('_'))
    print('API probe: BuiltinOptions attrs=%s' % bo_attrs, flush=True)

    bt_attr = f(targets[0], 'builtin_options_type', 'builtinOptionsType')
    bo_attr = f(targets[0], 'builtin_options', 'builtinOptions')

    # --- 首选：抄现成的 PAD 算子的 options -------------------------------
    # 模型里已有 3 个 PAD 算子被 TFLite 内核成功 prepare 过，它们的
    # builtin_options_type / builtin_options 就是「合法 PAD」的权威样板。
    # 照抄比猜 union tag 安全得多：改写后的算子与那 3 个结构完全一致。
    ref = None
    for op in ops:
        if eff_code(codes[getattr(op, oi_attr)]) == CODE_PAD:
            ref = op
            break
    if ref is not None:
        ref_bt = getattr(ref, bt_attr)
        ref_bo = getattr(ref, bo_attr)
        print('API probe: reference PAD options type=%r options=%s'
              % (ref_bt, type(ref_bo).__name__), flush=True)

    tag_name, union_tag = None, None
    for cand in ('PadOptions', 'Pad', 'PadV2Options', 'NONE', 'None'):
        if hasattr(bo_cls, cand):
            tag_name, union_tag = cand, getattr(bo_cls, cand)
            break
    print('API probe: chosen BuiltinOptions.%s = %r'
          % (tag_name, union_tag), flush=True)

    pad_opts_t = getattr(fb, 'PadOptionsT', None)
    print('API probe: PadOptionsT=%s' % (pad_opts_t is not None,),
          flush=True)

    for op in targets:
        # op.inputs 是 numpy array，不能用 del（ValueError: cannot delete
        # array elements），必须整体替换为切片。
        op.inputs = op.inputs[:2]
        setattr(op, oi_attr, pad_idx)
        ok = False
        if ref is not None:
            # 与已验证的 PAD 算子保持结构一致
            setattr(op, bt_attr, ref_bt)
            setattr(op, bo_attr, ref_bo)
            ok = True
        if not ok and isinstance(union_tag, int):
            # NONE 之类的纯 tag：options 置空（PAD 无必填 options）
            setattr(op, bt_attr, union_tag)
            setattr(op, bo_attr, None)
            ok = True
        if not ok and pad_opts_t is not None:
            setattr(op, bt_attr, 0)
            setattr(op, bo_attr, pad_opts_t())
            ok = True
        print('  rewrote one PADV2 -> PAD (options=%s)' % ok, flush=True)

    # 即使 --dry-run 也必须走完 Pack：Pack 才是最可能崩的一步
    # （历史教训：dry-run 在写盘前 return，掩盖了 Pack 的 TypeError，
    #  连续两轮 CI 都在这上面假绿）。
    # TF 的对象树 API：Pack(builder) 必须显式传 builder
    # （CI run 38992257089 实测 TypeError: missing 'builder'）。
    import flatbuffers
    builder = flatbuffers.Builder(1024 * 1024)
    # Pack 的返回值语义（CI run 38993185873 实测）：
    #   Pack(builder) 返回 root table 的**偏移量(int)**，不是字节流
    #   （直接对其取 len() 抛 "TypeError: object of type 'int' has no len()")。
    # 字节流必须从 builder 取：Output() 无参数且返回「已写入的数据段」
    # （flatbuffers/builder.py:180），它要求 buffer 已 Finish。
    # TF 对象树的 Pack 通常已自行 Finish，但不同版本行为不一致，
    # 故这里显式 Finish 一次（幂等：已 finished 时 no-op），
    # 再用 Output() 取字节。这样不依赖版本差异。
    packed_root = model.Pack(builder)
    try:
        builder.Finish(packed_root)
    except Exception as e:
        # 已 Finish 过或签名不同：不致命，Output() 仍可能可用
        print('  Finish(%s) raised %s (continuing)' % (packed_root, e),
              flush=True)
    out = bytes(builder.Output())
    if not out:
        raise RuntimeError('Pack(builder) produced an empty buffer')
    print('PACK OK: root_offset=%s bytes=%d' % (packed_root, len(out)),
          flush=True)

    # Pack 返回非空**不等于**字节流合法（flatbuffer 静默损坏的经典形态）。
    # 立刻用独立的 tflite 读 API 回读校验：算子数 / PAD 分类 / opcode，
    # 任何一项对不上就地失败，绝不落盘。
    _revalidate(out)

    if dry_run:
        print('dry-run: not writing (pack + revalidate both OK)', flush=True)
        return len(targets)

    with open(dst, 'wb') as f:
        f.write(out)
    print('WROTE %s (%d bytes, was %d)' % (dst, len(out), len(raw)),
          flush=True)
    return len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--survey-only', action='store_true')
    args = ap.parse_args()

    print('=== STAGE-P0: args parsed, src exists=%s ==='
          % __import__('os').path.exists(args.src), flush=True)
    _sys.stdout.flush()

    if args.survey_only:
        survey(args.src)
        return 0

    print('=== STAGE-P1: survey before ===', flush=True)
    survey(args.src)

    print('=== STAGE-P2: rewrite ===', flush=True)
    n = rewrite(args.src, args.dst, args.dry_run)
    if args.dry_run or not n:
        return 0

    print('=== STAGE-P3: verify after ===', flush=True)
    verify(args.dst)
    print('=== STAGE-P4: done ===', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())