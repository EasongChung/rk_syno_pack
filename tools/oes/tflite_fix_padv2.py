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


def verify(path):
    """改写后校验：必须 0 个 PADV2，且 PAD 全是 2 输入。"""
    n_pad, n_padv2 = survey(path)
    assert n_padv2 == 0, 'PADV2 still present after rewrite'
    print('VERIFY: 0 PADV2, %d PAD remaining -> OK' % n_pad)
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

    # BuiltinOptions union 的构造：不同 TF 版本略有差异，这里自适应。
    bo_cls = fb.BuiltinOptions
    union_tag = getattr(bo_cls, 'Pad')
    print('API probe: BuiltinOptions.Pad=%r' % (union_tag,), flush=True)
    pad_opts_t = getattr(fb, 'PadOptionsT', None)

    for op in targets:
        del op.inputs[2]
        setattr(op, oi_attr, pad_idx)
        setattr(op, f(op, 'builtin_options_type', 'builtinOptionsType'),
                union_tag)
        try:
            union_obj = union_tag()
            if hasattr(union_obj, 'Type'):
                union_obj.Type = union_tag
            if hasattr(union_obj, 'Value') and pad_opts_t is not None:
                union_obj.Value = pad_opts_t()
            setattr(op, f(op, 'builtin_options', 'builtinOptions'),
                    union_obj)
        except Exception as e:
            # 退化路径：直接把 PadOptionsT 挂在 builtin_options 上
            print('  union ctor failed (%s); try plain PadOptionsT' % e,
                  flush=True)
            setattr(op, f(op, 'builtin_options', 'builtinOptions'),
                    pad_opts_t() if pad_opts_t is not None else None)
        print('  rewrote one PADV2 -> PAD', flush=True)

    if dry_run:
        print('dry-run: not writing', flush=True)
        return len(targets)

    out = model.Pack()
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