#!/usr/bin/env python3
"""给所有未量化张量补上量化参数（mesa-24.3.0 delegate 兼容用）。

## 背景：och 真机段错误的根因（源码级已确认）

    tfl_device.c L238-239
        for (int i = 0; i < tf_context->tensors_size; i++)
            fill_tensor(delegate, tf_context, &tensors[i], i);

    fill_tensor 内只跳过 kTfLiteNoType，然后无条件：
        tensor->scale = quant->scale->data[0];

模型里有 20 个未量化张量 -> quant == NULL -> 解引用崩溃 -> EXIT=139。

## 本脚本做什么

给**每一个**缺量化参数的张量补 scale=[1.0] / zero_point=[0]，
使 fill_tensor 的 `quant->scale->data[0]` 永远有值可读。

同时可选执行 PADV2 -> PAD 改写（复用 tflite_fix_padv2 的逻辑），
让 main 版 delegate 也能接管该算子。

## 重要边界（不可省略，必须随产物一起传阅）

1. **这是为迁就旧版 delegate 而改变模型形态，不是修复。**
   TFLite 的 shape / paddings / axes 常量按惯例就是未量化 INT32，
   任何正常的全整数量化模型都会有一批。补量化参数对内核语义
   应当无副作用（内核用 GetTensorData<int32_t>() 读裸值），
   但这属**源码层推断**，本地无 TF 无法实证。
2. **TFLite VerifyModel 是否接受「INT32 张量带量化参数」未验证**，
   必须由 CI 的 tf.lite.Interpreter 加载 + regress 终值确认。
3. **本脚本不解决接管率**：24.3.0 接管上限仍是 56/201 = 27.86%。
   换 main 版 delegate 才能到 91.54%，且那才是正解。
   本脚本的唯一用途是**对照实验**：验证 och 的根因判断是否正确
   （若补全量化后 24.3.0 不崩，则根因成立）。
"""
import argparse
import os
import sys

CODE_PAD = 34
CODE_PADV2 = 60
FILE_IDENTIFIER = b'TFL3'

# 用 1.0/0 是语义上中性的选择：scale=1 意味着「原值即真值」，
# zp=0 对 INT32 无意义但能让 params 非空。内核不读 INT32 的量化参数，
# 所以取值本身不影响计算结果。
NEUTRAL_SCALE = 1.0
NEUTRAL_ZP = 0


def _attr(obj, snake, camel):
    """TF 的 schema_py_generated 用驼峰字段名（历史教训）。"""
    return camel if hasattr(obj, camel) else snake


def load_model_obj(raw):
    from tensorflow.lite.python import schema_py_generated as fb
    reader = fb.Model.GetRootAsModel(raw, 0)
    if not hasattr(fb.ModelT, 'InitFromObj'):
        raise RuntimeError('schema lacks ModelT.InitFromObj; TF schema '
                           'version mismatch')
    return fb, fb.ModelT.InitFromObj(reader)


def eff_code(oc, fb):
    bc = getattr(oc, _attr(oc, 'builtin_code', 'builtinCode'))
    dbc = getattr(oc, _attr(oc, 'deprecated_builtin_code',
                             'deprecatedBuiltinCode'))
    return bc if bc < 127 else dbc


def pack_and_check(fb, model, expect_padv2=0):
    """Pack + 落盘前的字节流校验（三个历史 bug 都死在这里，必须复用）。"""
    import flatbuffers
    builder = flatbuffers.Builder(1024 * 1024)
    packed_root = model.Pack(builder)
    # TFLite 的 file_identifier "TFL3" 固定在 [4:8]，漏了内核直接拒载
    builder.Finish(packed_root, file_identifier=FILE_IDENTIFIER)
    out = bytes(builder.Output())
    if not out:
        raise RuntimeError('Pack produced empty buffer')

    if out[4:8] != FILE_IDENTIFIER:
        raise RuntimeError('file_identifier at [4:8] is %r, expected %r'
                           % (bytes(out[4:8]), FILE_IDENTIFIER))
    _revalidate(out, expect_padv2)
    return out


def _revalidate(buf, expect_padv2=0):
    """用独立的 tflite 读 API 回读校验（Pack 非空 != 字节流合法）。"""
    import tflite as T
    if len(buf) < 8:
        raise RuntimeError('revalidate: buffer too short (%d B)' % len(buf))
    got = bytes(buf[4:8])
    if got != FILE_IDENTIFIER:
        raise RuntimeError('revalidate: file_identifier %r != %r'
                           % (got, FILE_IDENTIFIER))
    try:
        mo = T.Model.GetRootAsModel(buf, 0)
        g = mo.Subgraphs(0)
        codes = {}
        for i in range(mo.OperatorCodesLength()):
            oc = mo.OperatorCodes(i)
            bc = oc.BuiltinCode()
            codes[i] = bc if bc < 127 else oc.DeprecatedBuiltinCode()
        n_padv2 = sum(1 for i in range(g.OperatorsLength())
                      if codes.get(g.Operators(i).OpcodeIndex()) == CODE_PADV2)
        n_noq = 0
        for t in range(g.TensorsLength()):
            q = g.Tensors(t).Quantization()
            if q is None or q.ScaleLength() == 0:
                n_noq += 1
    except Exception as e:
        raise RuntimeError('revalidate: tflite cannot parse buffer: %s'
                           % str(e)[:200])
    print('REVALIDATE OK: file_id=%r tensors=%d no_quant=%d PADV2=%d'
          % (got, g.TensorsLength(), n_noq, n_padv2), flush=True)
    if expect_padv2 is not None and n_padv2 != expect_padv2:
        raise RuntimeError('revalidate: PADV2=%d, expected %d'
                           % (n_padv2, expect_padv2))
    if n_noq:
        raise RuntimeError('revalidate: %d tensor(s) still lack quantization '
                           'params -> fill_tensor would still segfault'
                           % n_noq)
    return True


def survey(path):
    """只读盘点：未量化张量的类型与分布。"""
    import tflite as T
    mo = T.Model.GetRootAsModel(open(path, 'rb').read(), 0)
    g = mo.Subgraphs(0)
    NAMES = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8',
             4: 'INT64', 6: 'BOOL', 9: 'INT8'}
    rows = []
    for t in range(g.TensorsLength()):
        tt = g.Tensors(t)
        q = tt.Quantization()
        if q is not None and q.ScaleLength() > 0:
            continue
        rows.append((t, NAMES.get(tt.Type(), 'T%d' % tt.Type()),
                     tt.Buffer()))
    print('--- survey %s ---' % path)
    print('  unquantized tensors: %d / %d' % (len(rows), g.TensorsLength()))
    by = {}
    for _, nm, _ in rows:
        by[nm] = by.get(nm, 0) + 1
    print('  by dtype: %s' % by)
    consts = [t for t, _, b in rows if b > 0]
    print('  with buffer (constant): %d   activation: %d'
          % (len(consts), len(rows) - len(consts)))
    return rows


def rewrite(src, dst, dry_run=False, also_fix_padv2=True):
    raw = open(src, 'rb').read()
    fb, model = load_model_obj(raw)
    sub = model.subgraphs[0]
    codes = getattr(model, _attr(model, 'operator_codes', 'operatorCodes'))
    tensors = getattr(sub, _attr(sub, 'tensors', 'tensors'))
    ops = getattr(sub, _attr(sub, 'operators', 'operators'))

    print('API probe: n_codes=%d n_tensors=%d n_ops=%d'
          % (len(codes), len(tensors), len(ops)), flush=True)

    # ---- 1. 给所有缺量化参数的张量补 scale/zp -------------------------
    QT = getattr(fb, 'QuantizationParametersT', None)
    if QT is None:
        raise RuntimeError('schema lacks QuantizationParametersT')
    qattr = _attr(tensors[0], 'quantization', 'quantization')

    filled = 0
    for tt in tensors:
        q = getattr(tt, qattr)
        if q is not None and getattr(q, 'scale', None) is not None \
                and len(q.scale) > 0:
            continue
        nq = QT()
        # numpy array 与 python list 都能被 flatbuffers 打包
        nq.scale = [NEUTRAL_SCALE]
        nq.zero_point = [NEUTRAL_ZP]
        nq.quantized_dimension = 0
        setattr(tt, qattr, nq)
        filled += 1
    print('QUANT-PARAM: filled %d tensor(s) with scale=%g zp=%d'
          % (filled, NEUTRAL_SCALE, NEUTRAL_ZP), flush=True)
    if filled == 0:
        print('WARNING: no tensor needed quantization params -- input model '
              'may already be fully quantized; the control experiment would '
              'be inconclusive.', flush=True)

    # ---- 2. 可选：PADV2 -> PAD ----------------------------------------
    n_padv2 = 0
    if also_fix_padv2:
        oi = _attr(ops[0], 'opcode_index', 'opcodeIndex')
        bt = _attr(ops[0], 'builtin_options_type', 'builtinOptionsType')
        bo = _attr(ops[0], 'builtin_options', 'builtinOptions')
        pad_idx = None
        for i, oc in enumerate(codes):
            if eff_code(oc, fb) == CODE_PAD:
                pad_idx = i
                break
        ref = None
        for op in ops:
            if eff_code(codes[getattr(op, oi)], fb) == CODE_PAD:
                ref = op
                break
        if pad_idx is not None and ref is not None:
            for op in ops:
                if eff_code(codes[getattr(op, oi)], fb) != CODE_PADV2:
                    continue
                op.inputs = op.inputs[:2]      # numpy array: 用切片不用 del
                setattr(op, oi, pad_idx)
                setattr(op, bt, getattr(ref, bt))
                setattr(op, bo, getattr(ref, bo))
                n_padv2 += 1
            print('PADV2->PAD: rewrote %d op(s)' % n_padv2, flush=True)
        else:
            print('PADV2->PAD: no PAD reference found, skipped', flush=True)

    out = pack_and_check(fb, model, expect_padv2=0)
    if dry_run:
        print('dry-run: not writing (pack + revalidate OK)', flush=True)
        return len(out)

    with open(dst, 'wb') as f:
        f.write(out)
    print('WROTE %s (%d bytes, was %d)' % (dst, len(out), len(raw)),
          flush=True)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--survey-only', action='store_true')
    ap.add_argument('--no-fix-padv2', action='store_true',
                    help='只补量化参数，不动 PADV2（纯对照变量）')
    args = ap.parse_args()

    print('=== STAGE-P0: src exists=%s ==='
          % os.path.exists(args.src), flush=True)
    if args.survey_only:
        survey(args.src)
        return 0
    print('=== STAGE-P1: survey before ===', flush=True)
    survey(args.src)
    print('=== STAGE-P2: rewrite ===', flush=True)
    rewrite(args.src, args.dst, args.dry_run, not args.no_fix_padv2)
    if not args.dry_run:
        print('=== STAGE-P3: verify after ===', flush=True)
        survey(args.dst)
    print('=== STAGE-P4: done ===', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())