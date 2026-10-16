#!/usr/bin/env python3
"""onnx2tf 整数量化输出：用 convert() 的 output_integer_quantized_tflite 开关。

发现（读 onnx2tf 1.29.24 源码）：
  CLI 无量化参数（__main__.py 走 argparse，但量化开关不在其中），
  而 convert() 签名里有：
    output_integer_quantized_tflite : 产出全整数(int8) TFLite 的总开关
    quant_type / input_quant_dtype / output_quant_dtype
    quant_norm_mean / quant_norm_std
  ops/QuantizeLinear.py 用 fake_quant 实现 Q 节点，ops/DequantizeLinear.py
  同理 -> 不带该开关时 QDQ 被翻成 float 伪量化，TFLite 里量化张量为 0。
  故必须走 convert() 直调，不能只靠命令行。

用法：
  python qdq_to_tflite_int.py --qdq work/concept_ort_qdq_clean.onnx \
      --out out/concept_ortq_clean.tflite [--quant-type int8|uint8]
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np


def _patch_for_onnx2tf():
    """复用 run_onnx2tf 的两个补丁（np.load pickle / 测试图下载）。"""
    import numpy as _np
    orig = _np.load

    def patched(*a, **k):
        k.setdefault('allow_pickle', True)
        return orig(*a, **k)

    _np.load = patched
    try:
        import importlib
        mod = importlib.import_module('onnx2tf.utils.common_functions')
        if hasattr(mod, 'download_test_image_data'):
            _o = mod.download_test_image_data

            def safe(*a, **k):
                try:
                    return _o(*a, **k)
                except Exception:
                    rng = _np.random.default_rng(0)
                    return (rng.random((395, 395, 3)) * 255).astype(_np.uint8)

            mod.download_test_image_data = safe
    except Exception as e:
        print('no common_functions patch: %s' % e, flush=True)


TFL_NAME = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8', 4: 'INT64',
            6: 'BOOL', 7: 'INT16', 9: 'INT8'}


def report(cand, tag):
    import collections
    import tflite
    raw = open(cand, 'rb').read()
    mo = tflite.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    nt = g.TensorsLength()
    nq = sum(1 for i in range(nt)
             if (q := g.Tensors(i).Quantization()) is not None
             and q.ScaleLength() > 0)
    tin = g.Tensors(g.Inputs(0))
    dt = collections.Counter(TFL_NAME.get(g.Tensors(i).Type(), '?')
                             for i in range(nt))
    ops = collections.Counter()
    codes = {}
    for i in range(mo.OperatorCodesLength()):
        oc = mo.OperatorCodes(i)
        codes[i] = (oc.BuiltinCode() if oc.BuiltinCode() < 127
                    else oc.DeprecatedBuiltinCode())
    for i in range(g.OperatorsLength()):
        ops[codes.get(g.Operators(i).OpcodeIndex())] += 1
    print('%s: in_type=%s quantized=%d/%d size=%d' % (
        tag, TFL_NAME.get(tin.Type(), tin.Type()), nq, nt, len(raw)))
    print('  dtype: %s' % dict(sorted(dt.items())))
    print('  ops:   %s' % dict(sorted(ops.items(), key=lambda kv: -kv[1])[:10]))
    ok = nq > 0 and tin.Type() in (3, 9)
    print('  -> %s' % ('INT8 OK' if ok else 'FLOAT (FAILED)'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qdq', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--quant-type', default='per-channel',
                    choices=['per-channel', 'per-tensor'],
                    help='onnx2tf 的量化粒度（源码实测 choices 如此，'
                         '注意不是 tf dtype）')
    ap.add_argument('--keep-nchw', action='store_true')
    args = ap.parse_args()

    _patch_for_onnx2tf()
    from onnx2tf import convert

    os.makedirs(args.workdir, exist_ok=True)
    odir = os.path.join(args.workdir, 'o2tf_int')
    # 注意：onnx2tf 的 quant_type 是 'per-channel'/'per-tensor' 字符串
    # （量化粒度选择），不是 tf dtype 编号。整型 dtype 由
    # input_quant_dtype / output_quant_dtype 控制。
    qt = args.quant_type

    kw = dict(
        input_onnx_file_path=args.qdq,
        output_folder_path=odir,
        output_signaturedefs=False,
        copy_onnx_input_output_names_to_tflite=True,
        output_integer_quantized_tflite=True,     # ← 关键开关
        quant_type=qt,
        input_quant_dtype='int8',                  # 源码默认即 int8
        output_quant_dtype='int8',
        non_verbose=True,
        not_use_onnxsim=False,
    )
    print('calling onnx2tf.convert with: integer_quantized=%s quant_type=%s '
          'dtype=int8' % (True, args.quant_type), flush=True)
    try:
        convert(**kw)
    except TypeError as e:
        # 老版本参数名不同，退回最小签名再试
        print('convert(**kw) TypeError (%s); retry minimal' % str(e)[:120],
              flush=True)
        convert(input_onnx_file_path=args.qdq,
                output_folder_path=odir,
                output_integer_quantized_tflite=True,
                non_verbose=True)

    cands = glob.glob(os.path.join(odir, '*.tflite'))
    if not cands:
        cands = glob.glob(os.path.join(odir, '*_int8.tflite')) or \
            glob.glob(os.path.join(odir, '**', '*.tflite'), recursive=True)
    if not cands:
        print('NO TFLITE produced; dir listing: %s' % os.listdir(odir))
        return 4
    cand = sorted(cands, key=os.path.getsize, reverse=True)[0]
    print('picked: %s (%d bytes)' % (cand, os.path.getsize(cand)), flush=True)
    ok = report(cand, 'o2tf-int')
    if not ok:
        return 4
    import shutil
    shutil.copy(cand, args.out)
    print('WROTE %s' % args.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())