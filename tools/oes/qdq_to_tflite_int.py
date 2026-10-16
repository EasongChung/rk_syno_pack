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
import sys as _sys

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


def build_calib_npy(calib_dir, num, name, out_dir=None, net=395):
    """为 onnx2tf 的 INT8 校准生成 (N,395,395,3) float32 npy。

    预处理必须与服务端、与 ort_quant 完全一致：resize(395,BILINEAR)
    + (v-127.5)/127.5，否则 onnx2tf 校准出的 scale 与实际推理分布不符。
    """
    from PIL import Image
    files = sorted(f for f in os.listdir(calib_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))[:num]
    assert files, 'calib dir empty: %s' % calib_dir
    arr = np.zeros((len(files), net, net, 3), dtype=np.float32)
    for i, f in enumerate(files):
        img = Image.open(os.path.join(calib_dir, f)) \
            .convert('RGB').resize((net, net), Image.BILINEAR)
        v = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        arr[i] = ((v - 127.5) * 0.00784313725).reshape(net, net, 3)
    d = out_dir or os.path.join('work', 'o2tf_calib')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'calib_%s.npy' % name)
    np.save(p, arr)
    return p


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
    ap.add_argument('--calib-dir', default='',
                    help='INT8 校准图目录（onnx2tf 要求自带校准数据：'
                         'custom_input_op_name_np_data_path）')
    ap.add_argument('--calib-num', type=int, default=50)
    ap.add_argument('--calib-name', default='inputs_0',
                    help='校准 npy 对应的输入张量名')
    ap.add_argument('--clean-k', type=float, default=3.0,
                    help='校准集离群剔除阈值（mag > k * median）')
    args = ap.parse_args()
    print('=== STAGE0: args parsed, qdq=%s exists=%s ===' % (
        args.qdq, os.path.exists(args.qdq)), flush=True)
    _sys.stdout.flush()

    # 注意：TF import 之前必须先打印，否则 import 失败时 CI 日志一片空白
    # （上一轮就是这样：子进程 stdout 全空，无法判断死在哪一步）。
    try:
        import tensorflow as tf
        print('=== STAGE1: tensorflow %s ===' % tf.__version__, flush=True)
        _sys.stdout.flush()
    except Exception as e:
        print('=== STAGE1 FAILED: tensorflow import: %s: %s ===' % (
            type(e).__name__, e), flush=True)
        _sys.stdout.flush()
        raise

    _patch_for_onnx2tf()
    try:
        from onnx2tf import convert
        print('=== STAGE2: onnx2tf imported ===', flush=True)
        _sys.stdout.flush()
    except Exception as e:
        print('=== STAGE2 FAILED: onnx2tf import: %s: %s ===' % (
            type(e).__name__, e), flush=True)
        _sys.stdout.flush()
        raise

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
        # 关闭 onnx2tf 内部所有"用 ORT 跑一遍图"的环节。
        # 原因（CI run 38895398097 实测）：转换本身已成功
        # （Saved artifact at 'work/o2tf_int'），但随后的 ORT 复核在
        # onnx2tf 动态生成的 `const_fold_opt__630_DequantizeLinear/duplicated`
        # 上崩溃：IsAxisInRange(axis, tensor_rank) was false, axis 1 not in
        # [-1,0]。本地审计原始 QDQ 图 251 个 DQ 全部 axis 合法（0 越界），
        # 故越界来自 onnx2tf 生成的副本，而非我们的输入图。
        # 该复核只影响精度验证报告，不影响产物本身。
        disable_strict_mode=True,
        check_gpu_delegate_compatibility=False,
        check_onnx_tf_outputs_elementwise_close=False,
        check_onnx_tf_outputs_elementwise_close_full=False,
        auto_generate_json=False,
        auto_generate_json_on_error=False,
    )
    # onnx2tf 不信任 QDQ 图里已有的 scale：开 output_integer_quantized_tflite
    # 后它会自己做一遍 INT8 校准，并强制要求提供校准数据，否则报
    #   "For models that have multiple input OPs and need to perform INT8
    #    quantization calibration ... specify the calibration data with
    #    --quant_calib_input_op_name_np_data_path"
    # 参数名（源码）：custom_input_op_name_np_data_path
    # 数据格式：{输入张量名: (N,H,W,3) float32 npy}
    if args.calib_dir:
        # 关键：onnx2tf 会用这份数据重新校准。若喂未清洗的集（含把激活
        # 推到 1633 的脏图），它会算回和 QDQ 图一样被撑爆的 scale，
        # 白费清洗成果。故强制走 ort_quant.clean_calib_files。
        # 注意：清洗很慢（逐图跑 float 子模型），故先打印，避免
        # 失败时看不到任何上下文。
        print('CLEAN-CALIB: start (dir=%s num*4=%d k=%.1f)' % (
            args.calib_dir, args.calib_num * 4, args.clean_k), flush=True)
        _sys.stdout.flush()
        import sys as _s
        _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ort_quant as _oq
        try:
            keep = _oq.clean_calib_files(args.qdq, args.calib_dir,
                                         args.calib_num * 4, args.clean_k,
                                         os.path.join(args.workdir, 'calib'))
        except Exception as e:
            print('CLEAN-CALIB failed (%s: %s); '
                  'falling back to raw calib dir' % (type(e).__name__,
                                                     str(e)[:160]), flush=True)
            _sys.stdout.flush()
            keep = None
        tmpdir = os.path.join(args.workdir, 'calib_clean')
        if os.path.isdir(tmpdir):
            import shutil as _sh2
            _sh2.rmtree(tmpdir, ignore_errors=True)
        os.makedirs(tmpdir, exist_ok=True)
        import shutil as _sh
        srcs = keep if keep else sorted(
            os.path.join(args.calib_dir, f) for f in os.listdir(args.calib_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for f in srcs[:args.calib_num]:
            _sh.copy(f, os.path.join(tmpdir, os.path.basename(f)))
        print('CLEAN-CALIB: prepared %d calibration images' %
              len(os.listdir(tmpdir)), flush=True)
        _sys.stdout.flush()
        npy = build_calib_npy(tmpdir, args.calib_num, args.calib_name,
                              out_dir=os.path.join(args.workdir,
                                                   'o2tf_calib'))
        # 源码实测：走 -oiqt 分支时必须给全 4 项，否则报
        #   "If you want to use custom input with the '-oiqt' option,
        #    {input_op_name}, {numpy_file_path}, {mean}, and {std}
        #    must all be entered. However, you have only entered 2 options."
        # 外层校验（L1192-1198）要求每项 len in (2,4)，故给 4 项。
        # mean/std 用 0/1：npy 已是归一化后的 [-1,1]，不再二次归一化。
        kw['custom_input_op_name_np_data_path'] = [[args.calib_name, npy,
                                                  '0.0', '1.0']]
        print('calib npy: %s shape=%s (from %d cleaned images)' % (
            npy, np.load(npy).shape, len(os.listdir(tmpdir))), flush=True)
        print('cind arg: %s' % kw['custom_input_op_name_np_data_path'],
              flush=True)
        _sys.stdout.flush()
    else:
        print('WARNING: no --calib-dir given; onnx2tf INT8 calibration '
              'will likely fail', flush=True)
    print('=== STAGE4: calling onnx2tf.convert ===', flush=True)
    _sys.stdout.flush()
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