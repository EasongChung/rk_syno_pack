#!/usr/bin/env python3
"""OES concept PTQ：model_float16.tflite -> concept_int8.tflite（Teflon 路线）.

管线（CI 内一站式）：
  .tflite --tf2onnx--> ONNX --smoke(onnxruntime)--> --onnx2tf--> SavedModel
  --TFLiteConverter PTQ--> INT8 .tflite（全整型 uint8 I/O + float-io 对照版）

铁律（与 OEC RKNN 线的区别，理由见 tools/oes/README_oes.md）：
  - 全程保持 NHWC [1,395,395,3]，**不插 NCHW-Transpose**。
  - 禁用 tflite2onnx（Inception 多分支错图）。
  - 版本钉死：tensorflow-cpu==2.15.1 + tf2onnx==1.16.1（OEC 验证组合）。

用法：
  python convert_to_int8.py --src tools/oes/models/model_float16.tflite \\
      --workdir work --outdir out --calib-dir calib --num-calib 300
"""
import argparse
import glob
import os
import subprocess
import sys

NET = 395
MEAN = 127.5
SCALE = 0.00784313725  # 2/255，与服务端预处理一致


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('$ %s' % ' '.join(cmd), flush=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr, flush=True)
        raise SystemExit('command failed: %s' % ' '.join(cmd[:4]))
    return r


def stage_tflite_to_onnx(src, onnx_path):
    sh([sys.executable, '-m', 'tf2onnx.convert', '--tflite', src,
        '--output', onnx_path, '--opset', '13'])
    assert os.path.exists(onnx_path), 'tf2onnx produced no onnx'
    import onnx
    m = onnx.load(onnx_path)
    g = m.graph
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    print('onnx input=%s dims=%s' % (g.input[0].name, dims), flush=True)
    # ★ 断言 NHWC，拒绝 NCHW：OES/Teflon 原生 NHWC，不插 Transpose
    assert len(dims) == 4 and dims[0] == 1 and dims[3] == 3 and dims[1] == 395, \
        'expect NHWC [1,395,395,3], got %s (do NOT insert NCHW Transpose)' % dims
    assert not any(n.op_type == 'Transpose' and n.name == 'nchw_to_nhwc'
                   for n in g.node), 'foreign NCHW Transpose detected, abort'
    n_transpose = sum(1 for n in g.node if n.op_type == 'Transpose')
    print('onnx nodes=%d transpose_in_graph=%d (model-native, kept)' % (
        len(g.node), n_transpose), flush=True)
    return onnx_path


def stage_smoke(onnx_path):
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    shape = [d if isinstance(d, int) and d > 0 else 1
             for d in sess.get_inputs()[0].shape]
    dummy = np.random.rand(*shape).astype(np.float32)
    out = sess.run(None, {name: dummy})
    assert np.isfinite(out[0]).all(), 'onnx smoke produced non-finite output'
    print('smoke OK: out_shapes=%s' % [o.shape for o in out], flush=True)
    del sess


def stage_onnx_to_savedmodel(onnx_path, sm_dir):
    sh([sys.executable, '-m', 'onnx2tf', '-i', onnx_path, '-o', sm_dir,
        '--non_verbose'])
    assert os.path.isdir(sm_dir), 'onnx2tf produced no SavedModel'


def representative_dataset(calib_dir, num, float_io):
    from PIL import Image
    import numpy as np
    files = sorted(glob.glob(os.path.join(calib_dir, '*')))
    files = [f for f in files
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    print('calib files available: %d (need %d)' % (len(files), num), flush=True)
    assert len(files) >= min(num, 50), 'too few calibration images'
    rng = np.random.default_rng(7)

    def gen():
        n = 0
        for f in files:
            try:
                img = Image.open(f).convert('RGB').resize(
                    (NET, NET), Image.BILINEAR)
            except Exception as e:
                print('skip bad image %s: %s' % (f, e), flush=True)
                continue
            arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
            if float_io:
                arr = (arr - MEAN) * SCALE
            yield [arr.reshape(1, NET, NET, 3)]
            n += 1
            if n >= num:
                return
        while n < num:  # 合成兜底（报告中注明 synthetic_fill=N）
            base = rng.normal(117.0, 60.0, size=(NET, NET, 3)).clip(0, 255)
            if float_io:
                base = (base - MEAN) * SCALE
            yield [base.astype(np.float32).reshape(1, NET, NET, 3)]
            n += 1
        print('representative_dataset: real=%d synthetic_fill=%d' % (
            min(len(files), num), max(0, num - len(files))), flush=True)
    return gen


def stage_quant(sm_dir, out_path, calib_dir, num_calib, float_io):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(
        calib_dir, num_calib, float_io)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if float_io:
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
    else:
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
    model = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(model)
    print('WROTE %s %d bytes (float_io=%s)' % (out_path, len(model), float_io),
          flush=True)
    # 产物自检：输入 NHWC + 输出 335
    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    ide = interp.get_input_details()[0]
    ode = interp.get_output_details()[0]
    print('quant model in shape=%s dtype=%s / out shape=%s dtype=%s' % (
        ide['shape'], ide['dtype'], ode['shape'], ode['dtype']), flush=True)
    assert list(ide['shape']) == [1, 395, 395, 3], 'input must stay NHWC'
    assert list(ode['shape']) == [1, 335], 'output must be 335'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--calib-dir', default='calib')
    ap.add_argument('--num-calib', type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    onnx_path = os.path.join(args.workdir, 'concept_mid.onnx')
    sm_dir = os.path.join(args.workdir, 'concept_float_sd')
    stage_tflite_to_onnx(args.src, onnx_path)
    stage_smoke(onnx_path)
    stage_onnx_to_savedmodel(onnx_path, sm_dir)
    stage_quant(sm_dir, os.path.join(args.outdir, 'concept_int8.tflite'),
                args.calib_dir, args.num_calib, float_io=False)
    stage_quant(sm_dir, os.path.join(args.outdir, 'concept_int8_floatio.tflite'),
                args.calib_dir, args.num_calib, float_io=True)
    print('CONVERT DONE', flush=True)


if __name__ == '__main__':
    main()
