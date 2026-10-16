#!/usr/bin/env python3
"""最小 QDQ -> TFLite 探针：先定生死，再谈大图。

背景：ORT QDQ 已达标（清洗后 covered_hi 35.2 vs 污染时 812.4），
但 onnx2tf 1.29.24 输出的 TFLite 里量化张量为 0（全 FLOAT32）。
在 544 节点大图上盲调参数代价高，故先用 11 节点小图判定：
  小图能出 int8 -> onnx2tf 支持 QDQ，问题在大图规模/预处理，值得继续调
  小图出 float  -> onnx2tf 不支持 QDQ 输入，必须换路线
                 （TFLiteConverter.from_onnx_model 或 ORT 直出 tflite）

同时探测路线 B：ORT quantize_static 能否把 model_output 指向 .tflite。

用法：python min_probe.py <outdir>
"""
import glob
import os
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out/minprobe'
os.makedirs(OUT, exist_ok=True)


def make_qdq_onnx(path):
    """最小 QDQ 图：X -Q-> -DQ-> Conv -Relu-> -Q-> -DQ-> Reshape -> Gemm
    -> Sigmoid -Q-> -DQ-> Y。含 per-channel 权重与 per-tensor 激活两类量化。"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8, 8, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    rs = np.random.RandomState(0)
    w = rs.randn(4, 3, 3, 3).astype(np.float32) * 0.1
    b = np.zeros(4, dtype=np.float32)
    wq = np.clip(np.round(w / (abs(w).max() / 127.0)), -128, 127).astype(np.int8)
    bq = np.zeros(4, dtype=np.int8)
    wsc = np.array([abs(w).max() / 127.0], dtype=np.float32)

    def t(name, dt, shape, vals):
        return helper.make_tensor(name, dt, shape,
                                 np.asarray(vals).ravel())

    inits = [
        t('wq', TensorProto.INT8, [4, 3, 3, 3], wq),
        t('bq', TensorProto.INT8, [4], bq),
        t('ws', TensorProto.FLOAT, [4], np.repeat(wsc, 4)),   # per-channel
        t('bs', TensorProto.FLOAT, [1], [1.0]),
        t('shape', TensorProto.INT64, [2], [1, 64]),
        t('gw', TensorProto.FLOAT, [2, 64], rs.randn(2, 64).astype(np.float32) * 0.1),
        t('as', TensorProto.FLOAT, [1], [0.05]),               # 激活 scale
        t('az', TensorProto.INT32, [1], [-128]),               # int8 zero point
        t('os', TensorProto.FLOAT, [1], [0.01]),              # 输出 scale
        t('oz', TensorProto.INT32, [1], [-128]),
    ]
    nodes = [
        helper.make_node('QuantizeLinear', ['X', 'as', 'az'], ['xq'], axis=1),
        helper.make_node('DequantizeLinear', ['xq', 'as', 'az'], ['xdq']),
        helper.make_node('Conv', ['xdq', 'wq', 'bq'], ['c'], axis=1),
        helper.make_node('Relu', ['c'], ['r']),
        helper.make_node('QuantizeLinear', ['r', 'as', 'az'], ['rq'], axis=1),
        helper.make_node('DequantizeLinear', ['rq', 'as', 'az'], ['rdq']),
        helper.make_node('Reshape', ['rdq', 'shape'], ['f'], axis=0),
        helper.make_node('Gemm', ['f', 'gw'], ['g'], transB=1),
        helper.make_node('Sigmoid', ['g'], ['s']),
        helper.make_node('QuantizeLinear', ['s', 'os', 'oz'], ['sq'], axis=1),
        helper.make_node('DequantizeLinear', ['sq', 'os', 'oz'], ['Y']),
    ]
    g = helper.make_graph(nodes, 'minqdq', [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


def report(tag, cand):
    import tflite
    raw = open(cand, 'rb').read()
    mo = tflite.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    nt = g.TensorsLength()
    nq = sum(1 for i in range(nt)
             if (q := g.Tensors(i).Quantization()) is not None
             and q.ScaleLength() > 0)
    tin = g.Tensors(g.Inputs(0))
    ok = nq > 0 and tin.Type() in (3, 9)
    print('  [%s] size=%d in_type=%d quantized=%d/%d -> %s' % (
        tag, len(raw), tin.Type(), nq, nt,
        'INT8 OK' if ok else 'FLOAT (FAILED)'))
    return ok


def probe_onnx2tf(p):
    d = os.path.join(OUT, 'o2tf')
    r = subprocess.run([sys.executable, '-m', 'onnx2tf', '-i', p, '-o', d,
                        '--non_verbose'], capture_output=True, text=True)
    if r.returncode:
        print('  onnx2tf rc=%d: %s' % (r.returncode,
                                       (r.stderr or r.stdout)[-260:]))
        return False
    cands = glob.glob(os.path.join(d, '*.tflite'))
    if not cands:
        print('  onnx2tf produced no tflite')
        return False
    return report('onnx2tf', sorted(cands, key=os.path.getsize)[-1])


def probe_ort_direct(p):
    """路线 B：ORT quantize_static 的 model_output 指向 .tflite。"""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class R:
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= 8:
                return None
            self.i += 1
            return {'X': np.random.RandomState(self.i)
                    .randn(1, 8, 8, 3).astype(np.float32)}

        def rewind(self):
            self.i = 0

    out = os.path.join(OUT, 'ort_direct.tflite')
    try:
        quantize_static(p, out, R(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, per_channel=True,
                        reduce_range=False, nodes_to_exclude=[],
                        calibrate_method=CalibrationMethod.MinMax)
    except Exception as e:
        print('  ORT->tflite raised: %s' % str(e)[:200])
        return False
    if not os.path.exists(out):
        print('  ORT->tflite produced nothing')
        return False
    return report('ort-direct', out)


def main():
    p = make_qdq_onnx(os.path.join(OUT, 'min_qdq.onnx'))
    m = onnx.load(p)
    qn = sum(1 for n in m.graph.node
             if n.op_type in ('QuantizeLinear', 'DequantizeLinear'))
    print('MIN-PROBE onnx: %d nodes (%d Q/DQ)' % (len(m.graph.node), qn))
    print('VERDICT-QDQ-PRESENT: %s' % ('YES' if qn >= 8 else 'NO'))

    print('\n[Q1] onnx2tf on small QDQ graph:')
    a = probe_onnx2tf(p)
    print('\n[Q2] ORT quantize_static -> .tflite (route B):')
    b = probe_ort_direct(p)

    print('\n=== VERDICT ===')
    print('onnx2tf preserves QDQ : %s' % ('YES' if a else 'NO'))
    print('ORT emits tflite direct: %s' % ('YES' if b else 'NO'))
    if not a:
        print('ACTION: abandon onnx2tf for QDQ; use TFLiteConverter'
              '.from_onnx_model (route A).')


if __name__ == '__main__':
    main()