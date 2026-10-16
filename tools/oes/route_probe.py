#!/usr/bin/env python3
"""TFLite 产出路线探针：一次性判定哪条路能真出 INT8。

## 背景
ORT QDQ 已达标（清洗后 covered_hi 35.2 vs 污染时 812.4，cos=0.9569 见
CI run 38872766517），卡在 QDQ ONNX -> TFLite：
onnx2tf 1.29.24 把 149Q/251DQ 压成全 FLOAT32（量化张量 0/337）。
源码级证据：onnx2tf/ops/DequantizeLinear.py 逐节点翻译成 TF float 伪量化
算子，common_functions.py 对 QDQ 零引用 -> QDQ 不被保留为真 int8。

## 探测四条路线（用小图，逐条给结论）
  R1 onnx2tf + TFLiteConverter.from_saved_model
     onnx2tf 只当 ONNX->SavedModel 转换器（其已验证 cos=1.0），量化交
     给 TFLiteConverter（自带 MLIR 量化器 + representative_dataset）。
  R2 onnx2tf 直出 + Optimize.DEFAULT
     onnx2tf 产出后再让 converter 自己做 int8 校准。
  R3 ORT quantize_static 直出 .tflite
     model_output 指向 .tflite，ORT 内部走受控转换。
  R4 QDQ ONNX -> onnx2tf -> 检查是否 int8（复现已知失败，做对照）

每条路线都必须报：量化张量数/总张量数、输入 dtype、dtype 直方图、文件大小。
判定标准（与 regress 的 --require-int8 一致）：
  量化张量数 > 0 且 图输入为 INT8/UINT8 -> 该路线可用

用法：python route_probe.py <outdir>
"""
import glob
import os
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out/route'
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 小图
def make_float_onnx(path, n=64):
    """构造一个小 float ONNX（与 concept 同构：Conv-ReLU 堆叠 + GAP + FC +
    Sigmoid），供各路线量化用。尺寸取小是为了探针快。"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 16, 16, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8])
    rs = np.random.RandomState(0)
    inits, nodes = [], []
    prev = 'X'
    chans = [8, 16, 24]
    for i, c in enumerate(chans):
        w = rs.randn(c, chans[i - 1] if i else 3, 3, 3).astype(np.float32) * 0.05
        b = rs.randn(c).astype(np.float32) * 0.01
        inits.append(helper.make_tensor('w%d' % i, TensorProto.FLOAT,
                                        list(w.shape), w.ravel()))
        inits.append(helper.make_tensor('b%d' % i, TensorProto.FLOAT, [c],
                                        b))
        nodes.append(helper.make_node('Conv', [prev, 'w%d' % i, 'b%d' % i],
                                      ['c%d' % i]))
        nodes.append(helper.make_node('Relu', ['c%d' % i], ['r%d' % i]))
        prev = 'r%d' % i
    nodes.append(helper.make_node('GlobalAveragePool', [prev], ['g']))
    nodes.append(helper.make_node('Flatten', ['g'], ['f'], axis=1))
    gf = rs.randn(8, chans[-1]).astype(np.float32) * 0.05
    inits.append(helper.make_tensor('gw', TensorProto.FLOAT,
                                    list(gf.shape), gf.ravel()))
    nodes.append(helper.make_node('Gemm', ['f', 'gw'], ['o'], transB=1))
    nodes.append(helper.make_node('Sigmoid', ['o'], ['y']))
    g = helper.make_graph(nodes, 'mini', [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return path, 'X', 16


class Reader:
    """校准/代表数据：随机但可复现。"""

    def __init__(self, input_name, size, n=16):
        self.nm, self.sz, self.n, self.i = input_name, size, n, 0

    def get_next(self):
        if self.i >= self.n:
            return None
        a = np.random.RandomState(1000 + self.i).randn(
            1, self.sz, self.sz, 3).astype(np.float32)
        self.i += 1
        return {self.nm: a}

    def rewind(self):
        self.i = 0

    def __len__(self):
        return self.n


# ---------------------------------------------------------------- 报告
TFL_NAME = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8',
            4: 'INT64', 6: 'BOOL', 7: 'INT16', 9: 'INT8'}


def report(tag, path):
    import tflite
    if not os.path.exists(path):
        print('  [%s] NOT PRODUCED' % tag)
        return False
    import collections
    raw = open(path, 'rb').read()
    mo = tflite.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    nt = g.TensorsLength()
    nq = sum(1 for i in range(nt)
             if (q := g.Tensors(i).Quantization()) is not None
             and q.ScaleLength() > 0)
    tin = g.Tensors(g.Inputs(0))
    dt = collections.Counter(TFL_NAME.get(g.Tensors(i).Type(),
                                         'T%d' % g.Tensors(i).Type())
                             for i in range(nt))
    ok = nq > 0 and tin.Type() in (3, 9)
    print('  [%s] size=%d in_type=%s quantized=%d/%d (%.1f%%)' % (
        tag, len(raw), TFL_NAME.get(tin.Type(), tin.Type()), nq, nt,
        100.0 * nq / max(nt, 1)))
    print('        dtype: %s' % dict(sorted(dt.items())))
    print('        -> %s' % ('INT8 OK' if ok else 'FLOAT (FAILED)'))
    return ok


# ---------------------------------------------------------------- 路线
def r1_o2tf_then_converter(float_onnx, iname, sz):
    """onnx2tf -> SavedModel -> TFLiteConverter(INT8)。"""
    import tensorflow as tf
    d = os.path.join(OUT, 'r1_sm')
    r = subprocess.run([sys.executable, '-m', 'onnx2tf', '-i', float_onnx,
                        '-o', d, '--non_verbose', '-kt', iname],
                       capture_output=True, text=True)
    if r.returncode:
        print('  [R1] onnx2tf rc=%d: %s' % (r.returncode,
                                           (r.stderr or r.stdout)[-200:]))
        return False
    sm = os.path.join(d, 'float32')
    if not os.path.isdir(sm):
        print('  [R1] no saved_model dir')
        return False
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: Reader(iname, sz).__iter__()
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        out = os.path.join(OUT, 'r1.tflite')
        open(out, 'wb').write(conv.convert())
    except Exception as e:
        print('  [R1] converter failed: %s' % str(e)[:200])
        return False
    return report('R1 o2tf+Converter', out)


def r2_ort_direct(float_onnx, iname, sz):
    """ORT quantize_static 直出 .tflite。"""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    out = os.path.join(OUT, 'r2.tflite')
    try:
        quantize_static(float_onnx, out, Reader(iname, sz),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True, reduce_range=False,
                        nodes_to_exclude=[],
                        calibrate_method=CalibrationMethod.MinMax)
    except Exception as e:
        print('  [R2] raised: %s' % str(e)[:220])
        return False
    return report('R2 ORT->tflite', out)


def r3_ort_qop_direct(float_onnx, iname, sz):
    """ORT QOperator 格式 + 直出 tflite。"""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    out = os.path.join(OUT, 'r3.tflite')
    try:
        quantize_static(float_onnx, out, Reader(iname, sz),
                        quant_format=QuantFormat.QOperator,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True, reduce_range=False,
                        nodes_to_exclude=[],
                        calibrate_method=CalibrationMethod.MinMax)
    except Exception as e:
        print('  [R3] raised: %s' % str(e)[:220])
        return False
    return report('R3 ORT-QOp->tflite', out)


def r4_control_o2tf_qdq(float_onnx, iname, sz):
    """对照：先 ORT 出 QDQ onnx，再 onnx2tf 看是否 int8（复现已知失败）。"""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    q = os.path.join(OUT, 'r4_qdq.onnx')
    quantize_static(float_onnx, q, Reader(iname, sz),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True, reduce_range=False,
                    nodes_to_exclude=[],
                    calibrate_method=CalibrationMethod.MinMax)
    import collections
    m = onnx.load(q)
    nq = sum(1 for n in m.graph.node
             if n.op_type in ('QuantizeLinear', 'DequantizeLinear'))
    print('  [R4] QDQ onnx has %d Q/DQ nodes' % nq)
    d = os.path.join(OUT, 'r4_sm')
    r = subprocess.run([sys.executable, '-m', 'onnx2tf', '-i', q, '-o', d,
                        '--non_verbose', '-kt', iname],
                       capture_output=True, text=True)
    if r.returncode:
        print('  [R4] onnx2tf rc=%d' % r.returncode)
        return False
    cands = glob.glob(os.path.join(d, '*.tflite'))
    if not cands:
        print('  [R4] no tflite')
        return False
    return report('R4 o2tf(QDQ) control', sorted(cands, key=os.path.getsize)[-1])


def main():
    fp, iname, sz = make_float_onnx(os.path.join(OUT, 'mini.onnx'))
    m = onnx.load(fp)
    print('mini float onnx: %d nodes' % len(m.graph.node))

    res = {}
    print('\n[R1] onnx2tf -> SavedModel -> TFLiteConverter(INT8):')
    res['R1'] = r1_o2tf_then_converter(fp, iname, sz)
    print('\n[R2] ORT quantize_static -> .tflite (QDQ):')
    res['R2'] = r2_ort_direct(fp, iname, sz)
    print('\n[R3] ORT quantize_static -> .tflite (QOperator):')
    res['R3'] = r3_ort_qop_direct(fp, iname, sz)
    print('\n[R4] control: ORT-QDQ -> onnx2tf (expect FLOAT):')
    res['R4'] = r4_control_o2tf_qdq(fp, iname, sz)

    print('\n=== ROUTE VERDICT ===')
    good = [k for k, v in res.items() if v]
    for k in sorted(res):
        print('  %s -> %s' % (k, 'INT8 OK' if res[k] else 'float/failed'))
    print('USABLE ROUTES: %s' % (good or 'NONE'))
    if good:
        print('CHOOSE: %s' % good[0])


if __name__ == '__main__':
    main()