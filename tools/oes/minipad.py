#!/usr/bin/env python3
"""最小 Pad 实验：转换器对 Pad(const=0) 生成几输入形态 + 标量尺度形态。

背景：concept INT8 的退化标量来自 onnx2tf 的 3 输入 TF-Pad lowering；
本实验确认原生 TF->TFLite 管线是否也会产生 3 输入 Pad（含标量），
以及标量张量的量化形态。纯结构审计 + 基线数值，不做破坏性改写。

输出 minipad/minipad_report.json。
"""
import os

import numpy as np


def build_float(sm_dir, explicit_scalar):
    import tensorflow as tf

    class M(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([1, 4, 4, 2], tf.float32)])
        def __call__(self, x):
            c = tf.constant(0.0, dtype=tf.float32) if explicit_scalar else 0.0
            return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]],
                          constant_values=c)

    tf.saved_model.save(M(), sm_dir)
    return sm_dir


def quantize(sm_dir, out_path):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(0)

    def gen():
        for _ in range(20):
            yield [rng.random((1, 4, 4, 2)).astype(np.float32) * 2 - 1]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    with open(out_path, 'wb') as f:
        f.write(conv.convert())
    print('WROTE %s' % out_path, flush=True)


def run_model(path, x):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_to_int8 import make_interpreter
    it = make_interpreter(path)['interp']
    ide = it.get_input_details()[0]
    if ide['dtype'] == np.uint8:
        q = ide['quantization_parameters']
        sc, zp = float(q['scales'][0]), int(q['zero_points'][0])
        xq = np.clip(np.round(x / sc) + zp, 0, 255).astype(np.uint8)
    else:
        xq = x.astype(np.float32)
    it.set_tensor(ide['index'], xq.reshape(ide['shape']))
    it.invoke()
    ode = it.get_output_details()[0]
    y = it.get_tensor(ode['index']).astype(np.float64)
    q = ode['quantization_parameters']
    if q is not None and len(q['scales']):
        y = (y - float(q['zero_points'][0])) * float(q['scales'][0])
    return y


def describe(base, x, ref):
    import tflite
    raw0 = bytearray(open(base, 'rb').read())
    model = tflite.Model.GetRootAsModel(bytes(raw0), 0)
    g = model.Subgraphs(0)
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() == 34:
            ins = []
            for k in range(op.InputsLength()):
                t = g.Tensors(op.Inputs(k))
                q = t.Quantization()
                sc = [q.Scale(j) for j in range(q.ScaleLength())] \
                    if q and q.ScaleLength() else []
                zp = [q.ZeroPoint(j) for j in range(q.ZeroPointLength())] \
                    if q and q.ZeroPointLength() else []
                val = None
                bb = model.Buffers(t.Buffer())
                if bb.DataLength() == 1:
                    val = bytes(bytearray(bb.DataAsNumpy())).hex()
                ins.append({'shape': [t.Shape(j)
                                      for j in range(t.ShapeLength())],
                            'scales': sc, 'zp': zp, 'byte': val})
            y = run_model(base, x)
            return {'pad_inputs': len(ins), 'detail': ins,
                    'bytes': len(raw0),
                    'max_abs_err': float(np.abs(y - ref).max())}
    return {'pad_inputs': -1, 'note': 'no PAD op found!'}


def main():
    import json
    import tensorflow as tf
    rep = {'variants': {}}
    rng = np.random.default_rng(3)
    x = (rng.random((1, 4, 4, 2)).astype(np.float32) * 2 - 1)
    ref = tf.pad(tf.constant(x), [[0, 0], [1, 1], [1, 1], [0, 0]],
                 constant_values=0.0).numpy()
    for tag, explicit in (('implicit', False), ('explicit_scalar', True)):
        work = 'minipad_' + tag
        os.makedirs(work, exist_ok=True)
        sm = os.path.join(work, 'sm')
        base = os.path.join(work, 'pad_int8.tflite')
        build_float(sm, explicit)
        quantize(sm, base)
        rep['variants'][tag] = describe(base, x, ref)
    os.makedirs('minipad', exist_ok=True)
    json.dump(rep, open('minipad/minipad_report.json', 'w'), indent=2)
    print(json.dumps(rep, indent=2)[:3000], flush=True)
    print('MINIPAD DONE', flush=True)


if __name__ == '__main__':
    main()
