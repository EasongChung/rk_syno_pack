#!/usr/bin/env python3
"""最小 Pad 内核实验：隔离退化 scale 下量化 PadV2 的行为。

复刻本案结构：Input[1,4,4,2] -> Pad(pads, scalar 0.0) -> out，全整型量化后，
逐个把有关张量改成退化 scale，观察输出是正确/全零/崩溃。
目标：判定 Pad 核对退化 scalar-scale 与退化 I/O-scale 各自的反应。

输出 minipad_report.json。任何崩溃都如实记录（同样是结论）。
"""
import os

import numpy as np


def build_float(sm_dir):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(4, 4, 2), batch_size=1, dtype=tf.float32)
    out = tf.pad(inp, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=0.0)
    model = tf.keras.Model(inp, out)
    tf.saved_model.save(model, sm_dir)
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
        sc, zp = None, None
    it.set_tensor(ide['index'], xq.reshape(ide['shape']))
    it.invoke()
    ode = it.get_output_details()[0]
    y = it.get_tensor(ode['index']).astype(np.float64)
    q = ode['quantization_parameters']
    if q is not None and len(q['scales']):
        y = (y - float(q['zero_points'][0])) * float(q['scales'][0])
    return y, (sc, zp)


def main():
    import json
    import struct
    import tflite
    work = 'minipad'
    os.makedirs(work, exist_ok=True)
    sm = os.path.join(work, 'sm')
    base = os.path.join(work, 'pad_int8.tflite')
    build_float(sm)
    quantize(sm, base)
    rng = np.random.default_rng(3)
    x = (rng.random((1, 4, 4, 2)).astype(np.float32) * 2 - 1).reshape(1, 4, 4, 2)
    y0, qinfo = run_model(base, x)
    # 期望：tf.pad 同输入 float 结果
    import tensorflow as tf
    ref = tf.pad(tf.constant(x),
                 [[0, 0], [1, 1], [1, 1], [0, 0]],
                 constant_values=0.0).numpy()
    rep = {'baseline_max_abs_err': float(np.abs(y0 - ref).max()),
           'qinfo': [qinfo[0], qinfo[1]] if qinfo[0] else None,
           'cases': {}}
    print('baseline max_abs_err=%.6g' % rep['baseline_max_abs_err'], flush=True)

    # 定位标量 const（Pad 第 3 输入）与 I/O 张量
    raw0 = bytearray(open(base, 'rb').read())
    model = tflite.Model.GetRootAsModel(bytes(raw0), 0)
    g = model.Subgraphs(0)
    padop = None
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() == 34:
            padop = op
            break
    assert padop is not None and padop.InputsLength() == 3
    pin, pout, pval = padop.Inputs(0), padop.Outputs(0), padop.Inputs(2)

    def tab_vec(tab, voff):
        off = tab.Offset(voff)
        assert off != 0
        return tab.Vector(off)

    # case A/B：见下循环（set_scale2 每次重建表对象，避免 stale 引用）

    # case A：标量 scale -> 1e36（值保持 zp，即反量化 0.0）
    for case, _ in [
        ('scalar_scale_1e36', None),
        ('io_scale_1e36', None),
    ]:
        raw = bytearray(open(base, 'rb').read())
        model = tflite.Model.GetRootAsModel(bytes(raw), 0)
        g = model.Subgraphs(0)
        # 重定位（表对象需重建）
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() == 34:
                padop = op
                break
        pin, pout, pval = padop.Inputs(0), padop.Outputs(0), padop.Inputs(2)

        def set_scale2(ti, sc):
            t = g.Tensors(ti)
            q = t.Quantization()
            assert q.ScaleLength() == 1
            struct.pack_into('<f', raw, tab_vec(q._tab, 8), float(sc))
        if case.startswith('scalar'):
            set_scale2(pval, 1.3344405750530544e+36)
        else:
            set_scale2(pin, 1.3344405750530544e+36)
            set_scale2(pout, 1.3344405750530544e+36)
        p = os.path.join(work, case + '.tflite')
        open(p, 'wb').write(raw)
        try:
            y, _ = run_model(p, x)
            rep['cases'][case] = {
                'max_abs_err': float(np.abs(y - ref).max()),
                'out_const': bool((y == y.flat[0]).all()),
                'out_sample': [float(v) for v in y.flat[:4]]}
        except Exception as e:
            rep['cases'][case] = {'CRASH': str(e)[:300]}
        print(case, rep['cases'][case], flush=True)
    json.dump(rep, open(os.path.join(work, 'minipad_report.json'), 'w'),
              indent=2)
    print('MINIPAD DONE', flush=True)


if __name__ == '__main__':
    main()
