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


def _const_values(m, name, limit=16):
    import onnx
    from onnx import numpy_helper
    for init in m.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init).flatten()
            return arr[:limit].tolist()
    return None


def diagnose_zero_channel(onnx_path):
    """打印导致 [*,*,*,0] 的上游节点链 + Slice 参数（纯诊断，不改图）。"""
    import onnx
    from onnx import shape_inference
    try:
        inferred = shape_inference.infer_shapes_path(onnx_path) \
            if hasattr(shape_inference, 'infer_shapes_path') else None
        m = onnx.load(onnx_path)
    except Exception as e:
        print('shape inference failed: %s' % e, flush=True)
        m = onnx.load(onnx_path)
    out2node = {}
    for n in m.graph.node:
        for o in n.output:
            out2node[o] = n
    consts = {}
    for init in m.graph.initializer:
        consts[init.name] = list(init.dims)
    for n in m.graph.node:
        if 'convolution_4' in n.name and n.op_type in ('Conv',):
            print('suspect node: %s op=%s inputs=%s' % (
                n.name, n.op_type, list(n.input)), flush=True)
            for inp in n.input:
                p = out2node.get(inp)
                if p is not None:
                    print('  parent: %s op=%s inputs=%s' % (
                        p.name, p.op_type, list(p.input)), flush=True)
                    if p.op_type == 'Slice':
                        for inp2 in p.input:
                            if inp2 in consts:
                                v = _const_values(m, inp2)
                                print('    slice input const %s dims=%s vals=%s' % (
                                    inp2, consts[inp2], v), flush=True)
                            p2 = out2node.get(inp2)
                            if p2 is not None:
                                print('    slice input node %s op=%s' % (
                                    p2.name, p2.op_type), flush=True)
                                if p2.op_type == 'Reshape':
                                    for inp3 in p2.input[1:]:
                                        if inp3 in consts:
                                            print('    reshape shape const %s dims=%s vals=%s' % (
                                                inp3, consts[inp3],
                                                _const_values(m, inp3)),
                                                flush=True)
                elif inp in consts:
                    print('  const input %s dims=%s' % (inp, consts[inp]),
                          flush=True)


def repair_reshape_minus_one(onnx_path, repaired_path):
    """修 tf2onnx 的 Reshape[*, -1, ...]：-1 本应由下游 Slice 的 ends 反推。

    实测：Reshape shape=[1,-1,50,50]，下游 Slice axes=[0,2,3,1]
    starts=[0,0,0,122] ends=[1,50,50,244] → 通道维实为 244，
    切片得 122 通道，正好对上 Conv filter 的输入通道 122。
    规则：每个含 -1 的 Reshape，对其直接下游 Slice 在同一轴上的 ends 取 max
    回填；无 Slice 佐证的不碰。修完用 onnxruntime 新旧图余弦验证数值中性。
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    consumers_map = {}
    for n in m.graph.node:
        for i in n.input:
            consumers_map.setdefault(i, []).append(n)
    inits = {i.name: i for i in m.graph.initializer}
    patched = 0
    n_reshape = 0
    for n in m.graph.node:
        if n.op_type != 'Reshape' or len(n.input) < 2:
            continue
        n_reshape += 1
        shp_init = inits.get(n.input[1])
        if shp_init is None:
            print('reshape %s: shape input not const' % n.name, flush=True)
            continue
        shape = numpy_helper.to_array(shp_init).tolist()
        consumers = []
        for o in n.output:
            consumers.extend(consumers_map.get(o, []))
        cdesc = [(c.name, c.op_type) for c in consumers]
        if -1 in shape:
            print('reshape %s shape=%s consumers=%s' % (n.name, shape, cdesc),
                  flush=True)
        if shape.count(-1) != 1:
            continue
        k = shape.index(-1)
        need = 0
        for consumer in consumers:
            if consumer is None or consumer.op_type != 'Slice':
                continue
            d = _slice_axis_range(consumer, inits, k)
            if d is not None:
                need = max(need, d)
        if need > 0:
            shape[k] = need
            shp_init.CopyFrom(numpy_helper.from_array(
                np.array(shape, dtype=np.int64), name=shp_init.name))
            patched += 1
            print('patched Reshape %s: -1@axis%d -> %d' % (n.name, k, need),
                  flush=True)
    print('reshape_total=%d patched=%d' % (n_reshape, patched), flush=True)
    if patched == 0:
        print('no Reshape(-1) with Slice evidence, graph unchanged',
              flush=True)
        return onnx_path
    onnx.save(m, repaired_path)
    _validate_onnx_pair(onnx_path, repaired_path)
    return repaired_path


def _slice_axis_range(slice_node, inits, axis):
    """Slice 在某轴上的 ends（正数才可信），无则 None。"""
    import onnx
    from onnx import numpy_helper
    inp = {i: n for n, i in enumerate(
        ['x', 'starts', 'ends', 'axes', 'steps'])}
    try:
        starts = numpy_helper.to_array(
            inits[slice_node.input[inp['starts']]]).tolist()
        ends = numpy_helper.to_array(
            inits[slice_node.input[inp['ends']]]).tolist()
        axes = numpy_helper.to_array(
            inits[slice_node.input[inp['axes']]]).tolist()
    except (KeyError, IndexError):
        return None
    for s, e, a in zip(starts, ends, axes):
        if a == axis and isinstance(e, int) and e > 0:
            return e
    return None


def _validate_onnx_pair(a_path, b_path):
    """新旧 ONNX 同输入余弦（修图必须数值中性）。"""
    import numpy as np
    import onnxruntime as ort
    rng = np.random.default_rng(11)
    dummy = rng.random((1, 395, 395, 3), dtype=np.float32)
    outs = []
    for p in (a_path, b_path):
        sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        outs.append(sess.run(None, {sess.get_inputs()[0].name: dummy})[0])
        del sess
    fa, fb = outs[0].flatten(), outs[1].flatten()
    c = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-9))
    print('onnx-vs-repaired cosine=%.6f' % c, flush=True)
    assert c > 0.9999, 'repair changed numerics, cosine=%.6f' % c


def stage_onnx_to_savedmodel(onnx_path, sm_dir):
    # onnx2tf 对 TFLite 血统图的 Depthwise/Slice 分支常推出 0 通道
    # （本图首撞：Conv2D input [1,50,50,0]，-b/-ois 证实非动态 batch 问题）。
    # 依次尝试：静态 shape 钉死 → -kt 保 NHWC → onnxsim 化简 → onnx-tf 换后端。
    diagnose_zero_channel(onnx_path)
    repaired = os.path.join(os.path.dirname(onnx_path), 'concept_fix.onnx')
    graph_path = repair_reshape_minus_one(onnx_path, repaired)
    variants = [
        ['-b', '1'],
        ['-ois', 'inputs_0:1,395,395,3'],
        ['-b', '1', '-ois', 'inputs_0:1,395,395,3'],
        ['-kt', 'inputs_0'],
        ['-b', '1', '-kt', 'inputs_0'],
    ]
    last_err = ''
    for i, extra in enumerate(variants):
        trial = '%s.try%d' % (sm_dir, i)
        r = subprocess.run(
            [sys.executable, '-m', 'onnx2tf', '-i', graph_path, '-o', trial,
             '--non_verbose'] + extra, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isdir(trial):
            os.rename(trial, sm_dir)
            print('onnx2tf OK with flags=%s' % extra, flush=True)
            with open(sm_dir + '.flags.txt', 'w') as f:
                f.write(' '.join(extra) + '\n')
            return
        last_err = (r.stdout[-800:] + r.stderr[-800:]).replace('\n', ' | ')
        print('onnx2tf flags=%s FAILED: %s' % (extra, last_err[-500:]),
              flush=True)
    # 后备 A：onnxsim 化简后再转（常量折叠常能消掉产出 0 通道的 Slice/Reshape）
    try:
        import onnxsim
        sim_path = graph_path.replace('.onnx', '.sim.onnx')
        print('trying onnxsim simplify ...', flush=True)
        model_opt, ok = onnxsim.simplify(graph_path)
        if ok:
            import onnx
            onnx.save(model_opt, sim_path)
            print('onnxsim OK -> %s' % sim_path, flush=True)
            r = subprocess.run(
                [sys.executable, '-m', 'onnx2tf', '-i', sim_path, '-o', sm_dir,
                 '--non_verbose', '-b', '1'], capture_output=True, text=True)
            if r.returncode == 0 and os.path.isdir(sm_dir):
                print('onnx2tf OK via onnxsim simplified graph', flush=True)
                with open(sm_dir + '.flags.txt', 'w') as f:
                    f.write('onnxsim+-b 1\n')
                return
            print('onnx2tf on simplified graph FAILED', flush=True)
        else:
            print('onnxsim simplify check failed', flush=True)
    except Exception as e:
        print('onnxsim path unavailable: %s' % e, flush=True)
    # 后备 B：换 onnx-tf 后端（与 onnx2tf 不同代码库），并做数值交叉验证
    try:
        print('trying onnx-tf backend ...', flush=True)
        import types
        import numpy as np
        import onnx
        # OEC 同款 shim：onnx>=1.14 移除了 onnx.mapping，老后端（onnx-tf/rknn）
        # 引用的符号需补回（见 OEC convert-concept.yml apply_mapping_shim）
        if not hasattr(onnx, 'mapping'):
            mt = types.SimpleNamespace()
            mt.TENSOR_TYPE_TO_NP_TYPE = {
                1: np.dtype('float32'), 2: np.dtype('uint8'),
                3: np.dtype('int8'), 4: np.dtype('uint16'),
                5: np.dtype('int16'), 6: np.dtype('int32'),
                7: np.dtype('int64'), 9: np.dtype('bool'),
                10: np.dtype('float16'), 11: np.dtype('float64'),
                12: np.dtype('uint32'), 13: np.dtype('uint64'),
                14: np.dtype('complex64'), 15: np.dtype('complex128'),
            }
            mt.NP_TYPE_TO_TENSOR_TYPE = {
                v: k for k, v in mt.TENSOR_TYPE_TO_NP_TYPE.items()}
            mt.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = mt.TENSOR_TYPE_TO_NP_TYPE
            onnx.mapping = mt
            print('onnx.mapping shim injected', flush=True)
        from onnx_tf.backend import prepare
        m = onnx.load(graph_path)
        rep = prepare(m)
        rep.export_graph(sm_dir)
        assert os.path.isdir(sm_dir), 'onnx-tf produced no SavedModel'
        _validate_savedmodel(sm_dir, graph_path)
        print('onnx-tf backend OK (numerically validated)', flush=True)
        with open(sm_dir + '.flags.txt', 'w') as f:
            f.write('onnx-tf\n')
        return
    except Exception as e:
        print('onnx-tf path failed: %s' % e, flush=True)
    raise SystemExit('onnx->SavedModel all paths failed; last=%s' % last_err[-500:])


def _validate_savedmodel(sm_dir, onnx_path):
    """SavedModel vs ONNX 同输入输出余弦（防换后端静默错图）。"""
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf
    rng = np.random.default_rng(11)
    dummy = rng.random((1, 395, 395, 3), dtype=np.float32)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ref = sess.run(None, {sess.get_inputs()[0].name: dummy})[0].flatten()
    del sess
    mod = tf.saved_model.load(sm_dir)
    fn = mod.signatures['serving_default']
    key = list(fn.structured_input_signature[1].keys())[0]
    got = fn(**{key: tf.constant(dummy)})
    got = list(got.values())[0].numpy().flatten()
    denom = (np.linalg.norm(ref) * np.linalg.norm(got)) + 1e-9
    c = float(np.dot(ref, got) / denom)
    print('savedmodel-vs-onnx cosine=%.5f' % c, flush=True)
    assert c > 0.999, 'backend mismatch, cosine=%.5f' % c


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