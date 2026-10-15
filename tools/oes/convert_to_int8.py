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


def dump_graph(onnx_path, dump_path):
    """转储 ONNX 全图节点（名/op/关键属性/常量），供离线精确分析改写规则。"""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    inits = {i.name: i for i in m.graph.initializer}
    lines = ['graph=%s nodes=%d inits=%d' % (
        m.graph.name, len(m.graph.node), len(m.graph.initializer))]
    for n in m.graph.node:
        parts = ['NODE %s op=%s' % (n.name, n.op_type)]
        parts.append('in=%s' % ','.join(n.input))
        parts.append('out=%s' % ','.join(n.output))
        for a in n.attribute:
            if a.name == 'perm':
                parts.append('perm=%s' % list(a.ints))
            elif a.name in ('axis', 'axes'):
                parts.append('%s=%s' % (a.name, list(a.ints)))
        for inp in n.input:
            if inp in inits:
                arr = numpy_helper.to_array(inits[inp]).flatten()
                v = arr[:12].tolist()
                parts.append('const[%s]=%s%s' % (
                    inp, v, '...' if len(arr) > 12 else ''))
        lines.append(' | '.join(parts))
    with open(dump_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('graph dump: %d nodes -> %s' % (len(m.graph.node), dump_path),
          flush=True)


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
    dump_graph(onnx_path, os.path.join(os.path.dirname(onnx_path),
                                       'graph_dump.txt'))
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


def rewrite_nchw_slice_detour(onnx_path, rewritten_path):
    """消掉 NHWC→NCHW→Reshape(-1)→Slice→NHWC 迂回（TFLite 血统典型产物）。

    实测链：Transpose[0,3,1,2] → Reshape[1,-1,H,W] → Slice(axes NCHW 序)
             → Transpose[0,2,3,1] → Conv(NHWC)。
    等价改写：在首个 Transpose 的输入（NHWC）上直接 Slice（轴重映射
    NCHW a -> NHWC [0,2,3,1][a]），删掉 2 Transpose + Reshape。
    仅当 perm 对恰为 NHWC<->NCHW 逆对时改写；改完 ORT 新旧图余弦必须为 1。
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    consumers = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    inits = {i.name: i for i in m.graph.initializer}
    NCHW2NHWC = [0, 2, 3, 1]
    NHWC2NCHW = [0, 3, 1, 2]

    def perm_of(node):
        for a in node.attribute:
            if a.name == 'perm':
                return list(a.ints)
        return None

    def const_vals(name):
        init = inits.get(name)
        if init is None:
            return None
        return numpy_helper.to_array(init).tolist()

    removed = 0
    for t1 in list(m.graph.node):
        if t1.op_type != 'Transpose' or perm_of(t1) != NHWC2NCHW:
            continue
        t1_out = t1.output[0]
        r_users = consumers.get(t1_out, [])
        if len(r_users) != 1 or r_users[0].op_type != 'Reshape':
            continue
        r = r_users[0]
        if len(r.input) < 2:
            continue
        shape = const_vals(r.input[1])
        if not shape or shape.count(-1) != 1 or shape[0] != 1:
            continue
        r_out = r.output[0]
        s_users = consumers.get(r_out, [])
        if len(s_users) != 1 or s_users[0].op_type != 'Slice':
            continue
        s = s_users[0]
        if len(s.input) != 4:  # 只要无 steps 的标准 Slice，语义才可直搬
            continue
        starts, ends, axes = (const_vals(s.input[1]), const_vals(s.input[2]),
                              const_vals(s.input[3]))
        if starts is None or ends is None or axes is None:
            continue
        if any(v < 0 for v in starts + ends):
            continue  # 负索引需结合维度解释，不碰
        s_out = s.output[0]
        t_users = consumers.get(s_out, [])
        if len(t_users) != 1 or t_users[0].op_type != 'Transpose':
            continue
        t2 = t_users[0]
        if perm_of(t2) != NCHW2NHWC:
            continue
        # 轴重映射到 NHWC
        nstarts = [0, 0, 0, 0]
        nends = [0, 0, 0, 0]
        naxes = [0, 1, 2, 3]
        for st, e, a in zip(starts, ends, axes):
            na = NCHW2NHWC[a]
            nstarts[na] = st
            nends[na] = e
        # 改 Slice 输入为 t1 的输入（NHWC），参数换 NHWC 序：新建 const 挂载
        # （以前缀 s_out 保证唯一，避免多次改写重名）
        new_names = []
        for tag, vals in (('starts', nstarts), ('ends', nends),
                          ('axes', naxes)):
            nm = '%s_nhwc_%s' % (s_out, tag)
            ni = numpy_helper.from_array(np.array(vals, dtype=np.int64),
                                         name=nm)
            m.graph.initializer.append(ni)
            new_names.append(nm)
        del s.input[:]
        s.input.extend([t1.input[0]] + new_names)
        # t2 输出改接 Slice 输出（删 t2），t1/r 悬空后由清理 pass 去掉
        t2_out = t2.output[0]
        for n2 in m.graph.node:
            for ii, inp in enumerate(n2.input):
                if inp == t2_out:
                    n2.input[ii] = s_out
        # 从图中移除 t1/r/t2（悬空 Reshape shape const 随 initializer 保留，无害）
        for dead in (t1, r, t2):
            m.graph.node.remove(dead)
        # consumers 映射已失效，重建
        consumers = {}
        for n2 in m.graph.node:
            for i2 in n2.input:
                consumers.setdefault(i2, []).append(n2)
        removed += 1
        print('rewrote detour at %s (nhwc slice %s:%s)' % (s.name, nstarts,
                                                           nends),
              flush=True)
    if removed == 0:
        print('no NHWC/NCHW slice detour found', flush=True)
        return onnx_path
    # 清理悬空节点（输入含不存在张量的非输入节点不处理，仅去孤立 Transpose）
    onnx.save(m, rewritten_path)
    _validate_onnx_pair(onnx_path, rewritten_path)
    print('detour rewrite: removed=%d' % removed, flush=True)
    return rewritten_path


def rewrite_slice_to_split(onnx_path, rewritten_path):
    """把静态 Slice 改写为 Split+取段（onnx2tf 的 Slice 轴重映射会推出 0 通道）。

    实测：Slice(axes=[0,2,3,1], starts=[0,0,0,122], ends=[1,50,50,244])
    作用于静态 [1,244,50,50]，onnx2tf 推出下游 [1,50,50,0]。
    等价改写：Split(sizes=[122,122], axis=1) 取第 2 段（[122:244]），
    或取第 1 段（[0:122]）。纯机械等价（同张量、同轴、同区间），
    Split 的静态 shape 对 onnx2tf 毫无推理压力。改完 ORT 余弦必须为 1。
    仅改：starts/ends 全非负、被切轴静态已知、单轴非常值步长（无 steps 输入
    或 steps 全 1）的 Slice；其余不动。
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper, shape_inference
    m = onnx.load(onnx_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print('shape inference warn: %s' % e, flush=True)
    inits = {i.name: i for i in m.graph.initializer}

    def tshape(tname):
        for v in list(m.graph.input) + list(m.graph.value_info) + \
                list(m.graph.output):
            if v.name == tname and v.type.tensor_type.HasField('shape'):
                return [d.dim_value for d in v.type.tensor_type.shape.dim]
        return None

    def const_vals(name):
        init = inits.get(name)
        if init is None:
            return None
        return numpy_helper.to_array(init).tolist()

    rewritten = 0
    for n in list(m.graph.node):
        if n.op_type != 'Slice':
            continue
        if len(n.input) == 5:
            steps = const_vals(n.input[4])
            if steps is None or any(s != 1 for s in steps):
                continue
        elif len(n.input) != 4:
            continue
        starts, ends, axes = (const_vals(n.input[1]), const_vals(n.input[2]),
                              const_vals(n.input[3]))
        if starts is None or ends is None or axes is None:
            continue
        if any(v < 0 for v in starts + ends):
            continue
        dims = tshape(n.input[0])
        if dims is None:
            continue
        # 仅处理"恰切一个轴的连续区间"（本图全部 26 个 Slice 皆属此类）
        if len(axes) != len(set(axes)):
            continue
        # 要求非切轴的 starts/ends 取全量（即只实质切一个轴）
        cut = None
        ok = True
        for s, e, a in zip(starts, ends, axes):
            d = dims[a] if a < len(dims) else 0
            if d <= 0 or e <= 0:
                ok = False
                break
            if s == 0 and e >= d:
                continue  # 该轴取全量，非实质切割
            if cut is not None:
                ok = False
                break
            cut = (a, s, min(e, d), d)
        if not ok or cut is None:
            continue
        a, s, e, d = cut
        segs, take = [], None
        if s > 0:
            segs.append(s)
        segs.append(e - s)
        take = len(segs) - 1
        if e < d:
            segs.append(d - e)
        if any(x <= 0 for x in segs):
            continue
        # 输出名先在普通 list 里定好（不依赖 protobuf 容器的 __setitem__）：
        # 被取段直接沿用原 Slice 输出名，下游 Conv/Concat 无需任何改动
        out_names = ['%s_split_%d' % (n.output[0], i)
                     for i in range(len(segs))]
        out_names[take] = n.output[0]
        # Split-13 起 split 走输入（attribute 形式已非法），挂初始化器
        split_nm = '%s_split_sizes' % n.output[0]
        m.graph.initializer.append(numpy_helper.from_array(
            np.array(segs, dtype=np.int64), name=split_nm))
        sp = onnx.helper.make_node(
            'Split', [n.input[0], split_nm], out_names,
            name='%s_as_split' % n.name, axis=a)
        # 用 Split 原位替换 Slice
        idx = list(m.graph.node).index(n)
        m.graph.node.remove(n)
        m.graph.node.insert(idx, sp)
        rewritten += 1
        print('slice->split %s axis=%d [%d:%d] of %d' % (n.name, a, s, e, d),
              flush=True)
    if rewritten == 0:
        print('no static Slice rewritten', flush=True)
        return onnx_path
    onnx.save(m, rewritten_path)
    _validate_onnx_pair(onnx_path, rewritten_path)
    print('slice->split rewritten=%d' % rewritten, flush=True)
    return rewritten_path


def sanitize_names(onnx_path, sanitized_path):
    """把图中一切名字里的 ';' 换成 '_'（onnx2tf 导出 SavedModel 的正侧。

    实测：plain -kt 整图转换本可成功，仅导出时被
    `model/.../convolution;.../Add/y/kernel` 之类 tf2onnx 融合名（含 ;）拦下。
    本改写纯机械（节点名/张量名/初始化器名/图出入名全量替换），改完 ORT
    新旧图余弦必须为 1；若替换后重名则放弃改写。
    """
    import onnx
    m = onnx.load(onnx_path)
    all_names = []

    def coll(lst):
        for v in lst:
            if v.name:
                all_names.append(v.name)

    coll(m.graph.input)
    coll(m.graph.output)
    coll(m.graph.value_info)
    for n in m.graph.node:
        if n.name:
            all_names.append(n.name)
        all_names.extend([i for i in n.input if i])
        all_names.extend([o for o in n.output if o])
    for init in m.graph.initializer:
        all_names.append(init.name)
    if not any(';' in x for x in all_names):
        print('no semicolon names, sanitize skipped', flush=True)
        return onnx_path
    mapped = {x: x.replace(';', '_') for x in set(all_names) if ';' in x}
    if len(set(mapped.values())) != len(mapped):
        print('semicolon rename collides, sanitize aborted', flush=True)
        return onnx_path

    def fix(name):
        return mapped.get(name, name)

    for v in list(m.graph.input) + list(m.graph.output) + \
            list(m.graph.value_info):
        if v.name in mapped:
            v.name = mapped[v.name]
    for n in m.graph.node:
        if n.name in mapped:
            n.name = mapped[n.name]
        for i, inp in enumerate(n.input):
            if inp in mapped:
                n.input[i] = mapped[inp]
        for i, out in enumerate(n.output):
            if out in mapped:
                n.output[i] = mapped[out]
    for init in m.graph.initializer:
        if init.name in mapped:
            init.name = mapped[init.name]
    for sp in m.graph.sparse_initializer:
        if sp.values.name in mapped:
            sp.values.name = mapped[sp.values.name]
        if sp.indices.name in mapped:
            sp.indices.name = mapped[sp.indices.name]
    onnx.save(m, sanitized_path)
    _validate_onnx_pair(onnx_path, sanitized_path)
    n_hit = len(mapped)
    print('sanitized %d semicolon names' % n_hit, flush=True)
    return sanitized_path


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
    """新旧 ONNX 同输入余弦（修图必须数值中性）。输入形状取自图本身。"""
    import numpy as np
    import onnx
    import onnxruntime as ort
    m = onnx.load(a_path)
    dims = []
    for d in m.graph.input[0].type.tensor_type.shape.dim:
        v = d.dim_value if d.dim_value > 0 else 4
        dims.append(v)
    rng = np.random.default_rng(11)
    dummy = rng.random(tuple(dims), dtype=np.float32)
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
    san_path = os.path.join(os.path.dirname(onnx_path), 'concept_san.onnx')
    onnx_path = sanitize_names(onnx_path, san_path)
    detour_path = os.path.join(os.path.dirname(onnx_path), 'concept_detour.onnx')
    onnx_path = rewrite_nchw_slice_detour(onnx_path, detour_path)
    repaired = os.path.join(os.path.dirname(onnx_path), 'concept_fix.onnx')
    graph_path = repair_reshape_minus_one(onnx_path, repaired)
    split_path = os.path.join(os.path.dirname(onnx_path), 'concept_split.onnx')
    graph_path = rewrite_slice_to_split(graph_path, split_path)
    with open(os.path.join(os.path.dirname(onnx_path),
                           'graph_path.txt'), 'w') as f:
        f.write(graph_path + '\n')
    variants = [
        ['-kt', 'inputs_0', '-osd'],
        ['-b', '1', '-kt', 'inputs_0', '-osd'],
        ['-kt', 'inputs_0'],
        ['-b', '1', '-kt', 'inputs_0'],
        ['-kt', 'inputs_0', '--disable_group_convolution'],
        ['-b', '1'],
        ['-ois', 'inputs_0:1,395,395,3'],
        ['-b', '1', '-ois', 'inputs_0:1,395,395,3'],
    ]
    last_err = ''
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_onnx2tf.py')
    for i, extra in enumerate(variants):
        trial = '%s.try%d' % (sm_dir, i)
        r = subprocess.run(
            [sys.executable, runner, '-i', graph_path, '-o', trial,
             '--non_verbose'] + extra, capture_output=True, text=True)
        sm_pb = os.path.join(trial, 'saved_model.pb')
        # onnx2tf 先落盘 SavedModel、再跑精度复核；复核在 numpy>=1.24 下会因
        # np.load(pickle) 自爆而 exit!=0。只要 pb 已落盘，就用自有校验裁决。
        if os.path.exists(sm_pb):
            try:
                _validate_savedmodel(trial, graph_path)
            except Exception as e:
                print('onnx2tf flags=%s SAVED BUT NUMERIC CHECK FAILED: %s' % (
                    extra, e), flush=True)
                continue
            os.rename(trial, sm_dir)
            print('onnx2tf OK with flags=%s (rc=%d, numerically validated)' % (
                extra, r.returncode), flush=True)
            with open(sm_dir + '.flags.txt', 'w') as f:
                f.write(' '.join(extra) + '\n')
            return
        last_err = (r.stdout[-800:] + r.stderr[-800:]).replace('\n', ' | ')
        errlines = [l.strip()[:300] for l in (r.stdout + r.stderr).splitlines()
                    if 'error' in l.lower() and 'computation_placer' not in l
                    and 'cuda' not in l.lower()][:6]
        print('onnx2tf flags=%s FAILED: %s || %s' % (
            extra, last_err[-300:], ' // '.join(errlines)), flush=True)
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
            runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'run_onnx2tf.py')
            r = subprocess.run(
                [sys.executable, runner, '-i', sim_path, '-o', sm_dir,
                 '--non_verbose', '-b', '1'], capture_output=True, text=True)
            if os.path.exists(os.path.join(sm_dir, 'saved_model.pb')):
                try:
                    _validate_savedmodel(sm_dir, sim_path)
                except Exception as e:
                    print('sim graph SAVED BUT NUMERIC CHECK FAILED: %s' % e,
                          flush=True)
                    import shutil
                    shutil.rmtree(sm_dir, ignore_errors=True)
                else:
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
    # ★ 校准分布必须与服务端真实输入一致：归一化 float32 NHWC [-1,1]。
    # （float_io 只决定模型 I/O 的 dtype，不改变校准分布；此前 full-int 路
    # 误喂 0~255 原始值，导致 activation 量化尺度全错，回归 top1=0.01。）
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
            arr = (arr - MEAN) * SCALE  # 与服务端/回归输入严格一致
            yield [arr.reshape(1, NET, NET, 3)]
            n += 1
            if n >= num:
                return
        while n < num:  # 合成兜底（报告中注明 synthetic_fill=N）
            base = rng.normal(117.0, 60.0, size=(NET, NET, 3)).clip(0, 255)
            base = (base - MEAN) * SCALE
            yield [base.astype(np.float32).reshape(1, NET, NET, 3)]
            n += 1
        print('representative_dataset: real=%d synthetic_fill=%d' % (
            min(len(files), num), max(0, num - len(files))), flush=True)
    return gen


def stage_quant_legacy(sm_dir, out_path, calib_dir, num_calib):
    """诊断/备选：旧量化器（experimental_new_quantizer=False）全整型。

    背景（实测）：MLIR 新量化器对 0 值 Pad 常量产出退化 scale 1.33e36
    （pseudo_qconst112 单值 128/zp127 → 反量化 1e36，本应为 0），
    毒化整网。若旧量化器对退化张量处理不同，本产物即为可交付 INT8。
    """
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(
        calib_dir, num_calib, False)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    try:
        conv.experimental_new_quantizer = False
    except AttributeError as e:
        print('legacy quantizer flag gone: %s' % e, flush=True)
        return None
    model = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(model)
    print('WROTE %s %d bytes (legacy quantizer)' % (out_path, len(model)),
          flush=True)
    return out_path


def stage_quant_mixed(sm_dir, out_path, calib_dir, num_calib):
    """混合量化：INT8 为主，个别算子允许回落 float。

    背景：全整型在 Pad 标量退化上传染整网；混合模式让转换器把
    不宜量化的算子留在 float（CPU 核跑），其余仍走 INT8。
    若门限通过，即为可交付 v1（后台识别任务对延迟不敏感），
    并用 diagnose_quant_model 公布回落算子清单。
    """
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(
        calib_dir, num_calib, False)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                      tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    model = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(model)
    print('WROTE %s %d bytes (mixed int8+float fallback)' % (
        out_path, len(model)), flush=True)
    interp = make_interpreter(out_path, invoke=False)
    ide = interp['input']
    ode = interp['output']
    print('mixed model in shape=%s dtype=%s / out shape=%s dtype=%s' % (
        ide['shape'], ide['dtype'], ode['shape'], ode['dtype']), flush=True)
    assert list(ide['shape']) == [1, 395, 395, 3], 'input must stay NHWC'
    assert list(ode['shape']) == [1, 335], 'output must be 335'
    diagnose_quant_model(out_path)


def stage_quant_oiqt(onnx_path, out_path, calib_dir):
    """已判死（HOST-VERDICT §1）：onnx2tf 1.29.24 的 -oiqt 在此依赖组合下
    直接产出 FLOAT 模型（316 F32 + 114 F16，0/450 量化），cos=1.0 系 float
    自比较假阳性。保留函数体作罢，防止后人重蹈；调用方直接跳过。"""
    print('OIQT SKIP: route dead per HOST-VERDICT §1 (emits float)', flush=True)
    return None


def stage_quant_dynamic(sm_dir, out_path):
    """诊断用：动态范围量化（仅权重量化，激活保持 float）。

    目的：隔离故障——若 dyn 版回归正常而全整型崩，则故障在 activation
    校准传播（高度怀疑对象：--disable_group_convolution 拆出的 ~730 个
    碎 conv + transpose 边界）；若 dyn 版同样崩，则故障在权重量化。
    产物 Teflon 不可用，仅作诊断。
    """
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    model = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(model)
    print('WROTE %s %d bytes (dynamic-range, diagnostic only)' % (
        out_path, len(model)), flush=True)


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
    # 产物自检：输入 NHWC + 输出 335（不做 invoke，见 diagnose 注释）
    interp = make_interpreter(out_path, invoke=False)
    ide = interp['input']
    ode = interp['output']
    print('quant model in shape=%s dtype=%s / out shape=%s dtype=%s' % (
        ide['shape'], ide['dtype'], ode['shape'], ode['dtype']), flush=True)
    assert list(ide['shape']) == [1, 395, 395, 3], 'input must stay NHWC'
    assert list(ode['shape']) == [1, 335], 'output must be 335'
    diagnose_quant_model(out_path)


def tflite_op_name(code):
    return {2: 'CONCATENATION', 3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D',
            5: 'DEPTH_TO_SPACE', 6: 'DEQUANTIZE', 9: 'FULLY_CONNECTED',
            14: 'LOGISTIC', 17: 'MAX_POOL_2D', 22: 'RESHAPE', 28: 'TANH',
            34: 'PAD', 39: 'TRANSPOSE', 40: 'MEAN', 45: 'STRIDED_SLICE',
            49: 'SPLIT', 65: 'SLICE', 79: 'ARG_MIN', 114: 'QUANTIZE',
            133: 'BROADCAST_TO'}.get(code, 'OP_%d' % code)


def diagnose_quant_model(path):
    """量化产物结构诊断：op 直方图 + 输入输出量化参数（纯静态，需 tflite 包）。"""
    try:
        import tflite
    except ImportError:
        print('diagnose skipped (no tflite pkg)', flush=True)
        return
    raw = open(path, 'rb').read()
    model = tflite.Model.GetRootAsModel(raw, 0)
    g = model.Subgraphs(0)
    hist = {}
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        c = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        hist[c] = hist.get(c, 0) + 1
    print('quant graph: %s' % {tflite_op_name(k): v
                               for k, v in sorted(hist.items())}, flush=True)
    tname = {0: 'FLOAT32', 3: 'UINT8', 9: 'INT8', 2: 'INT32'}
    for li, getter, n in (('in', g.Inputs, g.InputsLength()),
                          ('out', g.Outputs, g.OutputsLength())):
        for k in range(n):
            t = g.Tensors(getter(k))
            q = t.Quantization()
            print('  %s tensor=%r type=%s shape=%s q_n=%d' % (
                li, t.Name(), tname.get(t.Type(), t.Type()),
                [t.Shape(j) for j in range(t.ShapeLength())],
                q.ScaleLength() if q else 0), flush=True)


def make_interpreter(path, invoke=True, preserve=False):
    """建 TFLite 解释器：默认（XNNPACK）→ 去 delegate 重试序列。

    背景：量化产物中个别算子（如 Split 系）可能让 XNNPACK prepare 失败；
    验证机只需 CPU ref 核跑通做数值回归，逐级降级并如实记录。
    preserve=True 时透出全部中间张量（剖面用）。
    返回 {'interp','input','output','delegate'}。
    """
    import numpy as np
    import tensorflow as tf
    attempts = [('default', {})]
    try:
        members = [(m, getattr(tf.lite.experimental.OpResolverType, m))
                   for m in dir(tf.lite.experimental.OpResolverType)
                   if not m.startswith('_')]
        print('OpResolverType members=%s' % [m for m, _ in members],
              flush=True)
        for m, v in members:
            attempts.append((m, {'experimental_op_resolver_type': v}))
    except Exception as e:
        print('resolver enum unavailable: %s' % e, flush=True)
    last = None
    for tag, kw in attempts:
        try:
            it = tf.lite.Interpreter(model_path=path, **kw,
                                     experimental_preserve_all_tensors=preserve)
            it.allocate_tensors()
            if invoke:
                ide = it.get_input_details()[0]
                shp = [int(x) for x in ide['shape']]
                dt = ide['dtype']
                dummy = (np.zeros(shp, dtype=dt) if dt == np.uint8
                         else np.zeros(shp, dtype=np.float32))
                it.set_tensor(ide['index'], dummy)
                it.invoke()
            print('interpreter OK via %s' % tag, flush=True)
            return {'interp': it, 'input': it.get_input_details()[0],
                    'output': it.get_output_details()[0], 'delegate': tag}
        except Exception as e:
            last = e
            print('interpreter via %s FAILED: %s' % (tag, str(e)[:300]),
                  flush=True)
    raise RuntimeError('no interpreter backend works; last=%s' % last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--calib-dir', default='calib')
    ap.add_argument('--num-calib', type=int, default=300)
    ap.add_argument('--eval-dir', default='eval')
    ap.add_argument('--frontend-num', type=int, default=50)
    ap.add_argument('--skip-frontend-check', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    onnx_path = os.path.join(args.workdir, 'concept_mid.onnx')
    sm_dir = os.path.join(args.workdir, 'concept_float_sd')
    stage_tflite_to_onnx(args.src, onnx_path)
    stage_smoke(onnx_path)
    if not args.skip_frontend_check:
        import compare_tflite_onnx as cmp
        ok = cmp.run(args.src, onnx_path, args.eval_dir, args.frontend_num,
                     os.path.join(args.outdir, 'compare.json'))
        if not ok:
            raise SystemExit(3)
    stage_onnx_to_savedmodel(onnx_path, sm_dir)
    stage_quant(sm_dir, os.path.join(args.outdir, 'concept_int8.tflite'),
                args.calib_dir, args.num_calib, float_io=False)
    stage_quant(sm_dir, os.path.join(args.outdir, 'concept_int8_floatio.tflite'),
                args.calib_dir, args.num_calib, float_io=True)
    stage_quant_dynamic(sm_dir, os.path.join(args.outdir, 'concept_dyn.tflite'))
    try:
        gp = open(os.path.join(args.workdir, 'graph_path.txt')).read().strip()
        stage_quant_oiqt(gp, os.path.join(args.outdir, 'concept_oiqt.tflite'),
                         args.calib_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('OIQT SKIPPED: %s' % e, flush=True)
    stage_quant_mixed(sm_dir, os.path.join(args.outdir, 'concept_mixed.tflite'),
                      args.calib_dir, args.num_calib)
    leg = stage_quant_legacy(sm_dir, os.path.join(args.outdir, 'concept_legacy.tflite'),
                             args.calib_dir, args.num_calib)
    print('CONVERT DONE legacy=%s' % leg, flush=True)
    try:
        import transplant as tp
        tp_targets = tp.find_float_targets(args.src)
        tp_ranges = tp.measure_ranges(args.src, args.calib_dir, 50, tp_targets)
        tp_qidx = tp.find_quant_targets(
            os.path.join(args.outdir, 'concept_int8.tflite'))
        tp.patch_quant(os.path.join(args.outdir, 'concept_int8.tflite'),
                       os.path.join(args.outdir, 'concept_int8_fixed.tflite'),
                       tp_qidx, tp_ranges)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('TRANSPLANT SKIPPED: %s' % e, flush=True)
    try:
        import repair_bias as rb
        import sys as _sys
        _sys.argv = ['repair_bias',
                     '--dyn', os.path.join(args.outdir, 'concept_dyn.tflite'),
                     '--fixed', os.path.join(args.outdir, 'concept_int8_fixed.tflite'),
                     '--out', os.path.join(args.outdir, 'concept_int8_fixed2.tflite')]
        rb.main()
    except SystemExit as e:
        print('REBIAS exit=%s' % e, flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('REBIAS SKIPPED: %s' % e, flush=True)
    try:
        import depad as dp
        import sys as _sys2
        _sys2.argv = ['depad',
                      '--model', os.path.join(args.outdir, 'concept_int8_fixed2.tflite'),
                      '--out', os.path.join(args.outdir, 'concept_nodepad.tflite')]
        dp.main()
    except SystemExit as e:
        print('DEPAD exit=%s' % e, flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('DEPAD SKIPPED: %s' % e, flush=True)
    print('CONVERT DONE', flush=True)


if __name__ == '__main__':
    main()