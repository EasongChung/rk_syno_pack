#!/usr/bin/env python3
"""偏置重灌：把被毒化 bias_scale 压成 0 的 INT32 偏置，按真值重算。

根因（实测）：Pad 毒标量致其下游激活 scale 退化，MLIR 由此算出的
bias_scale（= input_scale × weight_scale）同样退化，op#7 DW 与 op#9 CONV
（消费者：MaxPool[1,99,99,24] 输出）的真偏置（dyn 模型：DW [-16.4,17.5]、
CONV [-8.6,52.1]，均非零）被压成全 0 int32。
修复：bias_int32[c] = round(true_bias[c] / (in_scale × w_scale[c]))，
bias scale 向量同步为 in_scale × w_scale（逐通道，zp=0 不变）。
真值取自同轮 dyn 模型（float 偏置）；映射键：拓扑角色
（MaxPool[1,99,99,24] 的 DW/CONV 消费者 + 通道数），不断言固定编号。

用法：
  python repair_bias.py --dyn out/concept_dyn.tflite \\
      --fixed out/concept_int8_fixed.tflite --out out/concept_int8_fixed2.tflite
"""
import argparse
import struct

import numpy as np


def load(path):
    import tflite
    raw = bytearray(open(path, 'rb').read())
    model = tflite.Model.GetRootAsModel(bytes(raw), 0)
    return model, raw


def shp(g, ti):
    t = g.Tensors(ti)
    return [t.Shape(j) for j in range(t.ShapeLength())]


def buf_bytes(model, t):
    bb = model.Buffers(t.Buffer())
    n = bb.DataLength()
    if n == 0:
        return None
    a = bb.DataAsNumpy()
    return bytes(bytearray(a)) if a is not None and len(a) else None


def find_branch(model):
    """找 MaxPool[1,99,99,24] 及其 DW/CONV 消费者。返回 dict。"""
    import tflite  # noqa
    g = model.Subgraphs(0)
    codes = [model.OperatorCodes(
        g.Operators(i).OpcodeIndex()).BuiltinCode()
        for i in range(g.OperatorsLength())]
    mps = [i for i, c in enumerate(codes)
           if c == 17 and shp(g, g.Operators(i).Outputs(0)) == [1, 99, 99, 24]]
    assert len(mps) == 1, 'maxpool mapping not unique: %s' % mps
    mpi = mps[0]
    mout = g.Operators(mpi).Outputs(0)
    roles = {}
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if op.InputsLength() < 3 or op.Inputs(0) != mout:
            continue
        if codes[i] == 4:  # DEPTHWISE
            roles['dw'] = (i, op.Inputs(1), op.Inputs(2), op.Inputs(0))
        elif codes[i] == 3:  # CONV
            roles['conv'] = (i, op.Inputs(1), op.Inputs(2), op.Inputs(0))
    assert set(roles) == {'dw', 'conv'}, roles.keys()
    dw_nch = shp(g, roles['dw'][2])[0]
    cv_nch = shp(g, roles['conv'][2])[0]
    assert (dw_nch, cv_nch) == (24, 122), (dw_nch, cv_nch)
    print('branch: maxpool op#%d dw_bias_ch=%d conv_bias_ch=%d' % (
        mpi, dw_nch, cv_nch), flush=True)
    return roles


def read_f32(model, ti):
    t = model.Subgraphs(0).Tensors(ti)
    pb = buf_bytes(model, t)
    assert pb is not None, 'tensor %d has no data' % ti
    if t.Type() == 1:
        return np.frombuffer(pb, dtype=np.float16).astype(np.float64)
    return np.array(struct.unpack('<%df' % (len(pb) // 4), pb))


def vec_data(tab, field_voff):
    off = tab.Offset(field_voff)
    assert off != 0, 'field missing'
    return tab.Vector(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dyn', required=True)
    ap.add_argument('--fixed', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dmodel, _ = load(args.dyn)
    fmodel, fraw = load(args.fixed)
    dg, fg = dmodel.Subgraphs(0), fmodel.Subgraphs(0)
    droles = find_branch(dmodel)
    froles = find_branch(fmodel)

    # fixed 输入 scale（移植后真值）
    in_scale = None
    for role in ('dw', 'conv'):
        ti = froles[role][3]
        q = fg.Tensors(ti).Quantization()
        sc = q.Scale(0)
        assert sc < 1e6, 'input scale still degenerate: %s' % sc
        in_scale = sc if in_scale is None else in_scale
        assert abs(in_scale - sc) < 1e-9, (in_scale, sc)
    print('branch input scale=%.6g' % in_scale, flush=True)

    patched = 0
    for role in ('dw', 'conv'):
        _, _, dbias_ti, _ = droles[role]
        _, fw_ti, fbias_ti, _ = froles[role]
        true_b = read_f32(dmodel, dbias_ti)
        ft = fg.Tensors(fw_ti)
        fq = ft.Quantization()
        assert fq.ScaleLength() == len(true_b), \
            (role, fq.ScaleLength(), len(true_b))
        wsc = np.array([fq.Scale(j) for j in range(fq.ScaleLength())])
        assert np.all((wsc > 0) & (wsc < 1e6)), (role, wsc[:4])
        bsc = in_scale * wsc
        new_i32 = np.round(true_b / bsc).astype(np.int64)
        mx = int(np.abs(new_i32).max())
        assert mx < 2 ** 31, (role, mx)
        if mx >= 2 ** 30:
            print('WARN %s: bias int32 near limit (%d), tiny weight scales?' % (
                role, mx), flush=True)
        # patch bias buffer（等长 int32）
        bt = fg.Tensors(fbias_ti)
        bb = fmodel.Buffers(bt.Buffer())
        doff = vec_data(bb._tab, 4)
        old = np.frombuffer(bytes(fraw[doff:doff + 4 * len(true_b)]),
                            dtype=np.int32)
        assert len(old) == len(true_b), (role, len(old), len(true_b))
        struct.pack_into('<%di' % len(true_b), fraw, doff,
                         *[int(v) for v in new_i32.tolist()])
        back = np.frombuffer(bytes(fraw[doff:doff + 4 * len(true_b)]),
                             dtype=np.int32)
        assert (back == new_i32.astype(np.int32)).all(), role
        # patch bias scale 向量（等长 float32）+ zp 保持 0
        bq = bt.Quantization()
        assert bq.ScaleLength() == len(true_b), (role, bq.ScaleLength())
        soff = vec_data(bq._tab, 8)
        struct.pack_into('<%df' % len(true_b), fraw, soff,
                         *[float(v) for v in bsc.tolist()])
        back_sc = struct.unpack_from('<%df' % len(true_b), fraw, soff)
        assert all(abs(a - b) < 1e-9 for a, b in zip(back_sc, bsc)), role
        print('rebiased %s: ch=%d true_bias=[%.3g,%.3g] i32range=[%d,%d]' % (
            role, len(true_b), true_b.min(), true_b.max(),
            new_i32.min(), new_i32.max()), flush=True)
        patched += 1
    assert patched == 2
    open(args.out, 'wb').write(fraw)
    import tflite
    m2 = tflite.Model.GetRootAsModel(open(args.out, 'rb').read(), 0)
    assert m2.Subgraphs(0).OperatorsLength() == \
        fg.OperatorsLength()
    print('REBIAS DONE -> %s' % args.out, flush=True)


if __name__ == '__main__':
    main()
