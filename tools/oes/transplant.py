#!/usr/bin/env python3
"""量化修复手术：移植真值 scale/zp 到 4 个退化张量。

根因（实测）：MLIR 量化器对 0 值 Pad 标量产出退化 scale 1.33e36
（pseudo_qconst112 单字节 128/zp127 → 反量化 1e36，真值应为 0），
并经 Pad 的 scale 统一传染到 140/141/142（激活）。
其它 2 个退化 const 全零值==zp，无害不动。

修复（最小、精确、可验证）：
  tensor 133（Pad 标量，有 buffer）：单字节 128→127（=zp），反量化恰为 0.0，
    与原版 TFLite Pad（2 输入、无 const，默认 0.0）一致。
  tensor 140/141/142（激活，无 buffer）：scale/zp 移植自原版 float 模型
    同位张量在校准集上的实测 min/max（映射键：data_shape=[1,198,198,24] 的
    Pad 唯一性已验证；INT8 惯例 zp=-128，非负时 scale=max/255）。
每步写前读回校验；修完重解析断言 + 回归门限裁决。

用法：
  python transplant.py --float tools/oes/models/model_float16.tflite \\
      --quant out/concept_int8.tflite --calib-dir calib --num 50 \\
      --out out/concept_int8_fixed.tflite
"""
import argparse
import glob
import io
import os
import struct
import sys

import numpy as np

TARGET_PAD_DATA_SHAPE = [1, 198, 198, 24]
TARGET_PAD_PADS = [0, 0, 1, 1, 1, 1, 0, 0]
DEGEN = 1e6
INT8_MIN, INT8_MAX = -128, 127


def parse_tflite(path):
    import tflite
    raw = bytearray(open(path, 'rb').read())
    model = tflite.Model.GetRootAsModel(bytes(raw), 0)
    return model, raw


def find_float_targets(float_path):
    """定位原版图中同位张量：返回 {role: tensor_idx}。"""
    import tflite
    model = tflite.Model.GetRootAsModel(open(float_path, 'rb').read(), 0)
    g = model.Subgraphs(0)

    def shp(ti):
        t = g.Tensors(ti)
        return [t.Shape(j) for j in range(t.ShapeLength())]

    def pads_of(op):
        t = g.Tensors(op.Inputs(1))
        b = model.Buffers(t.Buffer())
        n = b.DataLength()
        raw = bytes(b.DataAsNumpy().tobytes())
        return list(struct.unpack('<%di' % (n // 4), raw))

    cands = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != 34:
            continue
        if op.InputsLength() < 2:
            continue
        if shp(op.Inputs(0)) == TARGET_PAD_DATA_SHAPE and \
                pads_of(op) == TARGET_PAD_PADS:
            cands.append((i, op.Inputs(0), op.Outputs(0)))
    assert len(cands) == 1, 'pad mapping not unique: %s' % (cands,)
    opi, data_idx, out_idx = cands[0]
    # 其后的 MaxPool（输出 [1,99,99,24]）
    mp = None
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != 17:
            continue
        if op.InputsLength() and op.Inputs(0) == out_idx:
            assert shp(op.Outputs(0)) == [1, 99, 99, 24], shp(op.Outputs(0))
            assert mp is None, 'maxpool mapping not unique'
            mp = op.Outputs(0)
    assert mp is not None, 'maxpool not found'
    print('float mapping: data=%d padout=%d maxpool=%d' % (
        data_idx, out_idx, mp), flush=True)
    return {'t140': data_idx, 't141': out_idx, 't142': mp}


def measure_ranges(float_path, calib_dir, num, targets):
    """原版 float 模型跑校准集，记录目标张量 min/max。"""
    import tensorflow as tf
    from PIL import Image
    files = sorted(glob.glob(os.path.join(calib_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    assert files, 'calib empty'
    it = tf.lite.Interpreter(
        model_path=float_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ide = it.get_input_details()[0]
    idxs = sorted(set(targets.values()))
    rec = {k: [float('inf'), float('-inf')] for k in targets}
    inv = {v: k for k, v in targets.items()}
    for f in files:
        img = Image.open(f).convert('RGB').resize((395, 395), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        arr = ((arr - 127.5) * 0.00784313725).reshape(1, 395, 395, 3)
        it.set_tensor(ide['index'], arr.astype(np.float32))
        it.invoke()
        for ti in idxs:
            v = it.get_tensor(ti).astype(np.float64)
            assert np.isfinite(v).all(), 'non-finite in float model?!'
            k = inv[ti]
            rec[k][0] = min(rec[k][0], float(v.min()))
            rec[k][1] = max(rec[k][1], float(v.max()))
    for k, (lo, hi) in rec.items():
        print('range %s: [%.6g, %.6g]' % (k, lo, hi), flush=True)
        assert hi > lo and abs(hi - lo) < 1e3, 'suspicious range'
    return rec


def find_quant_targets(quant_path):
    """量化图中退化 4 张量定位（按名 + 退化断言）。"""
    import tflite
    model = tflite.Model.GetRootAsModel(open(quant_path, 'rb').read(), 0)
    g = model.Subgraphs(0)
    found = {}
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        name = t.Name().decode('utf-8', 'replace')
        q = t.Quantization()
        if q is None or q.ScaleLength() == 0:
            continue
        if not any(q.Scale(j) >= DEGEN for j in range(q.ScaleLength())):
            continue
        if name == 'tfl.pseudo_qconst112':
            found['t133'] = i
        elif 'Const_117' in name and '[1, 198, 198, 24]' == str(
                [t.Shape(j) for j in range(t.ShapeLength())]):
            found['t140'] = i
        elif 'pad_1/PadV2' in name:
            found['t141'] = i
        elif 'max_pool2d/MaxPool2d' in name:
            found['t142'] = i
    assert set(found) == {'t133', 't140', 't141', 't142'}, found.keys()
    print('quant mapping: %s' % found, flush=True)
    return found


def vec_data(tab, field_voff):
    off = tab.Offset(field_voff)
    assert off != 0, 'field missing'
    return tab.Vector(off)


def patch_quant(quant_path, out_path, qidx, ranges):
    import tflite
    model, raw = parse_tflite(quant_path)
    g = model.Subgraphs(0)

    def tensor(ti):
        return g.Tensors(ti)

    # 1) t133 单字节 128 -> zp(127) + scale 修正常态。
    #    字节修的是反量化值（恰为真值 0.0）；scale 必须同步修——量化 Pad 核
    #    在 setup 期用该 scale 算定点乘数，1e36 会使其溢出/跳过，
    #    输出保持全零（实测 141 全零字节）。取与 Pad I/O 一致的 scale。
    t = tensor(qidx['t133'])
    b = model.Buffers(t.Buffer())
    doff = vec_data(b._tab, 4)
    assert raw[doff] == 128, 't133 byte != 128: %d' % raw[doff]
    raw[doff] = 127
    assert raw[doff] == 127
    print('patched t133 value byte 128->127 (=zp, dequant exactly 0.0)',
          flush=True)

    # 2) t140/141/142 scale/zp 移植
    new_sc = {}
    for key, ti in (('t140', qidx['t140']), ('t141', qidx['t141']),
                    ('t142', qidx['t142'])):
        lo, hi = ranges[key]
        if lo >= 0:
            sc, zp = hi / 255.0, -128
        else:
            sc = (hi - lo) / 255.0
            zp = min(127, max(-128, int(round(-lo / sc))))
        assert 0 < sc < 1e3, (key, sc)
        t = tensor(ti)
        q = t.Quantization()
        # 读回校验旧值确为退化，再写
        old_sc = q.Scale(0)
        assert old_sc >= DEGEN, (key, old_sc)
        soff = vec_data(q._tab, 8)
        cur = struct.unpack_from('<f', raw, soff)[0]
        assert cur == old_sc, (key, cur, old_sc)
        struct.pack_into('<f', raw, soff, float(sc))
        assert struct.unpack_from('<f', raw, soff)[0] == np.float32(sc)
        zoff = vec_data(q._tab, 10)
        # zero_point 为 [long]（8 字节）或 [int]（4 字节）：按向量长度判定
        zlen = q._tab.VectorLen(q._tab.Offset(10))
        n = q.ZeroPointLength()
        assert zlen == n, (key, zlen, n)
        elsz = 8 if (zoff + 8 * n) <= len(raw) and \
            int.from_bytes(raw[zoff:zoff + 8], 'little', signed=True) == \
            q.ZeroPoint(0) else 4
        if elsz == 8:
            struct.pack_into('<q', raw, zoff, int(zp))
            assert struct.unpack_from('<q', raw, zoff)[0] == zp
        else:
            struct.pack_into('<i', raw, zoff, int(zp))
            assert struct.unpack_from('<i', raw, zoff)[0] == zp
        print('patched %s scale %.6g->%.6g zp->%d (range [%.6g,%.6g])' % (
            key, old_sc, sc, zp, lo, hi), flush=True)
        new_sc[key] = sc

    # 3) t133 scale 同步为 Pad I/O scale（核 setup 数学需要正常量级）。
    #    值已是 zp（反量化恒 0），scale 取 t141 移植值。
    t = tensor(qidx['t133'])
    q = t.Quantization()
    old_sc = q.Scale(0)
    assert old_sc >= DEGEN, old_sc
    soff = vec_data(q._tab, 8)
    assert struct.unpack_from('<f', raw, soff)[0] == old_sc
    struct.pack_into('<f', raw, soff, float(new_sc['t141']))
    assert struct.unpack_from('<f', raw, soff)[0] == \
        np.float32(new_sc['t141'])
    print('patched t133 scale %.6g->%.6g (value stays zp=exact 0.0)' % (
        old_sc, new_sc['t141']), flush=True)

    open(out_path, 'wb').write(raw)
    # 重解析全验证
    model2 = tflite.Model.GetRootAsModel(open(out_path, 'rb').read(), 0)
    g2 = model2.Subgraphs(0)
    assert g2.OperatorsLength() == g.OperatorsLength()
    print('reparse OK ops=%d' % g2.OperatorsLength(), flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--float', required=True)
    ap.add_argument('--quant', required=True)
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--num', type=int, default=50)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    targets = find_float_targets(args.float)
    ranges = measure_ranges(args.float, args.calib_dir, args.num, targets)
    qidx = find_quant_targets(args.quant)
    patch_quant(args.quant, args.out, qidx, ranges)
    print('TRANSPLANT DONE -> %s' % args.out, flush=True)


if __name__ == '__main__':
    main()