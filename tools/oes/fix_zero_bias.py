#!/usr/bin/env python3
"""把被量化成 0 的 bias 数值按浮点真值重算回来（等长原地字节补丁）。

## 为什么需要这个工具（正本：报告 66 §14）

`concept_int8_art13.tflite` 的两个卷积 bias 在 int8 转换里被烘焙成了 0：

    op#6  DEPTHWISE_CONV_2D  浮点 bias max = 17.5    -> int8 全零
    op#8  CONV_2D            浮点 bias max = 52.09   -> int8 全零

成因链：标定没观测到 stem 后头 3 个激活 ⇒ in_scale = FLT_MAX/255 = 1.3344e36
⇒ bias 的 int32 按 round(bias_float / (in_scale x w_scale)) 烘焙 ⇒ 归零。
`transplant_scales.py` 事后只修了 **scale**，没有重算 **数值**，所以
`0 x 正确 scale` 仍然是 0 —— 这就是它修了 6 个退化张量仍然不好用的原因。

## 安全设计（三重护栏）

1. **等长原地补丁**：bias 是定长 int32 数组，改写前后字节数完全相同，
   不重打包 flatbuffer。
2. **只允许被点名的 buffer 变化**：写回前逐字节比对，除补丁区间外必须完全一致。
3. **回读断言**：写完重新解析，bias 数值必须等于目标值；越界直接拒绝。

## 拒绝条件（宁可不动，也不做坏）

- bias scale 退化（>1e10）⇒ 该模型还没跑过 transplant_scales，先跑那个
- 目标 int32 越界
- 浮点侧 bias 本身为零（那不是丢失，是本来就没有）

用法：
  python tools/oes/fix_zero_bias.py --quant art/out/concept_int8_art13.tflite \
      --float art/out/baseline_float_concept_split_padsafe_float32.tflite \
      --out art/out/concept_int8_art14.tflite
"""
import argparse
import os
import struct
import sys

INT32_MAX = 2 ** 31 - 1


def _u32(b, p):
    return struct.unpack_from('<I', b, p)[0]


def _i32(b, p):
    return struct.unpack_from('<i', b, p)[0]


def _u16(b, p):
    return struct.unpack_from('<H', b, p)[0]


def _field_pos(b, table, fid):
    """表字段的绝对位置；字段缺省返回 None（flatbuffer vtable 语义）。"""
    vt = table - _i32(b, table)
    vlen = _u16(b, vt)
    off = 4 + 2 * fid
    if off + 2 > vlen:
        return None
    fo = _u16(b, vt + off)
    return None if fo == 0 else table + fo


def _vector(b, pos):
    """字段处是指向 vector 的相对偏移；返回 (元素数据起始, 元素个数)。"""
    vp = pos + _u32(b, pos)
    return vp + 4, _u32(b, vp)


def _table_vector(b, pos):
    """表的 vector：每个元素是相对它自己位置的 u32 偏移。"""
    data, n = _vector(b, pos)
    return [data + i * 4 + _u32(b, data + i * 4) for i in range(n)]


def buffer_data_offsets(buf):
    """返回 [(数据起始偏移 或 None, 字节长度)]，下标 = buffer 索引。

    Model.buffers 是 field 4；Buffer.data 是 field 0（vector<ubyte>）。
    """
    model = _u32(buf, 0)
    fp = _field_pos(buf, model, 4)
    if fp is None:
        return []
    out = []
    for bt in _table_vector(buf, fp):
        dp = _field_pos(buf, bt, 0)
        if dp is None:
            out.append((None, 0))
        else:
            data, n = _vector(buf, dp)
            out.append((data, n))
    return out


def _schema():
    from tensorflow.lite.python import schema_py_generated as fb
    return fb


def _builtin_name(fb, model, opcode_index):
    codes = model.OperatorCodes(opcode_index)
    code = max(codes.BuiltinCode(), codes.DeprecatedBuiltinCode())
    try:
        return fb.BuiltinOperatorNames[code]
    except Exception:
        return 'OP_%d' % code


def _scale_of(t):
    q = t.Quantization()
    if q is None:
        return []
    return [float(q.Scale(i)) for i in range(q.ScaleLength())]


def _ints_of(buf, offsets, buf_index, count):
    """从模型的某个 buffer 读出 count 个 int32。"""
    pos, length = offsets[buf_index]
    if pos is None or length < count * 4:
        return None
    return list(struct.unpack_from('<%di' % count, buf, pos))


def _floats_of(buf, offsets, buf_index, count):
    pos, length = offsets[buf_index]
    if pos is None or length < count * 4:
        return None
    return list(struct.unpack_from('<%df' % count, buf, pos))


def main(argv=None):
    ap = argparse.ArgumentParser(description='重算被量化成 0 的 bias 数值')
    ap.add_argument('--quant', required=True, help='待修的 int8 模型')
    ap.add_argument('--float', required=True, help='浮点参照（bias 真值来源）')
    ap.add_argument('--out', help='输出路径（不给则只报告）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    for p in (args.quant, args.float):
        if not os.path.isfile(p):
            print('missing file: %s' % p, file=sys.stderr)
            return 2

    fb = _schema()
    qbuf = open(args.quant, 'rb').read()
    fbuf = open(args.float, 'rb').read()
    qm = fb.Model.GetRootAsModel(qbuf, 0)
    fm = fb.Model.GetRootAsModel(fbuf, 0)
    qg, fg = qm.Subgraphs(0), fm.Subgraphs(0)
    qoff = buffer_data_offsets(qbuf)
    foff = buffer_data_offsets(fbuf)

    import perop_diff as P

    def seq_of(model, g):
        names = [g.Tensors(i).Name().decode('utf-8', 'replace')
                 for i in range(g.TensorsLength())]
        seq, act, seen = [], [], set()
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            # ★ 必须挡住 -1：TFLite 用 -1 表示「可选输入/输出缺省」。
            # 第一版直接 names[op.Inputs(j)]，-1 会取到**最后一个**张量名
            # （Python 负索引），把激活输入子序列搞乱 ⇒ 结构对齐覆盖 0%，
            # 被工具自己的护栏拒掉（CI run 40283888816 实测）。
            outs = [names[op.Outputs(j)] if op.Outputs(j) >= 0 else ''
                    for j in range(op.OutputsLength())]
            ins = [names[op.Inputs(j)] if op.Inputs(j) >= 0 else ''
                   for j in range(op.InputsLength())]
            shp = []
            for j in range(op.OutputsLength()):
                oi = op.Outputs(j)
                if oi < 0:
                    shp.append(())
                    continue
                t = g.Tensors(oi)
                shp.append(tuple(t.Shape(k) for k in range(t.ShapeLength())))
            seq.append((i, _builtin_name(fb, model, op.OpcodeIndex()), outs, ins, shp))
            for o in outs:
                if o not in seen:
                    seen.add(o)
                    act.append(o)
        return seq, act

    fseq, fact = seq_of(fm, fg)
    qseq, qact = seq_of(qm, qg)
    pairs, info = P.align_struct(fseq, qseq, fact, qact)
    print('=== 对齐: %s' % info['reason'])
    # cover 必须自己算：align_struct 不带这个字段（那是 align() 的返回值），
    # 直接取 info['cover'] 会恒为 0 而被护栏误拒（本地实测踩到）。
    cover = len(pairs) / float(len(fact)) if fact else 0.0
    if info['conflict'] != 0 or cover < 0.9:
        print('!! 对齐不可信（同名冲突 %d，覆盖 %.1f%% < 90%%）⇒ 拒绝动手'
              % (info['conflict'], cover * 100), file=sys.stderr)
        return 4

    fop_of, qop_of = {}, {}
    for s in fseq:
        for n in s[2]:
            fop_of[n] = s[0]
    for s in qseq:
        for n in s[2]:
            qop_of[n] = s[0]

    patches, skipped = [], []
    for fname, qname in pairs:
        fi, qi = fop_of.get(fname), qop_of.get(qname)
        if fi is None or qi is None:
            continue
        fop, qop = fg.Operators(fi), qg.Operators(qi)
        fty = _builtin_name(fb, fm, fop.OpcodeIndex())
        qty = _builtin_name(fb, qm, qop.OpcodeIndex())
        if fty != qty or fty not in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'):
            continue
        if fop.InputsLength() < 3 or qop.InputsLength() < 3:
            continue
        fbi, qbi = fop.Inputs(2), qop.Inputs(2)
        if fbi < 0 or qbi < 0:
            continue
        qt = qg.Tensors(qbi)
        count = 1
        for k in range(qt.ShapeLength()):
            count *= qt.Shape(k)
        fvals = _floats_of(fbuf, foff, fg.Tensors(fbi).Buffer(), count)
        qvals = _ints_of(qbuf, qoff, qt.Buffer(), count)
        sc = _scale_of(qt)
        if fvals is None or qvals is None or not sc:
            continue
        if len(fvals) != count or len(qvals) != count:
            continue
        if max(abs(x) for x in sc) > 1e10:
            skipped.append((fi, fname, 'bias scale 退化（先跑 transplant_scales）'))
            continue
        if max(abs(x) for x in fvals) < 1e-6:
            continue
        if any(x != 0 for x in qvals):
            continue
        new, bad = [], False
        for i, fv in enumerate(fvals):
            s = sc[min(i, len(sc) - 1)]
            q = int(round(fv / s))
            if abs(q) > INT32_MAX:
                bad = True
                break
            new.append(q)
        if bad:
            skipped.append((fi, fname, '目标 int32 越界'))
            continue
        patches.append((qt.Buffer(), count, new, fi, fty, fname,
                        max(abs(x) for x in fvals)))

    if skipped:
        print('=== 跳过 %d 项' % len(skipped))
        for fi, nm, why in skipped:
            print('   ref#%-4d %-44s  %s' % (fi, nm[-44:], why))

    print('=== 需要修 bias 的算子: %d 个' % len(patches))
    for pb, count, new, fi, fty, fname, mx in patches:
        print('   ref#%-4d %-18s buffer=%-4d n=%-4d 浮点|b|max=%8.4f int32[0..2]=%s'
              % (fi, fty, pb, count, mx, new[:3]))

    if not patches:
        print('=== 无需修改（这是正控制该有的输出）')
        return 0

    out = bytearray(qbuf)
    for pb, count, new, fi, fty, fname, mx in patches:
        pos, length = qoff[pb]
        if pos is None or length < count * 4:
            print('!! buffer %d 长度不足，拒绝' % pb, file=sys.stderr)
            return 4
        struct.pack_into('<%di' % count, out, pos, *new)

    if len(out) != len(qbuf):
        print('!! 长度变化，拒绝', file=sys.stderr)
        return 4
    allowed = bytearray(len(out))
    for pb, count, new, fi, fty, fname, mx in patches:
        pos, length = qoff[pb]
        for k in range(count * 4):
            allowed[pos + k] = 1
    diff = [i for i in range(len(out)) if out[i] != qbuf[i]]
    stray = [i for i in diff if not allowed[i]]
    if stray:
        print('!! %d 个字节落在补丁区间外，拒绝（样例 %s）' % (len(stray), stray[:5]),
              file=sys.stderr)
        return 4
    print('=== 护栏: 变化字节 %d 个，全部落在被点名的 bias 区间内 ✓' % len(diff))

    bb = bytes(out)
    for pb, count, new, fi, fty, fname, mx in patches:
        got = _ints_of(bb, qoff, pb, count)
        if got != new:
            print('!! 回读不一致 buffer=%d，拒绝' % pb, file=sys.stderr)
            return 4
    print('=== 护栏: 回读断言全部命中 ✓')

    if args.dry_run or not args.out:
        print('dry-run: 未写文件')
        return 0
    with open(args.out, 'wb') as f:
        f.write(bb)
    print('WROTE %s (%d bytes)' % (args.out, len(bb)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
