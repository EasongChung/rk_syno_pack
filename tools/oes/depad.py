#!/usr/bin/env python3
"""PadV2(3 输入) -> Pad(2 输入)：消掉退化标量 const 依赖。

根因链（实测）：onnx2tf 把 ONNX 2 输入 Pad lower 成 TF 3 输入 Pad
（scalar constant_values=0.0）；MLIR 对该 0/0 标量产出退化 scale；
即使事后把标量值/scale 全修正常，量化 PadV2 核仍输出全零（剖面实测）。
而原生 2 输入量化 Pad 经 minipad 与本模型 stem（op#2）双重验证正常。
数学等价性：3 输入 Pad(data, pads, 0.0) ≡ 2 输入 Pad(data, pads)（默认 0），
仅当标量反量化恰为 0.0 时改写，否则报错退出（不静默改图）。

实现（等长原位改写，无重建）：
  1. op#N.opcode_index := 已有 PAD opcode 表项下标（与 op#2 同项，保证版本兼容）。
  2. op#N.inputs 向量改为 [data, paddings]：在文件尾追加新向量
     [len=2, i0, i1]（4 字节对齐），重写 Operator.inputs 的 uoffset。
  3. 断言双方 builtin_options 均为空/None（Pad/PadV2 options 皆空表）。
改完重解析断言 + 回归门限裁决。

用法：
  python depad.py --model out/concept_int8_fixed2.tflite --out out/concept_nodepad.tflite
"""
import argparse
import struct


def load(path):
    import tflite
    raw = bytearray(open(path, 'rb').read())
    model = tflite.Model.GetRootAsModel(bytes(raw), 0)
    return model, raw


def field_slot(tab, field_id):
    """返回某字段槽位的绝对文件偏移（voffset -> slot pos）。"""
    voff = 4 + field_id * 2
    off = tab.Offset(voff)
    assert off != 0, 'field %d missing' % field_id
    return tab.Pos + off


def read_uoffset(raw, slot_pos):
    return slot_pos + struct.unpack_from('<i', raw, slot_pos)[0]


def main():
    import tflite  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    model, raw = load(args.model)
    g = model.Subgraphs(0)

    # 找 PAD opcode 表项（与现存 2 输入 Pad 同项）
    pad_code_idx = None
    for j in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(j)
        if oc.BuiltinCode() == 34:
            pad_code_idx = j
            break
    assert pad_code_idx is not None, 'no PAD opcode entry!'

    nfix = 0
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != 60:
            continue
        assert op.InputsLength() == 3, 'PadV2 without 3 inputs?!'
        data_ti, pads_ti, val_ti = (op.Inputs(0), op.Inputs(1), op.Inputs(2))
        # 标量必须反量化恰为 0，否则拒绝
        vt = g.Tensors(val_ti)
        vq = vt.Quantization()
        assert vq is not None and vq.ScaleLength() == 1, 'scalar not per-tensor'
        bb = model.Buffers(vt.Buffer())
        vbytes = bytes(bytearray(bb.DataAsNumpy()))
        assert len(vbytes) == 1, 'scalar not single byte?!'
        v = (vbytes[0] - vq.ZeroPoint(0)) * vq.Scale(0)
        assert v == 0.0, 'scalar dequant=%s, refuse!' % v
        # builtin_options：Pad/PadV2 皆无参数，缺席或空表皆可，仅记录
        try:
            bo = op.BuiltinOptions()
            print('  op#%d builtin_options=%r' % (i, bo), flush=True)
        except Exception as e:
            print('  op#%d builtin_options unreadable: %s' % (i, e), flush=True)
        # 改 opcode
        oc_slot = field_slot(op._tab, 0)
        struct.pack_into('<I', raw, oc_slot, pad_code_idx)
        # 新 inputs 向量追加到文件尾（4 对齐）
        new_vec = struct.pack('<III', 2, data_ti, pads_ti)
        while len(raw) % 4:
            raw.append(0)
        new_pos = len(raw)
        raw.extend(new_vec)
        in_slot = field_slot(op._tab, 1)
        struct.pack_into('<i', raw, in_slot, new_pos - in_slot)
        nfix += 1
        print('depad op#%d: PadV2->Pad inputs [%d,%d,%d]->[%d,%d]' % (
            i, data_ti, pads_ti, val_ti, data_ti, pads_ti), flush=True)

    assert nfix >= 1, 'no PadV2 converted?!'
    open(args.out, 'wb').write(raw)
    m2 = tflite.Model.GetRootAsModel(open(args.out, 'rb').read(), 0)
    g2 = m2.Subgraphs(0)
    assert g2.OperatorsLength() == g.OperatorsLength()
    for i in range(g2.OperatorsLength()):
        op = g2.Operators(i)
        code = m2.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert code != 60, 'PadV2 remains!'
    print('DEPAD DONE ops=%d fixed=%d -> %s' % (
        g2.OperatorsLength(), nfix, args.out), flush=True)


if __name__ == '__main__':
    main()
