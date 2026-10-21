#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_tiny_model.py —— 生成用于真机验证 NPU 通路的**小 int8 模型**。

## 为什么

大模型 `pertensor.tflite`（6 MB / 350 tensor / 82 partition）上，
`675a1091` 在 Invoke 的输出回读阶段 SIGSEGV（`read_outputs` 里 `src_map[k]` 野指针）。
即便修好，**第 ④ 级（与 CPU 逐字节一致）仍是巨大挑战** —— NPU 与 CPU 的累加顺序、
舍入、int8 量化点都可能不同。

小模型的价值：**若它能跑通 ④ 级**，就证明「NPU 通路本身可用」，
问题只在这个大模型的特定结构上 ⇒ 直接改变后续投入方向。

## 设计（刻意避开已知不支持的算子）

从真机接管日志里统计出的 `unsupported` 类型：
`STRIDED_SLICE 26 / TRANSPOSE 16 / CONV 2 / unknown 1 / MEAN 1 / MAXPOOL 1 / FC 1`。

⇒ **只用 CONV_2D 与 ABSOLUTE**，其余一律不用。

结构：

```
X  : int8 [1, 32, 32, 3]     输入
→ CONV_2D  f[8,3,3,3]   stride 1  pad SAME  ReLU      ⇒ Y1 int8 [1,32,32,8]
→ CONV_2D  f[8,3,3,8]   stride 2  pad SAME  ReLU      ⇒ Y2 int8 [1,16,16,8]
→ ABSOLUTE                                          ⇒ Y  int8 [1,16,16,8]  输出
```

**为什么第二个 conv 用 stride=2**：stride>1 会走 `strided_to_normal` → `reshape` → `subsample`，
正是大模型崩溃所在的**同一条代码路径**。小模型也走它，才能验证这条路径。

`padding = SAME`，`activation = RELU`（融合进 Conv2DOptions，不引入额外算子）。

## 量化（必须是 per-tensor，teflon 准入要求）

所有张量用 per-tensor 量化，`quantized_dimension = 0`。
`scale` 取 1/127 量级，`zero_point` = -128（int8 对称便于对比）。

⚠️ weight/bias 的量化参数**故意取规整值**（0.00784314 = 1/127、0.0 等），
让 CPU 与 NPU 的算术更容易对齐；但**不保证一定逐字节相同** —— 那要真机测。

## 自检

生成后用**独立实现** `verify/tflite_schema.py` 解析回来，逐项核对：
tensor 类型/shape/量化、算子拓扑、builtin_options、buffer 字节数。
解析不通过就**直接报错退出**，不产出文件 —— 避免把坏模型送上设备。

用法：
    python3 tools/make_tiny_model.py [-o models/tiny_int8.tflite]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tflite_write import Builder  # noqa: E402

# ── TFLite 常量（取自 schema.fbs / schema_generated.h）────────────────────
TENSOR_TYPE_INT8 = 9
BUILTIN_CONV_2D = 3
BUILTIN_ABS = 4
CONV2D_PADDING_SAME = 0
ACTIVATION_RELU = 2

# ── 量化参数（规整值，便于 CPU/NPU 对齐）────────────────────────────────
Z_IN = -128
S_IN = 1.0 / 127.0            # 输入
S_W1 = 1.0 / 127.0
S_B1 = 0.0
S_A1 = 1.0 / 127.0            # Y1
S_W2 = 1.0 / 127.0
S_B2 = 0.0
S_A2 = 1.0 / 127.0            # Y2 / Y（ABSOLUTE 不改量化）

# 张量布局：0=输入 1..4=W1,B1,W2,B2  5=Y1  6=Y2  7=输出
T_IN, T_W1, T_B1, T_Y1, T_W2, T_B2, T_Y2, T_OUT = range(8)


def _qparams(b: Builder, scale: float, zero_point: int):
    """构造 QuantizationParameters。"""
    f = b.create_float_vector([scale])
    z = b.create_long_vector([zero_point])
    b.start_object(5)     # min(0) max(1) scale(2) zero_point(3) quantized_dimension(4)
    b.add_uoffset(2, f)
    b.add_uoffset(3, z)
    return b.end_object()


def _tensor(b: Builder, shape, ttype, buffer_idx, name, qp_off):
    # ⚠️ 子对象必须在 start_object **之前**建好，否则会被算进 table_size。
    v_shape = b.create_int32_vector(list(shape))
    s_name = b.create_string(name)
    b.start_object(9)     # shape(0) type(1) buffer(2) name(3) quantization(4) is_variable(5) …
    b.add_uoffset(0, v_shape)
    b.prepend_int8(1, ttype)
    b.prepend_uint8(2, buffer_idx)
    b.add_uoffset(3, s_name)
    b.add_uoffset(4, qp_off)
    return b.end_object()


def _conv_options(b: Builder, stride, padding, activation):
    """Conv2DOptions：padding(0) stride_w(1) stride_h(2) fused_activation(3) …"""
    b.start_object(5)
    b.prepend_int8(0, padding)
    b.prepend_int32(1, stride)
    b.prepend_int32(2, stride)
    b.prepend_int8(3, activation)
    return b.end_object()


def _buffer(b: Builder, data: bytes):
    b.start_object(3)     # data(0) offset(1) size(2)
    if data:
        b.add_uoffset(0, b.create_byte_vector(data))
    return b.end_object()


def _operator(b: Builder, opcode_index, inputs, outputs, builtin_options=None):
    """Operator：opcode_index(0) inputs(1) outputs(2) builtin_options(3) …

    ⚠️ **没有** `builtin_options_type` 字段 —— TFLite 用 BuiltinOptions **union**，
       类型由 `builtin_options` 指向的表的 vtable 内容推断。
       （本项目 `verify/tflite_schema.py` 文件头也明确记了这一条；
         早先按「有 type 字段」写会把所有字段 id 错位一位。）
    """
    # ⚠️ 同 _tensor：所有子对象先建好再 start_object
    v_in = b.create_int32_vector(list(inputs))
    v_out = b.create_int32_vector(list(outputs))
    b.start_object(8)
    b.prepend_uint8(0, opcode_index)
    b.add_uoffset(1, v_in)
    b.add_uoffset(2, v_out)
    if builtin_options is not None:
        b.add_uoffset(3, builtin_options)
    return b.end_object()


def _opcode(b: Builder, code, custom=None, version=1):
    """OperatorCode：deprecated_builtin_code(0) custom_code(1) version(2) builtin_code(3)"""
    b.start_object(4)
    b.prepend_int8(0, code if code < 127 else 127)   # deprecated 字段，只放低 8 位
    if custom:
        b.add_uoffset(1, b.create_string(custom))
    b.prepend_int32(2, version)
    b.prepend_int32(3, code)          # v2.13+ 用 builtin_code
    return b.end_object()


def build(seed: int = 7) -> bytes:
    """构造模型字节流。权重用**确定性伪随机**（固定 seed），保证可复现。"""
    def rnd(n: int) -> bytes:
        # xorshift32，避免依赖 random 的实现细节
        out = bytearray()
        s = seed
        for _ in range(n):
            s ^= (s << 13) & 0xFFFFFFFF
            s ^= s >> 17
            s ^= (s << 5) & 0xFFFFFFFF
            out.append(s & 0xFF)
        return bytes(out)

    b = Builder()

    # buffers：0 = 空（约定），1..4 = 权重/偏置
    w1 = rnd(8 * 3 * 3 * 3)
    b1 = rnd(8)
    w2 = rnd(8 * 3 * 3 * 8)
    b2 = rnd(8)

    buf_offsets = [_buffer(b, b"")]                      # 0: 空
    buf_offsets.append(_buffer(b, w1))                   # 1
    buf_offsets.append(_buffer(b, b1))                   # 2
    buf_offsets.append(_buffer(b, w2))                   # 3
    buf_offsets.append(_buffer(b, b2))                   # 4

    # 量化参数
    qp_in = _qparams(b, S_IN, Z_IN)
    qp_w1 = _qparams(b, S_W1, 0)
    qp_b1 = _qparams(b, S_B1, 0)
    qp_a1 = _qparams(b, S_A1, -128)
    qp_w2 = _qparams(b, S_W2, 0)
    qp_b2 = _qparams(b, S_B2, 0)
    qp_a2 = _qparams(b, S_A2, -128)

    # tensors
    tensors = [
        _tensor(b, [1, 32, 32, 3], TENSOR_TYPE_INT8, 0, "in", qp_in),
        _tensor(b, [8, 3, 3, 3], TENSOR_TYPE_INT8, 1, "w1", qp_w1),
        _tensor(b, [8], TENSOR_TYPE_INT8, 2, "b1", qp_b1),
        _tensor(b, [1, 32, 32, 8], TENSOR_TYPE_INT8, 0, "y1", qp_a1),
        _tensor(b, [8, 3, 3, 8], TENSOR_TYPE_INT8, 3, "w2", qp_w2),
        _tensor(b, [8], TENSOR_TYPE_INT8, 4, "b2", qp_b2),
        _tensor(b, [1, 16, 16, 8], TENSOR_TYPE_INT8, 0, "y2", qp_a2),
        _tensor(b, [1, 16, 16, 8], TENSOR_TYPE_INT8, 0, "out", qp_a2),
    ]

    # operators：CONV2D(3) / CONV2D(3) / ABS(4)
    ops = [
        _operator(b, 0, [T_IN, T_W1, T_B1], [T_Y1],
                  _conv_options(b, 1, CONV2D_PADDING_SAME, ACTIVATION_RELU)),
        _operator(b, 0, [T_Y1, T_W2, T_B2], [T_Y2],
                  _conv_options(b, 2, CONV2D_PADDING_SAME, ACTIVATION_RELU)),
        _operator(b, 1, [T_Y2], [T_OUT]),
    ]

    # opcode：0 = CONV_2D，1 = ABS
    opcodes = [_opcode(b, BUILTIN_CONV_2D, version=6), _opcode(b, BUILTIN_ABS, version=2)]

    # ⚠️ **所有子对象必须在 start_object 之前构建好**。
    #    在 start_object 之后才 create_* 的话，那些字节会被 prepend 到 table 字段
    #    区域**之前**（文件更靠前），于是被算进 table_size —— vtable 里的
    #    table_size 虚高、field_off 看似越界。这个坑踩过一次（table_len 报 158）。
    sg_name = b.create_string("tiny_int8")
    v_sg_tensors = b.create_offset_vector(tensors)
    v_sg_inputs = b.create_int32_vector([T_IN])
    v_sg_outputs = b.create_int32_vector([T_OUT])
    v_sg_ops = b.create_offset_vector(ops)

    b.start_object(5)     # tensors(0) inputs(1) outputs(2) operators(3) name(4)
    b.add_uoffset(0, v_sg_tensors)
    b.add_uoffset(1, v_sg_inputs)
    b.add_uoffset(2, v_sg_outputs)
    b.add_uoffset(3, v_sg_ops)
    b.add_uoffset(4, sg_name)
    subgraph = b.end_object()

    m_desc = b.create_string("och-tiny-int8")
    v_opcodes = b.create_offset_vector(opcodes)
    v_subgraphs = b.create_offset_vector([subgraph])
    v_buffers = b.create_offset_vector(buf_offsets)

    b.start_object(5)     # version(0) operator_codes(1) subgraphs(2) description(3) buffers(4)
    b.prepend_uint8(0, 3)
    b.add_uoffset(1, v_opcodes)
    b.add_uoffset(2, v_subgraphs)
    b.add_uoffset(3, m_desc)
    b.add_uoffset(4, v_buffers)
    root = b.end_object()
    # TFLite 模型的文件标识（flatbuffer file_identifier，文件偏移 4..8）
    return b.finish(root, identifier=b"TFL3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="models/tiny_int8.tflite")
    args = ap.parse_args()

    data = build()

    # ── 用独立实现自检（解析不通过就不落盘）────────────────────────────
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "verify"))
    import tflite_schema  # noqa: E402

    try:
        m = tflite_schema.parse_model(data, "<generated>")
    except Exception as exc:                       # noqa: BLE001
        print("自检失败：独立解析器无法解析生成的字节流：", exc, file=sys.stderr)
        return 1

    sg = m.subgraphs[0]
    checks = [
        ("子图数", len(m.subgraphs) == 1),
        ("tensor 数", len(sg.tensors) == 8),
        ("算子数", len(sg.operators) == 3),
        ("输入 shape", list(sg.tensors[0].shape) == [1, 32, 32, 3]),
        ("输出 shape", list(sg.tensors[7].shape) == [1, 16, 16, 8]),
        ("w1 字节数", m.buffers[1].data == (8 * 3 * 3 * 3)),
        ("w2 字节数", m.buffers[3].data == (8 * 3 * 3 * 8)),
        ("算子类型", [sg.operators[i].builtin_name for i in range(3)]
         == ["CONV", "CONV", "ABS"]),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'✅' if ok else '★'} {n}")
    if bad:
        print("自检失败项：" + ", ".join(bad), file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"\n已生成 {args.out}  {len(data)} 字节")
    print(f"  sha256 = {__import__('hashlib').sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())