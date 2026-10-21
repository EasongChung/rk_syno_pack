#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零依赖最小 TFLite/FlatBuffers 读取器（只读，够用即止）。

## 为什么要自己写

本机没有 `flatbuffers` 模块，也没有 `pip`。而 `tools/tflite_write.py` 的 docstring
已经把 FlatBuffers 的表/向量规则写清楚了（:20-21），照同一条规则反着读即可，
且**不引入任何依赖**。

## 规则（与 tflite_write.py 的注释同源）

1. root table 位置 = `u32 at 0`（相对 0 的 uoffset）
2. table 起始 `P` 的头 4 字节是 **soffset**（有符号）⇒ vtable 在 `P - soffset`
3. vtable：`[vtable_size(u16), table_size(u16), field0_off(u16), field1_off(u16), ...]`
4. `field_off == 0` ⇒ 该字段缺席
5. 字段值位置 = `P + field_off`
6. **间接字段**（string / vector / table）：值位置处是 uoffset，**相对于值位置自身**
   ⇒ 目标 = 值位置 + uoffset

## 只读 TFLite 需要的字段号（tflite_schema.fbs）

```
Model        : version=0  operator_codes=1  subgraphs=2  description=3  buffers=4
SubGraph     : tensors=0  inputs=1  outputs=2  operators=3  name=4
Tensor       : shape=0  type=1  buffer=2  name=3  quantization=4
Operator     : opcode_index=0  inputs=1  outputs=2
OperatorCode : deprecated_builtin_code=0  custom_code=1  version=2  builtin_code=3
Buffer       : data=0
```

★ **`TensorType` 是 `byte` 不是 int32。** 在 TFLite 的 schema 里它是
`enum TensorType : byte`。按 4 字节读会取到错位的垃圾值（本读取器第一版就栽在这：
`type` 读出 21513 / 8190 这类数）。**字段偏移是奇数（如 23）正是 1 字节字段的特征**——
看到奇数偏移就该怀疑那里是个 byte/bool。

★ 本文件只解析上述字段。遇到没见过的字段**直接跳过**（FlatBuffers 天生向前兼容）。
"""

import struct


# ---------------------------------------------------------------- 基础读取

def _u16(b, p):
    return struct.unpack_from("<H", b, p)[0]


def _i32(b, p):
    return struct.unpack_from("<i", b, p)[0]


def _u32(b, p):
    return struct.unpack_from("<I", b, p)[0]


class FB:
    """FlatBuffers 只读视图。所有 `pos` 都是缓冲区内的绝对偏移。"""

    def __init__(self, buf: bytes):
        self.b = buf
        self.root = _u32(buf, 0)          # 规则 1

    # ---- 表字段 ----
    def _vtable(self, table):
        return table - _i32(self.b, table)   # 规则 2

    def field(self, table, field_id):
        """返回字段值的绝对位置；缺席返回 None。"""
        vt = self._vtable(table)
        vt_size = _u16(self.b, vt)
        fo = 4 + 2 * field_id
        if fo + 2 > vt_size:                  # 规则 3/4
            return None
        off = _u16(self.b, vt + fo)
        if off == 0:
            return None
        return table + off

    def scalar(self, table, field_id, fmt=None):
        p = self.field(table, field_id)
        if p is None:
            return None
        if fmt == "u8":
            return self.b[p]
        if fmt == "i8":
            return struct.unpack_from("<b", self.b, p)[0]
        if fmt == "u16":
            return _u16(self.b, p)
        if fmt == "i32":
            return _i32(self.b, p)
        return _u32(self.b, p)                 # 默认 u32（含 uoffset）

    def indirect(self, table, field_id):
        """间接字段（string/vector/table）的**目标绝对位置**；缺席返回 None。"""
        p = self.field(table, field_id)
        if p is None:
            return None
        return p + _u32(self.b, p)             # 规则 6

    def table_at(self, table, field_id):
        return self.indirect(table, field_id)

    # ---- 向量 ----
    def vec_len(self, vec):
        return _u32(self.b, vec)

    def vec_scalar(self, vec, i, fmt="i32"):
        p = vec + 4 + 4 * i
        if fmt == "i32":
            return _i32(self.b, p)
        return _u32(self.b, p)

    def vec_table(self, vec, i):
        p = vec + 4 + 4 * i
        return p + _u32(self.b, p)

    def string(self, table, field_id):
        s = self.indirect(table, field_id)
        if s is None:
            return None
        n = _u32(self.b, s)
        return self.b[s + 4: s + 4 + n].decode("utf-8", "replace")


# ---------------------------------------------------------------- TFLite 语义

# 只列本项目用得到的 builtin op
BUILTIN = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D", 5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP", 9: "FULLY_CONNECTED", 14: "LOGISTIC", 17: "MAX_POOL_2D",
    22: "RESHAPE", 25: "SOFTMAX", 28: "SUB", 34: "PAD", 35: "PADV2",
    39: "TRANSPOSE", 40: "MEAN", 45: "STRIDED_SLICE", 50: "SPLIT",
    51: "SPLIT_V", 54: "SPACE_TO_DEPTH", 95: "LOG", 97: "EXP",
}

# 会走 etnaviv 卷积分支的算子（R5GUARD 只挂在这条分支上）
CONV_OPS = {3, 4}


def op_name(code):
    return BUILTIN.get(code, "op#%d" % code)


def load_model(path):
    """返回 (fb, meta)。meta 含 tensors / operators / buffers 的扁平结构。"""
    with open(path, "rb") as f:
        fb = FB(f.read())

    sub_count = 0
    subgraphs_vec = fb.indirect(fb.root, 2)
    if subgraphs_vec is not None:
        sub_count = fb.vec_len(subgraphs_vec)
    if sub_count == 0:
        raise SystemExit("ERROR: 模型里没有 subgraph")
    sg = fb.vec_table(subgraphs_vec, 0)        # 只看第 0 个（本项目是单图）

    # --- tensors ---
    tensors = []
    tv = fb.indirect(sg, 0)
    n_t = fb.vec_len(tv) if tv is not None else 0
    for i in range(n_t):
        t = fb.vec_table(tv, i)
        shape = []
        sv = fb.indirect(t, 0)
        if sv is not None:
            shape = [fb.vec_scalar(sv, k, "i32") for k in range(fb.vec_len(sv))]
        tensors.append({
            "i": i,
            "shape": shape,
            "rank": len(shape),
            "type": fb.scalar(t, 1, "i8"),      # ★ byte，不是 u32！
            "buffer": fb.scalar(t, 2),
            "name": fb.string(t, 3) or "",
            "sig": (lambda v: [fb.vec_scalar(v, k, "i32") for k in range(fb.vec_len(v))]
                    if v is not None else [])(
                        fb.indirect(t, 7)),   # shape_signature（保留 -1 表示动态维）
        })

    # --- buffers（原始字节；“有没有数据”= nbytes > 0）---
    buffers = []
    bv = fb.indirect(fb.root, 4)
    n_b = fb.vec_len(bv) if bv is not None else 0
    for i in range(n_b):
        bt = fb.vec_table(bv, i)
        dv = fb.indirect(bt, 0)
        if dv is None:
            buffers.append({"nbytes": 0, "data": None})
        else:
            n = fb.vec_len(dv)
            buffers.append({"nbytes": n, "data": fb.b[dv + 4: dv + 4 + n]})

    # --- operator_codes ---
    ocodes = []
    ov = fb.indirect(fb.root, 1)
    n_o = fb.vec_len(ov) if ov is not None else 0
    for i in range(n_o):
        ot = fb.vec_table(ov, i)
        code = fb.scalar(ot, 3)
        if code is None:                       # 老模型只有 deprecated_builtin_code
            code = fb.scalar(ot, 0, "i8") or 0
        ocodes.append(code)

    # --- operators ---
    ops = []
    pv = fb.indirect(sg, 3)
    n_p = fb.vec_len(pv) if pv is not None else 0
    for i in range(n_p):
        p = fb.vec_table(pv, i)
        oci = fb.scalar(p, 0) or 0
        ins = []
        iv = fb.indirect(p, 1)
        if iv is not None:
            ins = [fb.vec_scalar(iv, k) for k in range(fb.vec_len(iv))]
        outs = []
        uv = fb.indirect(p, 2)
        if uv is not None:
            outs = [fb.vec_scalar(uv, k) for k in range(fb.vec_len(uv))]
        ops.append({
            "i": i, "code": ocodes[oci] if oci < len(ocodes) else -1,
            "inputs": ins, "outputs": outs,
        })

    return {"n_tensors": n_t, "tensors": tensors, "buffers": buffers,
            "operators": ops, "n_operators": n_p}


def is_activation(meta, ti):
    """激活张量：buffer 索引为 0 或指向没有数据的 buffer（驱动侧 data == NULL）。"""
    if ti < 0 or ti >= meta["n_tensors"]:
        return False
    b = meta["tensors"][ti]["buffer"]
    if b is None:
        return True
    if b == 0:
        return True
    return b >= len(meta["buffers"]) or meta["buffers"][b]["nbytes"] == 0


# TensorType 枚举（只列本项目见得到的）
TTYPE = {0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
         5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
         10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
         14: "VARIANT", 15: "UINT32", 16: "UINT16"}

_TT_STRUCT = {0: "f", 1: "e", 2: "i", 3: "B", 4: "q", 6: "?", 7: "h",
              9: "b", 10: "d", 15: "I", 16: "H"}
_TT_SIZE = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 6: 1, 7: 2, 9: 1, 10: 8, 15: 4, 16: 2}


def tensor_data(meta, ti):
    """返回该张量 buffer 的**原始字节**；激活张量返回 None。

    ★ 刻意返回原始字节而不是解析后的数组：调用方自己决定怎么解释，
      这样「有没有数据」「多少字节」这类判定不会被我这里的解析悄悄改写。
    """
    if ti < 0 or ti >= meta["n_tensors"]:
        return None
    t = meta["tensors"][ti]
    b = t["buffer"]
    if b is None or b >= len(meta["buffers"]):
        return None
    return meta["buffers"][b]["data"]


def tensor_values(meta, ti):
    """把常量张量解析成 Python 数值列表（按 type 决定元素宽度）。"""
    raw = tensor_data(meta, ti)
    if raw is None:
        return None
    t = meta["tensors"][ti]
    fmt = _TT_STRUCT.get(t["type"])
    if fmt is None:
        return None                      # STRING / 未支持类型：不猜
    sz = _TT_SIZE[t["type"]]
    n = len(raw) // sz
    return list(struct.unpack_from("<%d%s" % (n, fmt), raw, 0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("用法: tflite_min_read.py <model.tflite>")
    m = load_model(sys.argv[1])
    print("tensors=%d operators=%d" % (m["n_tensors"], m["n_operators"]))
    from collections import Counter
    print("rank 分布:", dict(sorted(Counter(t["rank"] for t in m["tensors"]).items())))
    print("算子分布:", dict(sorted(Counter(op_name(o["code"]) for o in m["operators"]).items())))
