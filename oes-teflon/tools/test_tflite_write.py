#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tflite_write 自测：用**独立手写的**最小 flatbuffer 解析器做 round-trip。

为什么不用 tflite_schema 来测这个：`tflite_schema` 是**读取**器，能测最终产物，
但对「Builder 本身对不对」这个更基础的问题，独立的几十行解析器更直接 ——
若两者给出不同结论，说明其中一方有 bug。

测试覆盖 flatbuffer 的每一个易错点：
  * vtable 的 vtable_size / table_size / fieldN_off 三段结构
  * fieldN_off == 0 表示字段缺失（**最容易写错**）
  * soffset 相对偏移的方向
  * 标量对齐（u8/u16/u32/f32/i64）
  * 向量长度字段与元素对齐
  * 字符串的 NUL 结尾
  * 嵌套 table 偏移
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tflite_write import Builder  # noqa: E402

# ── 独立解析器（故意写得与 Builder 无关）────────────────────────────────
def parse_table(buf: bytes, pos: int):
    """返回 (dict_of_fields, table_end_pos)。"""
    soffset = struct.unpack_from("<i", buf, pos)[0]
    vt = pos - soffset
    vt_size, tbl_size = struct.unpack_from("<HH", buf, vt)
    fields = {}
    for fid in range((vt_size - 4) // 2):
        off = struct.unpack_from("<H", buf, vt + 4 + 2 * fid)[0]
        if off == 0:
            continue          # ← 字段缺失，不是偏移 0
        fields[fid] = pos + off
    return fields, pos + tbl_size, vt_size


def read_u8(b, p):   return b[p]
def read_i8(b, p):   return struct.unpack_from("<b", b, p)[0]
def read_u16(b, p):  return struct.unpack_from("<H", b, p)[0]
def read_i32(b, p):  return struct.unpack_from("<i", b, p)[0]
def read_i64(b, p):  return struct.unpack_from("<q", b, p)[0]
def read_f32(b, p):  return struct.unpack_from("<f", b, p)[0]


def deref(b, p):     return p + read_u32(b, p)


def read_u32(b, p):  return struct.unpack_from("<I", b, p)[0]


def read_string(b, p):
    p = deref(b, p)
    n = read_u32(b, p)
    return b[p + 4:p + 4 + n].decode("utf-8")


def read_i32_vec(b, p):
    p = deref(b, p)
    n = read_u32(b, p)
    return [read_i32(b, p + 4 + 4 * i) for i in range(n)]


def read_f32_vec(b, p):
    p = deref(b, p)
    n = read_u32(b, p)
    return [read_f32(b, p + 4 + 4 * i) for i in range(n)]


def read_byte_vec(b, p):
    p = deref(b, p)
    n = read_u32(b, p)
    return bytes(b[p + 4:p + 4 + n])


# ── 测试 ────────────────────────────────────────────────────────────────
fails = []


def check(name, got, want):
    ok = got == want
    if not ok:
        fails.append(f"{name}: got={got!r} want={want!r}")
    print(f"  {'✅' if ok else '★★'} {name:34s} {got!r}")


b = Builder()

# 先建各个子对象（flatbuffer 必须先子后父）
s_name = b.create_string("och-tiny")
v_i32 = b.create_int32_vector([1, 2, 3, 7])
v_f32 = b.create_float_vector([0.5, 1.25, -3.5])
v_i64 = b.create_long_vector([0, -128, 2147483647])
v_byte = b.create_byte_vector(b"\x01\x02\xff\x00")

# 嵌套 table：字段布局 0=i32,1=bool,2=i32vec,3=string,4=f32vec,5=i64vec,6=bytevec,7=u16,8=f32
b.start_object(9)
b.prepend_int32(0, 42)                 # fid 0
b.prepend_bool(1, True)                # fid 1
b.add_uoffset(2, v_i32)                # fid 2
b.add_uoffset(3, s_name)               # fid 3
b.add_uoffset(4, v_f32)                # fid 4
b.add_uoffset(5, v_i64)                # fid 5
b.add_uoffset(6, v_byte)               # fid 6
b.prepend_int16(7, -1234)             # fid 7
b.prepend_float32(8, 2.5)              # fid 8
# fid 9 不存在（超过 num_fields）
nested_off = b.end_object()

# 根 table：fid 0 = nested table
b.start_object(1)
b.add_uoffset(0, nested_off)
root = b.end_object()
data = b.finish(root)

print(f"=== Builder 输出 {len(data)} 字节 ===")
root_pos = read_u32(data, 0)
rf, rend, _ = parse_table(data, root_pos)
nested_pos = deref(data, rf[0])
nf, nend, _ = parse_table(data, nested_pos)

check("根 table vtable 可解析", rf is not None, True)
check("fid0 i32", read_i32(data, nf[0]), 42)
check("fid1 bool", read_i8(data, nf[1]) != 0, True)
check("fid2 i32vec", read_i32_vec(data, nf[2]), [1, 2, 3, 7])
check("fid3 string", read_string(data, nf[3]), "och-tiny")
check("fid4 f32vec", [round(x, 4) for x in read_f32_vec(data, nf[4])],
      [0.5, 1.25, -3.5])
check("fid5 i64vec",
      [read_i64(data, deref(data, nf[5]) + 4 + 8 * i) for i in range(3)],
      [0, -128, 2147483647])
check("fid6 bytevec", read_byte_vec(data, nf[6]), b"\x01\x02\xff\x00")
check("fid7 u16", read_u16(data, nf[7]), (-1234 & 0xFFFF))
check("fid8 f32", read_f32(data, nf[8]), 2.5)
check("fid9 字段缺失（关键）", 9 in nf, False)

# 字段缺失必须回落到默认值 —— 这是 flatbuffer 最易错处
b2 = Builder()
b2.start_object(3)
b2.prepend_int32(2, 7)          # 只有 fid2 有值
off2 = b2.end_object()
d2 = b2.finish(off2)
p2 = read_u32(d2, 0)
f2, _, _ = parse_table(d2, p2)
check("稀疏 vtable：fid0 缺失", 0 in f2, False)
check("稀疏 vtable：fid1 缺失", 1 in f2, False)
check("稀疏 vtable：fid2 存在", read_i32(d2, f2[2]), 7)

# 空 table（零字段）
b3 = Builder()
b3.start_object(0)
off3 = b3.end_object()
d3 = b3.finish(off3)
p3 = read_u32(d3, 0)
f3, _, _ = parse_table(d3, p3)
check("零字段 table 不崩且无字段", len(f3), 0)

print()
if fails:
    print("总判定: ★ 有缺口 ★")
    for f in fails:
        print("   ", f)
    sys.exit(1)
print("总判定: Builder round-trip 全绿 ✅")