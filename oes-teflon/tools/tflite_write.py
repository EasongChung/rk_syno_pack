#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tflite_write.py —— 极简 TFLite flatbuffer **写入器**（纯标准库）。

## 为什么存在

本机 Alpine/musl，**无 pip、无 flatc、无 flatbuffers/tensorflow**，
但已有纯标准库的读取器 `verify/tflite_schema.py`（本项目自研）。
为了能离线生成小模型做真机验证，需要一个对应的写入器。

**不重复实现 flatbuffer 规范**，而是照 flatbuffers 官方 Python `Builder`
的核心算法实现（`Prep`/`Pad`/`Offset`/`StartObject`/`EndObject`/`Finish`
与向量、字符串的相对偏移编码）。算法正确性由 `tflite_schema.parse_model`
**反向自检**保证 —— 能被那个独立实现解析出来，才算编码正确。

## flatbuffer 布局要点（与 tflite_schema.py 的规则一一对应）

1. 文件第 0 字节起是 uint32「根偏移」，根 table 在 `buf[root_offset]`。
2. 构建**从文件末尾往前**（`Offset()` = 已构建字节数，越大越靠前）。
3. table 起始 P 的头 4 字节是 **soffset**（有符号），vtable 在 `P - soffset`。
4. vtable：`[vtable_size(u16), table_size(u16), field0_off(u16), ...]`，
   `fieldN_off` 相对 table 起点；**0 表示字段缺失**。
5. 表/字符串/向量字段存 **uint32 相对偏移**：真实位置 = `(P+fieldN_off) + 该值`。
6. 对齐：写标量前必须让「写完后的 Offset」满足该类型大小的对齐。
"""

from __future__ import annotations

import struct


class Builder:
    """照 flatbuffers 官方 Python Builder 的核心算法。"""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.minalign = 1
        self.current_vtable: list[int] | None = None
        self.object_end = 0
        self.head = 0

    # ── 基础 ────────────────────────────────────────────────────────────
    def offset(self) -> int:
        """已构建字节数。偏移从文件末尾算起。"""
        return len(self.buf)

    def pad(self, n: int) -> None:
        if n > 0:
            self.buf[0:0] = b"\x00" * n

    def prep(self, size: int, additional: int) -> None:
        """写标量前补齐对齐。

        要求：写完 `size` 字节后 `Offset()` 是 `minalign` 的倍数
        （`minalign` 取历史最大字段宽度，故 ≥ size）。

        ⚠️ 这里**不照抄** flatbuffers 官方那个 `while minalign > alignSize: Pad(1)`
        的写法 —— 本实现验证过它只 pad 1 字节、会让 4 字节字段落在 2 字节对齐的
        地址上（len=201 时得到 202 而非 204）。直接用取模一步到位，语义更清楚。
        """
        if size > self.minalign:
            self.minalign = size
        align = (-(len(self.buf) + additional)) & (self.minalign - 1)
        self.pad(align)

    def place(self, data: bytes, size: int) -> None:
        self.prep(size, 0)
        assert len(data) == size, (len(data), size)
        # ⚠️ head 必须是**字段起始**的 offset-from-end，也就是写入**之后**的 Offset()
        #    （prepend 之后新数据占据索引 0..size-1，其起始 offset = len）。
        #    用写入前的 offset 会让 vtable 里每个 field_off 都少 size（此坑踩过）。
        self.buf[0:0] = data
        self.head = self.offset()

    def prepend_sint(self, x: int, size: int) -> None:
        self.place(x.to_bytes(size, "little", signed=True), size)

    def prepend_uint(self, x: int, size: int) -> None:
        self.place(x.to_bytes(size, "little", signed=False), size)

    # ── 相对偏移（指向已构建的对象）────────────────────────────────────
    def prepend_uoffset(self, off: int) -> None:
        """把「指向 off 的相对偏移」写进去。

        公式对 **table 字段**与 **vector 元素**是**同一个**：
            真实位置 = 写入位置 + 值  ⇒  值 = off - 写入位置
        （标准实现里那个 nested 特判只是为了处理向量的长度占位，
          本实现用 `p = self.offset()`（写入前）直接表达，等价且更少状态。）
        """
        self.prep(4, 0)
        self.buf[0:0] = (0).to_bytes(4, "little")   # 占位，写入后回填
        self.head = self.offset()                  # 字段起始的 offset-from-end
        # 值 = 字段起始 - 目标起始（都是 offset-from-end）
        #   两者同为「目标的起始字节位置」，故差即文件里的距离。
        assert off <= self.head, (off, self.head)
        struct.pack_into("<I", self.buf, len(self.buf) - self.head,
                         self.head - off)

    def add_uoffset(self, field_id: int, off: int, default: int = 0) -> None:
        if off != default:
            self.prepend_uoffset(off)
            self.slot(field_id)

    # ── 对象（table）────────────────────────────────────────────────────
    def start_object(self, num_fields: int) -> None:
        self.current_vtable = [0] * num_fields
        self.object_end = self.offset()

    def slot(self, field_id: int) -> None:
        # 记**写入位置**（self.head），不是写完后的 offset ——
        # 否则 vtable 里的 field_off 会整体少 4，字段全部错位。
        self.current_vtable[field_id] = self.head

    def prepend_bool(self, field_id: int, x: bool, default: bool = False) -> None:
        if x != default:
            self.prepend_sint(1 if x else 0, 1)
            self.slot(field_id)

    def prepend_int8(self, field_id: int, x: int, default: int = 0) -> None:
        if x != default:
            self.prepend_sint(x, 1)
            self.slot(field_id)

    def prepend_uint8(self, field_id: int, x: int, default: int = 0) -> None:
        if x != default:
            self.prepend_uint(x, 1)
            self.slot(field_id)

    def prepend_int16(self, field_id: int, x: int, default: int = 0) -> None:
        if x != default:
            self.prepend_sint(x, 2)
            self.slot(field_id)

    def prepend_int32(self, field_id: int, x: int, default: int = 0) -> None:
        if x != default:
            self.prepend_sint(x, 4)
            self.slot(field_id)

    def prepend_float32(self, field_id: int, x: float, default: float = 0.0) -> None:
        if x != default:
            # 必须走 place()：它会设置 self.head，slot() 依赖它。
            # 早先这里直接 buf[0:0] = ...，导致 slot() 拿到上一次的 head，
            # vtable 里该字段偏移错误（此坑踩过一次）。
            self.place(struct.pack("<f", x), 4)
            self.slot(field_id)

    def end_object(self) -> int:
        """结束 table，返回 table 起点的 offset。

        顺序（照 flatbuffers 标准算法）：
          1. prepend soffset **占位** 4 字节 → 此时 offset 即 table 起点 P
          2. 构建 vtable（u16 对齐），字段偏移 = `object_offset - slot_offset`
          3. 回填 soffset = `vt_start - object_offset + 4`
             （依据 `tflite_schema` 的规则：vtable 位于 `P - soffset`）
        """
        self.prep(4, 0)
        self.buf[0:0] = b"\x00\x00\x00\x00"     # 占位，随后回填
        object_offset = self.offset()            # = P 对应的 offset

        vt = list(self.current_vtable or [])
        while vt and vt[-1] == 0:                # 裁掉尾部未使用字段
            vt.pop()

        vt_size = 4 + 2 * len(vt)
        table_size = object_offset - self.object_end
        assert table_size > 0, (table_size, object_offset, self.object_end)

        self.prep(2, 0)                          # vtable 以 u16 对齐
        for i in range(len(vt) - 1, -1, -1):
            # ⚠️ 未设置的字段必须写 **0**（flatbuffer 用 0 表示「字段缺失」）。
            #    若直接算 `object_offset - 0` 会得到一个非 0 的偏移，
            #    解析器就会把它当成「字段存在」—— 稀疏 table 全错（此坑踩过）。
            off = (object_offset - vt[i]) if vt[i] != 0 else 0
            assert 0 <= off <= 0xFFFF, ("field offset 越界", i, off, vt[i])
            self.buf[0:0] = off.to_bytes(2, "little")
        self.buf[0:0] = table_size.to_bytes(2, "little")
        self.buf[0:0] = vt_size.to_bytes(2, "little")
        # vtable 起点 offset = 写完后的 offset（prepend ⇒ 最后写的在最前）
        vt_end = self.offset()

        # 回填 soffset。⚠️ 索引不是 0，也不是 len-4：
        #    `buf[0]` 是文件的**最前**字节，soffset 占位写完那一刻位于索引 0，
        #    此后全部操作都是**前插** ⇒ 其当前索引 = len(buf) - object_offset。
        #    （把 buffer 索引与文件位置混为一谈，是本文件最坑的一个 bug。）
        # soffset = P - vtable_pos = (total-object_offset) - (total-vt_end)
        delta = vt_end - object_offset
        assert delta > 0, ("soffset 非正", delta, vt_end, object_offset)
        struct.pack_into("<i", self.buf, len(self.buf) - object_offset, delta)
        return object_offset

    # ── 字符串 ──────────────────────────────────────────────────────────
    def create_string(self, s: str) -> int:
        """**所有 create_* 都返回对象「起始」的 offset-from-end。**

        flatbuffer 里 prepend ⇒ **最后写入**的字节落在文件最前。
        长度字段是最后写的、占索引 0..3，因此**写完它之后的 `self.offset()`**
        就是索引 0 的 offset-from-end = 对象的起始位置。

        三个曾经的错法（都踩过）：
          * `return self.offset() - (4 + 元素总长)`（算术扣长度）⇒ 漏掉元素之间的
            对齐 padding，算出偏小的起始（差 4 字节，表现为长度读成半个）；
          * 在写长度字段**之前**记 `start` ⇒ 同样少 4（写完才 +4）；
          * `end_object` 用「写 soffset 之前」的 len ⇒ vtable 位置偏 4。
        """
        raw = s.encode("utf-8")
        # ⚠️ 顺序要紧：prepend 语义下「最后写的在最前」，长度必须**最后**写，
        #    否则会被内容挤到后面（这个坑踩过一次）。
        self.prep(4, len(raw) + 1)
        self.pad(1)                      # NUL 结尾
        self.buf[0:0] = raw              # ← 内容（曾被批量替换误删，导致串内容丢失）
        self.prep(4, 0)
        self.buf[0:0] = len(raw).to_bytes(4, "little")
        return self.offset()

    # ── 向量 ────────────────────────────────────────────────────────────
    def create_int32_vector(self, values: list[int]) -> int:
        self.prep(4, 4 * len(values))    # 为元素预留对齐空间
        start = None
        for v in reversed(values):       # prepend ⇒ 逆序插入才得到正序
            self.prep(4, 0)
            self.buf[0:0] = v.to_bytes(4, "little", signed=True)
        self.prep(4, 0)                  # 长度字段**最后**写（见 create_string 注释）
        self.buf[0:0] = len(values).to_bytes(4, "little")
        return self.offset()

    def create_float_vector(self, values: list[float]) -> int:
        self.prep(4, 4 * len(values))
        for v in reversed(values):
            self.prep(4, 0)
            self.buf[0:0] = struct.pack("<f", v)
        self.prep(4, 0)
        self.buf[0:0] = len(values).to_bytes(4, "little")
        return self.offset()

    def create_long_vector(self, values: list[int]) -> int:
        self.prep(4, 8 * len(values))
        for v in reversed(values):
            self.prep(8, 0)
            self.buf[0:0] = v.to_bytes(8, "little", signed=True)
        self.prep(4, 0)
        self.buf[0:0] = len(values).to_bytes(4, "little")
        return self.offset()

    def create_byte_vector(self, data: bytes) -> int:
        self.prep(4, len(data))
        self.buf[0:0] = data             # ← 内容（曾被批量替换误删）
        self.prep(4, 0)
        self.buf[0:0] = len(data).to_bytes(4, "little")
        return self.offset()

    def create_offset_vector(self, offsets: list[int]) -> int:
        self.prep(4, 4 * len(offsets))
        for off in reversed(offsets):
            self.prepend_uoffset(off)
        self.prep(4, 0)
        self.buf[0:0] = len(offsets).to_bytes(4, "little")
        return self.offset()

    # ── 收尾 ────────────────────────────────────────────────────────────
    def finish(self, root: int, identifier: bytes | None = None) -> bytes:
        """写根偏移（+ 可选文件标识）并返回完整文件。

        ⚠️ 根字段是文件的**前 4 字节**，存的是「文件开头 → 根 table」的**距离**，
        不是 Builder 内部的 offset-from-end 值。二者关系：
            根 table 文件位置 = 总长 - root，而 总长 = offset + 4

        `identifier`（TFLite 是 b"TFL3"）紧跟在根偏移**之后**，
        放在文件偏移 4..8 —— 这是 flatbuffer 的标准 file_identifier 位置。
        """
        ident = identifier or b""
        self.prep(self.minalign, 4 + len(ident))
        if ident:
            self.buf[0:0] = ident
        # p 已包含 ident（ident 是 prepend 进去的），所以总长 = p + 4（根字段）。
        p = self.offset()
        self.buf[0:0] = ((p + 4) - root).to_bytes(4, "little", signed=False)
        return bytes(self.buf)


def _is_aligned(n: int, align: int) -> bool:
    return align <= 0 or (n % align) == 0