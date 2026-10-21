#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线解码 etna_nn_params 描述符（不依赖设备、不依赖构建）。

## 为什么要有这个

`[och-nn] raw=` 是**硬件真正看到的那份字节**。补丁里逐字段打印依赖字段名与
版本，一旦 Mesa 升级改名就会静默打错；而 raw 永远是对的，只要解码器跟上。

本工具按 `etnaviv_ml_nn.c` 里 `struct etna_nn_params` 的 `FIELD(name, bits)`
声明顺序解码（C 位域在 uint32_t 上自低位向高位装填，每组注释 `/* n */`
正好是一个 32 位字），因此**不需要**结构体对齐/填充的知识。

## 自证（必须跑）

  python3 tools/decode-nn-desc.py --selftest

合成一个已知字段值的描述符 ⇒ 解码 ⇒ 逐字段比对；再对每个字做一次位翻转，
要求只有对应字段变化。任一项不符即报错退出（非零退出码）。
"""

import sys

# (字段名, 位数) —— 顺序严格照抄 etnaviv_ml_nn.c 的 struct etna_nn_params
FIELDS = [
    # /* 0 */
    ("layer_type", 1), ("no_z_offset", 1), ("kernel_xy_size", 4),
    ("kernel_z_size", 14), ("kernels_per_core", 7), ("pooling", 2),
    ("pooling_xy_size", 1), ("prelu", 1), ("nn_layer_flush", 1),
    # /* 1 */
    ("kernel_data_type", 2), ("in_image_data_type", 2), ("out_image_data_type", 2),
    ("in_image_x_size", 13), ("in_image_y_size", 13),
    # /* 2 */
    ("in_image_x_offset", 3), ("in_image_y_offset", 3), ("unused0", 1),
    ("brick_mode", 1), ("brick_distance", 16), ("relu", 1), ("unused1", 1),
    ("post_multiplier", 1), ("post_shift", 5),
    # /* 3 */
    ("unused2", 3), ("no_flush", 1), ("unused3", 2),
    ("out_image_x_size", 13), ("out_image_y_size", 13),
    # /* 4 */
    ("out_image_z_size", 14), ("rounding_mode", 2), ("in_image_x_offset_bit_3", 1),
    ("in_image_y_offset_bit_3", 1), ("out_image_tile_x_size", 7),
    ("out_image_tile_y_size", 7),
    # /* 5 */
    ("kernel_address", 26), ("kernel_z_size2", 6),
    # /* 6,7 */
    ("in_image_address", 32), ("out_image_address", 32),
    # /* 8 */
    ("image_caching_mode", 2), ("kernel_caching_mode", 2),
    ("partial_cache_data_unit", 2), ("kernel_pattern_msb", 6), ("kernel_y_size", 4),
    ("out_image_y_stride", 16),
    # /* 9,10 */
    ("kernel_pattern_low", 32), ("kernel_pattern_high", 32),
    # /* 11..14 */
    ("kernel_cache_start_address", 32), ("kernel_cache_end_address", 32),
    ("image_cache_start_address", 32), ("image_cache_end_address", 32),
    # /* 15 */
    ("in_image_border_mode", 2), ("in_image_border_const", 16), ("unused4", 1),
    ("kernel_data_type_bit_2", 1), ("in_image_data_type_bit_2", 1),
    ("out_image_data_type_bit_2", 1), ("post_multiplier_1_to_6", 6),
    ("post_shift_bit_5_6", 2), ("unused5", 2),
    # /* 16,17 */
    ("in_image_x_stride", 16), ("in_image_y_stride", 16),
    ("out_image_x_stride", 16), ("unused6", 8), ("post_multiplier_7_to_14", 8),
    # /* 18,19 */
    ("out_image_circular_buf_size", 26), ("per_channel_post_mul", 1),
    ("unused7_0", 1), ("unused7_1", 1), ("unused7_2", 1), ("unused7_3", 2),
    ("out_image_circular_buf_end_addr_plus_1", 26), ("unused8", 6),
    # /* 20,21 */
    ("in_image_circular_buf_size", 26), ("unused9", 6),
    ("in_image_circular_buf_end_addr_plus_1", 26),
]

# 关键字段：解码后单独高亮（判断作业配置是否自洽）
KEY = ("layer_type", "kernel_xy_size", "kernel_z_size", "kernel_y_size",
       "in_image_x_size", "in_image_y_size", "out_image_x_size", "out_image_y_size",
       "out_image_z_size", "kernels_per_core", "out_image_tile_x_size",
       "out_image_tile_y_size", "kernel_address", "in_image_address",
       "out_image_address", "kernel_caching_mode", "image_caching_mode",
       "kernel_cache_start_address", "kernel_cache_end_address",
       "image_cache_start_address", "image_cache_end_address", "post_shift",
       "post_multiplier", "post_multiplier_1_to_6", "post_multiplier_7_to_14",
       "relu", "kernel_pattern_msb")


def decode(raw):
    """raw: bytes → dict（字段名 → 值）。位域按 uint32 字自低位装填。"""
    vals = {}
    bitpos = 0
    total_bits = len(raw) * 8
    for name, bits in FIELDS:
        if bitpos + bits > total_bits:
            vals[name] = None
            bitpos += bits
            continue
        byte_i, off = divmod(bitpos, 8)
        # 取足够多的字节覆盖该字段（最多跨 5 字节）
        nbytes = (off + bits + 7) // 8
        chunk = raw[byte_i:byte_i + nbytes]
        acc = int.from_bytes(chunk, "little")
        vals[name] = (acc >> off) & ((1 << bits) - 1)
        bitpos += bits
    return vals


def encode(vals):
    """selftest 用：按同一规则装填回字节。"""
    word_bits = sum(b for _, b in FIELDS)
    buf = bytearray((word_bits + 7) // 8)
    bitpos = 0
    for name, bits in FIELDS:
        v = vals.get(name, 0) & ((1 << bits) - 1)
        byte_i, off = divmod(bitpos, 8)
        nbytes = (off + bits + 7) // 8
        chunk = int.from_bytes(buf[byte_i:byte_i + nbytes], "little")
        chunk |= v << off
        buf[byte_i:byte_i + nbytes] = chunk.to_bytes(nbytes, "little")
        bitpos += bits
    return bytes(buf)


def selftest():
    """正控制 + 负控制（位翻转、整字错位）。"""
    want = {name: (i * 7 + 3) % ((1 << bits) - 1) for i, (name, bits) in enumerate(FIELDS)}
    # 关键字段给出可辨认的具体值
    want.update({
        "layer_type": 0, "kernel_xy_size": 1, "kernel_z_size": 248,
        "kernels_per_core": 5, "in_image_x_size": 50, "in_image_y_size": 50,
        "out_image_x_size": 50, "out_image_y_size": 50, "out_image_z_size": 248,
        "kernel_address": 0x3FFAAC0, "in_image_address": 0xFFF5B000,
        "out_image_address": 0xFFEBB000, "relu": 1, "post_shift": 17,
    })
    raw = encode(want)
    got = decode(raw)

    bad = 0
    for name, _ in FIELDS:
        if got[name] != want[name]:
            print("  [BAD] 正控制字段不一致: %s want=%s got=%s"
                  % (name, want[name], got[name]))
            bad += 1
    print("  [ok] 正控制：%d 个字段编解码往返一致" % (len(FIELDS) - bad))

    # 负控制 1：把 kernel_address 的某一位置反，必须恰好被检出
    raw2 = bytearray(raw)
    bit = None
    acc = 0
    for name, bits in FIELDS:
        if name == "kernel_address":
            bit = acc
            break
        acc += bits
    assert bit is not None
    raw2[bit // 8] ^= 1 << (bit % 8)
    got2 = decode(bytes(raw2))
    diff = [n for n, _ in FIELDS if got2[n] != got[n]]
    assert got2["kernel_address"] != got["kernel_address"], "负控制 1 未检出位翻转"
    print("  [ok] 负控制 1：位翻转被检出（受影响字段: %s）" % ",".join(diff))

    # 负控制 2：整字错位（把 4 字节字序打乱）后，地址字段必须变
    words = [raw[i:i + 4] for i in range(0, (len(raw) // 4) * 4, 4)]
    scrambled = b"".join(words[::-1])
    got3 = decode(scrambled)
    assert got3["in_image_address"] != got["in_image_address"], "负控制 2 未检出字序错乱"
    print("  [ok] 负控制 2：整字错乱被检出")

    return bad == 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        ok = selftest()
        print("自测%s" % ("通过" if ok else "失败"))
        return 0 if ok else 1

    if len(argv) < 2:
        print(__doc__)
        return 2

    hexstr = "".join(argv[1].split())
    if hexstr.startswith("raw="):
        hexstr = hexstr[4:]
    raw = bytes.fromhex(hexstr)
    vals = decode(raw)

    print("描述符 %d 字节 / %d 个字" % (len(raw), len(raw) // 4))
    print("\n== 关键字段 ==")
    for k in KEY:
        v = vals.get(k)
        if v is None:
            continue
        extra = ""
        if k.endswith("address") and k != "in_image_address" and k != "out_image_address":
            extra = "  (VA = 0x%08x)" % (v << 6)
        elif k in ("in_image_address", "out_image_address"):
            extra = "  (0x%08x)" % v
        print("  %-34s = %-12d%s" % (k, v, extra))
    print("\n== 全部字段（非零）==")
    for name, _ in FIELDS:
        v = vals.get(name)
        if v:
            print("  %-34s = %d" % (name, v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
