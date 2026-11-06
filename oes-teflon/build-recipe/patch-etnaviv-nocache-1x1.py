#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：1x1 卷积强制关闭 image cache（NO_CACHE）。

## 为什么（2026-11-06 真机实测，全部为一手数据）

image cache 的 tile 是 tile_x x tile_y（A311D 上常见 64x12），DMA 按
in_image_x_stride = in_x 逐行取。当 in_x < tile_x 时，一个 tile 需要读
tile_x x tile_y x in_c 字节，而图像一行只有 in_x x in_c 字节 => 读/写越界。

实测（交付件 e095a36f，单算子 1x1 32->64）：

| 形状(HxW) | in_x | tile_x | image cache | 结果 |
|---|---|---|---|---|
| 4x4   | 4   | 4  | NO_CACHE | 正常（±1 LSB） |
| 64x64 | 64  | 64 | FULL     | 正常 |
| 12x128| 128 | 64 | FULL     | 正常 |
| 12x65 | 65  | 64 | FULL     | 正常 |
| 65x64 | 64  | 64 | FULL     | 正常 |
| 112x64| 64  | 64 | FULL     | 正常 |
| 65x32 | 32  | 64 | FULL     | 每次推理 4 次 hang、0 次 MMU fault、输出全零 |
| 65x12 | 12  | 64 | FULL     | 同上 |
| 112x112 | 112 | 64 | FULL   | MMU fault 到野地址 0x1f238600、整机失联 |

越界地址落在 GPU 那个约 40 MB 的已映射窗口内时不报错（静默写坏别的 BO），
落在窗口外才 MMU fault —— 这解释了「同样的 bug 有时全零、有时整机失联」。

## 修法

在 create_nn_config() 决定 image cache 之前，把 1x1（weight_width==1 &&
weight_height==1 && !depthwise）的 image_cache_size 置 0 => 走 NO_CACHE 分支，
tile DMA 不再发生，Mode 1（窄图越界）与 Mode 2（多块）同时消失。

判定用 kernel 尺寸，不用 operation->pointwise：实测部署件里该 flag 为 0
（1x1 走的是 interleaved 码流、zrl=2），依赖它会静默失效。

## 代价（不藏着）

1x1 失去 tile 级输入复用，superblocks > 1 时会多次回读 DRAM。
已知 NO_CACHE 在 0.25-128 模型上仍是 NPU 0.804 ms vs CPU 3.35 ms（4.2x），
但 224 模型的代价需实测记录。

用法：
    python3 patch-etnaviv-nocache-1x1.py <etnaviv_ml_nn.c 路径>
    python3 patch-etnaviv-nocache-1x1.py --selftest
"""

import argparse
import sys

MARKER = "och patch: 1x1 强制 NO_CACHE"
COMPILE_MARKER = "och-patch-v7-nocache-1x1"
TARGET_FN = "create_nn_config("

OLD = """   if (image_cache_size == 0) {
      map->image_caching_mode = SRAM_CACHE_MODE_NO_CACHE;
"""

NEW = """   /* och patch: 1x1 强制 NO_CACHE
    *
    * 真机实测：image cache 的 tile 固定 tile_x x tile_y，DMA 按 in_image_x_stride=in_x
    * 逐行取；in_x < tile_x 时一个 tile 要读 tile_x*tile_y*in_c 字节，而一行只有
    * in_x*in_c 字节 => 越界读/写。
    *   in_x=12/32 (tile 64)：每次推理 4 次 hang、0 次 MMU fault、输出全零
    *   in_x=112（2x10 块）：MMU fault 到野地址、整机失联
    * 关闭 image cache 后不再发生 tile DMA（4x4 NO_CACHE 实测数值仅 ±1 LSB）。
    * 判定用 kernel 尺寸而非 operation->pointwise：后者实测为 0，不可靠。
    */
   if (weight_width == 1 && weight_height == 1 && !operation->depthwise)
      image_cache_size = 0;

#if defined(__GNUC__) || defined(__clang__)
   __attribute__((used)) static const char och_nocache_1x1_marker[] =
      "och-patch-v7-nocache-1x1";
   (void)och_nocache_1x1_marker;
#endif

   if (image_cache_size == 0) {
      map->image_caching_mode = SRAM_CACHE_MODE_NO_CACHE;
"""

# 结构校验：这些 token 必须在目标函数体里出现（打补丁前后都要满足）
STRUCT = [
    ("image_cache_size", 3),
    ("SRAM_CACHE_MODE_NO_CACHE", 1),
    ("SRAM_CACHE_MODE_FULL_CACHE", 1),
]


def _check_structure(src: str, path: str) -> bool:
    at = src.find(TARGET_FN)
    if at < 0:
        sys.stderr.write("ERROR: %s 里找不到 %s（Mesa 结构可能已变）\n" % (path, TARGET_FN))
        return False
    end = src.find("\n}\n", at)
    body = src[at:end if end > 0 else len(src)]
    for token, want in STRUCT:
        got = body.count(token)
        if got < want:
            sys.stderr.write("ERROR: 目标函数体内 %s 出现 %d 处，至少应有 %d 处\n"
                             % (token, got, want))
            return False
    return True


def apply(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: %s 已包含本补丁，拒绝重复应用。\n" % path)
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write("ERROR: 锚点匹配到 %d 处，期望恰好 1 处（Mesa 可能已改动，请人工核对）\n" % n)
        return 1

    if not _check_structure(src, path):
        return 1

    before_assign = src.count("image_cache_size = 0;")
    out = src.replace(OLD, NEW, 1)

    # 自检：marker 各 1 处；新增的置零语句恰好 +1；NO_CACHE 分支没被改动
    if out.count(MARKER) != 1 or out.count(COMPILE_MARKER) != 1:
        sys.stderr.write("ERROR: 自检失败：marker 出现次数异常\n")
        return 1
    if out.count("image_cache_size = 0;") != before_assign + 1:
        sys.stderr.write("ERROR: 自检失败：置零语句没有恰好 +1（%d -> %d）\n"
                         % (before_assign, out.count("image_cache_size = 0;")))
        return 1
    if out.count("SRAM_CACHE_MODE_NO_CACHE") != src.count("SRAM_CACHE_MODE_NO_CACHE"):
        sys.stderr.write("ERROR: 自检失败：NO_CACHE 分支数量变化\n")
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print("  [nocache-1x1] 已注入：1x1 => image_cache_size = 0（NO_CACHE）")
    print("  [nocache-1x1] 编译期标记：%s" % COMPILE_MARKER)
    return 0


def selftest() -> int:
    sample = ("static void\ncreate_nn_config(void)\n{\n"
              "   unsigned image_cache_size = 1234;\n"
              "   if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size) {\n"
              "      map->kernel_cache_end_address = 1;\n"
              "   }\n"
              "   if (image_cache_size == 0) {\n"
              "      map->image_caching_mode = SRAM_CACHE_MODE_NO_CACHE;\n"
              "   } else {\n"
              "      map->image_caching_mode = SRAM_CACHE_MODE_FULL_CACHE;\n"
              "   }\n}\n")
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".c")
    os.write(fd, sample.encode())
    os.close(fd)
    try:
        if apply(p) != 0:
            print("SELFTEST FAIL: 首次应用失败"); return 1
        again = apply(p)
        if again == 0:
            print("SELFTEST FAIL: 重复应用未被拒绝"); return 1
        print("SELFTEST OK: 首次应用成功、重复应用被拒绝")
        return 0
    finally:
        os.unlink(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.path:
        ap.error("需要 <etnaviv_ml_nn.c> 或 --selftest")
    return apply(args.path)


if __name__ == "__main__":
    sys.exit(main())
