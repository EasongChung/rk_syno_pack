#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：write_core_interleaved() 写出权重前打印循环变量，out_channel 越界时报警。

## 背景（addr2line 硬证据，不再靠反推）

CI 里 `addr2line -e libteflon.so 0x256a60` 直接给出：

    write_core_interleaved()   etnaviv_ml_nn_v7.c:345（打 FIX_V7 补丁后的行号）

崩溃行正是 `wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);`。
而 `weights_map` 只是把权重缓冲区**强转**成四维视图：

    uint8_t (*weights_map)[input_channels][weight_width][weight_height] = (void *)input;

`out_channel` 一旦超过 `output_channels`，`weights_map[out_channel]` 就指向缓冲区之外
→ 越界读 → SIGSEGV；越界多少取决于堆布局，所以表现为**非确定性**。

**待证实/证伪的就是这一点** —— 所以本补丁只把数字打出来，不改任何行为。

## 本补丁做什么

在写出权重前打印：core / kernel / z / superblock、superblocks、kernels_in_superblock、
kernels_per_core、cores_used、out_channel、output_channels、weight_width/height、
weight_tensor_size；并在 `out_channel >= output_channels` 时追加 `*** OOB ***`。

**刻意不做**：不改行为、不钳界、不跳过循环。
打印紧挨崩溃行发生，而 `tf_load_check` 已 `setvbuf(_IONBF)`（日志无缓冲），
所以即使下一秒仍然 SIGSEGV，**这一行已经落进日志** ——
我们要的是「机制被证实」，不是「让它跑过去」。

## 用法

    python3 patch-etnaviv-write-core-diag.py <etnaviv_ml_nn_v7.c 路径>

由 `build-teflon.sh` 的 `TEFLON_WCDIAG=1` 启用，默认关。
"""

import argparse
import sys

MARKER = "och patch: write_core_interleaved out_channel diagnostics"

# ↓↓↓ 逐字锚点（由 tools/gen-wcdiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = '                  for (unsigned y = 0; y < stride; y++) {\n                     wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);\n'
NEW = '                  for (unsigned y = 0; y < stride; y++) {\n/* och patch: write_core_interleaved out_channel diagnostics */\n                        fprintf(stderr,\n                                "[och-wc] core=%u kernel=%u z=%u sb=%u/%u "\n                                "kpc=%u/%u cores=%u out_ch=%u out_channels=%u "\n                                "w=%u h=%u wt_size=%u%s\\n",\n                                core, kernel, z, superblock, superblocks,\n                                kernels_in_superblock, kernels_per_core, cores_used,\n                                out_channel, output_channels,\n                                operation->weight_width, operation->weight_height,\n                                operation->weight_tensor_size,\n                                (out_channel >= output_channels) ? "  *** OOB ***" : "");\n                     wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);\n'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁（标记 \"" + MARKER +
                         "\"），拒绝重复应用。\n")
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处锚点，期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对后重新生成补丁"
            "（python3 tools/gen-wcdiag-patch.py），不要跳过校验\n"
            "       （会静默产出未打补丁的 .so，诊断结论无效）。\n")
        return 1

    if "weights_map[out_channel][z][x][y]" not in src:
        sys.stderr.write("ERROR: 源文件里找不到 weights_map[out_channel][z][x][y]，"
                         "结构与预期不符\n")
        return 1

    src = src.replace(OLD, NEW, 1)

    if MARKER not in src or "[och-wc]" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-wc] 已注入 out_channel 诊断: " + args.path)
    print("  [och-wc] 日志形如 [och-wc] ... out_ch=N out_channels=M *** OOB ***")
    print("  [och-wc] 不改行为：崩溃行仍会照旧触发，但这一行已在日志里")
    return 0


if __name__ == "__main__":
    sys.exit(main())