#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：修正 write_core_interleaved() 的 out_channel 映射，并加越界钳位护栏。

## 根因（真机数据证实，不是推理）

1. 真机 per-tensor 模型 PrepareDelegate 阶段 SIGSEGV。本地对着 unstripped .so 跑
   `addr2line -f -i -C -e libteflon.so <addr>`，定位到 `write_core_interleaved()` 里的
   `wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);`
2. `weights_map` 只是把权重缓冲区强转成 `[out_channel][z][x][y]` 四维视图，**无边界检查**；
   同一函数第 319 行的 `biases[out_channel]` 同样无边界检查。
3. 诊断补丁 `[och-wc]` 打出的 **136 个独立组合**里，现用公式

       out_channel = core*kernels_in_superblock + kernel
                   + superblock*DIV_ROUND_UP(kernels_per_core, superblocks)*cores_used

   **精确复现全部实测值（0 个反例）**，并确实出现 `out_ch >= output_channels`
   （实测 122 / 123，而 output_channels = 122）→ 越界读 → SIGSEGV。

## 本补丁做两件事

1. **修正映射**：core 拥有 `kernels_per_core` 个 kernel（不是 `kernels_in_superblock`）；
   superblock 偏移 = 前序 superblock 已消耗的 kernel 数之和（不能写
   `sb*kernels_in_superblock`，最后一个 superblock 的 kernel 数会被减少）；去掉多余的
   `* cores_used`。
2. **越界钳位**：`out_channel >= output_channels` 时钳到 `output_channels - 1`。
   之所以不用 `continue`：函数开头 `append_bits(kernels_per_core, 16, ...)` 已把
   「每 core 槽位数」写进位流头，跳过会让位流少槽而整体错位。

## ⚠️ 验收不靠「不崩」

本补丁必须通过四级判据第 ④ 级：`--invoke` 输出与 x86 基准**逐字节一致**
（cos ≥ 0.99、top-1 相同）。不崩只说明越界消失，不代表系数排布正确。

## 同款公式还有 3 处未修

`write_core_6()` 两处、`write_core_sequential()` 一处含同一公式，属同一缺陷类，
但本轮无任何真机证据，故**不混入本次产物**，已记录为待修项。

## 用法

    python3 patch-etnaviv-outchannel-fix.py <etnaviv_ml_nn_v7.c 路径>

由 `build-teflon.sh` 的 `TEFLON_OCFIX=1` 启用，默认关。
"""

import argparse
import sys

MARKER = "och patch: out_channel 映射修正 + 越界钳位护栏"

# 目标函数（结构性检查限定在此函数体内，不做全局子串搜索）
TARGET_FN = "write_core_interleaved(struct etna_ml_subgraph"

# ↓↓↓ 逐字锚点（由 tools/gen-ocfix-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = '      for (unsigned z = 0; z < input_channels; z++) {\n         for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {\n            unsigned out_channel = core * kernels_in_superblock + kernel + superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;\n'
NEW = '      for (unsigned z = 0; z < input_channels; z++) {\n         for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {\n            /* och patch: out_channel 映射修正 + 越界钳位护栏 */\n            unsigned och_sb_base = 0;\n            for (unsigned och_p = 0; och_p < superblock; och_p++)\n               och_sb_base += DIV_ROUND_UP(kernels_per_core, superblocks);\n            /* 原式 core*kernels_in_superblock + kernel\n             *       + superblock*DIV_ROUND_UP(kernels_per_core,superblocks)*cores_used\n             * 已由真机 [och-wc] 日志证实：136 组合 0 反例地复现实测值，\n             * 且会出现 out_channel >= output_channels（实测 123 vs 122）→ 越界 SIGSEGV。\n             * core 拥有 kernels_per_core 个 kernel；superblock 偏移取前序累计值。 */\n            unsigned out_channel = core * kernels_per_core + och_sb_base + kernel;\n            /* 护栏：cores_used*kernels_per_core 常 > output_channels（实测 8*16=128 > 122），\n             * 尾部 kernel 槽对应不存在的输出通道。位流头声明了 kernels_per_core 个槽，\n             * 故不能 continue 跳过（会错位），只能钳位到最后一个合法通道：\n             * 这样 weights_map[out_channel] 与 biases[out_channel] 都不越界。 */\n            if (output_channels > 0 && out_channel >= output_channels)\n               out_channel = output_channels - 1;\n'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁，拒绝重复应用。\n")
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处 out_channel 算式，"
            "期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对并重新生成补丁"
            "（python3 tools/gen-ocfix-patch.py），不要跳过校验\n"
            "       （会静默产出未修复的 .so，崩溃会照旧）。\n")
        return 1

    # 结构性检查必须**限定在目标函数体内**，且用**精确计数**而非存在性。
    # 全局子串搜会被同款缺陷的其它函数满足（已实测：改掉本函数的 biases[out_channel] 被放行）；
    # 只做存在性也抓不到「两处同款调用改了一处」（已实测：weights_map[...] 改一处仍被放行）。
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    nxt = src.find("\nstatic", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    # (特征串, 期望出现次数) —— 期望值来自真实源码统计，见 tools/gen-ocfix-patch.py 注释
    for token, want in (
        ("weights_map[out_channel][z][x][y]", 2),   # 行 315 / 328 两个写点
        ("biases[out_channel]", 1),                  # 行 319，钳位同样要保护它
        ("append_bits(kernels_per_core, 16", 1),     # 位流头，护栏「不能 continue」的前提
        ("calculate_bias_correction((uint8_t *)weights_map[out_channel]", 1),
    ):
        got = fn_body.count(token)
        if got != want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token
                + "` 出现 " + str(got) + " 处，期望 " + str(want) + " 处。\n"
                "       Mesa 结构已变（或有人只改了一处同款调用）—— 护栏设计依赖这些点，\n"
                "       请人工核对后重新生成补丁，不要跳过校验\n"
                "       （会静默产出护栏失效的 .so，越界崩溃会照旧）。\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    # 打完补丁后自检：标记、钳位、修正映射、helper 都要在
    for token in (MARKER, "och_sb_base", "out_channel = core * kernels_per_core",
                  "out_channel >= output_channels", "out_channel = output_channels - 1"):
        if token not in src:
            sys.stderr.write("ERROR: 打完补丁后自检失败，缺少: " + token + "\n")
            return 1
    # 反向自检：**这个锚点**必须已被替换掉。
    # 注意不能用「旧公式子串」做检查 —— 同款公式在文件里另有 3 处（write_core_6 ×2、
    # write_core_sequential），那些本轮有意不动，误判会让补丁永远失败（已踩过）。
    if src.count(OLD) != 0:
        sys.stderr.write("ERROR: 锚点仍在源文件中，替换未生效\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 打完补丁后新代码出现次数为 "
                         + str(src.count(NEW)) + "（期望 1）\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-ocfix] 已修正 out_channel 映射并加越界钳位: " + args.path)
    print("  [och-ocfix] ⚠️ 只算「越界消失」；能否用要看四级判据第 ④ 级（逐字节比对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())