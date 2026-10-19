#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：把 NN core 描述符的 kernel_pattern 改成**运行期可切换**的强制值。

## 背景（报告 28 + 29 的实测与源码复算）

真机实测（verify/reports/28）：
  挂死最小单元 = **单个 1×1 CONV_2D（244→244 通道）**；
  122→122 的同型 1×1 conv **不挂**（14 ms，零复位）。

源码复算（mesa_src/etnaviv_ml_nn_v7.c + etnaviv_ml_nn.c）发现一处**完美相关**：

  权重写出分派（v7）：
      if (operation->pointwise && output_channels > 8)  write_core_6(...)      ← 1×1 conv 全走这条
      else if (input_channels > 1)                      write_core_interleaved(...)
      else                                              write_core_sequential(...)

  描述符的 kernel_pattern（etnaviv_ml_nn.c 的 create_nn_config，PARTIAL_CACHE 分支）
  却按**输出通道数**分档挑：

      z ≥ 1024 → 0x13 / 0x80000 / 0x0
      z ≥  512 → 0x3d / 0x0 / 0x2aaaaaa0
      z ≥  256 → 0x3e / 0xffffaaaa / 0x7fffffff
      z ≥  160 → 0x6  / 0x7e / 0x0        ←★ 本模型唯一落在这里的通道数 = 244
      else     → 0x3f / 0xfffffffe / 0xffffffff

  ⇒ 244 是全模型**唯一**落进 [160,256) 档的通道数（24/122/488/976/2048 都不在这一档），
    也**唯一**挂死 ⇒ 6 个通道数里 5 正常 1 挂，完美相关。

## 本补丁做什么

  在 create_nn_config() 的 PARTIAL_CACHE 分支里，把**分档选择**换成运行期开关：

      TEFLON_KPAT=6   强制 kernel_pattern = 0x6  / 0x7e        / 0x0
      TEFLON_KPAT=3   强制 kernel_pattern = 0x3f / 0xfffffffe  / 0xffffffff
      TEFLON_KPAT 未设 或 =0   保持**原分档逻辑**（默认，行为完全不变）

  ⇒ 一次构建、两个实验；**默认行为与原产物完全一致**。

## 这是「改行为」的诊断补丁（不是纯打印）

  必须说清楚：本补丁**会改变下发��硬件的描述符**，因此
  · 产物**不可交付**（数值不可信：pattern 与权重实际布局的配对被人为改过）；
  · 目的**只有一个**：做**单变量对照** ——
      同一个 122 通道的 1×1 conv（当前 pattern 0x3f，跑得飞快），
      只把 pattern 换成 0x6，其余字段一字不动：
        · 仍然不挂 ⇒ **pattern 不是元凶**（去查别的字段/系数/地址）
        · 立刻挂     ⇒ **pattern 0x6 就是元凶**（A311D 的 VIPNano-QI NN core 不支持该 pattern）
  · 反向对照同样成立：244 通道的 conv 用 TEFLON_KPAT=3（换成 0x3f）若不挂 ⇒ 同样确证。

## 用法

      python3 patch-etnaviv-kernel-pattern.py <etnaviv_ml_nn.c 路径>

  由 `build-teflon.sh` 的 `TEFLON_KPAT=1` 启用（编译期把补丁打进去），
  **运行期**再用 `TEFLON_KPAT=6|3` 选择强制值；不设运行期变量 ⇒ 行为与原产物一致。
  """

import argparse
import re
import sys

MARKER = "och patch: kernel_pattern 强制开关"

# ↓↓↓ 逐字锚点（取自真实源码 etnaviv_ml_nn.c 的 create_nn_config PARTIAL_CACHE 分支）↓↓↓
OLD = """      if (map->out_image_z_size >= 1024) {
         map->kernel_pattern_msb = 0x13;
         map->kernel_pattern_low = 0x80000;
         map->kernel_pattern_high = 0x0;
      } else if (map->out_image_z_size >= 512) {
         map->kernel_pattern_msb = 0x3d;
         map->kernel_pattern_low = 0x0;
         map->kernel_pattern_high = 0x2aaaaaa0;
      } else if (map->out_image_z_size >= 256) {
         map->kernel_pattern_msb = 0x3e;
         map->kernel_pattern_low = 0xffffaaaa;
         map->kernel_pattern_high = 0x7fffffff;
      } else if (map->out_image_z_size >= 160) {
         map->kernel_pattern_msb = 0x6;
         map->kernel_pattern_low = 0x7e;
         map->kernel_pattern_high = 0x0;
      } else {
         map->kernel_pattern_msb = 0x3f;
         map->kernel_pattern_low = 0xfffffffe;
         map->kernel_pattern_high = 0xffffffff;
      }
"""

NEW = """      /* och patch: kernel_pattern 强制开关 */
      {
         const char *kpat = getenv("TEFLON_KPAT");
         if (kpat && kpat[0] == '6') {
            map->kernel_pattern_msb = 0x6;
            map->kernel_pattern_low = 0x7e;
            map->kernel_pattern_high = 0x0;
            fprintf(stderr, "[och-kpat] 强制 kernel_pattern=0x6/0x7e/0x0 (z=%u)\\n",
                    map->out_image_z_size);
         } else if (kpat && kpat[0] == '3') {
            map->kernel_pattern_msb = 0x3f;
            map->kernel_pattern_low = 0xfffffffe;
            map->kernel_pattern_high = 0xffffffff;
            fprintf(stderr, "[och-kpat] 强制 kernel_pattern=0x3f/0xfffffffe/0xffffffff (z=%u)\\n",
                    map->out_image_z_size);
         } else {
            if (map->out_image_z_size >= 1024) {
               map->kernel_pattern_msb = 0x13;
               map->kernel_pattern_low = 0x80000;
               map->kernel_pattern_high = 0x0;
            } else if (map->out_image_z_size >= 512) {
               map->kernel_pattern_msb = 0x3d;
               map->kernel_pattern_low = 0x0;
               map->kernel_pattern_high = 0x2aaaaaa0;
            } else if (map->out_image_z_size >= 256) {
               map->kernel_pattern_msb = 0x3e;
               map->kernel_pattern_low = 0xffffaaaa;
               map->kernel_pattern_high = 0x7fffffff;
            } else if (map->out_image_z_size >= 160) {
               map->kernel_pattern_msb = 0x6;
               map->kernel_pattern_low = 0x7e;
               map->kernel_pattern_high = 0x0;
            } else {
               map->kernel_pattern_msb = 0x3f;
               map->kernel_pattern_low = 0xfffffffe;
               map->kernel_pattern_high = 0xffffffff;
            }
            fprintf(stderr, "[och-kpat] 原分档逻辑 (z=%u -> msb=0x%x)\\n",
                    map->out_image_z_size, map->kernel_pattern_msb);
         }
      }
"""
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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
            "       Mesa 可能已改动此处 —— 请人工核对后重新生成补丁，不要跳过校验\n"
            "       （会静默产出未打补丁的 .so，诊断结论无效）。\n")
        return 1

    for token in ("kernel_pattern_msb", "out_image_z_size >= 160"):
        if token not in src:
            sys.stderr.write("ERROR: 源文件里找不到 " + token + "，结构与预期不符\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    # getenv 需要 <stdlib.h>；该文件只间接用到 stdlib（malloc），没有直接 include。
    # 不显式补上，CI 的新编译器会按 C99 隐式声明报错 ⇒ 这里补一行（幂等：已存在则跳过）。
    if "#include <stdlib.h>" not in src:
        m_inc = re.search(r'^#include .*$', src, re.M)
        if not m_inc:
            sys.stderr.write("ERROR: 源文件里找不到 #include 行，无法补 <stdlib.h>\n")
            return 1
        src = (src[:m_inc.end()] + "\n#include <stdlib.h> /* och patch: kernel_pattern 强制开关需要 getenv */"
               + src[m_inc.end():])

    if MARKER not in src or "[och-kpat]" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1
    # 自检：原分档逻辑必须**逐字保留**在 else 分支里（默认行为不变）
    if "if (map->out_image_z_size >= 160) {" not in src:
        sys.stderr.write("ERROR: 原分档逻辑没有保留在 else 分支里 —— 默认行为被改了\n")
        return 1
    if "getenv(\"TEFLON_KPAT\")" not in src:
        sys.stderr.write("ERROR: 运行期开关缺失\n")
        return 1
    if "#include <stdlib.h>" not in src:
        sys.stderr.write("ERROR: <stdlib.h> 没补上（getenv 无声明）\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-kpat] 已注入 kernel_pattern 强制开关: " + args.path)
    print("  [och-kpat] 运行期: TEFLON_KPAT=6 强制 0x6/0x7e/0x0；=3 强制 0x3f/...；不设=原分档逻辑")
    print("  [och-kpat] ⚠️ 本产物改行为，**不可交付**；默认（不设运行期变量）行为与原产物一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
