#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：强制 NN core 描述符走 PARTIAL cache 分支（运行期开关）

## 背景（报告 28/30/32 + oca 报告 31 的第三轮复核）

真机实测：挂死最小单元 = **单个 1×1 CONV_2D，244→244 通道**；
122→122、488→488、976→2048 等同型**都不挂**。

独立复核算出的关键分组（FULL/PARTIAL 判据见 `etnaviv_ml_nn.c:1188`）：

  | C   | total = 0x800 + coef + image | 分支   | 占 512KB SRAM | 真机 |
  |-----|----------------------------|--------|---------------|------|
  | 122 | 93,184                     | FULL   | 18%           | 不挂 |
  | **244** | **213,376**             | **FULL** | **41%**     | **★挂** |
  | 488 | 540,928                    | PARTIAL| —             | 不挂 |

⇒ **244 是唯一「FULL cache 且占用大」的通道数** ⇒ 新的头号候选：
  **判别量 = FULL-cache 路径 + 大占用**（不是 pattern、不是 kpc、不是 superblocks、不是 tile）。

## 本补丁做什么

在 `create_nn_config()` 的 FULL/PARTIAL 判据处加一个**运行期开关**：

  TEFLON_PARTIAL=1  ⇒ **强制走 PARTIAL 分支**（把 oc_sram 判据调小）
  不设               ⇒ 保持原逻辑（默认行为完全不变）

⇒ 单变量对照：
  · 244 通道的 1×1 conv，若改走 PARTIAL 后**不再挂** ⇒ **该候选成立**（根因确证）
  · 若**仍挂** ⇒ 该候选被证伪 ⇒ 判别量在别处（继续查系数流/落位/输出偏移）

## 本补丁不改变行为？—— **默认路径不变**

  不设 TEFLON_PARTIAL 时，判据与原代码**逐字相同** ⇒ 产物可当基线用。
  ⚠️ 设了 `TEFLON_PARTIAL=1` 时**会改行为**（描述符的 kernel_caching_mode 与 pattern 全变）⇒
  **该模式下的数值不可信**，只用于「挂/不挂」的判定。

## 用法

      python3 patch-etnaviv-force-partial.py <etnaviv_ml_nn.c 路径>

由 `build-teflon.sh` 的 `TEFLON_FPARTIAL=1` 启用（编译期把补丁打进去），
**运行期**再用 `TEFLON_PARTIAL=1` 强制；不设运行期变量 ⇒ 行为与原产物一致。
"""

import argparse
import sys

MARKER = "och patch: 强制 PARTIAL cache 分支"

# ↓↓↓ 逐字锚点（取自真实源码 etnaviv_ml_nn.c 的 create_nn_config）↓↓↓
OLD = """   /* Get all the image tiles in the cache, then use the rest for the kernels */
   if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size) {
"""

NEW = """   /* Get all the image tiles in the cache, then use the rest for the kernels */
   /* och patch: 强制 PARTIAL cache 分支（运行期开关，默认不变） */
   {
      const char *fp = getenv("TEFLON_PARTIAL");
      int force_partial = (fp && fp[0] == '1');
      unsigned long long total_needed =
         (unsigned long long)map->kernel_cache_start_address + coef_cache_size + image_cache_size;
      if (force_partial)
         fprintf(stderr,
                 "[och-fp] 强制 PARTIAL: total_needed=%llu oc_sram=%u coef=%u image=%u\n",
                 total_needed, oc_sram_size, coef_cache_size, image_cache_size);
   }
   if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size
       && !(getenv("TEFLON_PARTIAL") && getenv("TEFLON_PARTIAL")[0] == '1')) {
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

    for token in ("oc_sram_size", "kernel_cache_start_address", "SRAM_CACHE_MODE_PARTIAL_CACHE"):
        if token not in src:
            sys.stderr.write("ERROR: 源文件里找不到 " + token + "，结构与预期不符\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    if MARKER not in src or "[och-fp]" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1
    if 'getenv("TEFLON_PARTIAL")' not in src:
        sys.stderr.write("ERROR: 运行期开关缺失\n")
        return 1
    # 原判据必须**逐字保留**（默认路径行为不变）
    if "map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size" not in src:
        sys.stderr.write("ERROR: 原判据没有保留 —— 默认行为被改了\n")
        return 1
    if "#include <stdlib.h>" not in src:
        m_inc = None
        import re
        m_inc = re.search(r'^#include .*$', src, re.M)
        if not m_inc:
            sys.stderr.write("ERROR: 找不到 #include 行，无法补 <stdlib.h>\n")
            return 1
        src = (src[:m_inc.end()] + "\n#include <stdlib.h> /* och patch: 强制 PARTIAL cache 分支需要 getenv */"
               + src[m_inc.end():])

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-fp] 已注入 强制 PARTIAL 开关: " + args.path)
    print("  [och-fp] 运行期: TEFLON_PARTIAL=1 强制走 PARTIAL 分支；不设=原逻辑（默认行为不变）")
    print("  [och-fp] ⚠️ 强制模式改行为、数值不可信，只用于「挂/不挂」判定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
