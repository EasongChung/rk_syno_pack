#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：create_nn_config() 填完 NN core 描述符后，把**整份描述符的原始字节**打出来。

## 背景（报告 27/28 的实测结论）

  挂死的最小单元已缩到**单个 1×1 CONV_2D（244→244 通道）**：
  * 段 #9 前 3 算子（RESHAPE,TRANSPOSE,CONV）→ 挂（520 ms/次，内核复位 4 次）
  * 段 #12 前 4 算子（…,CONV）               → 挂（512 ms/次）
  * 对照：段 #4 的 122→122 同型 1×1 conv     → **不挂**（14 ms，零复位）

  驱动侧判定挂死的依据是「**前端 DMA 指针不再前进**」
  （etnaviv_sched.c: etnaviv_sched_timedout_job 读 VIVS_FE_DMA_ADDRESS）
  ⇒ **NPU 连作业都没开始推进**，不是算得慢 ⇒ 极可能是**下发给硬件的描述符本身不合法**。

  ⇒ 本补丁把「下发给 NN core 的那份描述符」原样打出来：
  与不挂的 122 通道作业**逐字节 diff**，即可锁定是哪个字段越界/错位。

## 为什么打「原始字节」而不是逐字段打印

  `struct etna_nn_params` 是**位域结构体**（FIELD(field, bits) 宏），
  逐字段打印依赖每个字段名与版��，任一字段名在 Mesa 升级后改名就会**静默打错**。
  原始字节是**硬件真正看到的东西**：一次 dump 就能覆盖全部字段，
  之后离线用脚本解码即可，不必再改一次补丁、跑一次 CI。

## 本补丁不改变行为

  * 只**读**已填好的 map（`etna_bo_map()` 得到的 CPU 映射），不改任何字段；
  * 不新增分支、不改控制流、不改任何计算；
  * 打印走 stderr，`tf_load_check` 已 setvbuf(_IONBF) ⇒ 日志无缓冲，**当场可见**。

## 输出格式

  [och-nn] type=0 dw=0 pw=1 relu=1 in=50x50x244 out=50x50x244 kw=1 kh=1
           superblocks=1 tile=50x50 kpc=31 img_cache=0 coef_cache=610000
           conv_scale=0.001637 shift=112 post_mul=0/12/34
           raw=<整份描述符的十六进制字节>

  「raw=」后面是 `sizeof(struct etna_nn_params)` 个字节，两份作业的 raw 直接异或即可。

## 用法

      python3 patch-etnaviv-nn-desc-dump.py <etnaviv_ml_nn.c 路径>

  由 `build-teflon.sh` 的 `TEFLON_NNDUMP=1` 启用，默认关。
  ⚠️ 该产物是**诊断产物**：带取证打印，**不可交付**。
  """

import argparse
import sys

MARKER = "och patch: NN core descriptor dump"

# ↓↓↓ 逐字锚点（取自真实源码 etnaviv_ml_nn.c 的 create_nn_config() 收尾，勿手改）↓↓↓
OLD = """   map->per_channel_post_mul = 0x0;

   etna_bo_cpu_fini(bo);

   return bo;
}
"""

NEW = """   map->per_channel_post_mul = 0x0;

   /* och patch: NN core descriptor dump */
   {
      const unsigned char *raw = (const unsigned char *)map;
      fprintf(stderr,
              "[och-nn] type=%u dw=%d pw=%d relu=%d in=%ux%ux%u out=%ux%ux%u kw=%u kh=%u "
              "superblocks=%u tile=%ux%u kpc=%u img_cache=%u coef_cache=%u conv_scale=%g shift=%u "
              "post_mul=%u/%u/%u raw=",
              operation->type, operation->depthwise, operation->pointwise, operation->relu,
              operation->input_width, operation->input_height, operation->input_channels,
              operation->output_width, operation->output_height, operation->output_channels,
              operation->weight_width, operation->weight_height,
              superblocks, tile_x, tile_y, map->kernels_per_core,
              image_cache_size, coef_cache_size, conv_scale, shift,
              map->post_multiplier, map->post_multiplier_1_to_6, map->post_multiplier_7_to_14);
      for (unsigned och_i = 0; och_i < sizeof(*map); och_i++)
         fprintf(stderr, "%02x", raw[och_i]);
      fprintf(stderr, "\\n");
   }

   etna_bo_cpu_fini(bo);

   return bo;
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

    # 结构自检：确认锚点附近确实是 NN core 描述符的收尾
    for token in ("map->per_channel_post_mul", "etna_bo_cpu_fini(bo)", "struct etna_nn_params"):
        if token not in src:
            sys.stderr.write("ERROR: 源文件里找不到 " + token + "，结构与预期不符\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    # 打完补丁后的自检
    if MARKER not in src or "[och-nn]" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1
    if "sizeof(*map)" not in src:
        sys.stderr.write("ERROR: 补丁里缺少原始字节 dump\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-nn] 已注入 NN core 描述符 dump: " + args.path)
    print("  [och-nn] 日志形如 [och-nn] type=.. in=.. raw=<十六进制字节>")
    print("  [och-nn] 不改行为：只读已填好的描述符并打印")
    return 0


if __name__ == "__main__":
    sys.exit(main())
