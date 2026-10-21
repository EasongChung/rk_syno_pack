#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：把 v8 的 depthwise/addition 特判 backport 到 v7 的 calculate_bias_correction()。

## 根因（已静态证明，无需重新编译即可确认）

真机崩溃（26.2.5 + teflon 双补丁 + per-tensor 模型）：

```
PrepareDelegate 阶段 SIGSEGV（未加 --invoke，推理从未开始）
libteflon.so(+0x256a30) → etna_ml_calculate_tiling_v7 与 etna_ml_create_coeffs_v7 之间的
                          static 函数区间（static 不在 .dynsym，故范围归属≠函数归属）
```

故障指令（objdump 实测）：

```asm
256a30: ldrb  w1, [x0], #1      ; weights[i]
256a34: sub   w1, w1, w5        ; - weight_zero_point   ([x19,#268])
256a38: madd  w2, w1, w4, w2    ; * input_zero_point    ([x19,#132])
256a3c: cmp   x0, x3 / b.ne     ; 循环
```

五个偏移与 `struct etna_operation`（aarch64 布局精确计算）**逐一对应**：
`#128=input_channels` `#132=input_zero_point` `#260=weight_width`
`#264=weight_height` `#268=weight_zero_point`。函数身份唯一确定为：

    src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c  calculate_bias_correction()

**缺陷**：该函数（v7）直接用 `operation->input_channels` 决定扫描长度，
**没有 depthwise 特判**；而同仓库里：

| 函数 | 位置 | depthwise → input_channels=1 |
|---|---|---|
| `calc_weight_quant` | etnaviv_ml_nn.c:202-203 | 有 |
| `requantize_weights` | etnaviv_ml_nn.c:239-240 | 有 |
| `calculate_bias_correction` | **etnaviv_ml_nn_v7.c:114** | **无（缺陷）** |
| `calculate_bias_correction` | etnaviv_ml_nn_v8.c:170-171 | 有（**v8 已修**） |

深度卷积的权重张量形状是 `[1,kh,kw,out_c]`，每输出通道只有 `W*H*1` 个权重。
v7 按 `W*H*input_channels` 扫描且以此为步长：
对 `W=H=3, input_channels=24` 的 DWCONV，缓冲区仅 `out_c*9 = 216` 字节，
而 v7 会以 216 字节为步长、每步扫 216 字节 → **第 2 个输出通道起点即越界**，
之后一路向前扫到不可达页 → SIGSEGV；撞到哪一页取决于堆布局 → **非确定性**。

## 本补丁做什么

1. 打印修复前的扫描长度与越界倍数（**取证**，判断是否确实越界）；
2. 采用与 v8 **完全一致**的特判逻辑（depthwise→1 / addition→2*out_c / 其余→input_channels）；
3. 用修正后的长度做扫描。

即：**一次构建同时拿到「证据」与「修复」**。

## ⚠️ 适用范围

只改 `etnaviv_ml_nn_v7.c`。若某 Mesa 版本的该文件结构不同，脚本会**中止**（返回 1），
不要放宽校验硬打。由 `build-teflon.sh` 的 `TEFLON_FIX_V7=1` 开关启用，
默认关闭（不设该开关时构建流程与产物完全不变）。

## 用法

    python3 patch-etnaviv-v7-biasfix.py <etnaviv_ml_nn_v7.c 路径>
"""

import argparse
import sys

MARKER = "och patch: v7 calculate_bias_correction depthwise fix"

OLD = """static uint32_t
calculate_bias_correction(uint8_t *weights, const struct etna_operation *operation)
{
   int32_t correction = 0;

   for (unsigned i = 0; i < operation->weight_width * operation->weight_height * operation->input_channels; i++) {
      correction += (weights[i] - operation->weight_zero_point) * operation->input_zero_point;
   }

   return correction;
}
"""

# 与 v8（etnaviv_ml_nn_v8.c:164-191）逻辑逐条对齐；打印放在修复之前，
# 这样日志里「修复前的越界倍数」与「修复后的长度」同时可见，一次上机取证闭环。
NEW = """static uint32_t
calculate_bias_correction(uint8_t *weights, const struct etna_operation *operation)
{
   /* 全部声明前置，保持 C89 兼容（Mesa 主体是 C89 风格，别引入语句后声明） */
   unsigned input_channels;
   unsigned och_buggy_bound;
   unsigned och_bound;
   int32_t correction = 0;

   /* %s */
   och_buggy_bound = operation->weight_width * operation->weight_height *
                      operation->input_channels;

   /* 与 v8（etnaviv_ml_nn_v8.c:164-191）逐条对齐 */
   if (operation->depthwise)
      input_channels = 1;
   else if (operation->addition)
      input_channels = 2 * operation->output_channels;
   else
      input_channels = operation->input_channels;

   och_bound = operation->weight_width * operation->weight_height * input_channels;

   /* 取证打印：修复前的 buggy_bound 与修复后的 fixed_bound 同时可见 */
   fprintf(stderr,
           "[och-fix] calc_bias_correction depthwise=%%d addition=%%d in_c=%%u w=%%u h=%%u "
           "| buggy_bound=%%u fixed_bound=%%u wt_size=%%u wt=%%p\\n",
           operation->depthwise ? 1 : 0, operation->addition ? 1 : 0,
           operation->input_channels, operation->weight_width, operation->weight_height,
           och_buggy_bound, och_bound, operation->weight_tensor_size,
           (void *)operation->weight_tensor);
   if (operation->weight_tensor_size && och_bound > operation->weight_tensor_size)
      fprintf(stderr, "[och-fix] still OOB after fix: bound=%%u > wt_size=%%u\\n",
              och_bound, operation->weight_tensor_size);

   for (unsigned i = 0; i < och_bound; i++) {
      correction += (weights[i] - operation->weight_zero_point) * operation->input_zero_point;
   }

   return correction;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write(
            f"ERROR: {args.path} 已包含本补丁（标记 \"{MARKER}\"），拒绝重复应用。\n")
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            f"ERROR: 在 {args.path} 中匹配到 {n} 处完整的 calculate_bias_correction()，期望恰好 1 处。\n"
            "       Mesa 可能已改动该函数（若上游已修，请直接升级 Mesa 而不是打本补丁）——\n"
            "       不要跳过校验强行构建（会静默产出未修复的 .so，崩溃会照旧）。\n")
        return 1

    src = src.replace(OLD, NEW % MARKER, 1)

    # 自检：修复逻辑与标记都在，原函数体确实被换掉。
    if MARKER not in src or "if (operation->depthwise)" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1
    if "operation->weight_width * operation->height" in src:  # 不该出现的拼写
        sys.stderr.write("ERROR: 补丁内容异常\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"  [och-fix] 已修复 v7 calculate_bias_correction: {args.path}")
    print("  [och-fix] depthwise/addition 特判已按 v8 对齐；日志会打印 buggy_bound/fixed_bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())