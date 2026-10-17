#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：在 calculate_bias_correction() 里打印尺寸/指针，并在预测到越界时**钳住**循环。

## 为什么需要这个补丁（真机崩溃的定位过程）

26.2.5 双补丁 + per-tensor 模型，`AllocateTensors`（PrepareDelegate）阶段 SIGSEGV，
未加 `--invoke`（推理从未开始）。原始故障地址 `libteflon.so(+0x256a30)`：

```asm
256a30: ldrb  w1, [x0], #1      ; weights[i]
256a34: sub   w1, w1, w5        ; - weight_zero_point   ([x19,#268])
256a38: madd  w2, w1, w4, w2    ; * input_zero_point    ([x19,#132])
256a3c: cmp   x0, x3
256a40: b.ne  256a30
```

五个内存偏移与 `struct etna_operation` 字段**逐一对应**（aarch64 布局精确计算）：
`#128=input_channels` `#132=input_zero_point` `#260=weight_width`
`#264=weight_height` `#268=weight_zero_point`。
因此故障函数唯一确定为：

    etnaviv_ml_nn_v7.c:109  calculate_bias_correction()

```c
for (unsigned i = 0; i < operation->weight_width * operation->weight_height * operation->input_channels; i++)
   correction += (weights[i] - operation->weight_zero_point) * operation->input_zero_point;
```

**越界的必要条件 = 循环界 > `weight_tensor` 实际分配字节数**，而不是 scale 的数值。
两个待分辨子情形：

* (1) 分配偏小：`input_channels` 与权重张量末维不一致
  （DWCONV 的 TFLite 权重是 `[1,kh,kw,out_c]`，末维是**输出**通道）；
* (2) 指针野：`weight_tensor` 在某条路径上未初始化 / 已被 free 后又被赋值。

一次上机打印就能把两者分开，不必靠猜。

## 本补丁做什么

在 `calculate_bias_correction()` 函数体开头插入一段诊断：

1. 打印 `weight_width / weight_height / input_channels / weight_tensor_size /
   weight_tensor(指针) / depthwise / pointwise`；
2. 计算 `bound = w*h*c`，若 `bound > weight_tensor_size` 则打印
   `*** OOB PREDICTED ***`（这就是越界的直接证据）；
3. **把循环界钳到 `weight_tensor_size`**（仅在预测到越界时），让进程不再当场崩掉，
   从而把**后续**是否还有别的问题也一并暴露出来。

## ⚠️ 产物限制

打了本补丁的 `libteflon.so` **只能用于诊断**，绝不可交付：
它会改变越界处的行为（钳界），数值与正常构建不同。
默认关闭：由 `build-teflon.sh` 的 `TEFLON_DIAG=1` 开关启用，
不设该开关时构建流程完全不变（仍是干净产物）。

## 用法

    python3 patch-etnaviv-diag-bias.py <etnaviv_ml_nn_v7.c 路径> [--expect-symbol N]

匹配数不为 1 即**中止**（返回 1），不静默产出未打补丁的二进制。
"""

import argparse
import sys

MARKER = "och patch: bias-correction diagnostics"

# 插入内容：C89 兼容（声明集中在块首），只依赖已有的 stdio（Mesa 本就用 fprintf）。
# ⚠️ 标记必须是 **C 注释**：写成裸文本会直接让编译失败（已由本地纯文本自检抓到）。
DIAG = """   /* %s */
   {
      unsigned diag_bound = operation->weight_width * operation->weight_height *
                            operation->input_channels;
      fprintf(stderr,
              "[och-diag] calc_bias_correction: w=%%u h=%%u ic=%%u bound=%%u "
              "wt_size=%%u wt=%%p depthwise=%%d pointwise=%%u\\n",
              operation->weight_width, operation->weight_height,
              operation->input_channels, diag_bound,
              operation->weight_tensor_size, (void *)operation->weight_tensor,
              operation->depthwise ? 1 : 0, operation->pointwise ? 1u : 0u);
      if (diag_bound > operation->weight_tensor_size) {
         fprintf(stderr,
                 "[och-diag] *** OOB PREDICTED: bound(%%u) > wt_size(%%u) ===\\n",
                 diag_bound, operation->weight_tensor_size);
         fprintf(stderr,
                 "[och-diag] *** 越界读已钳住，进程不会崩；这是诊断构建，产物不可交付 ===\\n");
         operation->input_channels = operation->weight_tensor_size /
                                     (operation->weight_width * operation->weight_height ?
                                      operation->weight_width * operation->weight_height : 1);
      }
   }
"""

# 插入点：函数体第一行（`{` 之后的空行 + 第一条语句 `int32_t correction = 0;`）。
ANCHOR = "   int32_t correction = 0;\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-symbol", default="calculate_bias_correction",
                    help="期望存在的函数名（用于补丁后自检）")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    # 防重入：打过就必须拒绝，避免把诊断块套进诊断块。
    if MARKER in src:
        sys.stderr.write(
            f"ERROR: {args.path} 已包含本补丁（标记 \"{MARKER}\"），拒绝重复应用。\n")
        return 1

    n = src.count(ANCHOR)
    if n != 1:
        sys.stderr.write(
            f"ERROR: 在 {args.path} 中匹配到 {n} 处 \"{ANCHOR.strip()}\"，期望恰好 1 处。\n"
            "       Mesa 可能已改动 calculate_bias_correction —— 请人工核对后更新本补丁，\n"
            "       不要跳过校验（会静默产出未打补丁的 .so，诊断结论无效）。\n")
        return 1

    if args.expect_symbol not in src:
        sys.stderr.write(
            f"ERROR: {args.path} 里找不到 {args.expect_symbol}，文件结构与预期不符。\n")
        return 1

    # 注意：DIAG 模板用 %s 占位 + %% 转义，因此这里必须传**字符串**而不是 dict
    # （传 dict 时 %s 不会替换，会把 "{'marker': ...}" 字面写进 C 注释 —— 已由自检抓到）。
    src = src.replace(ANCHOR, ANCHOR + (DIAG % MARKER), 1)

    # 自检：诊断块与原函数都还在。
    if MARKER not in src or "int32_t correction = 0;" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"  [och-diag] 已注入诊断: {args.path}")
    print("  [och-diag] 将在真机打印尺寸/指针，并在预测到越界时钳住循环")
    # ⚠️ 提示语不带 emoji：Windows 的 GBK 控制台编码不了 U+26A0，
    # 会在**注入成功之后**抛 UnicodeEncodeError 并以 rc=1 退出，
    # 把成功误判成失败（本地自检实测）。CI 是 UTF-8 不受影响，
    # 但工具行为不应依赖终端编码。
    print("  [och-diag] WARNING: 这是诊断构建，产物不可交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
