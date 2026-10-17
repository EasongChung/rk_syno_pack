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

在 `calculate_bias_correction()` 的循环**之前**插入诊断：

1. 打印 `weight_width / weight_height / input_channels / weight_tensor_size /
   weight_tensor(指针) / depthwise / pointwise`；
2. 计算 `bound = w*h*c`，若 `bound > weight_tensor_size` 则打印
   `*** OOB PREDICTED ***`（这就是越界的直接证据）；
3. 钳界：算出安全的迭代上界 `diag_limit`，并把**原循环的终止条件**替换为
   `i < (diag_limit < bound ? diag_limit : bound)`。

## ⚠️ 两个曾让本补丁编译失败的坑（已修，勿回退）

1. **不能写 `operation->input_channels = ...`**
   `calculate_bias_correction()` 的参数是
   `const struct etna_operation *operation`，赋值给 const 对象必然编译失败：
   `error: assignment of member 'input_channels' in read-only object`
   （CI run 39037044419 实测）。故改为局部 `diag_limit`。
2. **布尔上下文不要用整数三元**
   `depthwise ? 1 : 0` 触发 `-Wint-in-bool-context` 警告；改为
   `operation->depthwise ? "yes" : "no"`（打印用）避免该告警。

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

# 注入位置：原循环语句之前。锚点是**原样的 for 语句**（含其后的函数体）。
# 用完整 for 行做锚点，能同时保证：
#   - 只匹配一次（该 for 在文件里唯一）
#   - 我们能把它整体替换成「诊断 + 钳界后的 for」
LOOP = ("   for (unsigned i = 0; i < operation->weight_width * "
        "operation->weight_height *\n"
        "                            operation->input_channels; i++)\n"
        "      correction += (weights[i] - operation->weight_zero_point) *\n"
        "                    operation->input_zero_point;")

# 备用锚点：Mesa 各版本该 for 的缩进/换行可能不同，故再提供一种更宽松的形态。
LOOP_ALT = ("   for (unsigned i = 0; i < operation->weight_width * "
            "operation->weight_height *\n"
            "                            operation->input_channels; i++)\n"
            "      correction += (weights[i] - operation->weight_zero_point) *\n"
            "                     operation->input_zero_point;")

# 插入的诊断块（C89 兼容，声明集中在块首）。
# 关键：只用局部变量，绝不写 operation 的 const 成员。
DIAG = """   /* {marker} */
   {{
      const unsigned diag_bound = operation->weight_width *
                                  operation->weight_height *
                                  operation->input_channels;
      unsigned diag_limit = diag_bound;
      fprintf(stderr,
              "[och-diag] calc_bias_correction: w=%u h=%u ic=%u bound=%u "
              "wt_size=%u wt=%p depthwise=%s pointwise=%u\\n",
              operation->weight_width, operation->weight_height,
              operation->input_channels, diag_bound,
              operation->weight_tensor_size, (void *)operation->weight_tensor,
              operation->depthwise ? "yes" : "no",
              (unsigned)(operation->pointwise ? 1 : 0));
      if (diag_bound > operation->weight_tensor_size) {{
         fprintf(stderr,
                 "[och-diag] *** OOB PREDICTED: bound(%u) > wt_size(%u) ***\\n",
                 diag_bound, operation->weight_tensor_size);
         fprintf(stderr,
                 "[och-diag] *** OOB loop clamped; DIAGNOSTIC build, "
                 "NOT deliverable ***\\n");
         diag_limit = operation->weight_tensor_size;
      }}
   }}

   /* {marker}: loop bound uses diag_limit (clamped when OOB predicted) */
   for (unsigned i = 0;
        i < (diag_limit < operation->weight_width *
                          operation->weight_height *
                          operation->input_channels
             ? diag_limit
             : operation->weight_width * operation->weight_height *
               operation->input_channels);
        i++)
      correction += (weights[i] - operation->weight_zero_point) *
                    operation->input_zero_point;"""


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

    if args.expect_symbol not in src:
        sys.stderr.write(
            f"ERROR: {args.path} 里找不到 {args.expect_symbol}，文件结构与预期不符。\n")
        return 1

    # 依次尝试两个锚点，任一命中恰好 1 次即用。
    hit, used = 0, None
    for cand in (LOOP, LOOP_ALT):
        n = src.count(cand)
        if n == 1:
            hit, used = 1, cand
            break
        if n > 1:
            sys.stderr.write(
                f"ERROR: 锚点在 {args.path} 中出现 {n} 次（期望 1 次）。\n"
                "       Mesa 可能已改动 calculate_bias_correction —— 请人工核对后\n"
                "       更新本补丁，不要跳过校验（会静默产出未打补丁的 .so）。\n")
            return 1

    if not hit:
        sys.stderr.write(
            f"ERROR: 在 {args.path} 中匹配到 0 处目标 for 循环（已尝试 2 种形态），"
            "期望恰好 1 处。\n"
            "       Mesa 可能已改动 calculate_bias_correction —— 请人工核对后更新本补丁，\n"
            "       不要跳过校验（会静默产出未打补丁的 .so，诊断结论无效）。\n")
        return 1

    src = src.replace(used, DIAG.format(marker=MARKER), 1)

    # 自检：诊断块与函数都还在，且**没有**对 const 成员的赋值。
    if MARKER not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败（标记缺失）\n")
        return 1
    if args.expect_symbol not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败（函数名丢失）\n")
        return 1
    if "operation->input_channels =" in src:
        sys.stderr.write(
            "ERROR: 补丁后仍存在对 operation->input_channels 的赋值 ——\n"
            "       该参数是 const，会编译失败。请检查补丁模板。\n")
        return 1
    if "diag_limit" not in src:
        sys.stderr.write("ERROR: 补丁后自检失败（无 diag_limit）\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"  [och-diag] 已注入诊断: {args.path}")
    print("  [och-diag] 将在真机打印尺寸/指针，并在预测到越界时钳住循环")
    # 提示语不带 emoji：Windows 的 GBK 控制台编码不了 U+26A0，
    # 会在**注入成功之后**抛 UnicodeEncodeError 并以 rc=1 退出，
    # 把成功误判成失败（本地自检实测）。CI 是 UTF-8 不受影响。
    print("  [och-diag] WARNING: 这是诊断构建，产物不可交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
