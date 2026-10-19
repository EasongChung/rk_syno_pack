#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能补丁：R5GUARD —— rank-5 激活 guard。

## 为什么需要（och 报告 11）

 contracted 里没有「维度个数」字段，delegate 写入的最后 4 维；
驱动一律算 `dims[1]*dims[2]*dims[3]`，**恒定丢 `dims[0]`**。

| rank | 真实形状 | 写入 dims[] | 驱动算出 | 真实字节 | 丢掉的 dims[0] |
|---|---|---|---|---|---|
| 4 | [1,50,50,244]   | [1,50,50,244]   | 50·50·244 = 610000 | 610000 | 1（batch）⇒ 无害 |
| 5 | [1,122,2,50,50] | [122,2,50,50]   | 2·50·50 = 5000     | 610000 | **122** ⇒ 致命 |

⇒ 含 rank-5 的子图**不接管**，回退 CPU，不再 abort。

## 本补丁做的事

在 `etna_ml_operation_supported` 的 CONVOLUTION 分支，追加两个条件：
    !(input_tensor->data  == NULL && input_tensor->dims[0] != 1) &&
    !(output_tensor->data == NULL && output_tensor->dims[0] != 1)

`data != NULL` 的张量（权重/常量）被排除 —— 它们不走这条路径。

## 代价（och 要求如实报告）

接管数会下降。och 报告 11 §7 给出的**保守上界 = 13/201 = 6%**（假设所有传递
下游的子图都不接管）。**真实值必须真机实测**，本补丁只保证「不再 abort」，
不保证接管数不跌到不可接受。若跌到不可接受，**停下来报告数字**，不要自行换模型。

## 本补丁**不**做的事

* 不改模型、不改阈值（产物语义不变）
* 不修 size 本身（修 size 也救不了 job 几何，见 och 报告 11 §2.2）
* 不修 RANK-5 表达能力 —— 这是硬件作业模型的边界，打补丁修不了

## ★ 本补丁的 predicate 是 och 给的，需要先在真机上验证误命中率（P0）。
  误命中率不为零时，本补丁必须撤回。P0 由 och 负责。
"""

import argparse
import sys

MARKER = "och patch: R5GUARD — rank-5 activation guard"
COMPILE_MARK = "och-patch-r5guard"

OLD = ("         if (tensor_quantization_supported(input_tensor) &&\n"
       "             tensor_quantization_supported(weight_tensor) &&\n"
       "             tensor_quantization_supported(bias_tensor) &&\n"
       "             tensor_quantization_supported(output_tensor) &&\n"
       "             operation->conv.dilation_width_factor == 1 &&\n"
       "             operation->conv.dilation_height_factor == 1) {\n"
       "            supported = true;\n"
       "         }\n")

NEW = ("         if (tensor_quantization_supported(input_tensor) &&\n"
       "             tensor_quantization_supported(weight_tensor) &&\n"
       "             tensor_quantization_supported(bias_tensor) &&\n"
       "             tensor_quantization_supported(output_tensor) &&\n"
       "             operation->conv.dilation_width_factor == 1 &&\n"
       "             operation->conv.dilation_height_factor == 1 &&\n"
       "             /* och patch: R5GUARD — rank-5 activation guard (och report 11) */\n"
       "             /*   data == NULL 的张量是激活（不是权重/常量）；\n"
       "                * dims[0] != 1 说明 rank-5 被压进 4 槽，驱动算出的几何/size 都错。\n"
       "                * 不接管 ⇒ 回退 CPU，不再 abort。 */\n"
       "             !(input_tensor->data == NULL && input_tensor->dims[0] != 1) &&\n"
       "             !(output_tensor->data == NULL && output_tensor->dims[0] != 1)) {\n"
       "            supported = true;\n"
       "         }\n")

DIAG = ('         /* och patch: R5GUARD 诊断（print-only） */\n'
        '         ML_DBG("R5GUARD: in dims[0]=%u (data=%p) out dims[0]=%u (data=%p) -> %s\\n",\n'
        '                input_tensor->dims[0], input_tensor->data,\n'
        '                output_tensor->dims[0], output_tensor->data,\n'
        '                ((input_tensor->data == NULL && input_tensor->dims[0] != 1) ||\n'
        '                 (output_tensor->data == NULL && output_tensor->dims[0] != 1))\n'
        '                ? "REJECT" : "allow");\n')


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
            "ERROR: R5GUARD 锚点匹配到 %d 处，期望恰好 1 处（Mesa 结构可能已变，"
            "或 FSGUARD 未先打）。\n" % n)
        return 1
    if OLD == NEW:
        sys.stderr.write("ERROR: 替换未生效\n")
        return 1

    src = src.replace(OLD, NEW, 1)

    # 诊断打印插在 NEW 之前（只加打印，不改控制流）
    src = src.replace(NEW, DIAG + NEW, 1)

    # 编译期标记
    if COMPILE_MARK not in src:
        src = src.rstrip("\n") + "\n\n/* och patch: compile-time marker */\n"
        src = src + "__attribute__((used)) static const char r5guard_marker[] = \"%s\";\n" % COMPILE_MARK

    if src.count(MARKER) != 1:
        sys.stderr.write("ERROR: MARKER 写入异常\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [r5guard] 已加 rank-5 激活 guard：含 rank-5 的子图不接管、回退 CPU")
    print("  [r5guard] predicate: data == NULL && dims[0] != 1")
    print("  [r5guard] ⚠️ 本补丁只保证「不再 abort」，不保证接管数不跌到不可接受")
    print("  [r5guard] ⚠️ 接管数必须真机实测；跌到不可接受时停下来报告，不要自行换模型")
    return 0


if __name__ == "__main__":
    sys.exit(main())