#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：把 conv.pointwise 的赋值提到 ReLU6 提前 return 之前。

## 根因（源码 + 真机双重证据）

tfl_device.c 的 fill_operation() 里：

    if (activation == kTfLiteActRelu6)
       return fused_relu6_supported(...);                 // 行 125 / 157：提前返回
    ...
    operation->conv.pointwise = dims[1]==1 && dims[2]==1;  // 行 159：被跳过

=> 任何 fused ReLU6 的卷积 pointwise 都不会被赋值，保持 0。

后果（v7）：pointwise=0 时选 write_core_interleaved 而不是 write_core_6，
且 calculate_zrl_bits() 不再提前返回 0（码流带零游程压缩）。

真机单变量实验（2026-11-06，同几何 64x64x32->64、同权重、同库）：
  ACT_NONE -> pointwise=1 -> core_6(zrl=0)  -> 9.78 ms，与 CPU 逐字节相同
  RELU6    -> pointwise=0 -> interleaved    -> 503.98 ms，4 次 MMU fault 0x00edad00，输出全零

=> writer/码流布局就是 1x1 卷积 fault 的直接原因。

## 修法

在 bias_tensor 赋值之后立刻算 pointwise（早于两处 ReLU6 return），
原来的赋值保留（同一表达式，幂等）。

用法：
    python3 patch-etnaviv-pointwise-relu6.py <tfl_device.c 路径>
    python3 patch-etnaviv-pointwise-relu6.py --selftest
"""

import argparse
import sys

MARKER = "och patch: pointwise 先于 ReLU6 判定"
COMPILE_MARKER = "och-patch-v7-pointwise-relu6"

OLD = (
    "      operation->conv.weight_tensor = &tensors[node->inputs->data[1]];\n"
    "      operation->conv.bias_tensor = &tensors[node->inputs->data[2]];\n"
)

NEW = (
    "      operation->conv.weight_tensor = &tensors[node->inputs->data[1]];\n"
    "      operation->conv.bias_tensor = &tensors[node->inputs->data[2]];\n"
    "      /* och patch: pointwise 先于 ReLU6 判定\n"
    "       * 两处 fused ReLU6 的提前 return（:125/:157）会跳过函数末尾的赋值，\n"
    "       * 导致所有 ReLU6 卷积 pointwise=0 => interleaved + zrl 压缩 => 1x1 fault。\n"
    "       * 真机单变量实验（2026-11-06）：同几何同权重，ACT_NONE(core_6) 正常、RELU6 必挂。\n"
    "       */\n"
    "      operation->conv.pointwise = operation->conv.weight_tensor->dims[1] == 1 &&\n"
    "                                   operation->conv.weight_tensor->dims[2] == 1;\n"
    "#if defined(__GNUC__) || defined(__clang__)\n"
    "      __attribute__((used)) static const char och_pointwise_relu6_marker[] =\n"
    "         \"och-patch-v7-pointwise-relu6\";\n"
    "      (void)och_pointwise_relu6_marker;\n"
    "#endif\n"
)


def apply(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        sys.stderr.write("ERROR: %s 已包含本补丁，拒绝重复应用。\n" % path)
        return 1
    n = src.count(OLD)
    if n != 1:
        sys.stderr.write("ERROR: 锚点匹配到 %d 处，期望恰好 1 处\n" % n)
        return 1
    if src.count("return fused_relu6_supported(") != 2:
        sys.stderr.write("ERROR: fused_relu6_supported 的 return 不是 2 处，结构已变\n")
        return 1
    out = src.replace(OLD, NEW, 1)
    if out.count(MARKER) != 1 or out.count(COMPILE_MARKER) != 1:
        sys.stderr.write("ERROR: 自检失败：marker 次数异常\n")
        return 1
    if out.count("operation->conv.pointwise =") != src.count("operation->conv.pointwise =") + 1:
        sys.stderr.write("ERROR: 自检失败：pointwise 赋值没有恰好 +1\n")
        return 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print("  [pointwise-relu6] 已在 ReLU6 return 之前插入 pointwise 计算")
    print("  [pointwise-relu6] 编译期标记：%s" % COMPILE_MARKER)
    return 0


def selftest():
    sample = (
        "static bool fill_operation(void)\n{\n"
        "      operation->conv.weight_tensor = &tensors[node->inputs->data[1]];\n"
        "      operation->conv.bias_tensor = &tensors[node->inputs->data[2]];\n"
        "      if (a) {\n"
        "         if (act == 3)\n"
        "            return fused_relu6_supported(t);\n"
        "      } else {\n"
        "         if (act == 3)\n"
        "            return fused_relu6_supported(t);\n"
        "      }\n"
        "      operation->conv.pointwise = w->dims[1] == 1 && w->dims[2] == 1;\n"
        "}\n")
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".c")
    os.write(fd, sample.encode())
    os.close(fd)
    try:
        if apply(p) != 0:
            print("SELFTEST FAIL: 首次应用失败"); return 1
        if apply(p) == 0:
            print("SELFTEST FAIL: 重复应用未被拒绝"); return 1
        print("SELFTEST OK")
        return 0
    finally:
        os.unlink(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.path:
        ap.error("需要 <tfl_device.c> 或 --selftest")
    return apply(a.path)


if __name__ == "__main__":
    sys.exit(main())
