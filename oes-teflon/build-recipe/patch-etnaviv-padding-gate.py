#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控补丁：`stride>1` 的 depthwise 且 `padding_top/left>0` ⇒ **不接管**，回落 CPU。

## 为什么需要它（报告 56）

v7 上 `stride>1` 的 depthwise 一律走 `ETNA_NN_POOLING_FIRST_PIXEL`
（`etnaviv_ml_nn.c:1141`：取每 2×2 块的**第一像素**，`out_image_*_size *= 2`）。
而 `operation->padding_same` 的**唯一**消费点在 `strided_to_normal()`
（`etnaviv_ml_nn.c:476`），该函数只在 **非** pooling 路径被调用
（`etnaviv_ml_nn.c:619`：`if (operation->stride > 1 && !operation->pooling_first_pixel)`）。

⇒ pooling 路径**完全不补偿 `pad_top`**：硬件按 `pad_top = 0` 采样。

对 `in=99, k=3, s=2, SAME`：`pad_total = 2 ⇒ pad_top = 1` ⇒ 输出整体偏 **1 个输入像素**。
对 `in=100`（偶数）：`pad_total = 1 ⇒ pad_top = 0` ⇒ 恰好正确。

### 证据（报告 56）

* 脉冲栅格真机实验：CPU 一个脉冲只影响 **1 个**输出像素，NPU 铺成 **2×2 邻域**
* 离线精确复算（`verify/op7-sim.py`）：
  * `pad_top=1` vs CPU = **100.00% ±1**
  * `pad_top=0` vs 真机 NPU = **99.64% ±1 / max 2**
  * 偶数边长（100×100）真机 NPU vs CPU = **99.63% ±1 / max 3**（剩余为 per-channel 折叠的已知损失）
* 真实 concept 模型（201 op）静态扫描：4 个 stride-2 depthwise 里
  **3 个受影响**（op#6 99×99、op#9 99×99、op#151 25×25，均为奇数边长）

## 判据为什么是「depthwise ∧ stride>1 ∧ padding_top/left>0」

* 只有 depthwise 会走 `calc_pooling_first_pixel()` 的**无条件** `return true`
  （`etnaviv_ml_nn.c:504`）；非 depthwise 的 stride-2 卷积在 v7 上返回 false
  ⇒ 走 `strided_to_normal()`，padding 被正确处理 ⇒ **不需要门控**。
* 1×1 卷积的 `pad_top/left` 恒为 0 ⇒ 不会被误伤。
* 偶数边长的 stride-2 depthwise（如官方 MobileNetV1 的 224→112）`pad_top=0`
  ⇒ **不受影响**，已交付的回归不变。

## 运行期门（交付语义）

    TEFLON_PADGATE 不设  ⇒ 开
    TEFLON_PADGATE=0    ⇒ 关（与上游行为一致，用于对照构建/负控制）
    TEFLON_PADGATE=其它  ⇒ 开

## 与既有补丁的关系

注入点与 `patch-etnaviv-phantom-gate.py`（PHANTOM_GATE）、
`patch-etnaviv-filter-shape-guard.py`（FSGUARD）**同一个函数**，但：

* 三者都用**各自的逐字锚点**，且锚点互不为前缀 ⇒ 可任意顺序共存；
* 本补丁的守卫块断言「块内出现 `return false;` 且**不得**出现 `supported = true;`」
  （fail-open 检查），与 PHANTOM_GATE 同款。

## 用法

    python3 patch-etnaviv-padding-gate.py <etnaviv_ml.c 路径>

产物判据：`grep -a 'och-patch-v7-padding-gate' libteflon.so`
"""

import argparse
import sys

MARKER = "och patch: pooling-first-pixel padding 门控"
COMPILE_MARKER = "och-patch-v7-padding-gate"
TARGET_FN = "etna_ml_operation_supported(struct pipe_ml_device"

# (特征串, 至少出现次数) —— 用「下限」而非「恰好」：同函数里可能已有别的补丁注入
STRUCT = (
    ("case PIPE_ML_OPERATION_TYPE_CONVOLUTION:", 1),
    ("operation->conv.weight_tensor", 1),
    ("operation->conv.dilation_width_factor == 1", 1),
)

# ↓↓↓ 逐字锚点（与 patch-etnaviv-phantom-gate.py 相同，取自真实源码）↓↓↓
OLD = ('      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n'
       '         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n')
NEW = ('      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n'
       '         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n'
       '\n'
       '         /* och patch: pooling-first-pixel padding 门控\n'
       '          *\n'
       '          * v7 上 stride>1 的 depthwise 走 ETNA_NN_POOLING_FIRST_PIXEL，该路径\n'
       '          * 只在 strided_to_normal() 里消费 operation->padding_same；而 pooling\n'
       '          * 路径根本不调用它 ⇒ 硬件按 pad_top=0 采样。\n'
       '          *\n'
       '          * TFLite SAME 在「奇数边长 + k=3 + stride=2」时要求 pad_top=1\n'
       '          * ⇒ 输出整体偏 1 个输入像素（报告 56：脉冲栅格 + 离线复算双重确认）。\n'
       '          *\n'
       '          * 真机 concept 模型（201 op）里 op#6/#9/#151 三个层属于此类。\n'
       '          * 这里宁可不接管（回落 CPU），也不给「偏 1 像素的加速」。\n'
       '          * 偶数边长（pad_top=0，如 MobileNetV1 224→112）不受影响。\n'
       '          *\n'
       '          * TEFLON_PADGATE=0 ⇒ 关掉本门（与上游行为一致，用于对照/负控制）。\n'
       '          */\n'
       '         {\n'
       '            const char *och_pad_env = getenv("TEFLON_PADGATE");\n'
       '            if ((och_pad_env == NULL || och_pad_env[0] != \'0\') &&\n'
       '                operation->conv.depthwise &&\n'
       '                (operation->conv.stride_x > 1 || operation->conv.stride_y > 1) &&\n'
       '                (operation->conv.padding_top > 0 || operation->conv.padding_left > 0)) {\n'
       '               ML_DBG("stride>1 depthwise 且 padding_top/left>0："\n'
       '                      "pooling 路径不补偿 pad_top ⇒ 回退 CPU\\n");\n'
       '               /* return false 而不是 break：break 依赖「此刻 supported 仍是 false」\n'
       '                * 这个隐含前提；return false 是显式失败出口（同 PHANTOM_GATE）。 */\n'
       '               return false;\n'
       '            }\n'
       '         }\n'
       '#if defined(__GNUC__) || defined(__clang__)\n'
       '         __attribute__((used)) static const char och_padding_gate_marker[] =\n'
       '            "och-patch-v7-padding-gate";\n'
       '         (void)och_padding_gate_marker;\n'
       '#endif\n')
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处锚点，期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对并重新生成补丁，不要跳过校验\n"
            "       （会静默产出**未门控**的 .so，数值偏差会照旧）。\n")
        return 1

    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    nxt = src.find("\n}\n", fn_at)
    fn_body = src[fn_at:nxt if nxt > 0 else len(src)]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got < want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 " + str(got)
                + " 处，至少应有 " + str(want) + " 处。\n"
                "       Mesa 结构已变 —— 守卫逻辑依赖这些点，请人工核对后重新生成补丁。\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    for token in (MARKER, COMPILE_MARKER, 'getenv("TEFLON_PADGATE")', "return false;"):
        if token not in src:
            sys.stderr.write("ERROR: 打完补丁后自检失败，缺少: " + token + "\n")
            return 1

    # fail-open 检查（同 PHANTOM_GATE）：`return false;` 必须在守卫块内，
    # 且该块内不得出现 `supported = true;`。
    g_at = src.index(MARKER, src.index(TARGET_FN))
    guard = src[g_at:g_at + len(NEW)]
    if "return false;" not in guard:
        sys.stderr.write("ERROR: 守卫块内没有 return false —— 补丁形态不对\n")
        return 1
    if "supported = true;" in guard:
        sys.stderr.write("ERROR: 守卫块内出现 `supported = true;` —— 守卫会 fail-open\n")
        return 1
    if "padding_top > 0" not in guard or "depthwise" not in guard:
        sys.stderr.write("ERROR: 守卫块内没有判据 —— 补丁形态不对\n")
        return 1

    # NEW 以 OLD 为前缀 ⇒ 只能断言两者各出现恰好 1 次（同 PHANTOM_GATE 的注释）
    if src.count(OLD) != 1:
        sys.stderr.write("ERROR: 锚点出现次数异常（期望 1），替换形态不对\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 打完补丁后新代码出现次数为 " + str(src.count(NEW)) + "（期望 1）\n")
        return 1

    # 共存检查：FSGUARD / PHANTOM_GATE 的锚点必须都还在
    for anchor, name in (
        ('         struct pipe_tensor *output_tensor = operation->output_tensors[0];\n', "FSGUARD"),
        ('            unsigned och_out_c = operation->output_tensors[0]->dims[3];\n', "PHANTOM_GATE"),
    ):
        if anchor in src and src.count(anchor) != 1:
            sys.stderr.write("ERROR: " + name + " 的锚点出现次数异常 —— 共存前提失效\n")
            return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-padgate] 已加 padding 门控：stride>1 且 pad_top/left>0 的 depthwise 不接管")
    print("  [och-padgate] 预期效果：这些层回落 CPU（数值正确），少一点加速")
    print("  [och-padgate] 运行期：TEFLON_PADGATE=0 可关（对照/负控制）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
