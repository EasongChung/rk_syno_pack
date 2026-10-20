#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控补丁：输出通道数不是 8 的倍数的卷积**不接管**，回落 CPU。

## 为什么需要它（零设备推导 + 真机实测）

Mesa `etnaviv_ml_nn_v7.c` 的三个写入器都按
    kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)      (v7:193)
分配通道 ⇒ 通道总数上取整到 cores_used 的倍数 ⇒ 末批必然多出

    m = cores_used*ceil(C/cores_used) - C        (A311D: cores_used = 8 ⇒ m = 8*ceil(C/8)-C)

个**幻影通道**（不存在的通道），它们会被真实执行 ⇒ 三处越界：
    v7:226 / v7:391  权重张量越界读（host）
    v7:238 / v7:407  biases[out_channel] 越界读（host）
    v7:251 / v7:426  输出偏移越界写（GPU，写到输出张量之外）

真机实测（1×1 conv）：C=244、284 挂死（244 只挂 GPU 可 recover，284 整机失联）；
C=122/196/220/240/248/252/260/280 不挂。
⇒ `m > 0` 是**必要条件**（不是充分条件：≥3 个判据同时 10/10 拟合，判别律仍欠定）。

⇒ **本补丁把「m>0」的形状整体挡在 NPU 之外** ⇒ 该机制**不可能再触发**。
   代价：这些形状回落 CPU（损失加速），换来 **零挂死风险**。

## 判据为什么取「C % 8 != 0」这个最简形式

  · `m > 0` ⇔ `C % cores_used != 0` ⇒ 对 A311D（cores_used = 8，NNDUMP 实测确认）
    就是 `C % 8 != 0`。
  · 若换一颗 cores_used 不同的芯片，这个判据会**过度门控**（把本来安全的形状也挡掉）
    ⇒ 只会少加速，**不会漏放行** ⇒ 仍然安全。（要精确到 cores_used 需要 screen，
    而 `etna_ml_operation_supported()` 里拿不到 —— 见下「已知局限」。）

## 落点与既有补丁的关系

注入点与 `patch-etnaviv-filter-shape-guard.py`（FSGUARD）**同一个函数的不同位置**：

  · FSGUARD 注入在 `struct pipe_tensor *output_tensor = ...;` 之后；
  · 本补丁注入在 `case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {` 之后、
    FSGUARD 的锚点**之前** ⇒ 两个补丁可共存、互不冲突（各自锚点唯一）。

## 已知局限（写清楚，避免被当成"已根治"）

  1. 这是**兜底**，不是根治：被门控的形状只是"不挂"，没有 NPU 加速。
     要拿回加速 ⇒ 模型侧 pad-to-8（见 tools/pad-to-8.py）或长期移植 v8 分配（P5）。
  2. 判据只用了 `m > 0`（必要条件），**没有用**更紧的
     `rem==1 ∨ FP≥3 ∨ Aff≥3`（那需要 screen 算 sb/d/rem）。
  3. 只有**卷积**被门控；其它算子类型不受影响（但它们不产生幻影通道）。

## 用法

    python3 patch-etnaviv-phantom-gate.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_PHANTOM_GATE=1` 启用，默认关。
产物判据：`strings libteflon.so | grep och-patch-v7-phantom-gate`
"""

import argparse
import sys

MARKER = "och patch: 幻影通道形状门控"
COMPILE_MARKER = "och-patch-v7-phantom-gate"
TARGET_FN = "etna_ml_operation_supported(struct pipe_ml_device"

# (特征串, 期望出现次数) —— 期望值来自真实源码统计
STRUCT = (
    ('case PIPE_ML_OPERATION_TYPE_CONVOLUTION:', 1),
    ('operation->conv.weight_tensor', 1),
    ('tensor_quantization_supported(input_tensor)', 1),
    ('operation->conv.dilation_width_factor == 1', 1),
)

# ↓↓↓ 逐字锚点（取自真实源码，勿手改）↓↓↓
OLD = ('      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n'
       '         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n')
NEW = ('      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n'
       '         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n'
       '\n'
       '         /* och patch: 幻影通道形状门控\n'
       '          *\n'
       '          * v7 的写入器按 kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)\n'
       '          * 分配通道 ⇒ 通道总数上取整到 cores_used 的倍数 ⇒ 末批多出\n'
       '          *   m = cores_used*ceil(C/cores_used) - C\n'
       '          * 个「幻影通道」（不存在的通道），它们会被真实执行 ⇒ 三处越界：\n'
       '          *   v7:226/v7:391 权重张量越界读（host）\n'
       '          *   v7:238/v7:407 biases[out_channel] 越界读（host）\n'
       '          *   v7:251/v7:426 输出偏移越界写（GPU，写到输出张量之外）\n'
       '          *\n'
       '          * 真机实测（1×1 conv）：C=244、284 挂死；122/196/220/240/248/252/260/280 不挂。\n'
       '          * ⇒ m>0 是必要条件（不是充分条件：判别律仍欠定）。\n'
       '          *\n'
       '          * 这里把 m>0 的形状整体挡在 NPU 之外 ⇒ 该机制不可能再触发。\n'
       '          * 代价只是这些形状回落 CPU（少加速），换来零挂死风险。\n'
       '          *\n'
       '          * ⚠️ 这是**兜底**不是根治：要拿回加速 ⇒ 模型侧 pad-to-8 或长期移植 v8 分配。\n'
       '          * ⚠️ 判据用固定的 8（A311D 的 nn_core_count，NNDUMP 实测确认）：换芯片若\n'
       '          *    cores_used 不同，本判据只会**过度门控**（少加速），不会漏放行 ⇒ 仍安全。\n'
       '          */\n'
       '         {\n'
       '            unsigned och_out_c = operation->output_tensors[0]->dims[3];\n'
       '            if (och_out_c % 8 != 0) {\n'
       '               ML_DBG("幻影通道形状 (out_c=%u %% 8 != 0)，回退 CPU\\n", och_out_c);\n'
       '               /* return false 而不是 break：break 依赖「此刻 supported 仍是 false」\n'
       '                * 这个隐含前提；将来若有人把 supported = true 提到守卫之前，\n'
       '                * break 就会变成 fail-open。return false 是显式失败出口。 */\n'
       '               return false;\n'
       '            }\n'
       '         }\n'
       '#if defined(__GNUC__) || defined(__clang__)\n'
       '         /* 编译期标记：只有 __attribute__((used)) 的对象才会留在 .rodata，\n'
       '          * 于是 `strings libteflon.so | grep och-patch-v7-phantom-gate`\n'
       '          * 才能真正判出「这个 .so 里有没有本补丁」。 */\n'
       '         __attribute__((used)) static const char och_phantom_gate_marker[] =\n'
       '            "och-patch-v7-phantom-gate";\n'
       '         (void)och_phantom_gate_marker;\n'
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
            "       （会静默产出**未门控**的 .so，挂死会照旧）。\n")
        return 1

    # 结构性检查：限定目标函数体 + 精确计数
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    nxt = src.find("\n}\n", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    # ⚠️ 这里用「下限」而不是「恰好」：本函数里可能已被**另一个**补丁（FSGUARD）
    #    注入了含同样 token 的代码 ⇒ 精确计数会把「先打 FSGUARD 后打门控」误判为
    #    「Mesa 结构已变」。真正的主防线是**锚点唯一性**（上面的 OLD 计数 == 1）。
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got < want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 " + str(got)
                + " 处，至少应有 " + str(want) + " 处。\n"
                "       Mesa 结构已变 —— 守卫逻辑依赖这些点，请人工核对后重新生成补丁。\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    for token in (MARKER, COMPILE_MARKER, "och_out_c % 8 != 0", "return false;"):
        if token not in src:
            sys.stderr.write("ERROR: 打完补丁后自检失败，缺少: " + token + "\n")
            return 1
    # 关键结构自检：`return false;` 必须在守卫块内、且在 `supported = true;` 之前，
    # 否则守卫会 fail-open（漏放行 = 挂死照旧）。
    # 守卫区间 = **本补丁插入的那一块自身**（不能按 case 边界取：同一 case 里
    # 还有 FSGUARD 的 supported = true，会被误判成 fail-open）
    g_at = src.index(MARKER, src.index(TARGET_FN))
    guard = src[g_at:g_at + len(NEW)]
    if "return false;" not in guard:
        sys.stderr.write("ERROR: 守卫块内没有 return false —— 补丁形态不对\n")
        return 1
    if "supported = true;" in guard:
        sys.stderr.write("ERROR: 守卫块内出现 `supported = true;` —— 守卫会 fail-open\n")
        return 1
    if "och_out_c % 8 != 0" not in guard:
        sys.stderr.write("ERROR: 守卫块内没有判据 —— 补丁形态不对\n")
        return 1
    # ⚠️ 本补丁的 NEW **以 OLD 为前缀**（只追加不替换）⇒ 「锚点不再出现」这个检查
    #    在数学上永远不成立（OCFIX 补丁已实测踩到过这个坑：正例被全部误拒）。
    #    正确的断言是：OLD 与 NEW 各出现恰好 1 次。
    if src.count(OLD) != 1:
        sys.stderr.write("ERROR: 锚点/新代码出现次数异常（期望 1），替换形态不对\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 打完补丁后新代码出现次数为 " + str(src.count(NEW)) + "（期望 1）\n")
        return 1
    # 共存检查：FSGUARD 的锚点必须还在（两个补丁同函数、不同锚点）
    fsguard_anchor = ('         struct pipe_tensor *output_tensor = operation->output_tensors[0];\n')
    if fsguard_anchor not in src:
        sys.stderr.write("ERROR: FSGUARD 的锚点不见了 —— 共存前提失效\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-pgate] 已加幻影通道形状门控：out_c %% 8 != 0 的卷积不接管（回落 CPU）")
    print("  [och-pgate] 预期效果：不挂 + 接管数下降；**兜底不是根治**")
    print("  [och-pgate] 要拿回加速 ⇒ 模型侧 pad-to-8（tools/pad-to-8.py）或长期 P5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
