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
     要拿回加速 ⇒ F4 钳位修复（patch-etnaviv-f4-phantom-fix.py）或模型侧 pad-to-8。
  2. 判据只用了 `m > 0`（必要条件），**没有用**更紧的
     `rem==1 ∨ FP≥3 ∨ Aff≥3`（那需要 screen 算 sb/d/rem）。
  3. 只有**卷积**被门控；其它算子类型不受影响（但它们不产生幻影通道）。

## 用法

    python3 patch-etnaviv-phantom-gate.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_PHANTOM_GATE=1` 启用，默认关。
产物判据：`strings libteflon.so | grep och-patch-v7-phantom-gate`

## 运行期开关（2026-11-09 F4 线增补，oca 两轮复核采纳）

编译进产物后，门的行为再由运行期环境变量 `TEFLON_PHANTOM_GATE` 门控：

  · unset 或 =1 ⇒ 门开（与本补丁历史行为逐字节一致，交付默认）；
  · =0          ⇒ 关门（F4 取证/判决实验用：让 out_c%8!=0 的形状进 NPU）。

关门时打 ML_DBG 通告（不静默）；门开拒绝形状时仍打 ML_DBG。
判决实验协议（第一批）：TEFLON_PHANTOM_GATE=0 + ETNA_MESA_DEBUG=ml_msgs。
⚠️ 关门运行只可取证，数值判决通过前不得交付。
旧版补丁（无运行期开关）已应用的树：本补丁器自动原地升级（不拒绝）。
"""

import argparse
import re
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
       '          * ⚠️ 这是**兜底**不是根治：要拿回加速 ⇒ F4 钳位修复或模型侧 pad-to-8。\n'
       '          * ⚠️ 判据用固定的 8（A311D 的 nn_core_count，NNDUMP 实测确认）：换芯片若\n'
       '          *    cores_used 不同，本判据只会**过度门控**（少加速），不会漏放行 ⇒ 仍安全。\n'
       '          */\n'
       '         {\n'
       '            /* och patch: 幻影门运行期开关（F4 线，2026-11-09）\n'
       '             * unset/1 = 门开（默认，与历史行为一致）；0 = 关门（取证/判决用）。\n'
       '             * 关门通告走 ML_DBG 不静默；拒绝形状 likewise。 */\n'
       '            const char *och_pgate_env = getenv("TEFLON_PHANTOM_GATE");\n'
       '            int och_pgate_on = !(och_pgate_env && och_pgate_env[0] == \'0\' && och_pgate_env[1] == \'\\0\');\n'
       '            unsigned och_out_c = operation->output_tensors[0]->dims[3];\n'
       '            if (!och_pgate_on) {\n'
       '               ML_DBG("och-pgate: 幻影门运行期关闭 (TEFLON_PHANTOM_GATE=0, out_c=%u)\\n", och_out_c);\n'
       '            } else if (och_out_c % 8 != 0) {\n'
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

# 旧版守卫块头部（无运行期开关）——升级路径的定位锚点
LEGACY_HEAD = ('         {\n'
               '            unsigned och_out_c = operation->output_tensors[0]->dims[3];\n'
               '            if (och_out_c % 8 != 0) {\n')
UPGRADED_HEAD = ('         {\n'
                 '            /* och patch: 幻影门运行期开关（F4 线，2026-11-09）\n'
                 '             * unset/1 = 门开（默认，与历史行为一致）；0 = 关门（取证/判决用）。\n'
                 '             * 关门通告走 ML_DBG 不静默；拒绝形状 likewise。 */\n'
                 '            const char *och_pgate_env = getenv("TEFLON_PHANTOM_GATE");\n'
                 '            int och_pgate_on = !(och_pgate_env && och_pgate_env[0] == \'0\' && och_pgate_env[1] == \'\\0\');\n'
                 '            unsigned och_out_c = operation->output_tensors[0]->dims[3];\n'
                 '            if (!och_pgate_on) {\n'
                 '               ML_DBG("och-pgate: 幻影门运行期关闭 (TEFLON_PHANTOM_GATE=0, out_c=%u)\\n", och_out_c);\n'
                 '            } else if (och_out_c % 8 != 0) {\n')


def ensure_stdlib(src):
    """getenv 需要 <stdlib.h>：幂等补全（KPAT 补丁器先例）。"""
    if "#include <stdlib.h>" in src:
        return src
    m_inc = re.search(r'^#include .*$', src, re.M)
    if not m_inc:
        sys.stderr.write("ERROR: 源文件里找不到 #include 行，无法补 <stdlib.h>（getenv 无声明）\n")
        return None
    return (src[:m_inc.end()] + "\n#include <stdlib.h> /* och-pgate: getenv 需要 */"
            + src[m_inc.end():])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        if 'getenv("TEFLON_PHANTOM_GATE")' in src:
            sys.stderr.write("ERROR: " + args.path + " 已包含本补丁（含运行期开关），拒绝重复应用。\n")
            return 1
        # 旧版（无运行期开关）升级路径：守卫块头部原地替换；其余部分（拒绝日志/
        # return false/编译期标记）逐字保留。CI 从 pristine mesa 构建走不到这里。
        if src.count(LEGACY_HEAD) != 1:
            sys.stderr.write("ERROR: 旧版守卫块定位失败（出现 %d 次，期望 1）—— 人工核对，不要跳过\n"
                             % src.count(LEGACY_HEAD))
            return 1
        src = src.replace(LEGACY_HEAD, UPGRADED_HEAD, 1)
        src = ensure_stdlib(src)
        if src is None:
            return 1
        for token in ('getenv("TEFLON_PHANTOM_GATE")', "och-pgate: 幻影门运行期关闭",
                      "och_out_c % 8 != 0", "return false;", COMPILE_MARKER):
            if token not in src:
                sys.stderr.write("ERROR: 升级后自检失败，缺少: " + token + "\n")
                return 1
        with open(args.path, "w", encoding="utf-8") as f:
            f.write(src)
        print("  [och-pgate] 旧版已升级：运行期开关 TEFLON_PHANTOM_GATE（unset/1=门开，0=关）")
        return 0

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
                "       Mesa 结构已变 —— 守卫逻辑依赖这些点，请人工核对并重新生成补丁。\n")
            return 1

    src = src.replace(OLD, NEW, 1)
    src = ensure_stdlib(src)
    if src is None:
        return 1

    for token in (MARKER, COMPILE_MARKER, "och_out_c % 8 != 0", "return false;",
                  'getenv("TEFLON_PHANTOM_GATE")', "och-pgate: 幻影门运行期关闭"):
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
    if 'getenv("TEFLON_PHANTOM_GATE")' not in guard:
        sys.stderr.write("ERROR: 守卫块内没有运行期开关 —— 补丁形态不对\n")
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
    print("  [och-pgate] 运行期开关：TEFLON_PHANTOM_GATE unset/1=门开（默认），0=关（取证/判决用）")
    print("  [och-pgate] 预期效果：不挂 + 接管数下降；**兜底不是根治**")
    print("  [och-pgate] 要拿回加速 ⇒ F4 钳位修复（patch-etnaviv-f4-phantom-fix.py）或模型侧 pad-to-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
