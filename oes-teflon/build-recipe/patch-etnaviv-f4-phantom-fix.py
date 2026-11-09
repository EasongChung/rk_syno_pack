#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch-etnaviv-f4-phantom-fix.py —— F4：幻影通道钳位补全（转正候选，取证件专用）

## 背景（交接包 HANDOFF-20261109 §4 / 报告105 / 报告34）

v7 三个写入器共用通道映射式：

    out_channel = core * kernels_in_superblock + kernel
                + superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used

而 kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)（v7:193）
把通道总数上取整到 cores_used(=8) 的倍数 ⇒ 末批多出
    m = 8 * ceil(C/8) - C
个幻影通道，被真实执行 ⇒ 三处越界：
    v7:226/391 权重越界读、v7:238/407 biases[out_channel] 越界读（host）、
    v7:251/426 输出偏移越界写（GPU，写出输出张量之外）。

c335 的 PHANTOM_GATE（out_c%8!=0 回落 CPU）就是为兜底本缺陷 ⇒ 约 90% 算力在 CPU。
F4 = 修掉缺陷本体，拿回 122/244 族卷积的 NPU 接管。

## 本补丁做什么

在原式算出的 out_channel 后加尾部钳位（原式逐字保留）：

    out_channel = och_f4_clamp_out_channel(out_channel, output_channels);

覆盖站点（与 PHANTOM_CLAMP 相同，锚点已实证）：
  · write_core_6          —— 1×1 pointwise 路径，公式行 2 次（权重指针 + 偏置/输出偏移）
    ★ 两处必须同钳：只钳一处 ⇒ 权重取 C-1 而偏置/输出取 C ⇒ 不越界但数值全错。
  · write_core_sequential —— depthwise 路径，公式行 1 次。
  · write_core_interleaved —— **不动**，由 OCFIX 覆盖（本补丁强制要求 OCFIX 已在）。

## 与 PHANTOM_CLAMP 的关系（为什么另立补丁而不是直接启用旧件）

  · 语义相同（钳到 C-1），锚点相同；但 PHANTOM_CLAMP 从未转正（报告47：被 GATE 取代），
    store/add 语义未决、无运行期遥测、无 F4 验证链绑定。
  · F4 增加：
    1) 专属编译期标记 och-patch-v7-f4-phantom-fix（产物判据）；
    2) helper 内 ML_DBG 遥测：每次钳位打印原值→钳后值（ETNA_MESA_DEBUG=ml_msgs 时可见，
       供 dump 场次逐作业核对幻影槽数量与落点）；
    3) fail-closed 前置：OCFIX 必须已应用（interleaved 有人管）；
       PHANTOM_CLAMP 必须未应用（同站点互斥，防双重钳位/锚点冲突）。
  · 两者互斥：本补丁检测到 och-patch-v7-phantom-clamp 标记即拒绝。

## ⚠️ 数值安全未决（不得跳过判决直接交付）

钳位后幻影核 = 重算通道 C-1（同权重/同偏置/同输出偏移）：
  · NN core 输出为**存储**语义 ⇒ 两次写同值 ⇒ 幂等无害；
  · 若为**累加**语义 ⇒ 通道 C-1 翻倍 ⇒ 数值全错。
判决实验 = 合成 C=122 单 conv（非退化量化，已知不挂点）NPU vs CPU 逐元素（重点通道 121）。
**判决通过前本补丁产物只能用于取证，不得交付。**

## 用法

    python3 patch-etnaviv-f4-phantom-fix.py <etnaviv_ml_nn_v7.c 路径>
    python3 patch-etnaviv-f4-phantom-fix.py --check <...>   # 只检查是否已打上

由 build-teflon.sh 的 TEFLON_F4=1 启用（要求 TEFLON_OCFIX=1，默认关）。
产物判据：strings libteflon.so | grep och-patch-v7-f4-phantom-fix
"""

import argparse
import os
import sys

MARKER = "och patch: F4 幻影通道钳位补全"
COMPILE_MARKER = "och-patch-v7-f4-phantom-fix"
HELPER_FN = "och_f4_clamp_out_channel"
OCFIX_MARKER = "och-patch-v7-outchannel-clamp"
PCLAMP_MARKER = "och-patch-v7-phantom-clamp"

FORMULA = ("unsigned out_channel = core * kernels_in_superblock + kernel + "
           "superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;")

# (函数头, 该函数体内应出现的公式行次数)
TARGETS = [
    ("write_core_6(struct etna_ml_subgraph", 2),
    ("write_core_sequential(struct etna_ml_subgraph", 1),
]
# 本补丁不得触碰（OCFIX 已覆盖）
PROTECTED = "write_core_interleaved(struct etna_ml_subgraph"

HELPER_BLOCK = """/* och patch: F4 幻影通道钳位补全
 *
 * v7 写入器按 kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)
 * 均分通道 ⇒ 总槽位 8*ceil(C/8) > C ⇒ 末批 m 个幻影槽三处越界
 * （权重读 / biases[out_channel] 读 / 输出偏移写）。
 * 钳到 output_channels-1 ⇒ 幻影核重算最后一个合法通道（同权重/同偏置/同输出偏移），
 * 存储语义下幂等、越界归零。
 *
 * ⚠️ 数值安全取决于 NN core 输出 store/add 语义（未决）：
 *    add 语义下末通道会翻倍 ⇒ 判决实验（合成 C=122 单 conv 逐元素比对）
 *    通过前不得交付。见 HANDOFF-20261109 §4 与报告 34 §2。
 *
 * ⚠️ write_core_6 两处公式必须同钳（权重指针 + 偏置/输出偏移），
 *    只钳一处会「不越界但数值全错」。
 * ⚠️ write_core_interleaved 由 OCFIX 覆盖，本补丁不动它（前置强制检查）。
 */
static inline unsigned
och_f4_clamp_out_channel(unsigned out_channel, unsigned output_channels)
{
   if (output_channels > 0 && out_channel >= output_channels) {
      ML_DBG("och-f4: phantom out_channel %u -> %u (output_channels=%u)\n",
             out_channel, output_channels - 1, output_channels);
      return output_channels - 1;
   }

   return out_channel;
}
#if defined(__GNUC__) || defined(__clang__)
   /* 编译期标记：__attribute__((used)) 让对象留在 .rodata ⇒
    * strings libteflon.so | grep och-patch-v7-f4-phantom-fix 可判补丁在不在。
    * ★ 文件作用域不能写 (void)var; 语句（CI run 39402555413 实测编译错误）。 */
   __attribute__((used)) static const char och_f4_phantom_fix_marker[] =
      "och-patch-v7-f4-phantom-fix";
#endif
"""

HELPER_ANCHOR = "#define MAX_TILE_WIDTH 64"


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    return 1


def fn_body(src, header):
    """取函数体：从函数头到下一个顶层 static 定义（或文件尾）。"""
    i = src.find(header)
    if i < 0:
        return None, None
    j = src.find("\nstatic", i + 1)
    return i, src[i: j if j > 0 else len(src)]


def indent_of(text, pos):
    line_start = text.rfind("\n", 0, pos) + 1
    seg = text[line_start:pos]
    return seg if seg.strip() == "" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--check", action="store_true", help="只检查是否已打上，不修改")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        return die("找不到文件: " + args.path)
    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    already = COMPILE_MARKER in src
    if args.check:
        print("  [f4-phantom-fix] %s: %s" % (args.path,
              "已打上 (" + COMPILE_MARKER + ")" if already else "未打上"))
        return 0 if already else 1

    if already:
        return die(args.path + " 已包含本补丁，拒绝重复应用。")

    # ---------- 0. 前置/互斥守卫（fail-closed） ----------
    if PCLAMP_MARKER in src:
        return die("检测到 PHANTOM_CLAMP 标记（" + PCLAMP_MARKER + "）—— 同站点互斥，"
                   "两者只能启用其一（防双重钳位/锚点冲突）。")
    if OCFIX_MARKER not in src:
        return die("未检测到 OCFIX 标记（" + OCFIX_MARKER + "）—— write_core_interleaved "
                   "无人钳位。请先应用 patch-etnaviv-outchannel-clamp.py（TEFLON_OCFIX=1），"
                   "再应用 F4。不要跳过：interleaved 路径的越界会照旧。")

    # ---------- 1. 结构自检：函数体与公式行次数 ----------
    for header, want in TARGETS:
        at, body = fn_body(src, header)
        if at < 0:
            return die("源文件找不到函数 " + header + "（Mesa 结构可能已变）")
        got = body.count(FORMULA)
        if got != want:
            return die("在 %s 函数体内公式行出现 %d 次，期望 %d 次。\n"
                       "       Mesa 结构已变 —— 钳位护栏依赖这些点，请人工核对后重新生成补丁，"
                       "不要跳过校验（会静默产出护栏失效的 .so，越界会照旧）。"
                       % (header, got, want))
        if "unsigned output_channels" not in body:
            return die("在 %s 函数体内找不到 unsigned output_channels 声明 —— "
                       "钳位行引用该变量，上游改名则产出编译不过的 .so。" % header)
    _, prot_body = fn_body(src, PROTECTED)
    if prot_body is None:
        return die("源文件找不到函数 " + PROTECTED + "（无法确认 OCFIX 覆盖范围）")
    if prot_body.count(FORMULA) != 1:
        return die("在 %s 函数体内公式行出现 %d 次，期望 1 次；保护范围假设失效。"
                   % (PROTECTED, prot_body.count(FORMULA)))
    # OCFIX 的钳位必须真的在 interleaved 函数体内（标记可能在别处 ⇒ 双重确认）
    if "out_channel = output_channels - 1;" not in prot_body:
        return die("OCFIX 标记在，但 write_core_interleaved 体内没有钳位行 —— "
                   "OCFIX 应用形态异常，拒绝继续。")

    # ---------- 2. 插入 helper（文件作用域，锚点唯一） ----------
    if src.count(HELPER_ANCHOR) != 1:
        return die("锚点 %s 出现 %d 次，期望恰好 1 次。"
                   % (HELPER_ANCHOR, src.count(HELPER_ANCHOR)))
    src = src.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)

    # ---------- 3. 目标函数体内逐处加钳位（按函数体定位，缩进保留） ----------
    for header, want in TARGETS:
        at, body = fn_body(src, header)
        out = []
        pos = 0
        added = 0
        while True:
            k = body.find(FORMULA, pos)
            if k < 0:
                out.append(body[pos:])
                break
            ind = indent_of(body, k)
            out.append(body[pos:k])
            out.append(FORMULA)
            out.append("\n" + ind + "out_channel = " + HELPER_FN +
                       "(out_channel, output_channels);")
            pos = k + len(FORMULA)
            added += 1
        if added != want:
            return die("内部错误：%s 加了 %d 处，期望 %d 处" % (header, added, want))
        src = src[:at] + "".join(out) + src[at + len(body):]

    # ---------- 4. 反向自检 ----------
    for header, want in TARGETS:
        _, body = fn_body(src, header)
        n_formula = body.count(FORMULA)
        n_clamp = body.count("out_channel = " + HELPER_FN + "(out_channel, output_channels);")
        if n_formula != want:
            return die("打完补丁后 %s 的公式行变成 %d 次（期望 %d）" % (header, n_formula, want))
        if n_clamp != want:
            return die("打完补丁后 %s 的钳位行出现 %d 次（期望 %d）—— 替换形态不对"
                       % (header, n_clamp, want))
    _, prot_body = fn_body(src, PROTECTED)
    if HELPER_FN in prot_body:
        return die("本补丁污染了 %s（OCFIX 已覆盖，不应重复改）" % PROTECTED)
    if src.count("static inline unsigned\n" + HELPER_FN + "(") != 1:
        return die("helper 函数定义不唯一")
    for tok in (MARKER, COMPILE_MARKER, HELPER_FN, "output_channels - 1", "och-f4: phantom"):
        if tok not in src:
            return die("打完补丁后自检失败，缺少: " + tok)
    if src.count(FORMULA) != 4:
        return die("原公式行总数变成 %d（期望 4：core_6 两处 + sequential 一处 + interleaved 一处）"
                   % src.count(FORMULA))

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [f4-phantom-fix] 已加 F4 幻影通道钳位: " + args.path)
    print("  [f4-phantom-fix]   write_core_6          : 2 处（权重指针 / 偏置+输出偏移）")
    print("  [f4-phantom-fix]   write_core_sequential : 1 处（depthwise 路径）")
    print("  [f4-phantom-fix]   write_core_interleaved : 未动（OCFIX 前置已验证）")
    print("  [f4-phantom-fix] 遥测: ML_DBG och-f4 phantom（需 ETNA_MESA_DEBUG=ml_msgs）")
    print("  [f4-phantom-fix] ⚠️ store/add 语义判决通过前只可取证，不得交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
