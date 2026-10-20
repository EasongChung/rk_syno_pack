#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：write_core_6() / write_core_sequential() 的幻影通道(out_channel)尾部钳位。

## 背景（离线推导 + 真机 10 点实测）

`etnaviv_ml_nn_v7.c` 的三个写入器都用同一条通道映射式：

    out_channel = core * kernels_in_superblock + kernel
                + superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used

其中 `kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)`（v7:193）
把通道总数**向上取整到 cores_used(=8) 的倍数** ⇒ 末批必然多出

    m = cores_used * DIV_ROUND_UP(output_channels, cores_used) - output_channels
      = 8 * ceil(C/8) - C        (0..7)

个**不存在的通道**（幻影通道），而它们会被**真实执行**，造成三处越界：

    v7:226 / v7:391  权重张量越界读        (host 侧)
    v7:238 / v7:407  biases[out_channel] 越界读 (host 侧)
    v7:251 / v7:426  输出偏移越界写        (GPU 侧，写到输出张量之外)

## 本补丁做什么

在原式算出的 `out_channel` 后面加一条尾部钳位；**原式逐字保留**。

    out_channel = och_clamp_out_channel(out_channel, output_channels);

## 覆盖范围（关键：漏一处就会静默算错）

  · write_core_6          —— 1×1 pointwise 路径，公式行出现 **2 次**（v7:225 / v7:231）
      ★ 两处**必须同时**钳：v7:225 决定权重指针，v7:231 决定偏置与输出偏移。
        只钳一处 ⇒ 权重取自通道 C-1、而偏置/输出偏移取自通道 C ⇒ 不越界但**数值全错**。
  · write_core_sequential —— depthwise 路径（in_c==1），公式行 1 次（v7:389）
  · write_core_interleaved —— **不动**。已由 patch-etnaviv-outchannel-clamp.py（OCFIX）覆盖，
      两补丁都改同一行会导致锚点冲突与重复钳位。

## 钳位后的语义（重要，不要跳过）

钳位到 `output_channels - 1` 后，幻影 kernel 等价于「**再算一遍最后一个合法通道**」：
权重指针、偏置、输出偏移三者都指向通道 C−1，且该通道的合法 kernel 也会写同一处。
若 NN core 对输出是**存储**语义 ⇒ 两次写同值 ⇒ 幂等、结果不变；
若是**累加**语义 ⇒ 通道 C−1 会被写两次 ⇒ 末通道数值翻倍（**必须实测，见报告 34 的 C 段判据**）。

## 与其它方案的关系（为什么是钳位而不是别的）

  · **P1（末批 core 间距 rem→d）已废止**：会让合法通道成空洞（244 下丢通道 241/242/243），
    越界反而从 4 通道涨到 7 通道 ⇒ 用错误结果换不挂。**不要"顺手优化"原式。**
  · **P6（按核截断幻影 kernel）结构性不可行**：`etnaviv_ml_nn.c:37/1168` 的描述符只有
    **一个全局** `kernels_per_core` 字段（7 bit，值 = d）⇒ 表达不了 per-core 异构 kpc。
  · **模型侧 pad-to-8**（把 C 补到 8 的倍数）是更彻底的零驱动方案，但只对 dense/pointwise
    成立（depthwise 输出通道与输入通道 1:1 绑定），且需要改模型图。

## 验收不靠「不崩」

必须过四级判据第 ④ 级（与 CPU 参考逐字节一致）。不崩只说明越界消失。
另需实测「通道 C−1 的数值是否被写坏」（store/add 语义）。

## 用法

    python3 patch-etnaviv-phantom-clamp.py <etnaviv_ml_nn_v7.c 路径>
    python3 patch-etnaviv-phantom-clamp.py --check <...>   # 只检查是否已打上

由 `build-teflon.sh` 的 `TEFLON_PHANTOM_CLAMP=1` 启用，默认关。
产物判据：`strings libteflon.so | grep och-patch-v7-phantom-clamp`
"""

import argparse
import os
import sys

MARKER = "och patch: 幻影通道尾部钳位"
COMPILE_MARKER = "och-patch-v7-phantom-clamp"
HELPER_FN = "och_clamp_out_channel"

FORMULA = ("unsigned out_channel = core * kernels_in_superblock + kernel + "
           "superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;")

# (函数头, 该函数体内应出现的公式行次数)
TARGETS = [
    ("write_core_6(struct etna_ml_subgraph", 2),
    ("write_core_sequential(struct etna_ml_subgraph", 1),
]
# 本补丁不得触碰（OCFIX 已覆盖）
PROTECTED = "write_core_interleaved(struct etna_ml_subgraph"

HELPER_BLOCK = """/* och patch: 幻影通道尾部钳位
 *
 * 背景：write_core_6 / write_core_sequential 用
 *   out_channel = core * kernels_in_superblock + kernel
 *               + superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used
 * 分配通道，而 kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)（v7:193）
 * 把通道总数向上取整到 cores_used 的倍数 ⇒ 末批必然多出
 *   m = cores_used * ceil(output_channels/cores_used) - output_channels
 * 个「幻影通道」(out_channel >= output_channels)，且会被真实执行：
 *   - v7:226 / v7:391  权重张量越界读（host 侧）
 *   - v7:238 / v7:407  biases[out_channel] 越界读（host 侧）
 *   - v7:251 / v7:426  输出偏移越界写（GPU 侧，写到输出张量之外）
 *
 * 钳到 output_channels-1 后，幻影 kernel 等价于「重算最后一个合法通道」：
 * 权重/偏置/输出偏移三者一致 ⇒ 存储语义下幂等、数值语义不变，同时消除全部越界。
 *
 * ⚠️ write_core_6 里公式行出现两次，**两处都必须钳**：一处定权重指针，
 *    一处定偏置与输出偏移；只钳一处会「不越界但数值全错」。
 *
 * ⚠️ write_core_interleaved 的同一处已由 patch-etnaviv-outchannel-clamp.py（OCFIX）覆盖，
 *    语义完全相同 ⇒ 此处**不重复**改它，避免两个补丁锚点冲突。
 */
static inline unsigned
och_clamp_out_channel(unsigned out_channel, unsigned output_channels)
{
   if (output_channels > 0 && out_channel >= output_channels)
      return output_channels - 1;

   return out_channel;
}
#if defined(__GNUC__) || defined(__clang__)
   /* 编译期标记：注释里的字符串编译后会消失（已实测），只有 __attribute__((used))
    * 的对象才会留在 .rodata ⇒ `strings libteflon.so | grep och-patch-v7-phantom-clamp`
    * 才能真正判出「这个 .so 里有没有本补丁」。反过来：若 strings 找不到它，说明补丁
    * 没打进去，构建流程与产物都不对 —— 必须当场失败，而不是交付。 */
   __attribute__((used)) static const char och_phantom_clamp_marker[] =
      "och-patch-v7-phantom-clamp";
   (void)och_phantom_clamp_marker;
#endif
"""

HELPER_ANCHOR = "#define MAX_TILE_WIDTH 64"


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    return 1


def fn_body(src, header):
    """取函数体：从函数头到下一个顶层 `static` 定义（或文件尾）。"""
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
        print("  [phantom-clamp] %s: %s" % (args.path,
              "已打上 (" + COMPILE_MARKER + ")" if already else "未打上"))
        return 0 if already else 1

    if already:
        return die(args.path + " 已包含本补丁，拒绝重复应用。")

    # ---------- 1. 结构自检：三个函数体必须都在，且公式行次数符合预期 ----------
    for header, want in TARGETS:
        at, body = fn_body(src, header)
        if at < 0:
            return die("源文件找不到函数 " + header + "（Mesa 结构可能已变）")
        got = body.count(FORMULA)
        if got != want:
            return die("在 %s 函数体内公式行出现 %d 次，期望 %d 次。\n"
                       "       Mesa 结构已变（或有人只改了一处同款调用）—— 钳位护栏依赖这些点，"
                       "请人工核对后重新生成补丁，不要跳过校验\n"
                       "       （会静默产出护栏失效的 .so，越界会照旧）。"
                       % (header, got, want))
        # 变量存在性检查：钳位行要引用 output_channels。若上游把它改名了，
        # 补丁会插入**编译不过**的代码 ⇒ 必须当场失败，而不是产出一个编不过的 .so。
        if "unsigned output_channels" not in body:
            return die("在 %s 函数体内找不到 `unsigned output_channels` 声明 ——\n"
                       "       钳位行引用该变量，若上游改名则补丁会产出编译不过的 .so。\n"
                       "       请人工核对后重新生成补丁。" % header)
    _, prot_body = fn_body(src, PROTECTED)
    if prot_body is None:
        return die("源文件找不到函数 " + PROTECTED + "（无法确认 OCFIX 覆盖范围）")
    if prot_body.count(FORMULA) != 1:
        return die("在 %s 函数体内公式行出现 %d 次，期望 1 次；"
                   "本补丁的保护范围假设失效。" % (PROTECTED, prot_body.count(FORMULA)))

    # ---------- 2. 插入 helper（文件作用域，锚点必须唯一） ----------
    if src.count(HELPER_ANCHOR) != 1:
        return die("锚点 `%s` 出现 %d 次，期望恰好 1 次。"
                   % (HELPER_ANCHOR, src.count(HELPER_ANCHOR)))
    src = src.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)

    # ---------- 3. 在两个目标函数体内逐处加钳位 ----------
    # ⚠️ 缩进不可用来区分站点：write_core_6 有 9/12 两种缩进，write_core_sequential 是 9，
    #    write_core_interleaved 是 12 ⇒ 必须**按函数体**定位。
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
    for tok in (MARKER, COMPILE_MARKER, HELPER_FN, "output_channels - 1"):
        if tok not in src:
            return die("打完补丁后自检失败，缺少: " + tok)
    # 原公式必须仍在（本补丁只允许追加）
    if src.count(FORMULA) != 4:
        return die("原公式行总数变成 %d（期望 4：write_core_6 两处 + sequential 一处 + interleaved 一处）"
                   % src.count(FORMULA))

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [phantom-clamp] 已加幻影通道钳位: " + args.path)
    print("  [phantom-clamp]   write_core_6          : 2 处（v7:225 权重指针 / v7:231 偏置+输出偏移）")
    print("  [phantom-clamp]   write_core_sequential : 1 处（v7:389，depthwise 路径）")
    print("  [phantom-clamp]   write_core_interleaved : 未动（OCFIX 已覆盖）")
    print("  [phantom-clamp] 原式逐字保留（未改成 d —— 改 d 会丢合法通道，已废止）")
    print("  [phantom-clamp] ⚠️ 只算「越界消失」；可用性看四级判据 ④，且必须实测通道 C-1 的数值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
