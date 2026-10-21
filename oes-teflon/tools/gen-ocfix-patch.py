#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-outchannel-clamp.py。

## ⚠️ 本文件取代 gen-ocfix-patch.py（重映射版）

前一版补丁把 `out_channel` 改成
`core*kernels_per_core + 前序superblock累计偏移 + kernel`，
并删掉末尾的 `* cores_used`。**那个判断被上游自己的 v8 实现否证了**（见下），
所以重映射版作废，只保留「钳位」这一个改动。

## v8 的权威参照（`etnaviv_ml_nn_v8.c` 的 `fill_weights()`，同仓库、同用途）

```c
unsigned full_superblock = DIV_ROUND_UP(output_channels, nn_core_count * superblocks);   // :722
unsigned channel_per_superblock[superblocks];
for (unsigned superblock = 0; superblock < superblocks; superblock++)
   channel_per_superblock[superblock] = superblock * full_superblock * cores_used;        // :726
...
for (unsigned core = 0; core < cores_used; core++) {
   unsigned kernels_per_core = output_channels / cores_used;                              // :729
   if (core < output_channels % cores_used) kernels_per_core++;                           // :730-731
   ...
   for (unsigned superblock = 0; superblock < superblocks; superblock++) {
      ...
      unsigned first_channel = channel_per_superblock[superblock];                        // :750
      encode_superblock(..., kernels_in_superblock, first_channel);
      channel_per_superblock[superblock] += kernels_in_superblock;                        // :754
   }
}
```

由此确立三件事（v7 的公式结构与之一致，**不是笔误**）：

1. `:726` **原样保留了** `superblock * full_superblock * cores_used`
   —— 正是我曾判为「多余的 `* cores_used`」。**它是有意的 superblock 步长。**
2. 每个 core 推进 `kernels_in_superblock`（`:754` 的累计），**不是 `kernels_per_core`**。
   —— 我改成 `core*kernels_per_core` 是错的。
3. **v7 真正的缺陷在别处**：`:729-731` 把 `output_channels % cores_used` 的余数
   分给前几个 core，于是 `Σ kernels_per_core == output_channels`（**没有幽灵槽**）；
   而 v7 是 `DIV_ROUND_UP(output_channels, cores_used)` **均分**，
   实测 `8*16 = 128 > 122` → 尾部 core 的 kernel 槽算出越界的 `out_channel`。

## 本补丁做什么（只有一个改动）

保留 v7 原式不动，仅在 `out_channel` 算出后加一条尾部钳位：

```c
if (output_channels > 0 && out_channel >= output_channels)
   out_channel = output_channels - 1;
```

为什么这样就够（不是「姑且一试」，是算出来的）：

- `cores_used * DIV_ROUND_UP(output_channels, cores_used)` 多出来的那些槽
  对应的通道**根本不存在**，其输出会被丢弃；
- 钳位后这些槽写入的都是 `output_channels-1` 这**同一个通道**的权重 ——
  写同一个地址、写同样的数据，**幂等且无害**；
- 因此通道 0..output_channels-1 拿到的权重与上游意图**逐字节一致**。

本地仿真（`verify/sim_coeffs_v7.py`）在真机实测参数下验证：
越界槽数 0、`wb_stream_write` 次数与原版相等、每 core 槽数仍 == `kernels_per_core`。

## 为什么锚点要程序化提取

1. 手敲模板缩进极易错（已踩过 15 vs 18 空格 → 匹配 0 处）。
2. 单行公式在文件里出现 4 次；锚点必须带函数内唯一上下文
   （`for z { for kernel { <公式行>` 三行）。
3. 结构性检查必须**限定函数体 + 精确计数** —— 全局子串搜会被同款缺陷的其它函数满足，
   只查存在性也抓不到「两处同款调用改一处」（两个反例曾被放行，已修）。
"""

import os
import sys

SRC = "/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
OUT = "/root/workspace/oes-teflon/build-recipe/patch-etnaviv-outchannel-clamp.py"

TARGET_FN = "write_core_interleaved(struct etna_ml_subgraph"
FORMULA_HEAD = "unsigned out_channel = core * kernels_in_superblock + kernel"
MARKER = "och patch: out_channel 尾部越界钳位"

# 目标函数体内必须精确匹配的结构特征（token, 期望出现次数）
STRUCT = (
    ("weights_map[out_channel][z][x][y]", 2),
    ("biases[out_channel]", 1),
    ("append_bits(kernels_per_core, 16", 1),
    ("calculate_bias_correction((uint8_t *)weights_map[out_channel]", 1),
)


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def extract_anchor(src: str):
    """在目标函数里定位公式行，向上取两行组成唯一锚点。"""
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        die("源码里找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）")
    nxt = src.find("\nstatic", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    fn_lines = fn_body.split("\n")

    hit = -1
    for i, l in enumerate(fn_lines):
        if FORMULA_HEAD in l:
            if hit >= 0:
                die("函数内公式出现多次，锚点失效")
            hit = i
    if hit < 0:
        die("函数 " + TARGET_FN + " 内找不到 out_channel 算式")

    ctx = fn_lines[hit - 2: hit + 1]
    if not any("for (unsigned z" in c for c in ctx[:2]):
        die("公式行上方不是 z/kernel 双循环，锚点假设不成立:\n" + "\n".join(ctx))
    if not any("for (unsigned kernel" in c for c in ctx[:2]):
        die("公式行上方不是 kernel 循环，锚点假设不成立:\n" + "\n".join(ctx))

    old = "\n".join(ctx) + "\n"
    n = src.count(old)
    if n != 1:
        die("锚点在整个文件出现 " + str(n) + " 处（期望 1）")
    return old


def build_new(old: str) -> str:
    """锚点末行（公式）后面追加钳位；公式本身逐字保留。"""
    lines = old.rstrip("\n").split("\n")
    head, formula = "\n".join(lines[:-1]), lines[-1]
    indent = formula[: len(formula) - len(formula.lstrip())]
    guard = (
        indent + "/* och patch: out_channel 尾部越界钳位 */\n"
        + indent + "/* 保留原式不动。原式结构与上游 v8 的 fill_weights() 一致\n"
        + indent + " * （etnaviv_ml_nn_v8.c:726 同样用 superblock*full_superblock*cores_used\n"
        + indent + " *   作为 superblock 步长，:754 按 kernels_in_superblock 逐 core 推进）——\n"
        + indent + " * 所以它是**有意**的，不是笔误，不要\"修正\"它。\n"
        + indent + " *\n"
        + indent + " * v7 真正的缺陷是 kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)\n"
        + indent + " * **均分**：实测 8*16 = 128 > 122，尾部 core 的 kernel 槽算出越界的 out_channel。\n"
        + indent + " * v8 已在 :729-731 用「余数分给前几个 core」修掉，槽位数恰为 output_channels。\n"
        + indent + " *\n"
        + indent + " * 这里只钳位：多出来的槽对应的通道根本不存在（其输出会被丢弃），\n"
        + indent + " * 钳位后它们写的是同一个通道 output_channels-1 的同样权重 —— 幂等且无害，\n"
        + indent + " * 于是通道 0..output_channels-1 的权重与上游意图逐字节一致。\n"
        + indent + " *\n"
        + indent + " * ⚠️ 不用 continue：append_bits(kernels_per_core, 16, ...) 已把每 core 槽位数\n"
        + indent + " * 写进位流头，跳过会让位流少槽而整体错位。 */\n"
        + indent + "#if defined(__GNUC__) || defined(__clang__)\n"
        + indent + "   /* 编译期标记：注释里的字符串编译后会消失（已实测），\n"
        + indent + "    * 只有 __attribute__((used)) 的对象才会留在 .rodata，\n"
        + indent + "    * 于是 `strings libteflon.so | grep och-patch-v7-outchannel-clamp`\n"
        + indent + "    * 才能真正判出「这个 .so 里有没有本补丁」。\n"
        + indent + "    * 反过来：若 strings **找不到**它，说明补丁没打进去，\n"
        + indent + "    * 构建流程与产物都不对 —— 必须当场失败，而不是交付。 */\n"
        + indent + "   __attribute__((used)) static const char och_ocfix_marker[] =\n"
        + indent + "      \"och-patch-v7-outchannel-clamp\";\n"
        + indent + "   (void)och_ocfix_marker;\n"
        + indent + "#endif\n"
        + indent + "if (output_channels > 0 && out_channel >= output_channels)\n"
        + indent + "   out_channel = output_channels - 1;"
    )
    return head + "\n" + formula + "\n" + guard + "\n"


TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：write_core_interleaved() 的 out_channel 尾部越界钳位。

## 只有一个改动

在原式算出的 `out_channel` 后面加一条尾部钳位；**原式逐字保留**。

```c
if (output_channels > 0 && out_channel >= output_channels)
   out_channel = output_channels - 1;
```

## 根因（真机数据证实）

1. 本地 `addr2line`（对着 unstripped `.so`）把 SIGSEGV 定位到
   `etnaviv_ml_nn_v7.c` 里 `wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);`。
2. 诊断补丁 `[och-wc]` 在真机打出 **136 个独立组合**，原式 **0 反例地复现全部实测 out_ch**，
   并确实出现 `out_ch >= output_channels`（实测 123 vs 122）→ `weights_map[out_channel]`
   与 `biases[out_channel]` 越界读 → SIGSEGV。
3. 必然性：`cores_used * kernels_per_core = 8*16 = 128 > 122`。

## 为什么不改公式（这是本补丁最容易被人「顺手优化掉」的地方）

上游 v8 的同源实现 `etnaviv_ml_nn_v8.c:715-760`（`fill_weights()`）证明原式结构是有意的：

- `:726` `channel_per_superblock[superblock] = superblock * full_superblock * cores_used;`
  —— **原样保留了末尾的 `* cores_used`**；
- `:754` `channel_per_superblock[superblock] += kernels_in_superblock;`
  —— 逐 core 推进 `kernels_in_superblock`，**不是 `kernels_per_core`**；
- `:729-731` v8 把 `output_channels % cores_used` 的余数分给前几个 core，
  于是槽位总数恰为 `output_channels`，**没有幽灵槽** —— 这才是 v7 缺的修复。

v7 用 `DIV_ROUND_UP(output_channels, cores_used)` 均分，才产生越界槽。

**所以：原式不动，只钳位。** 通道 0..output_channels-1 的权重与上游意图逐字节一致。

## ⚠️ 验收不靠「不崩」

必须过四级判据第 ④ 级（与 x86 基准逐字节一致，cos ≥ 0.99、top-1 相同）。
不崩只说明越界消失。

## 用法

    python3 patch-etnaviv-outchannel-clamp.py <etnaviv_ml_nn_v7.c 路径>

由 `build-teflon.sh` 的 `TEFLON_OCFIX=1` 启用，默认关。
"""

import argparse
import sys

MARKER = "och patch: out_channel 尾部越界钳位"
TARGET_FN = "write_core_interleaved(struct etna_ml_subgraph"

# (特征串, 期望出现次数) —— 期望值来自真实源码统计
STRUCT = __STRUCT__

# ↓↓↓ 逐字锚点（由 tools/gen-ocfix-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = __OLD__
NEW = __NEW__
FORMULA_LINE = __FORMULA_LINE__
INDENT = __INDENT__
# 原公式的完整字面量：本补丁**只允许追加**，它必须逐字保留
FORMULA_LITERAL = __FORMULA_LITERAL__
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处 out_channel 算式，"
            "期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对并重新生成补丁"
            "（python3 tools/gen-ocfix-patch.py），不要跳过校验\n"
            "       （会静默产出未修复的 .so，崩溃会照旧）。\n")
        return 1

    # 结构性检查：**限定目标函数体 + 精确计数**。
    # 全局子串搜会被同款缺陷的其它函数满足；只查存在性抓不到「多处同款改一处」
    # （两个反例曾因此被放行）。
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    nxt = src.find("\nstatic", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got != want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 " + str(got)
                + " 处，期望 " + str(want) + " 处。\n"
                "       Mesa 结构已变（或有人只改了一处同款调用）—— 钳位护栏依赖这些点，\n"
                "       请人工核对后重新生成补丁，不要跳过校验\n"
                "       （会静默产出护栏失效的 .so，越界崩溃会照旧）。\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    for token in (MARKER, "out_channel >= output_channels",
                  "out_channel = output_channels - 1"):
        if token not in src:
            sys.stderr.write("ERROR: 打完补丁后自检失败，缺少: " + token + "\n")
            return 1
    # 反向自检。
    # ⚠️ 本补丁**只追加不替换**，所以 NEW 以 OLD 为前缀 → `src.count(OLD) == 0`
    #    在数学上永远不成立（已实测踩到：正例被全部误拒）。
    #    改为断言唯一形态：**公式行后面紧接着就是护栏首行**。
    joined = FORMULA_LINE + "\n" + INDENT + "/* och patch: out_channel 尾部越界钳位 */"
    if src.count(joined) != 1:
        sys.stderr.write("ERROR: 打完补丁后「公式行紧跟护栏」出现 "
                         + str(src.count(joined)) + " 次（期望 1），替换形态不对\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 打完补丁后新代码出现次数为 " + str(src.count(NEW)) + "（期望 1）\n")
        return 1
    # 本补丁只允许**追加**：原公式必须仍在，且未被改写
    if FORMULA_LITERAL not in src:
        sys.stderr.write("ERROR: 原公式不见了 —— 本补丁**只允许加钳位**，"
                         "若要改映射请先看 v8 的 fill_weights() 再另立补丁\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-ocfix] 已加 out_channel 尾部越界钳位: " + args.path)
    print("  [och-ocfix] 原式保留未动（与上游 v8 fill_weights() 结构一致）")
    print("  [och-ocfix] ⚠️ 只算「越界消失」；能否用要看四级判据第 ④ 级（逐字节比对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

FORMULA_LITERAL = ("unsigned out_channel = core * kernels_in_superblock + kernel + "
                    "superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;")


def main():
    src = open(SRC, encoding="utf-8").read()
    old = extract_anchor(src)
    new = build_new(old)

    if new == old:
        die("生成的新代码与原文相同（替换未生效）")
    if MARKER not in new:
        die("生成的新代码缺少标记串")
    if FORMULA_LITERAL not in new:
        die("生成的新代码丢掉了原公式（本补丁只允许加钳位）")

    # 生成前先确认结构特征在真实源码里确实是期望次数（防止期望值本身写错）
    fn_at = src.find(TARGET_FN)
    nxt = src.find("\nstatic", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got != want:
            die("结构特征期望值与真实源码不符: `" + token + "` 实际 " + str(got)
                + " 处，期望 " + str(want) + " 处")

    formula_line = old.rstrip("\n").split("\n")[-1]
    indent = formula_line[: len(formula_line) - len(formula_line.lstrip())]
    body = (TEMPLATE
            .replace("__OLD__", repr(old))
            .replace("__NEW__", repr(new))
            .replace("__FORMULA_LINE__", repr(formula_line))
            .replace("__INDENT__", repr(indent))
            .replace("__FORMULA_LITERAL__", repr(FORMULA_LITERAL))
            .replace("__STRUCT__", repr(STRUCT)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(OUT, 0o755)

    print("已生成:", OUT)
    print("--- 锚点（OLD）逐字预览 ---")
    print(old)
    print("--- 替换后（NEW）逐字预览 ---")
    print(new)


if __name__ == "__main__":
    main()
