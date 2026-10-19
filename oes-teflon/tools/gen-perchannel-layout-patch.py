#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成/校验 build-recipe/patch-etnaviv-perchannel-layout.py。

## 这个补丁修什么

per-channel 路线在 A311D（**nn_core_version == 7**）上的下降路径存在**布局契约违反**：
`requantize_weights()` 与 `calc_quantization()` 都把权重缓冲区声明成 `[out][W][H][in]`，
但它们跑在 `transpose()`（`etnaviv_ml_nn.c:377-411`）**之后**，而 `transpose()` 的输出是
`[out][in][W][H]` —— 正是两个 write_core_* 期望的布局。

后果分两种：

| 情形 | 读侧 | 后果 |
|---|---|---|
| 非 depthwise | 尺寸不变，只是元素错位 | **静默算错**（不崩） |
| depthwise | `nn.c:239-240` 强制 in=1，只读 `out_c*W*H` 字节 | 上机实测：v7 未写回 in_c ⇒ `write_core_interleaved` 按 `oc*W*H*in` 取 ⇒ **SIGSEGV** |

`calc_quantization` 因为**先**跑、且只求 min/max，还有第二重问题：
depthwise 时它只看得到 **out_channel 0 那一组**（2196/535824 = 0.41%），
⇒ 折叠出来的 scale 是错的 ⇒ 只修 `requantize_weights` 仍然输出静默错误。

## 为什么必须一次改齐

    calc_quantization ──scale/zp──▶ requantize_weights ──缓冲区+布局──▶ write_core_*

| 只改 | depthwise | 非 depthwise |
|---|---|---|
| 只改 requantize_weights + 写回 | 不崩，但 scale 错 ⇒ 数值错 | 正确 |
| 只改 calc_quantization + 写回 | 仍读错 ⇒ 仍错 | 仍错 |
| 三个都改 | **正确** | **正确** |

## 为什么不拆成第二个开关

门禁放开与布局修复**都只可能在 per-channel 被准入时才走到**。
拆成两个开关只会允许「开一个不启另一个」的坏组合 —— 那必然是错的。
单一 `TEFLON_PERCHANNEL=1` = 单一语义 = 单一产物来源。

## 用法

    # 校验（只读；CI 与本地都跑）
    python3 tools/gen-perchannel-layout-patch.py --verify \\
        <etnaviv_ml_nn.c> <etnaviv_ml_nn_v7.c> <补丁器路径>

    # 重新生成（会覆盖补丁器）
    python3 tools/gen-perchannel-layout-patch.py --src <etnaviv_ml_nn.c> \\
        --src-v7 <etnaviv_ml_nn_v7.c> --out <补丁器路径>
"""

import argparse
import importlib.util
import os
import sys

DEFAULT_NN = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
              "src/gallium/drivers/etnaviv/etnaviv_ml_nn.c")
DEFAULT_V7 = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
              "src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c")
DEFAULT_OUT = ("/root/workspace/oes-teflon/build-recipe/"
               "patch-etnaviv-perchannel-layout.py")

NL = chr(10)

MARKER = "och patch: per-channel layout contract (transpose-aware)"
COMPILE_MARK = "och-patch-perchannel-layout"

# 结构性断言：限定在函数体内，命中数不对就 fail-closed。
STRUCT_CQ = (
    ("if (poperation->conv.depthwise)", 1),
    ("input_channels = 1;", 1),
    ("[operation->weight_width][operation->weight_height][input_channels] = map;", 1),
    ("calc_quant_params(min, max, out_scale, out_zero_point);", 1),
)
STRUCT_RW = (
    ("if (poperation->conv.depthwise)", 1),
    ("input_channels = 1;", 1),
    ("[operation->weight_width][operation->weight_height][input_channels] = input;", 1),
    ("[operation->weight_width][operation->weight_height][input_channels] = (void *)output;", 1),
    ("operation->weight_tensor_size = new_size;", 1),
)
# 写回那一行：按内容定位，不写死缩进（缩进变了也不该让锚点失配）
KEY_INC = "if (nn_core_version >= 8 && poperation->conv.depthwise)"
KEY_INC_NEXT = "operation->input_channels = 1;"


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def load(path):
    return open(path, encoding="utf-8").read().replace("\r\n", "\n")


def body_of(src, fn):
    at = src.find("\n" + fn + "(")
    if at < 0:
        die("源码里找不到函数 " + fn)
    at += 1
    i = src.index("{", at)
    d = 0
    j = i
    while j < len(src):
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return at, j + 1
        j += 1
    die("函数 " + fn + " 的花括号不配平（源码可能被截断）")


def whole_fn(src, fn, decl):
    """逐字取出 `<decl>\\n<fn>(...)\\n{ ... }\\n`，含返回类型行。

    ★ 必须连 decl 一起取：入库的锚点带 `static void`，若这里从函数名起抽，
      --verify 会拿「去头的真源码」比「带头的锚点」，永远报不一致。
      （这个坑在 gen-perchannel-quant-patch.py 上踩过一次。）
    """
    at, end = body_of(src, fn)
    line_start = src.rfind("\n", 0, at - 1) + 1
    head = src[line_start:at]
    if head.strip() != decl.strip():
        die("%s 的返回类型是 %r，期望 %r —— 源码结构可能已变" % (fn, head, decl))
    full = head + src[at:end] + "\n"
    n = src.count(full)
    if n != 1:
        die("%s 的整函数锚点在文件里出现 %d 处（期望 1）" % (fn, n))
    return full


def check_struct(body, name, struct):
    for tok, want in struct:
        got = body.count(tok)
        if got != want:
            die("%s 内 `%s` 实得 %d（期望 %d）" % (name, tok, got, want))


def block_ending_at(text, key, nlines, name, must_contain=()):
    """以含 key 的那一行**结尾**，往上取 nlines 行（整行，含前导缩进）。

    ★ v7 的选路 if/else 出现两次且**内容不同**（试算传 NULL、正式传 map），
      所以定位串取各自独有的那个 write_core_XX 调用，块本身往上对齐到 if 行。
      漏改一段会让「预测压缩尺寸」与「实际写入」走不同分支。
    """
    n = text.count(key)
    if n != 1:
        die("%s 的定位串 `%s` 出现 %d 处（期望 1）" % (name, key, n))
    line_no = text.count(NL, 0, text.index(key))   # key 所在行号（0 基）
    lines = text.split(NL)
    lo = line_no - nlines + 1
    if lo < 0:
        die("%s 往上取 %d 行会越界，源码结构可能已变" % (name, nlines))
    blk_lines = lines[lo:line_no + 1]
    if "if (operation->pointwise" not in blk_lines[0]:
        die("%s 的 %d 行块首行不是选路 if（实得 %r）"
            % (name, nlines, blk_lines[0][:70]))
    blk = NL.join(blk_lines) + NL
    for want in must_contain:
        if want not in blk:
            die("%s 取到的块里没有 %r（实得 %r）" % (name, want, blk[:120]))
    if text.count(blk) != 1:
        die("%s 的块在文件里出现 %d 处（期望 1）" % (name, text.count(blk)))
    return blk


def line_block(text, key, nlines, name, must_contain=(), want_n=1):
    """从含 key 的那一行起取 nlines 行（整行，含前导缩进）。

    ★ 不写死缩进：缩进风格变了也不该让锚点失配；但**内容**必须逐字对上，
      所以同时断言 key 出现 want_n 次、且取到的行含有 must_contain 里的每个片段。

    ★ want_n > 1 的情形真实存在：create_coeffs_v7 的选路 if/else 在
      「试算压缩尺寸」与「正式写入」两个循环里各出现一次，必须一起改，
      漏一处就会留下 size 预测与实际不符的隐患。
    """
    n = text.count(key)
    if n != want_n:
        die("%s 的定位串 `%s` 在文件里出现 %d 处（期望 %d）"
            % (name, key, n, want_n))
    at = text.index(key)
    start = text.rfind("\n", 0, at) + 1
    lines = text[start:].split("\n")
    if len(lines) < nlines:
        die("%s 不足 %d 行，源码结构可能已变" % (name, nlines))
    blk = "\n".join(lines[:nlines]) + "\n"
    for want in must_contain:
        if want not in blk:
            die("%s 取到的 %d 行里没有 %r —— 源码结构可能已变" % (name, nlines, want))
    got = text.count(blk)
    if got != want_n:
        die("%s 的 %d 行锚点在文件里出现 %d 处（期望 %d）"
            % (name, nlines, got, want_n))
    return blk


def extract(nn, v7):
    at_cq, end_cq = body_of(nn, "calc_quantization")
    at_rw, end_rw = body_of(nn, "requantize_weights")
    b_cq = nn[at_cq:end_cq]
    b_rw = nn[at_rw:end_rw]
    check_struct(b_cq, "calc_quantization", STRUCT_CQ)
    check_struct(b_rw, "requantize_weights", STRUCT_RW)

    old_cq = whole_fn(nn, "calc_quantization", "static void")
    old_rw = whole_fn(nn, "requantize_weights", "static void")

    old_inc = line_block(nn, KEY_INC, 2, "input_channels 写回",
                         must_contain=(KEY_INC_NEXT,))

    # ★ 试算循环传 NULL、正式循环传 map —— 两段内容不同，必须分别锚住；
    #   漏改一段会让「预测压缩尺寸」与「实际写入」走不同的分支。
    old_sel_probe = block_ending_at(
        v7, "write_core_sequential(subgraph, NULL", 6,
        "create_coeffs_v7 选路（试算）",
        must_contain=("if (operation->pointwise", "write_core_6(subgraph, NULL",
                      "write_core_interleaved(subgraph, NULL"))
    old_sel_real = block_ending_at(
        v7, "write_core_sequential(subgraph, map", 6,
        "create_coeffs_v7 选路（正式）",
        must_contain=("if (operation->pointwise", "write_core_6(subgraph, map",
                      "write_core_interleaved(subgraph, map"))
    return old_cq, old_rw, old_inc, old_sel_probe, old_sel_real


# --------------------------------------------------------------------------
# 以下 NEW_* 是本工具作者写的替换文本（不是从源码抄的）——
# 缩进沿用 Mesa 的 3 空格风格，由 tools/check-patches-vs-source.py 复核。
# --------------------------------------------------------------------------

NEW_CQ = '''static void
calc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                  struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point)
{
   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;
   void *map = operation->weight_tensor;
   /* och patch: per-channel layout contract (transpose-aware)
    *
    * 原实现把 map 声明成 [out][W][H][input_channels]，并且对 depthwise 强行把
    * input_channels 压成 1。它跑在 transpose() **之后**，而 transpose() 的输出是
    * [out][in][W][H] —— 正是两个 write_core_* 期望的布局：
    *    etnaviv_ml_nn_v7.c:226  interleaved: blocks = input + oc*W*H*in
    *    etnaviv_ml_nn_v7.c:278  interleaved: (*weights_map)[in][W][H]
    *    etnaviv_ml_nn_v7.c:391  sequential : base   = input + oc*W*H（无 in 维）
    * 所以按 [out][W][H][in] 读是**读错元素**。
    *
    * depthwise 时更糟：声明 in=1 让它只读到 out_c*W*H 字节（244*3*3*1 = 2196，
    * 占 535824 的 0.41%），也就是只看得到 output_channel 0 那一组 ⇒ scale 是错的。
    *
    * 正确读法（与 transpose 的输出一致）：
    *   非 depthwise：[out][in][W][H]，覆盖整个缓冲区；min/max 与排列无关
    *                 ⇒ 行为不变
    *   depthwise   ：只取对角线 ic == oc（out=c 只用 in=c 的 filter，
    *                 那是 expand_depthwise() 把它放到 ic==c 处的）
    */
   unsigned real_in = operation->input_channels;
   unsigned depthwise = poperation->conv.depthwise ? 1u : 0u;
   unsigned oc_max = depthwise ? MIN2(operation->output_channels, real_in)
                               : operation->output_channels;
   uint8_t (*weights)[real_in][operation->weight_width][operation->weight_height] = map;
#ifdef OCH_PCL_DIAG
   unsigned visited = 0;
#endif

   float max = .0, min = FLT_MAX;
   for (unsigned oc = 0; oc < oc_max; oc++) {
      for (unsigned w = 0; w < operation->weight_width; w++) {
         for (unsigned h = 0; h < operation->weight_height; h++) {
            for (unsigned ic = 0; ic < real_in; ic++) {
               float dequant;
               if (depthwise && ic != oc)
                  continue;
               if (operation->weight_signed) {
                  int8_t val = weights[oc][ic][w][h];
                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
               } else {
                  uint8_t val = weights[oc][ic][w][h];
                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
               }
               max = MAX2(max, dequant);
               min = MIN2(min, dequant);
#ifdef OCH_PCL_DIAG
               visited++;
#endif
            }
         }
      }
   }

   calc_quant_params(min, max, out_scale, out_zero_point);
#ifdef OCH_PCL_DIAG
   fprintf(stderr, "[och-pcl] CQ-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u "
                   "wt_size=%u visited=%u min=%f max=%f scale=%f zp=%u\\n",
           depthwise, real_in, operation->output_channels, oc_max,
           operation->weight_width, operation->weight_height,
           (unsigned)operation->weight_tensor_size, visited, min, max,
           *out_scale, (unsigned)*out_zero_point);
#endif
}
'''

NEW_RW = '''static void
requantize_weights(struct etna_ml_subgraph *subgraph,
                   const struct pipe_ml_operation *poperation,
                   struct etna_operation *operation)
{
   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;
   void *input = operation->weight_tensor;
   /* och patch: per-channel layout contract (transpose-aware)
    *
    * 读侧按 transpose() 的真实输出 [out][in][W][H]；
    * 写侧**保持同一布局**（非 depthwise），depthwise 例外折成 in=1 的紧凑
    * [out][W][H]，好让 create_coeffs_v7（input_channels > 1 ? interleaved : sequential）
    * 选到 sequential（基址 oc*W*H）并与之严格对应。
    *
    * 非 depthwise 时元素偏移与改前**逐个相同**（(oc*in+ic)*W*H + w*H + h），
    * 变的只有「从哪儿读」⇒ 尺寸不变、write_core_interleaved 取值正确。
    */
   unsigned real_in = operation->input_channels;
   unsigned depthwise = poperation->conv.depthwise ? 1u : 0u;
   unsigned out_dim = depthwise ? 1u : real_in;
   unsigned oc_max = depthwise ? MIN2(operation->output_channels, real_in)
                               : operation->output_channels;
   unsigned new_size = oc_max * operation->weight_width * operation->weight_height * out_dim;
   uint8_t *output = malloc(new_size);
   uint8_t (*map_in)[real_in][operation->weight_width][operation->weight_height] = input;
   uint8_t (*map_out)[out_dim][operation->weight_width][operation->weight_height] = (void *)output;

   for (unsigned oc = 0; oc < oc_max; oc++) {
      for (unsigned w = 0; w < operation->weight_width; w++) {
         for (unsigned h = 0; h < operation->weight_height; h++) {
            for (unsigned ic = 0; ic < out_dim; ic++) {
               unsigned src_ic = depthwise ? oc : ic;
               if (operation->weight_signed) {
                  int8_t val = map_in[oc][src_ic][w][h];
                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
                  int8_t requantized = round(dequantized / operation->weight_scale);
                  map_out[oc][ic][w][h] = requantized + operation->weight_zero_point - 0x80;
               } else {
                  uint8_t val = map_in[oc][src_ic][w][h];
                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
                  uint8_t requantized = round(dequantized / operation->weight_scale);
                  map_out[oc][ic][w][h] = requantized + operation->weight_zero_point;
               }
            }
         }
      }
   }

   free(operation->weight_tensor);
   operation->weight_tensor = output;
   operation->weight_tensor_size = new_size;
#ifdef OCH_PCL_DIAG
   fprintf(stderr, "[och-pcl] RW-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u "
                   "new_size=%u head=", depthwise, real_in, operation->output_channels,
           oc_max, operation->weight_width, operation->weight_height, new_size);
   for (unsigned i = 0; i < 16 && i < new_size; i++)
      fprintf(stderr, "%02x ", output[i]);
   fprintf(stderr, "\\n");
#endif
}
'''

NEW_INC_BODY = '''/* och patch: v7 也要写回。requantize_weights 已把 depthwise 权重折成
 * in=1 的紧凑布局；不写回的话 create_coeffs_v7 仍按 in_c=244 选
 * write_core_interleaved，按 oc*W*H*in 取 244 倍 => 越界读。
 * calculate_bias_correction(v7:114) 同样按 W*H*input_channels 取字节，
 * 所以这一行必须与上面的布局改动**同时**生效。 */
if (poperation->conv.depthwise)
operation->input_channels = 1;'''


def make_new_inc(old_inc):
    """按真源码锚点的实际缩进生成新代码（缩进只从源码来，不写死）。"""
    lines = old_inc.rstrip("\n").split("\n")
    if len(lines) != 2:
        die("input_channels 写回锚点不是 2 行")
    i1 = len(lines[0]) - len(lines[0].lstrip())
    i2 = len(lines[1]) - len(lines[1].lstrip())
    if i2 <= i1:
        die("写回两行的缩进关系异常（%d / %d）" % (i1, i2))
    out = []
    for bl in NEW_INC_BODY.split("\n"):
        out.append((" " * i1 + bl) if bl else "")
    return "\n".join(out) + "\n"


NEW_SELECT_DIAG = [
    "/* och patch: print-only 诊断 —— 证明 depthwise 修好后退到了 sequential */",
    "#ifdef OCH_PCL_DIAG",
    'fprintf(stderr, "[och-pcl] SEL dw=%u input_channels=%u output_channels=%u "',
    '                "wt_size=%u -> %s\\n",',
    "operation->depthwise ? 1u : 0u, input_channels, output_channels,",
    "(unsigned)operation->weight_tensor_size,",
    '(operation->pointwise && output_channels > 8) ? "write_core_6" :',
    '(input_channels > 1) ? "write_core_interleaved" : "write_core_sequential");',
    "#endif",
]


def make_new_select(old_sel):
    """功能部分**逐字沿用**真源码的 5 行 if/else，只在其前插入诊断打印。

    ★ 不重打那 5 行：它们是函数本体，重打就等于手抄锚点 —— 上次栽过。
    """
    lines = old_sel.rstrip("\n").split("\n")
    if len(lines) != 6:
        die("v7 选路锚点不是 6 行（实得 %d）" % len(lines))
    ind = len(lines[0]) - len(lines[0].lstrip())
    diag = [(" " * ind + d) if d else d for d in NEW_SELECT_DIAG]
    return "\n".join(diag + lines) + "\n"

TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能补丁：修复 per-channel 下降路径与 transpose() 的**布局契约违反**。

本文件由 tools/gen-perchannel-layout-patch.py 从真实 Mesa 源码生成，
OLD_* 锚点逐字来自真源码（勿手改；要改请改生成器后重新生成）。
同一次构建还会注入 --diag 的 print-only 诊断。

## 修什么

在 nn_core_version == 7（A311D）上，权重到达 calc_quantization /
requantize_weights 时已经被 transpose() 排成 [out][in][W][H]，
而这两个函数按 [out][W][H][in] 读 ⇒ 读错元素。

* 非 depthwise：尺寸不变 ⇒ **静默算错**（实测 37 个 CONV 用错权重）
* depthwise    ：nn.c:239-240 强制 in=1，v7 又不写回 input_channels
                 ⇒ create_coeffs_v7 按 in_c=244 选 write_core_interleaved
                 ⇒ 越界读 ⇒ **SIGSEGV**（实测 EXIT=139）
* calc_quantization 还有第二重：depthwise 时只覆盖 0.41% 的权重
  ⇒ 折叠出来的 scale 本身就是错的 ⇒ 只修 requantize_weights 仍会静默错

## 改三处

1. calc_quantization  : 按 [out][in][W][H] 遍历；depthwise 取对角线 ic == oc
2. requantize_weights : 按 [out][in][W][H] 读；depthwise 折成紧凑 [out][W][H]
3. input_channels 写回 : 去掉 nn_core_version >= 8 条件（v7 也要）

## 代价（不藏着）

per-channel 仍被**折叠成单一 scale**（upstream 原话 "with a loss of accuracy"）。
本补丁修的是「读错元素」，**不消除折叠本身的精度损失**。
⇒ 只解决崩溃与取错权重；cos / top-1（门 e）必须设备实测，不能由本补丁推断。
"""

import argparse
import sys

MARKER = {marker!r}
COMPILE_MARK = {compile_mark!r}
NL = chr(10)

OLD_CALC_Q = {old_cq!r}
OLD_REQ_W = {old_rw!r}
OLD_INC = {old_inc!r}

NEW_CALC_Q = {new_cq!r}
NEW_REQ_W = {new_rw!r}
NEW_INC = {new_inc!r}

OLD_SEL_PROBE = {old_sel_probe!r}
OLD_SEL_REAL = {old_sel_real!r}

NEW_SEL_PROBE = {new_sel_probe!r}
NEW_SEL_REAL = {new_sel_real!r}

# ★ diag=1 时在两个 .c 顶部插入的前奏：定义 OCH_PCL_DIAG 并带上 stdio。
#   前奏里含 MARKER，顺带承担「重复应用」的拒绝锚点。
PROLOGUE = ("#include <stdio.h>" + NL +
            "#define OCH_PCL_DIAG 1" + NL +
            "/* " + MARKER + " */" + NL)


def _replace(text, tag, old, new, want=1):
    """逐字替换，命中数必须**恰好**等于 want（Mesa 结构变了就 fail-closed）。"""
    n = text.count(old)
    if n != want:
        sys.stderr.write("ERROR: %s 锚点匹配到 %d 处，期望恰好 %d 处（Mesa 结构可能已变）。" % (tag, n, want))
        return None
    if old == new:
        sys.stderr.write("ERROR: %s 替换未生效" % tag)
        return None
    return text.replace(old, new)


def _strip_diag_blocks(text):
    """摘掉全部 `#ifdef OCH_PCL_DIAG ... #endif` 块（含其中的打印）。

    diag=0 时必须整段删除而不是留成死代码：否则 --diag 0 与
    「打一个没有诊断的补丁」产出的二进制无法逐字比对。
    """
    out, depth = [], 0
    for line in text.split(NL):
        if depth == 0 and line.strip() == "#ifdef OCH_PCL_DIAG":
            depth = 1
            continue
        if depth:
            if line.strip() == "#endif":
                depth = 0
            continue
        out.append(line)
    if depth:
        sys.stderr.write("ERROR: 去诊断时遇到未闭合的 #ifdef OCH_PCL_DIAG")
        return None
    return NL.join(out)


def _strip_prologue(text):
    """diag=0 时把前奏摘掉，得到「无诊断」的干净补丁。"""
    if text.startswith(PROLOGUE):
        return text[len(PROLOGUE):]
    return text


def _check_nn(nn):
    """补丁后的关键不变量；任何一条不满足都说明修法没落到位。"""
    must = (
        ("uint8_t (*weights)[real_in]", 1, "calc_quantization 读侧未改成 real_in 布局"),
        ("uint8_t (*map_in)[real_in]", 1, "requantize_weights 读侧未改成 real_in 布局"),
        ("uint8_t (*map_out)[out_dim]", 1, "requantize_weights 写侧未改成 out_dim 布局"),
        ("unsigned src_ic = depthwise ? oc : ic;", 1, "缺 depthwise 对角线取权重"),
        ("unsigned real_in = operation->input_channels;", 2, "两处 real_in 定义不对"),
    )
    for tok, want, msg in must:
        got = nn.count(tok)
        if got != want:
            sys.stderr.write("ERROR: %s（实得 %d，期望 %d）" % (msg, got, want))
            return False
    if "nn_core_version >= 8 && poperation->conv.depthwise" in nn:
        sys.stderr.write("ERROR: 写回条件仍带 nn_core_version >= 8 —— v7 会漏写")
        return False
    if "calc_quant_params(min, max, out_scale, out_zero_point);" not in nn:
        sys.stderr.write("ERROR: calc_quant_params 调用丢了")
        return False
    if "operation->weight_tensor_size = new_size;" not in nn:
        sys.stderr.write("ERROR: weight_tensor_size 回写丢了")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nn_path")
    ap.add_argument("--v7", required=True, help="etnaviv_ml_nn_v7.c 路径")
    ap.add_argument("--diag", default="1", choices=["0", "1"],
                    help="1=启用 print-only 诊断（默认）；0=产出无诊断候选")
    args = ap.parse_args()
    diag = args.diag == "1"

    with open(args.nn_path, "r", encoding="utf-8") as f:
        nn = f.read()
    with open(args.v7, "r", encoding="utf-8") as f:
        v7 = f.read()

    if MARKER in nn or MARKER in v7:
        sys.stderr.write("ERROR: 已包含本补丁，拒绝重复应用。" + NL)
        return 1

    nn = _replace(nn, "calc_quantization", OLD_CALC_Q, NEW_CALC_Q)
    if nn is None:
        return 1
    nn = _replace(nn, "requantize_weights", OLD_REQ_W, NEW_REQ_W)
    if nn is None:
        return 1
    nn = _replace(nn, "input_channels 写回", OLD_INC, NEW_INC)
    if nn is None:
        return 1
    v7 = _replace(v7, "create_coeffs_v7 选路（试算）", OLD_SEL_PROBE, NEW_SEL_PROBE)
    if v7 is None:
        return 1
    v7 = _replace(v7, "create_coeffs_v7 选路（正式）", OLD_SEL_REAL, NEW_SEL_REAL)
    if v7 is None:
        return 1

    if not _check_nn(nn):
        return 1

    # ★ 两个文件都必须写回 —— 只写 nn 会让 v7 里的诊断与注释变成空转。
    if diag:
        nn = PROLOGUE + nn
        v7 = PROLOGUE + v7
    else:
        nn = _strip_diag_blocks(nn)
        v7 = _strip_diag_blocks(v7)
        if nn is None or v7 is None:
            return 1
        nn = _strip_prologue(nn)
        v7 = _strip_prologue(v7)
        for name, t in (("nn", nn), ("v7", v7)):
            if "#ifdef OCH_PCL_DIAG" in t:
                sys.stderr.write("ERROR: --diag 0 仍在 %s.c 里留了诊断块" % name)
                return 1
            if PROLOGUE in t:
                sys.stderr.write("ERROR: --diag 0 仍在 %s.c 里留了前奏" % name)
                return 1

    with open(args.nn_path, "w", encoding="utf-8") as f:
        f.write(nn)
    with open(args.v7, "w", encoding="utf-8") as f:
        f.write(v7)

    print("  [perchannel-layout] calc_quantization : 按 [out][in][W][H] 遍历，depthwise 取对角线")
    print("  [perchannel-layout] requantize_weights: 按 [out][in][W][H] 读，depthwise 折成紧凑")
    print("  [perchannel-layout] input_channels 写回: 去掉 nn_core_version >= 8 条件（v7 也写）")
    print("  [perchannel-layout] diag=%s（%s）" % ("1" if diag else "0",
          "已注入 print-only 诊断" if diag else "无诊断候选"))
    print("  [perchannel-layout] *仍折叠成单一 scale，精度损失未被本补丁消除*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _strip_diag(text):
    """去掉所有 #ifdef OCH_PCL_DIAG ... #endif 块（用于 --diag 0）。"""
    out = []
    depth = 0
    for line in text.split("\n"):
        if depth == 0 and line.strip() == "#ifdef OCH_PCL_DIAG":
            depth = 1
            continue
        if depth:
            if line.strip() == "#endif":
                depth = 0
            continue
        out.append(line)
    if depth:
        die("--diag 0 去诊断时遇到未闭合的 #ifdef OCH_PCL_DIAG")
    return "\n".join(out)


def build(nn_path, v7_path, out_path):
    nn = load(nn_path)
    v7 = load(v7_path)
    old_cq, old_rw, old_inc, old_sel_probe, old_sel_real = extract(nn, v7)
    text = TEMPLATE.format(marker=MARKER, compile_mark=COMPILE_MARK,
                           old_cq=old_cq, old_rw=old_rw, old_inc=old_inc,
                           new_cq=NEW_CQ, new_rw=NEW_RW,
                           new_inc=make_new_inc(old_inc),
                           old_sel_probe=old_sel_probe,
                           old_sel_real=old_sel_real,
                           new_sel_probe=make_new_select(old_sel_probe),
                           new_sel_real=make_new_select(old_sel_real))
    with open(out_path, "w", encoding="utf-8", newline=NL) as f:
        f.write(text)
    print("已生成 %s" % out_path)
    print("  OLD_CALC_Q   %d 字符" % len(old_cq))
    print("  OLD_REQ_W    %d 字符" % len(old_rw))
    print("  OLD_INC      %d 字符" % len(old_inc))
    print("  OLD_SEL_PROBE %d 字符" % len(old_sel_probe))
    print("  OLD_SEL_REAL  %d 字符" % len(old_sel_real))


def verify(nn_path, v7_path, out_path):
    nn = load(nn_path)
    v7 = load(v7_path)
    old_cq, old_rw, old_inc, old_sel_probe, old_sel_real = extract(nn, v7)
    spec = importlib.util.spec_from_file_location("pcl_committed", out_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    ok = True
    for name, real, committed in (("OLD_CALC_Q", old_cq, P.OLD_CALC_Q),
                                  ("OLD_REQ_W", old_rw, P.OLD_REQ_W),
                                  ("OLD_INC", old_inc, P.OLD_INC),
                                  ("OLD_SEL_PROBE", old_sel_probe, P.OLD_SEL_PROBE),
                                  ("OLD_SEL_REAL", old_sel_real, P.OLD_SEL_REAL)):
        if real == committed:
            print("  %-12s 与真实源码逐字一致 [OK]  (%d 字符)" % (name, len(real)))
        else:
            ok = False
            print("  %-12s 不一致" % name)
            print("    真实源码 : %r" % real[:200])
            print("    已入库   : %r" % committed[:200])
    print()
    print("VERIFY: PASS —— 锚点与真实源码完全一致" if ok
          else "VERIFY: FAIL —— 请用 --src/--out 重新生成")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_NN)
    ap.add_argument("--src-v7", default=DEFAULT_V7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        return verify(args.src, args.src_v7, args.out)
    build(args.src, args.src_v7, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())