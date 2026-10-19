#!/usr/bin/env python3
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

MARKER = 'och patch: per-channel layout contract (transpose-aware)'
COMPILE_MARK = 'och-patch-perchannel-layout'
NL = chr(10)

OLD_CALC_Q = 'static void\ncalc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,\n                  struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point)\n{\n   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;\n   void *map = operation->weight_tensor;\n   unsigned input_channels = operation->input_channels;\n\n   if (poperation->conv.depthwise)\n      input_channels = 1;\n\n   uint8_t (*weights)[operation->weight_width][operation->weight_height][input_channels] = map;\n\n   float max = .0, min = FLT_MAX;\n   for (unsigned oc = 0; oc < operation->output_channels; oc++) {\n      for (unsigned w = 0; w < operation->weight_width; w++) {\n         for (unsigned h = 0; h < operation->weight_height; h++) {\n            for (unsigned ic = 0; ic < input_channels; ic++) {\n               float dequant;\n               if (operation->weight_signed) {\n                  int8_t val = weights[oc][w][h][ic];\n                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n               } else {\n                  uint8_t val = weights[oc][w][h][ic];\n                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n               }\n               max = MAX2(max, dequant);\n               min = MIN2(min, dequant);\n            }\n         }\n      }\n   }\n\n   calc_quant_params(min, max, out_scale, out_zero_point);\n}\n'
OLD_REQ_W = 'static void\nrequantize_weights(struct etna_ml_subgraph *subgraph,\n                   const struct pipe_ml_operation *poperation,\n                   struct etna_operation *operation)\n{\n   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;\n   void *input = operation->weight_tensor;\n   unsigned input_channels = operation->input_channels;\n\n   if (poperation->conv.depthwise)\n      input_channels = 1;\n\n   unsigned new_size = operation->output_channels * operation->weight_width * operation->weight_height * input_channels;\n   uint8_t *output = malloc(new_size);\n   uint8_t (*map_in)[operation->weight_width][operation->weight_height][input_channels] = input;\n   uint8_t (*map_out)[operation->weight_width][operation->weight_height][input_channels] = (void *)output;\n\n   for (unsigned oc = 0; oc < operation->output_channels; oc++) {\n      for (unsigned w = 0; w < operation->weight_width; w++) {\n         for (unsigned h = 0; h < operation->weight_height; h++) {\n            for (unsigned ic = 0; ic < input_channels; ic++) {\n               if (operation->weight_signed) {\n                  int8_t val = map_in[oc][w][h][ic];\n                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n                  int8_t requantized = round(dequantized / operation->weight_scale);\n                  map_out[oc][w][h][ic] = requantized + operation->weight_zero_point - 0x80;\n               } else {\n                  uint8_t val = map_in[oc][w][h][ic];\n                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n                  uint8_t requantized = round(dequantized / operation->weight_scale);\n                  map_out[oc][w][h][ic] = requantized + operation->weight_zero_point;\n               }\n            }\n         }\n      }\n   }\n\n   free(operation->weight_tensor);\n   operation->weight_tensor = output;\n   operation->weight_tensor_size = new_size;\n}\n'
OLD_INC = '      if (nn_core_version >= 8 && poperation->conv.depthwise)\n         operation->input_channels = 1;\n'

NEW_CALC_Q = 'static void\ncalc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,\n                  struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point)\n{\n   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;\n   void *map = operation->weight_tensor;\n   /* och patch: per-channel layout contract (transpose-aware)\n    *\n    * 原实现把 map 声明成 [out][W][H][input_channels]，并且对 depthwise 强行把\n    * input_channels 压成 1。它跑在 transpose() **之后**，而 transpose() 的输出是\n    * [out][in][W][H] —— 正是两个 write_core_* 期望的布局：\n    *    etnaviv_ml_nn_v7.c:226  interleaved: blocks = input + oc*W*H*in\n    *    etnaviv_ml_nn_v7.c:278  interleaved: (*weights_map)[in][W][H]\n    *    etnaviv_ml_nn_v7.c:391  sequential : base   = input + oc*W*H（无 in 维）\n    * 所以按 [out][W][H][in] 读是**读错元素**。\n    *\n    * depthwise 时更糟：声明 in=1 让它只读到 out_c*W*H 字节（244*3*3*1 = 2196，\n    * 占 535824 的 0.41%），也就是只看得到 output_channel 0 那一组 ⇒ scale 是错的。\n    *\n    * 正确读法（与 transpose 的输出一致）：\n    *   非 depthwise：[out][in][W][H]，覆盖整个缓冲区；min/max 与排列无关\n    *                 ⇒ 行为不变\n    *   depthwise   ：只取对角线 ic == oc（out=c 只用 in=c 的 filter，\n    *                 那是 expand_depthwise() 把它放到 ic==c 处的）\n    */\n   unsigned real_in = operation->input_channels;\n   unsigned depthwise = poperation->conv.depthwise ? 1u : 0u;\n   unsigned oc_max = depthwise ? MIN2(operation->output_channels, real_in)\n                               : operation->output_channels;\n   uint8_t (*weights)[real_in][operation->weight_width][operation->weight_height] = map;\n#ifdef OCH_PCL_DIAG\n   unsigned visited = 0;\n#endif\n\n   float max = .0, min = FLT_MAX;\n   for (unsigned oc = 0; oc < oc_max; oc++) {\n      for (unsigned w = 0; w < operation->weight_width; w++) {\n         for (unsigned h = 0; h < operation->weight_height; h++) {\n            for (unsigned ic = 0; ic < real_in; ic++) {\n               float dequant;\n               if (depthwise && ic != oc)\n                  continue;\n               if (operation->weight_signed) {\n                  int8_t val = weights[oc][ic][w][h];\n                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n               } else {\n                  uint8_t val = weights[oc][ic][w][h];\n                  dequant = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n               }\n               max = MAX2(max, dequant);\n               min = MIN2(min, dequant);\n#ifdef OCH_PCL_DIAG\n               visited++;\n#endif\n            }\n         }\n      }\n   }\n\n   calc_quant_params(min, max, out_scale, out_zero_point);\n#ifdef OCH_PCL_DIAG\n   fprintf(stderr, "[och-pcl] CQ-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u "\n                   "wt_size=%u visited=%u min=%f max=%f scale=%f zp=%u\\n",\n           depthwise, real_in, operation->output_channels, oc_max,\n           operation->weight_width, operation->weight_height,\n           (unsigned)operation->weight_tensor_size, visited, min, max,\n           *out_scale, (unsigned)*out_zero_point);\n#endif\n}\n'
NEW_REQ_W = 'static void\nrequantize_weights(struct etna_ml_subgraph *subgraph,\n                   const struct pipe_ml_operation *poperation,\n                   struct etna_operation *operation)\n{\n   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;\n   void *input = operation->weight_tensor;\n   /* och patch: per-channel layout contract (transpose-aware)\n    *\n    * 读侧按 transpose() 的真实输出 [out][in][W][H]；\n    * 写侧**保持同一布局**（非 depthwise），depthwise 例外折成 in=1 的紧凑\n    * [out][W][H]，好让 create_coeffs_v7（input_channels > 1 ? interleaved : sequential）\n    * 选到 sequential（基址 oc*W*H）并与之严格对应。\n    *\n    * 非 depthwise 时元素偏移与改前**逐个相同**（(oc*in+ic)*W*H + w*H + h），\n    * 变的只有「从哪儿读」⇒ 尺寸不变、write_core_interleaved 取值正确。\n    */\n   unsigned real_in = operation->input_channels;\n   unsigned depthwise = poperation->conv.depthwise ? 1u : 0u;\n   unsigned out_dim = depthwise ? 1u : real_in;\n   unsigned oc_max = depthwise ? MIN2(operation->output_channels, real_in)\n                               : operation->output_channels;\n   unsigned new_size = oc_max * operation->weight_width * operation->weight_height * out_dim;\n   uint8_t *output = malloc(new_size);\n   uint8_t (*map_in)[real_in][operation->weight_width][operation->weight_height] = input;\n   uint8_t (*map_out)[out_dim][operation->weight_width][operation->weight_height] = (void *)output;\n\n   for (unsigned oc = 0; oc < oc_max; oc++) {\n      for (unsigned w = 0; w < operation->weight_width; w++) {\n         for (unsigned h = 0; h < operation->weight_height; h++) {\n            for (unsigned ic = 0; ic < out_dim; ic++) {\n               unsigned src_ic = depthwise ? oc : ic;\n               if (operation->weight_signed) {\n                  int8_t val = map_in[oc][src_ic][w][h];\n                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n                  int8_t requantized = round(dequantized / operation->weight_scale);\n                  map_out[oc][ic][w][h] = requantized + operation->weight_zero_point - 0x80;\n               } else {\n                  uint8_t val = map_in[oc][src_ic][w][h];\n                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);\n                  uint8_t requantized = round(dequantized / operation->weight_scale);\n                  map_out[oc][ic][w][h] = requantized + operation->weight_zero_point;\n               }\n            }\n         }\n      }\n   }\n\n   free(operation->weight_tensor);\n   operation->weight_tensor = output;\n   operation->weight_tensor_size = new_size;\n#ifdef OCH_PCL_DIAG\n   fprintf(stderr, "[och-pcl] RW-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u "\n                   "new_size=%u head=", depthwise, real_in, operation->output_channels,\n           oc_max, operation->weight_width, operation->weight_height, new_size);\n   for (unsigned i = 0; i < 16 && i < new_size; i++)\n      fprintf(stderr, "%02x ", output[i]);\n   fprintf(stderr, "\\n");\n#endif\n}\n'
NEW_INC = '      /* och patch: v7 也要写回。requantize_weights 已把 depthwise 权重折成\n       * in=1 的紧凑布局；不写回的话 create_coeffs_v7 仍按 in_c=244 选\n       * write_core_interleaved，按 oc*W*H*in 取 244 倍 => 越界读。\n       * calculate_bias_correction(v7:114) 同样按 W*H*input_channels 取字节，\n       * 所以这一行必须与上面的布局改动**同时**生效。 */\n      if (poperation->conv.depthwise)\n      operation->input_channels = 1;\n'

OLD_SEL_PROBE = '         if (operation->pointwise && output_channels > 8)\n            actual_size = write_core_6(subgraph, NULL, core, operation, zrl_bits);\n         else if (input_channels > 1)\n            actual_size = write_core_interleaved(subgraph, NULL, core, operation, zrl_bits);\n         else\n            actual_size = write_core_sequential(subgraph, NULL, core, operation, zrl_bits);\n'
OLD_SEL_REAL = '      if (operation->pointwise && output_channels > 8)\n         actual_size = write_core_6(subgraph, map, core, operation, zrl_bits);\n      else if (input_channels > 1)\n         actual_size = write_core_interleaved(subgraph, map, core, operation, zrl_bits);\n      else\n         actual_size = write_core_sequential(subgraph, map, core, operation, zrl_bits);\n'

NEW_SEL_PROBE = '         /* och patch: print-only 诊断 —— 证明 depthwise 修好后退到了 sequential */\n         #ifdef OCH_PCL_DIAG\n         fprintf(stderr, "[och-pcl] SEL dw=%u input_channels=%u output_channels=%u "\n                         "wt_size=%u -> %s\\n",\n         operation->depthwise ? 1u : 0u, input_channels, output_channels,\n         (unsigned)operation->weight_tensor_size,\n         (operation->pointwise && output_channels > 8) ? "write_core_6" :\n         (input_channels > 1) ? "write_core_interleaved" : "write_core_sequential");\n         #endif\n         if (operation->pointwise && output_channels > 8)\n            actual_size = write_core_6(subgraph, NULL, core, operation, zrl_bits);\n         else if (input_channels > 1)\n            actual_size = write_core_interleaved(subgraph, NULL, core, operation, zrl_bits);\n         else\n            actual_size = write_core_sequential(subgraph, NULL, core, operation, zrl_bits);\n'
NEW_SEL_REAL = '      /* och patch: print-only 诊断 —— 证明 depthwise 修好后退到了 sequential */\n      #ifdef OCH_PCL_DIAG\n      fprintf(stderr, "[och-pcl] SEL dw=%u input_channels=%u output_channels=%u "\n                      "wt_size=%u -> %s\\n",\n      operation->depthwise ? 1u : 0u, input_channels, output_channels,\n      (unsigned)operation->weight_tensor_size,\n      (operation->pointwise && output_channels > 8) ? "write_core_6" :\n      (input_channels > 1) ? "write_core_interleaved" : "write_core_sequential");\n      #endif\n      if (operation->pointwise && output_channels > 8)\n         actual_size = write_core_6(subgraph, map, core, operation, zrl_bits);\n      else if (input_channels > 1)\n         actual_size = write_core_interleaved(subgraph, map, core, operation, zrl_bits);\n      else\n         actual_size = write_core_sequential(subgraph, map, core, operation, zrl_bits);\n'

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
