#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能补丁：把 per-channel 折叠**提前**到所有布局变换之前，且改为原地折叠。

对应方案：`verify/reports/55-proposal-fold-before-expand.md`（方案 A，已过 OCA 评审）。

## 修什么（两个缺陷，必须一起修）

P1（内存安全，报告 54 已实测）
    upstream `requantize_weights()` 对 depthwise 强制 `input_channels = 1`，
    `new_size = oc·W·H·1` ⇒ 把展开后的 `C·k²·C` 缓冲区缩成 `C·k²` 并 free 掉原来的；
    而 `operation->input_channels` 仍是 C ⇒ 写入器（`create_coeffs_v7`）按 `C·k²·C` 读
    ⇒ **越界 C 倍**（24/122/244/488 四档实测，偏移恰在缓冲区末尾）。

P2（数值语义）
    `expand_depthwise()` / `pointwise_to_2x2()` 用 `operation->weight_zero_point` 填背景，
    而折叠排在它们**之后** ⇒ 同一张量里「对角=新域、背景=旧域」混域；
    更糟的是 depthwise 压成 in_c=1 后，硬件描述符让 C 个输出**共享同一输入通道**，
    而 v7 没有 depthwise 模式（`expand_depthwise` 注释：hardware doesn't support those）。

## 怎么修（净变化 3 条）

1. 折叠块整体**前移**到 `pointwise_to_2x2()` / `expand_depthwise()` / `strided_to_normal()` /
   `transpose()` 之前（紧跟 bias memcpy 之后）
2. 折叠改为**原地**：读写下标相同、无 malloc/free、**不改 `weight_tensor_size`**
   ⇒ 「缩容后仍按 C 读」这一类越界**结构性消失**
3. 删掉 depthwise 的 `input_channels = 1` 写回（v8 分支的收口保留）

元素→输出通道的映射只依赖「oc 是最外层维度」这一条：
    depthwise 原始布局 `[1][kh][kw][C]` ⇒ `i % C`
    普通卷积  原始布局 `[oc][kh][kw][ic]` ⇒ `i / (kh·kw·ic)`
逐元素运算与块内顺序无关 ⇒ **非 depthwise 的结果与上游逐字节相同**（见 §5 I2）。

## 定位方式（为什么不用逐字大段锚点）

按**函数名**定位 + 括号配对 + **body md5 钉死**。上游一改（哪怕空格）就 fail-closed，
且不会像逐字锚点那样在 Mesa 升级时留下「锚点还命中、但语义已变」的假绿。

## 时序纪律（必须写进注释，防后人"优化"）

`calculate_bias_correction()` 在 `etnaviv_ml_nn_v7.c` 的**写系数**路径里
（`v7.c:236/317/405`）读的是**折叠后**的权重，减的是**新域** `weight_zero_point`。
⇒ 折叠必须先于它，且**不得**提前预计算 correction（那会变成「旧域权重 − 新域 zp」）。
（评审曾以 R9 提出此点，经逐字核对为误判并撤回；结论保留在此以防复发。）

## 与其它补丁的关系

⚠️ 与 `patch-etnaviv-perchannel-layout.py` **互斥**：两者都整函数重写
`calc_quantization` / `requantize_weights`，且对 depthwise 的 `input_channels` 处置相反。
`build-teflon.sh` 必须 fail-closed（同时打开就报错），产物用二进制标记区分：
    foldearly 版 ⇒ 含 `och-patch-perchannel-foldearly`，**不含** `och-patch-perchannel-layout`
"""

import argparse
import hashlib
import re
import sys

MARKER = "och-patch-perchannel-foldearly"

# ── 上游原始函数体的 md5（未打任何补丁；上游一改就 fail-closed）────────────────
MD5_CALC_Q = "660a01edd6e26f88e128821065aec127"
MD5_REQ_W = "b648d88873bda693860424212471cdf3"
MD5_BLOCK = "712b3736c443a9a31796fd92d611f44f"

ANCHOR_INSERT = "   operation->bias_tensor_size = bias_size;\n"

# ── 新实现 ────────────────────────────────────────────────────────────────────

NEW_HELPER = r'''
/* ═══ och patch: per-channel fold-early ═══════════════════════════════════════
 * 元素 i 属于哪个**输出通道**。只依赖「oc 是最外层维度」这一事实：
 *   depthwise 原始布局 [1][kh][kw][C]   ⇒ 通道在内层，c = i % C
 *   普通卷积  原始布局 [oc][kh][kw][ic] ⇒ c = i / (kh·kw·ic)
 * 两种布局的 oc 分块大小都是 kh·kw·ic ⇒ 与块内顺序无关（逐元素运算）。
 * ══════════════════════════════════════════════════════════════════════════ */
static unsigned
och_fold_channel(const struct etna_operation *operation, unsigned i, bool depthwise)
{
   if (depthwise)
      return i % operation->output_channels;

   return i / (operation->weight_width * operation->weight_height * operation->input_channels);
}
'''

NEW_CALC_Q_BODY = r'''{
   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;
   const uint8_t *weights = operation->weight_tensor;
   const bool depthwise = poperation->conv.depthwise;
   const unsigned n = operation->weight_tensor_size;

   /* och patch: fold-early —— 逐元素、按元素所属通道取 scale/zp，不假设任何布局。
    * max 的初值 .0 是上游 quirk（把 max 下限钳到 0），**故意保留**：
    * 改它会同时改变非 depthwise 的结果，违反「与上游逐字节相同」这条回归控制。 */
   float max = .0, min = FLT_MAX;
   for (unsigned i = 0; i < n; i++) {
      unsigned c = och_fold_channel(operation, i, depthwise);
      float dequant;
      if (operation->weight_signed) {
         int8_t val = ((const int8_t *)weights)[i];
         dequant = weight_tensor->scales[c] * (val - weight_tensor->zero_points[c]);
      } else {
         uint8_t val = weights[i];
         dequant = weight_tensor->scales[c] * (val - weight_tensor->zero_points[c]);
      }
      max = MAX2(max, dequant);
      min = MIN2(min, dequant);
   }

   calc_quant_params(min, max, out_scale, out_zero_point);
}'''

NEW_REQ_W_BODY = r'''{
   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;
   uint8_t *weights = operation->weight_tensor;
   const bool depthwise = poperation->conv.depthwise;
   const unsigned n = operation->weight_tensor_size;

   /* och patch: fold-early —— **原地**折叠：读写下标相同 ⇒ 尺寸恒定。
    *
    * ★ 不得改回 malloc/free：上游就是因为 new_size 用 input_channels=1 缩容，
    *   而 operation->input_channels 仍是 C，才让写入器越界 C 倍（报告 54）。
    * ★ 时序纪律：calculate_bias_correction() 在 v7.c:236/317/405 的写系数路径里
    *   读的是**折叠后**的权重（减新域 weight_zero_point）⇒ 折叠必须先于它，
    *   且**不得**提前预计算 correction（会变成「旧域权重 − 新域 zp」）。 */
   for (unsigned i = 0; i < n; i++) {
      unsigned c = och_fold_channel(operation, i, depthwise);
      if (operation->weight_signed) {
         int8_t val = ((int8_t *)weights)[i];
         double dequantized = weight_tensor->scales[c] * (val - weight_tensor->zero_points[c]);
         int8_t requantized = round(dequantized / operation->weight_scale);
         weights[i] = requantized + operation->weight_zero_point - 0x80;
      } else {
         uint8_t val = weights[i];
         double dequantized = weight_tensor->scales[c] * (val - weight_tensor->zero_points[c]);
         uint8_t requantized = round(dequantized / operation->weight_scale);
         weights[i] = requantized + operation->weight_zero_point;
      }
   }
}'''

NEW_EARLY_BLOCK = r'''
   /* ═══ och patch: per-channel 折叠提前（fold-early）══════════════════════════
    * 位置是**语义要求**，不是风格：下面四个变换都会
    *   ① 用 operation->weight_zero_point 填背景（pointwise_to_2x2 / expand_depthwise /
    *      strided_to_normal）⇒ 折叠必须先更新 zp，否则同一张量里混域；
    *   ② 按 weight_tensor_size 重新 malloc（expand / transpose / strided_to_normal）
    *      ⇒ 折叠若排在后面，缓冲区尺寸就被折解决定，而写入器用的是 input_channels。
    * 折叠在此完成后：缓冲尺寸恒为「原始 TFLite 尺寸」的派生量，input_channels 全程 = C，
    * v7 写入器走 write_core_interleaved，与 expand_depthwise 的稠密 C×C 布局自洽。
    * ══════════════════════════════════════════════════════════════════════════ */
   if (poperation->conv.weight_tensor->scales) {
      float scale;
      uint8_t zero_point;

      assert(poperation->conv.weight_tensor->zero_points);

      ML_DBG("\nWARNING: The weights in this model are quantized using a per-channel scheme.\n"
             "The hardware doesn't support it natively, so it will be workarounded by requantizing\n"
             "on the fly, with a loss of accuracy. It is recommended to quantize the model with the\n"
             "per-tensor scheme. See https://ai.google.dev/edge/litert/models/quantization_spec.\n\n");

      calc_quantization(subgraph, poperation, operation, &scale, &zero_point);
      operation->weight_scale = scale;
      operation->weight_zero_point = zero_point;
      requantize_weights(subgraph, poperation, operation);
      requantize_bias(subgraph, poperation, operation);
   }
'''

NEW_LATE_BLOCK = r'''   /* och patch: fold-early —— 折叠已在上面的「权重/bias 拷贝之后」完成，
    * 这里只保留 v8 硬件原生 depthwise 的语义收口（v7 无 depthwise 模式，不需要）。
    * ★ 不要在这里重新调用 calc_quantization/requantize_*：
    *   折叠过一次就够了，二次折叠会再引入一次舍入。 */
   if (nn_core_version >= 8 && poperation->conv.depthwise)
      operation->input_channels = 1;
'''


# ── 源码定位 ──────────────────────────────────────────────────────────────────

def find_func_span(text, name):
    """返回 (start, end) —— 含签名与函数体，end 指向收尾大括号之后。"""
    m = re.search(r"^static\s+(?:void|bool|unsigned)\s*\n%s\s*\(" % re.escape(name), text, re.MULTILINE)
    if not m:
        m = re.search(r"^%s\s*\(" % re.escape(name), text, re.MULTILINE)
    if not m:
        raise SystemExit("找不到函数 %s" % name)
    i = text.index("{", m.end())
    depth = 0
    j = i
    in_str = in_chr = in_line_c = in_blk_c = False
    while j < len(text):
        ch = text[j]
        nxt = text[j + 1] if j + 1 < len(text) else ""
        if in_line_c:
            if ch == "\n":
                in_line_c = False
        elif in_blk_c:
            if ch == "*" and nxt == "/":
                in_blk_c = False
                j += 1
        elif in_str:
            if ch == "\\":
                j += 1
            elif ch == '"':
                in_str = False
        elif in_chr:
            if ch == "\\":
                j += 1
            elif ch == "'":
                in_chr = False
        elif ch == "/" and nxt == "/":
            in_line_c = True
            j += 1
        elif ch == "/" and nxt == "*":
            in_blk_c = True
            j += 1
        elif ch == '"':
            in_str = True
        elif ch == "'":
            in_chr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return m.start(), j + 1
        j += 1
    raise SystemExit("花括号不配对：%s" % name)


def find_block(text, start_literal):
    """从 start_literal 开始，按缩进+括号配对取出整块（含末尾换行）。"""
    if text.count(start_literal) != 1:
        raise SystemExit("锚点出现 %d 次（要求恰好 1 次）：%r"
                         % (text.count(start_literal), start_literal[:60]))
    i = text.index(start_literal)
    j = text.index("{", i)
    depth = 0
    in_str = in_line_c = in_blk_c = False
    k = j
    while k < len(text):
        ch = text[k]
        nxt = text[k + 1] if k + 1 < len(text) else ""
        if in_line_c:
            if ch == "\n":
                in_line_c = False
        elif in_blk_c:
            if ch == "*" and nxt == "/":
                in_blk_c = False
                k += 1
        elif in_str:
            if ch == "\\":
                k += 1
            elif ch == '"':
                in_str = False
        elif ch == "/" and nxt == "/":
            in_line_c = True
            k += 1
        elif ch == "/" and nxt == "*":
            in_blk_c = True
            k += 1
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = k + 1
                while end < len(text) and text[end] != "\n":
                    end += 1
                return i, end + 1
        k += 1
    raise SystemExit("块不配对：%r" % start_literal[:60])


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


LATE_START = "   if (poperation->conv.weight_tensor->scales) {"


# ── 主机 ──────────────────────────────────────────────────────────────────────

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="etnaviv_ml_nn.c")
    ap.add_argument("--verify", action="store_true",
                    help="只核对锚点/函数 md5 是否与钉死值一致，不写文件")
    ap.add_argument("--allow-md5-drift", action="store_true",
                    help="仅供调试：跳过 md5 钉死（CI 禁用）")
    args = ap.parse_args(argv[1:])

    text = open(args.src, encoding="utf-8").read()

    s_calc, e_calc = find_func_span(text, "calc_quantization")
    s_req, e_req = find_func_span(text, "requantize_weights")
    calc_src = text[s_calc:e_calc]
    req_src = text[s_req:e_req]
    # body = 从第一个 '{' 到收尾 '}'（不含签名）
    calc_body = calc_src[calc_src.index("{"):]
    req_body = req_src[req_src.index("{"):]

    b0, b1 = find_block(text, LATE_START)
    block_src = text[b0:b1]

    report = [
        ("calc_quantization  body", md5(calc_body), MD5_CALC_Q),
        ("requantize_weights body", md5(req_body), MD5_REQ_W),
        ("per-channel 折叠块", md5(block_src), MD5_BLOCK),
    ]
    bad = 0
    for name, got, want in report:
        ok = (got == want) or args.allow_md5_drift
        print("  [%-24s] md5=%s  %s" % (name, got, "OK" if ok else "MISMATCH（期望 %s）" % want))
        bad += 0 if ok else 1
    if bad:
        sys.stderr.write("ERROR: 上游源码与钉死的 md5 不符 —— 拒绝打补丁（fail-closed）。\n")
        sys.stderr.write("       请重读 verify/reports/55 的假设，更新补丁与 md5 后重跑。\n")
        return 2

    if text.count(ANCHOR_INSERT) != 1:
        sys.stderr.write("ERROR: 插入锚点出现 %d 次（要求 1）：%r\n"
                         % (text.count(ANCHOR_INSERT), ANCHOR_INSERT))
        return 2

    if args.verify:
        print("  [verify] 三个 md5 与插入锚点全部命中，未写文件。")
        return 0

    # 1) 插 helper（在 calc_quantization 之前）
    out = text[:s_calc] + NEW_HELPER.lstrip("\n") + "\n" + text[s_calc:]
    # 位置偏移：helper 插在前面 ⇒ 重新定位后两个函数
    s_calc2, e_calc2 = find_func_span(out, "calc_quantization")
    s_req2, e_req2 = find_func_span(out, "requantize_weights")

    # 2) 换 calc_quantization 的 body（保留签名）
    prefix = out[s_calc2:s_req2]
    sig_end = prefix.index("{")
    out = out[:s_calc2 + sig_end] + NEW_CALC_Q_BODY + out[e_calc2:]

    # 3) 换 requantize_weights 的 body
    s_req3, e_req3 = find_func_span(out, "requantize_weights")
    span = out[s_req3:e_req3]
    sig_end3 = span.index("{")
    out = out[:s_req3 + sig_end3] + NEW_REQ_W_BODY + out[e_req3:]

    # 4) 折叠块前移
    b0b, b1b = find_block(out, LATE_START)
    moved = out[b0b:b1b]
    out = out[:b0b] + NEW_LATE_BLOCK + out[b1b:]
    out = out.replace(ANCHOR_INSERT, ANCHOR_INSERT + NEW_EARLY_BLOCK, 1)

    # 5) 二进制标记
    if MARKER not in out:
        out += ('\n/* och patch: compile-time marker */\n'
                '__attribute__((used)) static const char och_foldearly_marker[] = "%s";\n' % MARKER)

    open(args.src, "w", encoding="utf-8").write(out)

    # ── 硬门（打完必查）──────────────────────────────────────────────────────
    chk = open(args.src, encoding="utf-8").read()
    errs = []
    if chk.count("och_fold_channel(") < 3:
        errs.append("och_fold_channel 调用点 < 3（helper 定义 + 两处调用）")
    s_req4, e_req4 = find_func_span(chk, "requantize_weights")
    body4 = chk[s_req4:e_req4]
    for bad_tok in ("malloc(", "free(", "weight_tensor_size ="):
        if bad_tok in body4:
            errs.append("requantize_weights 里仍有 %r（必须原地、不得改尺寸）" % bad_tok)

    # 迁移后在**函数作用域内**核对（不能再用 find_block(LATE_START)：迁移后该锚点
    # 只剩「前移后的那一处」，再解一次会取到新位置而误判）
    s_fn, e_fn = find_func_span(chk, "etna_ml_lower_convolution")
    scope = chk[s_fn:e_fn]
    n_blocks = scope.count(LATE_START)
    if n_blocks != 1:
        errs.append("函数内折叠块出现 %d 次（要求恰好 1，且必须已前移）" % n_blocks)
    else:
        p_block = scope.index(LATE_START)
        p_pw = scope.find("pointwise_to_2x2(subgraph")
        p_dw = scope.find("expand_depthwise(subgraph")
        p_tp = scope.find("transpose(subgraph")
        for nm, p in (("pointwise_to_2x2", p_pw), ("expand_depthwise", p_dw), ("transpose", p_tp)):
            if p < 0:
                errs.append("函数内找不到调用 %s" % nm)
            elif not (p_block < p):
                errs.append("折叠块没有排在 %s 之前" % nm)
    if scope.count("nn_core_version >= 8 && poperation->conv.depthwise") != 1:
        errs.append("v8 的 input_channels 收口丢了（应恰好 1 处）")
    # 防重复折叠：函数内三处调用各只能出现 1 次
    for call in ("calc_quantization(subgraph", "requantize_weights(subgraph", "requantize_bias(subgraph"):
        c = scope.count(call)
        if c != 1:
            errs.append("函数内 %s 出现 %d 次（要求 1，否则是重复折叠）" % (call, c))
    if MARKER not in chk:
        errs.append("缺二进制标记")
    if errs:
        for e in errs:
            sys.stderr.write("ERROR: %s\n" % e)
        return 3

    print("  [foldearly] calc_quantization : 逐元素、按元素所属通道取 scale/zp（原地）")
    print("  [foldearly] requantize_weights: 原地折叠，尺寸不变（不再 malloc/free/改 size）")
    print("  [foldearly] 折叠块前移        : 到 pointwise/expand/strided/transpose 之前")
    print("  [foldearly] v8 收口保留       : input_channels = 1 仅 v8")
    print("  [foldearly] 标记              : %s" % MARKER)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
