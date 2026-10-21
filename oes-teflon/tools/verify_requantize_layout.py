#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""requantize_weights 布局错位的**独立数学复核**（不需要上机、不需要 CI）。

背景
----
`etna_ml_lower_convolution()` 的执行顺序（etnaviv_ml_nn.c）：

    607  if (depthwise) { 612 expand_depthwise(...) }      # [out][W][H][in]
    622  transpose(...)                                     # → [out][in][W][H]
    632  if (weight_tensor->scales) {                       # 只有 per-channel 进
    646     requantize_weights(...)
    649     if (v>=8 && depthwise) input_channels = 1;
         }

争议点：`requantize_weights` 的 `map_in` 声明是
    uint8_t (*map_in)[W][H][input_channels] = input;
即按 **`[out][W][H][in]`** 索引，但它拿到的是 `transpose()` 之后的
**`[out][in][W][H]`** 缓冲区。

本脚本用「每个元素记住自己的坐标」的办法，直接判定：
  - 旧实现读到的元素，是否就是它想要的那个 (oc, ic, w, h)？
  - 提议的修法是否读到正确元素？

判据：只做整数索引算术，不依赖任何模型文件、不依赖设备。
正控制：in_c == 1 时，旧实现也必须是对的（维度顺序退化等价）。
负控制：in_c > 1 时，旧实现必须被本脚本判为错位。
"""
import sys

FAIL = 0


def off(oc, ic, w, h, in_c, W, H):
    """`[out][in][W][H]` 布局下的线性偏移（= transpose() 的输出）。"""
    return ((oc * in_c + ic) * W + w) * H + h


def build_transposed(out_c, in_c, W, H):
    """模拟 transpose() 之后的缓冲区：每个位置存放它自己的坐标。"""
    buf = [None] * (out_c * in_c * W * H)
    for oc in range(out_c):
        for ic in range(in_c):
            for w in range(W):
                for h in range(H):
                    buf[off(oc, ic, w, h, in_c, W, H)] = (oc, ic, w, h)
    return buf


def read_old(buf, oc, w, h, ic, in_c, W, H):
    """旧实现：map_in[oc][w][h][ic]，声明为 [W][H][in]。"""
    return buf[((oc * W + w) * H + h) * in_c + ic]


def read_new(buf, oc, w, h, ic, in_c, W, H):
    """提议修法：map_in[oc][ic][w][h]，声明为 [in][W][H]。"""
    return buf[off(oc, ic, w, h, in_c, W, H)]


def check(label, out_c, in_c, W, H, expect_old_ok, expect_new_ok):
    """返回 (old_ok, new_ok)，并与期望比对。"""
    global FAIL
    buf = build_transposed(out_c, in_c, W, H)
    old_ok = new_ok = True
    old_bad = new_bad = None
    for oc in range(out_c):
        for ic in range(in_c):
            for w in range(W):
                for h in range(H):
                    want = (oc, ic, w, h)
                    if read_old(buf, oc, w, h, ic, in_c, W, H) != want and old_ok:
                        old_ok = False
                        old_bad = (want, read_old(buf, oc, w, h, ic, in_c, W, H))
                    if read_new(buf, oc, w, h, ic, in_c, W, H) != want and new_ok:
                        new_ok = False
                        new_bad = (want, read_new(buf, oc, w, h, ic, in_c, W, H))

    status = "OK  " if (old_ok == expect_old_ok and new_ok == expect_new_ok) else "FAIL"
    if status == "FAIL":
        FAIL += 1
    print(f"  [{status}] {label}")
    print(f"          out={out_c} in={in_c} W={W} H={H}")
    print(f"          旧实现读到正确元素 : {old_ok}  (期望 {expect_old_ok})")
    if old_bad:
        print(f"            首个错位: 想要 {old_bad[0]}，实际读到 {old_bad[1]}")
    print(f"          提议修法读到正确元素: {new_ok}  (期望 {expect_new_ok})")
    if new_bad:
        print(f"            首个错位: 想要 {new_bad[0]}，实际读到 {new_bad[1]}")
    print()


def depthwise_semantics(out_c, in_c, W, H):
    """depthwise 修法取的是对角线 (out=c, in=c)，验证它取到的确实是 c 通道的 filter。"""
    buf = build_transposed(out_c, in_c, W, H)
    ok = True
    for c in range(out_c):
        for w in range(W):
            for h in range(H):
                got = read_new(buf, c, w, h, c, in_c, W, H)
                if got != (c, c, w, h):
                    ok = False
    print(f"  [{'OK  ' if ok else 'FAIL'}] depthwise 对角线语义 (out={out_c}, in={in_c})")
    print(f"          map_in[c][c][w][h] 取到的确实是通道 c 的 filter: {ok}")
    print()
    return ok


def main():
    global FAIL
    print("=" * 74)
    print("requantize_weights 布局错位 —— 数学复核")
    print("=" * 74)
    print()

    print("--- 负控制：in_c > 1 且 W*H > 1（旧实现必须被判为错位）---")
    check("普通 per-channel 卷积 (in=24, 3x3)", 24, 24, 3, 3, False, True)
    check("普通 per-channel 卷积 (in=122, 2x2)", 122, 122, 2, 2, False, True)
    check("depthwise 展开后 (in=244)", 244, 244, 3, 3, False, True)
    check("depthwise 展开后 (in=122)", 122, 122, 3, 3, False, True)

    # ★ 重要边界：W*H == 1 时两种布局的线性偏移恒等
    #   [out][in][1][1] : (oc*in + ic)*1 + 0 = oc*in + ic
    #   [out][1][1][in] : ((oc*1+0)*1+0)*in + ic = oc*in + ic
    #   ⇒ 1x1 卷积（pointwise）**不受此缺陷影响**，旧实现也是对的。
    print("--- 边界：in_c > 1 但 W*H == 1（两种布局重合，旧实现也必须对）---")
    check("1x1 卷积 (in=244)", 244, 244, 1, 1, True, True)
    check("1x1 卷积 (in=24)", 24, 24, 1, 1, True, True)

    print("--- 正控制：in_c == 1（维度顺序退化，旧实现也必须对）---")
    check("in_c=1 退化情形", 8, 1, 3, 3, True, True)
    check("in_c=1 退化情形 (W=H=1)", 8, 1, 1, 1, True, True)

    print("--- depthwise 修法语义 ---")
    if not depthwise_semantics(24, 24, 3, 3):
        FAIL += 1
    if not depthwise_semantics(244, 244, 3, 3):
        FAIL += 1

    print("=" * 74)
    if FAIL:
        print(f"结果: {FAIL} 项不符预期 —— 结论不成立，不要采信")
        print("=" * 74)
        return 1
    print("结果: 全部符合预期")
    print()
    print("  ⇒ 旧 requantize_weights 在 in_c>1 时**读错元素**（尺寸相同故不崩、但数据错）")
    print("  ⇒ 提议修法（map_in[oc][ic][w][h] + depthwise 取对角线）读到的元素全部正确")
    print("  ⇒ in_c==1 时两者等价（正控制）—— 这正是「depthwise 崩溃能消掉」的原因")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
