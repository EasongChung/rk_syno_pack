#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值自测：把 per-channel 下降路径**逐字镜像成 Python**，验证索引数学。

这不是 grep 门 —— 它把 expand_depthwise -> transpose -> calc_quantization ->
requantize_weights 整条链跑一遍，与「独立的正确答案」逐元素比对。
C 那边编译不了、也上不了机时，这是唯一能证明修法数学正确的东西。

## 镜像的四个函数（逐行对应 etnaviv_ml_nn.c）

| 函数 | 源码位置 |
|---|---|
| `expand_depthwise` | nn.c:328-355 |
| `transpose`          | nn.c:377-411 |
| `calc_quantization`  | nn.c:194-228（新旧各一份） |
| `requantize_weights` | nn.c:230-270（新旧各一份） |

## 跑哪些用例

| 编号 | 用例 | 断言 |
|---|---|---|
| A | depthwise 244x3x3 | 新实现逐元素等于「对角线权重重量化」；旧实现取到的是 (out=0, in=oc) |
| B | 非 depthwise 24x3x3 | 布局不变、尺寸不变；**min/max 与排列无关 ⇒ scale 不变**；元素从错位变对位 |
| C | 非 depthwise 1x1 | 新旧**逐字节相同** ⇒ 1x1 正控制只在非 depthwise 成立 |
| D | depthwise 1x1 | 新旧**不同** ⇒ 1x1 对 depthwise 并非「不受影响」 |
| E | depthwise 的 scale | 旧实现只看 0.41% 权重 ⇒ scale 与新实现不同 |

用法：python3 tools/selftest-perchannel-layout-math.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

fails = []


def ck(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def c_round(x):
    """C 的 round()：四舍五入、.5 远离零（Python 的 round 是银行家舍入）。"""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def MIN2(a, b):
    return a if a < b else b


def MAX2(a, b):
    return a if a > b else b


# ---------------------------------------------------------------- 镜像层

def expand_depthwise(buf, out_c, in_c, W, H, wzp, signed):
    """nn.c:328-355。输出布局 [out][(w*H+h)*in_c + ic]，真实值落在 ic == oc。"""
    out = [0] * (out_c * W * H * in_c)
    for channel in range(out_c):
        in_channel = channel // out_c          # 恒为 0
        in_depth = channel % out_c            # 恒为 channel
        base_in = in_channel * W * H * in_c   # 恒为 0
        base_out = channel * W * H * in_c
        for i in range(W * H * in_c):
            if i % in_c == in_depth:
                out[base_out + i] = buf[base_in + i]
            else:
                out[base_out + i] = wzp - 128 if signed else wzp
    return out


def transpose(buf, out_c, in_c, W, H):
    """nn.c:377-411（depthwise 走 else 分支，in_c 取 operation->input_channels）。
    输入按 [out][W][H][in] 读，输出按 [out][in][W][H] 写。
    """
    out = [0] * (out_c * W * H * in_c)
    i = 0
    # ★ 读侧遍历顺序必须是 d0,d1,d2,d3（即 [out][W][H][in]），
    #   写侧才是 [out][in][W][H]。顺序写反会让整个镜像失真。
    for d0 in range(out_c):
        for d1 in range(W):
            for d2 in range(H):
                for d3 in range(in_c):
                    out[(d0 * in_c + d3) * W * H + d1 * H + d2] = buf[i]
                    i += 1
    return out


def calc_quant_params(mn, mx):
    """nn.c:184-192。"""
    if mx < mn:
        mn, mx = mx, mn
    diff = mx - MIN2(mn, 0.0)
    scale = diff / 255.0
    zp = c_round(-(mn / scale))
    return scale, zp


def calc_quantization_old(buf, out_c, in_c, W, H, depthwise, scales, zps, signed):
    """nn.c:194-228 原版：声明 [out][W][H][input_channels]，depthwise 强制 in=1。"""
    ic_n = 1 if depthwise else in_c
    mn, mx, visited = float("inf"), 0.0, 0
    for oc in range(out_c):
        for w in range(W):
            for h in range(H):
                for ic in range(ic_n):
                    v = buf[((oc * W + w) * H + h) * ic_n + ic]
                    dq = scales[oc] * (v - zps[oc])
                    mx = MAX2(mx, dq)
                    mn = MIN2(mn, dq)
                    visited += 1
    scale, zp = calc_quant_params(mn, mx)
    return scale, zp, visited


def calc_quantization_new(buf, out_c, in_c, W, H, depthwise, scales, zps, signed):
    """补丁后：声明 [out][in][W][H]，depthwise 只取对角线 ic == oc。"""
    oc_max = MIN2(out_c, in_c) if depthwise else out_c
    mn, mx, visited = float("inf"), 0.0, 0
    for oc in range(oc_max):
        for w in range(W):
            for h in range(H):
                for ic in range(in_c):
                    if depthwise and ic != oc:
                        continue
                    v = buf[(oc * in_c + ic) * W * H + w * H + h]
                    dq = scales[oc] * (v - zps[oc])
                    mx = MAX2(mx, dq)
                    mn = MIN2(mn, dq)
                    visited += 1
    scale, zp = calc_quant_params(mn, mx)
    return scale, zp, visited


def requantize_weights_old(buf, out_c, in_c, W, H, depthwise, scales, zps,
                           wscale, wzp, signed):
    """nn.c:230-270 原版。"""
    ic_n = 1 if depthwise else in_c
    new_size = out_c * W * H * ic_n
    out = [0] * new_size
    for oc in range(out_c):
        for w in range(W):
            for h in range(H):
                for ic in range(ic_n):
                    v = buf[((oc * W + w) * H + h) * ic_n + ic]
                    dq = scales[oc] * (v - zps[oc])
                    out[((oc * W + w) * H + h) * ic_n + ic] = c_round(dq / wscale) + wzp
    return out, new_size


def requantize_weights_new(buf, out_c, in_c, W, H, depthwise, scales, zps,
                           wscale, wzp, signed):
    """补丁后：读 [out][in][W][H]；depthwise 折成紧凑 [out][W][H]。"""
    out_dim = 1 if depthwise else in_c
    oc_max = MIN2(out_c, in_c) if depthwise else out_c
    new_size = oc_max * W * H * out_dim
    out = [0] * new_size
    for oc in range(oc_max):
        for w in range(W):
            for h in range(H):
                for ic in range(out_dim):
                    src_ic = oc if depthwise else ic
                    v = buf[(oc * in_c + src_ic) * W * H + w * H + h]
                    dq = scales[oc] * (v - zps[oc])
                    out[(oc * out_dim + ic) * W * H + w * H + h] = c_round(dq / wscale) + wzp
    return out, new_size


# ---------------------------------------------------------------- 造数据

def make_case(out_c, in_c, W, H, depthwise, seed=12345):
    """构造一条完整的 per-channel 权重链，返回各阶段缓冲与逐项真值。

    ★ 布局依据（全部来自真源码）：
      - expand_depthwise 的输入指针恒为 0（nn.c:336/339，in_channel 恒为 0），
        所以它的输入是**一个** W*H*in_c 的块，不是每 out-channel 一块；
      - 其输出是 [out][(w*H+h)*in_c + ic]，真实权重落在 ic == oc；
      - transpose 把它读成 [out][in][W][H]（nn.c:377-411），对角线保持不变。
    非 depthwise 时 expand_depthwise 根本不参与，缓冲直接是 [out][W][H][in]。
    """
    def rnd():
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed

    truth = [[[[rnd() % 255 - 128 for _ in range(in_c)] for _ in range(H)]
              for _ in range(W)] for _ in range(out_c)]
    if depthwise:
        # out=c 只用 in=c 的 filter：把每个 ic 槽都填成同一个 (c) 的值
        for oc in range(min(out_c, in_c)):
            for w in range(W):
                for h in range(H):
                    for ic in range(in_c):
                        truth[oc][w][h][ic] = truth[oc][w][h][oc]

    # per-channel 的 scale 差异拉大，好让「只看一个通道」立刻显形
    scales = [0.002 + 0.01 * (i % 17) / 17.0 for i in range(out_c)]
    zps = [0] * out_c
    wzp = 0

    if depthwise:
        pre = [0] * (W * H * in_c)
        for oc in range(min(out_c, in_c)):
            for w in range(W):
                for h in range(H):
                    pre[(w * H + h) * in_c + oc] = truth[oc][w][h][oc]
        exp = expand_depthwise(pre, out_c, in_c, W, H, wzp, True)
    else:
        pre = [0] * (out_c * W * H * in_c)
        for oc in range(out_c):
            for w in range(W):
                for h in range(H):
                    for ic in range(in_c):
                        pre[(oc * W + w) * H * in_c + h * in_c + ic] = truth[oc][w][h][ic]
        exp = pre

    tr = transpose(exp, out_c, in_c, W, H)
    return dict(truth=truth, scales=scales, zps=zps, wzp=wzp,
                exp=exp, tr=tr, out_c=out_c, in_c=in_c, W=W, H=H)


def expected_new(c, depthwise):
    """按**语义**独立算出「新实现应该得到什么」，不看新实现的代码。"""
    o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
    s, z = c["scales"], c["zps"]
    if not depthwise:
        vals = [s[oc] * (c["truth"][oc][w][h][ic] - z[oc])
                for oc in range(o) for w in range(W) for h in range(H) for ic in range(i)]
    else:
        vals = [s[oc] * (c["truth"][oc][w][h][oc] - z[oc])
                for oc in range(min(o, i)) for w in range(W) for h in range(H)]
    scale, zp = calc_quant_params(min(vals), max(vals))
    n = o * W * H * (1 if depthwise else i)
    return scale, zp, n


# ---------------------------------------------------------------- 用例

print("=== A. depthwise 244x3x3（实测崩溃的那一档）===")
c = make_case(244, 244, 3, 3, True)
o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
tr, s, z, wzp = c["tr"], c["scales"], c["zps"], c["wzp"]

s_old, zp_old, v_old = calc_quantization_old(tr, o, i, W, H, True, s, z, True)
s_new, zp_new, v_new = calc_quantization_new(tr, o, i, W, H, True, s, z, True)
e_scale, e_zp, e_n = expected_new(c, True)

print("  CQ-OUT 预期: visited=%d  scale=%.8f" % (e_n, e_scale))
print("  旧: visited=%d scale=%.8f" % (v_old, s_old))
print("  新: visited=%d scale=%.8f" % (v_new, s_new))
ck(v_new == e_n, "新实现遍历字节数 = %d（= out_c*W*H，不是 535824）" % e_n)
ck(v_new != o * W * H * i, "新实现确实没有覆盖整个缓冲区（否则 A 的判据无意义）")
ck(abs(s_new - e_scale) < 1e-12, "新实现的 scale 与独立算出的期望一致")
ck(abs(s_old - s_new) > 1e-9,
   "★ 旧实现的 scale 与新的不同（只看了 %.2f%% 的权重）" % (100.0 * v_old / (o * W * H * i)))

out_new, n_new = requantize_weights_new(tr, o, i, W, H, True, s, z, s_new, zp_new, True)
out_old, n_old = requantize_weights_old(tr, o, i, W, H, True, s, z, s_new, zp_new, True)
ck(n_new == o * W * H, "新实现 new_size = %d（= 244*3*3）" % n_new)

ref = []
for oc in range(min(o, i)):
    for w in range(W):
        for h in range(H):
            dq = s[oc] * (c["truth"][oc][w][h][oc] - z[oc])
            ref.append(c_round(dq / e_scale) + e_zp)
ck(out_new == ref, "★ 新实现逐元素等于「对角线权重量化到正确 scale」")

# 旧实现读到的是 (out=0, in=oc)
wrong = []
for oc in range(o):
    for w in range(W):
        for h in range(H):
            v = tr[(0 * i + oc) * W * H + w * H + h]
            wrong.append(c_round((s[oc] * (v - z[oc])) / s_new) + zp_new)
ck(out_old == wrong, "旧实现确实读的是 (out=0, in=oc) 那 244 个切片")
ck(out_old != out_new, "★ 旧、新结果不同 ⇒ 只修 write-back 会得到静默的错误权重")

print()
print("=== B. 非 depthwise 24x3x3（不崩，但静默算错）===")
c = make_case(24, 24, 3, 3, False)
o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
tr, s, z = c["tr"], c["scales"], c["zps"]
s_old, zp_old, _ = calc_quantization_old(tr, o, i, W, H, False, s, z, True)
s_new, zp_new, v_new = calc_quantization_new(tr, o, i, W, H, False, s, z, True)
e_scale, e_zp, e_n = expected_new(c, False)
ck(v_new == o * W * H * i, "非 depthwise 遍历覆盖整个缓冲区（%d）" % (o * W * H * i))
ck(abs(s_new - s_old) < 1e-12,
   "★ min/max 与排列无关 ⇒ 旧、新 scale 相同（所以只有改前才静默错）")
ck(abs(s_new - e_scale) < 1e-12, "scale 与独立期望一致")

o_new, n_new = requantize_weights_new(tr, o, i, W, H, False, s, z, s_new, zp_new, True)
o_old, n_old = requantize_weights_old(tr, o, i, W, H, False, s, z, s_new, zp_new, True)
ck(n_new == n_old == o * W * H * i, "尺寸不变（%d），所以不会崩" % n_new)

# ★ 更正我自己的 18 号报告：非 depthwise 情形下旧实现**也是对的**。
#   两个索引 ((oc*W+w)*H+h)*in+ic 与 (oc*in+ic)*W*H+w*H+h 都是到
#   [0, out*W*H*in) 的**双射**，而且都是「同下标读写」；weight 侧的
#   requant 只用到 scales[oc]，而 oc = p // (W*H*in) 与两种下标都相容
#   ⇒ 这个置换是自逆的，数值逐元素不变。
#   真正的错只在 depthwise（ic_n 被压成 1，缓冲区变小、不再是双射）。
ck(o_old == o_new, "★ 非 depthwise：旧、新逐字节相同 ⇒ 这一路径原本就无害")

# 双射/自逆的论断对非方形核也成立，补一档 W != H 的抽样
for (o2, i2, w2, h2) in ((8, 8, 5, 3), (6, 6, 1, 7), (12, 12, 3, 5)):
    cc = make_case(o2, i2, w2, h2, False)
    t2, s2, z2 = cc["tr"], cc["scales"], cc["zps"]
    sn2, zpn2, _ = calc_quantization_new(t2, o2, i2, w2, h2, False, s2, z2, True)
    a2, _ = requantize_weights_new(t2, o2, i2, w2, h2, False, s2, z2, sn2, zpn2, True)
    b2, _ = requantize_weights_old(t2, o2, i2, w2, h2, False, s2, z2, sn2, zpn2, True)
    ck(a2 == b2, "out=%d in=%d %dx%d 非 depthwise：旧、新仍逐字节相同" % (o2, i2, w2, h2))

# ★ ref 的遍历顺序必须与输出缓冲区的线性顺序一致：非 depthwise 的缓冲区是
#   [out][in][W][H] ⇒ 下标顺序 oc, ic, w, h（写错顺序会误判）。
ref = [c_round((s[oc] * (c["truth"][oc][w][h][ic] - z[oc])) / e_scale) + e_zp
       for oc in range(o) for ic in range(i) for w in range(W) for h in range(H)]
ck(o_new == ref, "★ 新实现逐元素等于「按 [out][in][W][H] 就位重量化」")

print()
print("=== C. 非 depthwise 1x1（och 提议的正控制）===")
c = make_case(16, 16, 1, 1, False)
o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
tr, s, z = c["tr"], c["scales"], c["zps"]
_, _, _ = calc_quantization_new(tr, o, i, W, H, False, s, z, True)
sc = calc_quant_params(*(lambda vs: (min(vs), max(vs)))(
    [s[oc] * (c["truth"][oc][0][0][ic] - z[oc])
     for oc in range(o) for ic in range(i)]))
a, _ = requantize_weights_new(tr, o, i, W, H, False, s, z, sc[0], sc[1], True)
b, _ = requantize_weights_old(tr, o, i, W, H, False, s, z, sc[0], sc[1], True)
ck(a == b, "1x1 非 depthwise：新旧逐字节相同 ⇒ 正控制成立（仅限非 depthwise）")

print()
print("=== D. depthwise 1x1（对 och 那个正控制的更正）===")
c = make_case(16, 16, 1, 1, True)
o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
tr, s, z = c["tr"], c["scales"], c["zps"]
s_new, zp_new, _ = calc_quantization_new(tr, o, i, W, H, True, s, z, True)
a, _ = requantize_weights_new(tr, o, i, W, H, True, s, z, s_new, zp_new, True)
b, _ = requantize_weights_old(tr, o, i, W, H, True, s, z, s_new, zp_new, True)
ck(a != b, "★ depthwise 1x1 新旧不同 ⇒ 「1x1 不受影响」对 depthwise **不成立**")

print()
print("=== E. 边界：in_c == 1 / out_c == 1 / 不变量破坏 ===")
for (o_, i_, w_, h_, dw) in ((8, 1, 3, 3, False), (1, 8, 3, 3, False),
                             (1, 1, 1, 1, False), (4, 4, 1, 1, True)):
    c = make_case(o_, i_, w_, h_, dw)
    o, i, W, H = c["out_c"], c["in_c"], c["W"], c["H"]
    tr, s, z = c["tr"], c["scales"], c["zps"]
    sn, zpn, vn = calc_quantization_new(tr, o, i, W, H, dw, s, z, True)
    an, nn_ = requantize_weights_new(tr, o, i, W, H, dw, s, z, sn, zpn, True)
    ao, no = requantize_weights_old(tr, o, i, W, H, dw, s, z, sn, zpn, True)
    ck(nn_ == no, "out=%d in=%d %dx%d dw=%d：尺寸不变（%d）" % (o_, i_, w_, h_, dw, nn_))
    if i_ == 1 or w_ * h_ == 1 and not dw:
        ck(an == ao, "out=%d in=%d %dx%d dw=%d：布局退化 ⇒ 新旧应相同" % (o_, i_, w_, h_, dw))
    ck(len(an) == nn_, "out=%d in=%d %dx%d dw=%d：输出长度自洽" % (o_, i_, w_, h_, dw))

# depthwise 且 out_c > in_c（真实 Mesa 里不该出现）必须被 oc_max 护住而不是越界
o_, i_, w_, h_ = 8, 4, 3, 3
c = make_case(o_, i_, w_, h_, True)
tr, s, z = c["tr"], c["scales"], c["zps"]
sn, zpn, _ = calc_quantization_new(tr, o_, i_, w_, h_, True, s, z, True)
an, nn_ = requantize_weights_new(tr, o_, i_, w_, h_, True, s, z, sn, zpn, True)
ck(nn_ == min(o_, i_) * w_ * h_, "depthwise out_c(8) > in_c(4)：被 oc_max 护住（new_size=%d）" % nn_)
ck(len(an) == nn_, "护栏下不越界")

print()
print("MATH SELFTEST: " + ("PASS" if not fails else "FAIL (%d)" % len(fails)))
for f in fails:
    print("  - " + f)
sys.exit(0 if not fails else 1)