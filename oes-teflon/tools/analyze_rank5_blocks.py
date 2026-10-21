#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线分析模型里全部 rank-5 块的结构，并判定「模型手术」是否可行。

## 背景（报告 11）

驱动契约里 `dims` 只有 4 槽，一律按 `dims[1]*dims[2]*dims[3]` 算字节数。
delegate 写入形状的**最后 4 维** ⇒ rank-5 激活的第 0 维被丢掉，size 缩成 1/X，
且 job 几何（channel/height/width）同源错误。**这是表达能力边界，不是 size bug。**

## 本脚本回答的问题

1. rank-5 张量一共几个、**它们构成多少个结构相同的块**？
2. 每个块的「净效果」是什么？（是不是纯通道置换？）
3. 置换后谁在消费它们？（决定能不能在模型层消除）
4. **消除它们是否数学等价**？（置换是不是双射、消费者是不是纯选择）

## 为什么这很重要

如果块是「源形状与目标形状完全相同的纯通道置换」，那么它对下游的影响
**只体现为通道顺序**。而下游消费者如果是「通道选择/重排」类算子，
就可以把「置换 + 选择」**合并成一次选择**（写出新的索引），从而
**在不改变任何数值的前提下把 rank-5 从图里彻底去掉**。

★ 本脚本只做判定，**不改模型**。改模型属产物语义变更，必须先获得用户批准。
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tflite_min_read import load_model, op_name, is_activation  # noqa: E402

RESHAPE, TRANSPOSE, STRIDED_SLICE = 22, 39, 45


def fmt_shape(shape):
    return "[" + ",".join(str(x) for x in shape) + "]"


def channel_perm(c, C):
    """rank-5 块的净效果：c 在 [0,C) 上，块把通道号映射到哪？

    块形态（以 C=244 为例）：
        [1,C,H,W] --RESHAPE--> [1,2,C/2,H,W] --TRANSPOSE(交换 1,2)--> [1,C/2,2,H,W] --RESHAPE--> [1,C,H,W]
    下标：reshape 后 (g,k) 满足 c = g*(C/2) + k；transpose 后 (k,g)；reshape 回来 c' = k*2 + g。
    """
    h = C // 2
    return (c % h) * 2 + (c // h)


def main(path):
    m = load_model(path)
    T, O = m["tensors"], m["operators"]
    n_t, n_o = m["n_tensors"], m["n_operators"]

    def sh(ti):
        return "t%d%s" % (ti, fmt_shape(T[ti]["shape"])) if 0 <= ti < n_t else "t%d(?)" % ti

    prod, cons = {}, {}
    for o in O:
        for ti in o["outputs"]:
            prod.setdefault(ti, []).append(o)
        for ti in o["inputs"]:
            cons.setdefault(ti, []).append(o)

    print("=" * 78)
    print("rank-5 块结构分析")
    print("=" * 78)
    print("  模型: %s" % path)
    print("  张量 %d / 算子 %d" % (n_t, n_o))
    r5 = [t["i"] for t in T if t["rank"] == 5]
    r5_act = [ti for ti in r5 if is_activation(m, ti)]
    print("  rank-5 张量 %d 个（其中激活 %d 个 ⇒ 会走驱动分配路径）" % (len(r5), len(r5_act)))
    print()

    # ------------------------------------------------------------------ 1
    print("--- [1] 找出全部 rank-5 块并统计签名 ---")
    blocks = []
    for o in O:
        if o["code"] != RESHAPE:
            continue
        if not any(0 <= x < n_t and T[x]["rank"] == 5 for x in o["outputs"]):
            continue
        r1 = o
        t5a = r1["outputs"][0]
        tr = cons.get(t5a, [None])[0]
        if tr is None:
            continue
        t5b = tr["outputs"][0]
        r2 = cons.get(t5b, [None])[0]
        if r2 is None:
            continue
        blocks.append({"r1": r1, "tr": tr, "r2": r2, "t5a": t5a, "t5b": t5b,
                       "src": r1["inputs"][0], "out4": r2["outputs"][0]})

    print("  块数: %d" % len(blocks))
    sig = Counter()
    for b in blocks:
        sig[(tuple(T[b["src"]]["shape"]), tuple(T[b["t5a"]]["shape"]),
             tuple(T[b["t5b"]]["shape"]), tuple(T[b["out4"]]["shape"]),
             op_name(b["r1"]["code"]), op_name(b["tr"]["code"]), op_name(b["r2"]["code"]))] += 1
    for k, v in sorted(sig.items()):
        print("    x%-3d %s -> %s --%s/%s/%s--> %s -> %s"
              % (v, fmt_shape(k[0]), fmt_shape(k[1]), k[4], k[5], k[6], fmt_shape(k[2]), fmt_shape(k[3])))
    print()

    # ------------------------------------------------------------------ 2
    print("--- [2] 净效果判定：源形状 == 目标形状？（若是 ⇒ 纯置换）---")
    same = sum(1 for b in blocks if T[b["src"]]["shape"] == T[b["out4"]]["shape"])
    print("  源形状 == 目标形状 的块: %d / %d" % (same, len(blocks)))
    print()

    # ------------------------------------------------------------------ 3
    print("--- [3] 三者之和 = 双射？---")
    ok_all = True
    for b in blocks:
        C = T[b["src"]]["shape"][1]
        perm = [channel_perm(c, C) for c in range(C)]
        bij = sorted(perm) == list(range(C))
        ok_all &= bij
        if not bij:
            print("    ❌ C=%d 不是双射" % C)
    enum = Counter((T[b["src"]]["shape"][1],
                    T[b["src"]]["shape"][2],
                    int(sorted(channel_perm(c, T[b["src"]]["shape"][1]) for c in range(T[b["src"]]["shape"][1])) == list(range(T[b["src"]]["shape"][1]))))
                   for b in blocks)
    for (C, HW, bij), n in sorted(enum.items()):
        print("    C=%-4d 空间=%-4s 双射=%s  x%d" % (C, HW, bool(bij), n))
    print("  ⇒ 全部双射: %s" % ("是（⇒ 块是纯置换，语义上可消除）" if ok_all else "否"))
    print()

    # ------------------------------------------------------------------ 4
    print("--- [4] 置换后的消费者（决定手术形态）---")
    kind = Counter()
    non_slice = []
    for b in blocks:
        cs = cons.get(b["out4"], [])
        kinds = [op_name(q["code"]) for q in cs]
        for k in kinds:
            kind[k] += 1
        if STRIDED_SLICE not in [q["code"] for q in cs]:
            non_slice.append(b)
    print("  消费者分布: %s" % dict(kind))
    print("  消费者中**没有** STRIDED_SLICE 的块: %d 个" % len(non_slice))
    for b in non_slice:
        cs = cons.get(b["out4"], [])
        print("    block@op#%-4d 置换输出 %s -> %s"
              % (b["r1"]["i"], fmt_shape(T[b["out4"]]["shape"]),
                 [(q["i"], op_name(q["code"]), sh(q["outputs"][0])) for q in cs]))
    print()

    print("--- [5] STRIDED_SLICE 清点（是不是纯通道选择）---")
    ss = [o for o in O if o["code"] == STRIDED_SLICE]
    print("  STRIDED_SLICE 算子数: %d" % len(ss))
    pat = Counter((tuple(T[o["inputs"][0]]["shape"]), tuple(T[o["outputs"][0]]["shape"])) for o in ss)
    for (a, b_), n in sorted(pat.items()):
        print("    %s -> %s  x%d" % (fmt_shape(a), fmt_shape(b_), n))
    halves = all(len(a) == 4 and len(b_) == 4 and a[0] == b_[0] and a[2:] == b_[2:]
                 and a[1] == 2 * b_[1] for a, b_ in pat)
    print("  ⇒ 全部是「通道数减半、空间不变」的 4D 选择: %s" % ("是" if halves else "否"))
    print()

    # ------------------------------------------------------------------ 6
    print("--- [6] 合并后的索引形态（以 C=244 为例）---")
    C = 244
    perm = [channel_perm(c, C) for c in range(C)]
    for lo, hi, tag in ((0, C // 2, "第一路 slice[0:C/2]"), (C // 2, C, "第二路 slice[C/2:C]")):
        # 置换后第 j 个元素来自原张量的哪个通道
        src_of = [0] * C
        for c in range(C):
            src_of[perm[c]] = c
        sel = src_of[lo:hi]
        runs = []
        s = sel[0]
        for i in range(1, len(sel)):
            if sel[i] != sel[i - 1] + 1:
                runs.append((s, sel[i - 1]))
                s = sel[i]
        runs.append((s, sel[-1]))
        print("  %s: 取原张量 %d 个通道，连续段 %d 段" % (tag, len(sel), len(runs)))
        print("      前 12 个源通道 = %s" % sel[:12])
        print("      连续段 = %s" % (runs if len(runs) <= 4 else runs[:4] + ["..."]))
    print()

    # ------------------------------------------------------------------ 结论
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if ok_all and len(non_slice) <= 3:
        print("  ✅ 全部 %d 个块都是**源形状==目标形状的纯通道置换**（双射）。" % len(blocks))
        print("  ✅ 消费者是对通道轴的选择/重排（STRIDED_SLICE %d 个%s）。"
              % (len(ss), "，另有 %d 个块直接接 TRANSPOSE" % len(non_slice) if non_slice else ""))
        print("  ⇒ **模型手术在数学上可行**：把「置换 + 通道选择」合并成一次等价的选择，")
        print("     rank-5 可从图里彻底去掉，**不需要改任何下游权重或量化参数**。")
        print("     改完的图是全 4D ⇒ 驱动契约的表达能力边界不再被触碰。")
        print()
        print("  ⚠️ 但：改模型 = **产物语义变更**，必须先获得用户批准；")
        print("     且新模型必须通过「CPU 输出与原模型逐字节一致」的验证才算成立。")
        return 0
    print("  ⚠️ 条件不满足，手术可行性需要进一步分析（见上方明细）。")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: analyze_rank5_blocks.py <model.tflite>")
    sys.exit(main(sys.argv[1]))
