#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0：离线验证 R5GUARD 判据 `data == NULL && dims[0] != 1` 的命中情况。

## 为什么可以离线算

契约里 `dims` 是 delegate **从张量形状直接写进去**的（报告 11 §2.1），
所以「驱动看到的 `dims[0]`」完全由模型里的形状决定，不需要真机数据。
真机唯一能提供而离线提供不了的，是「驱动实际遍历到哪些 idx」——
但那不影响**误命中率**这一命题（它是形状的函数）。

## 判据的两个分支

    predicate = (data == NULL)  &&  (dims[0] != 1)
                 ^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                 激活张量（buffer 无数据）    驱动看到的第 0 槽

`dims` 的填法（报告 11）：delegate 写入形状的**最后 4 维**。

    rank 4: shape [1,H,W,C]      -> dims = [1,H,W,C]      dims[0] = 1   ⇒ 不命中
    rank 5: shape [1,X,2,H,W]    -> dims = [X,2,H,W]      dims[0] = X   ⇒ 命中 iff X != 1
    rank<4: 填法未知 -> 一并统计（这是**潜在误命中**的来源）

## 本脚本回答三个问题

1. **误命中**：有没有「不是 rank-5 被压进 4 槽」却命中的激活张量？
2. **真命中**：有多少个 rank-5 激活命中？
3. **★ 守卫会不会触发**：R5GUARD 只挂在 **CONVOLUTION 分支**上，
   所以必须看「有没有卷积的输入/输出是 rank-5 激活」。
   若一个都没有 ⇒ **这个守卫是空操作**（补丁再正确也不改变任何行为）。
"""

import sys
from collections import Counter

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from tflite_min_read import load_model, is_activation, op_name, CONV_OPS


def driver_dims(shape):
    """返回 delegate 写进 dims[4] 的 4 个槽（按「最后 4 维」规则）。"""
    if len(shape) >= 4:
        return shape[-4:]
    return shape + [None] * (4 - len(shape))     # 补 None 表示「填法未知」


def main(path):
    m = load_model(path)
    T = m["tensors"]

    print("=" * 78)
    print("P0 分析：R5GUARD 判据 `data == NULL && dims[0] != 1`")
    print("=" * 78)
    print("  模型: %s" % path)
    print("  张量 %d / 算子 %d" % (m["n_tensors"], m["n_operators"]))
    print()

    # ---------------------------------------------------------------- 1
    print("--- [1] 按 rank 清点「激活」张量（data == NULL 的那些）---")
    act = [t for t in T if is_activation(m, t["i"])]
    print("  激活张量数: %d / %d" % (len(act), len(T)))
    print("  激活的 rank 分布: %s" % dict(sorted(Counter(t["rank"] for t in act).items())))
    print()

    hits, false_hits, unknown = [], [], []
    for t in act:
        d = driver_dims(t["shape"])
        if d[0] is None:
            unknown.append((t, d))
        elif d[0] != 1:
            (hits if t["rank"] >= 5 else false_hits).append((t, d))

    print("--- [2] 命中清点 ---")
    print("  真命中（rank>=5 且 dims[0]!=1）      : %d" % len(hits))
    print("  ★ 误命中（rank<5 且 dims[0]!=1）     : %d" % len(false_hits))
    print("  填法未知（rank<4，dims[0] 不可判）  : %d" % len(unknown))
    print()
    if false_hits:
        print("  ★★ 误命中明细（前 20 条）—— 误命中不为零 ⇒ 守卫必须撤回:")
        for t, d in false_hits[:20]:
            print("     tensor %-4d rank=%d shape=%-22s -> dims=%s  name=%s"
                  % (t["i"], t["rank"], t["shape"], d, t["name"][:26]))
        print()
    if unknown:
        print("  填法未知明细（前 12 条）—— 这些是 rank<4 的激活，")
        print("  若 delegate 把它们**左对齐**写 dims[0]=shape[0]，则它们全都误命中:")
        for t, d in unknown[:12]:
            print("     tensor %-4d rank=%d shape=%-22s name=%s"
                  % (t["i"], t["rank"], t["shape"], t["name"][:34]))
        print()
    if hits:
        print("  真命中明细（前 20 条）:")
        for t, d in hits[:20]:
            print("     tensor %-4d rank=%d shape=%-24s -> dims=%s  name=%s"
                  % (t["i"], t["rank"], t["shape"], d, t["name"][:24]))
        print()

    # ---------------------------------------------------------------- 3
    print("--- [3] ★ R5GUARD 会不会触发？（它只挂在 CONVOLUTION 分支）---")
    convs = [o for o in m["operators"] if o["code"] in CONV_OPS]
    print("  卷积算子数: %d  （%s）"
          % (len(convs), dict(Counter(op_name(o["code"]) for o in convs))))
    fire, by_op = 0, Counter()
    fired_ops = []
    for o in convs:
        touched = []
        for ti in list(o["inputs"]) + list(o["outputs"]):
            if ti < 0 or ti >= m["n_tensors"]:
                continue
            t = T[ti]
            if not is_activation(m, ti):
                continue
            d = driver_dims(t["shape"])
            if t["rank"] >= 5 and d[0] not in (None, 1):
                touched.append((ti, t["rank"], t["shape"]))
            elif t["rank"] < 5 and d[0] is not None and d[0] != 1:
                touched.append((ti, t["rank"], t["shape"]))
        if touched:
            fire += 1
            by_op[op_name(o["code"])] += 1
            fired_ops.append((o, touched))
    print("  ⇒ 会被守卫拒绝的卷积数: %d / %d" % (fire, len(convs)))
    if fire:
        print("     分布: %s" % dict(by_op))
        for o, touched in fired_ops[:12]:
            print("     op#%-3d %-18s 触碰: %s" % (o["i"], op_name(o["code"]), touched))
    else:
        print("  ★★ 没有任何卷积的输入/输出是 rank-5 激活")
        print("     ⇒ 按当前补丁的挂载位置，**R5GUARD 是空操作**，")
        print("       它既不会减少接管数，也不会阻止那次 abort。")
    print()

    # ---------------------------------------------------------------- 4
    print("--- [4] 谁在“直接触碰” rank-5 激活？（对照报告 11 §7）---")
    touch = Counter()
    for o in m["operators"]:
        for ti in list(o["inputs"]) + list(o["outputs"]):
            if 0 <= ti < m["n_tensors"] and T[ti]["rank"] >= 5 and is_activation(m, ti):
                touch[op_name(o["code"])] += 1
                break
    print("  按算子类型（每个算子只计一次）: %s" % dict(sorted(touch.items())))
    print()

    # ---------------------------------------------------------------- 结论
    print("=" * 78)
    ok = (not false_hits) and fire > 0
    print("结论:")
    print("  误命中为零? %s" % ("是" if not false_hits else "★ 否 —— 判据必须修/撤回"))
    print("  守卫会触发? %s" % ("是" if fire else "★ 否 —— 空操作，需改挂载位置"))
    print("  ⇒ %s" % ("判据可用，且挂载位置正确，值得再派 CI 实测"
                      if ok else "当前形态下有阻断性问题，详见上方 ★ 条目"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: analyze_r5guard_p0.py <model.tflite>")
    sys.exit(main(sys.argv[1]))
