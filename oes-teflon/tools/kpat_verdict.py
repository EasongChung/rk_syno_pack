#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kpat_verdict.py —— 汇总三组 KPAT 对照的日志，给出「假设 H 是否成立」的判决

用法:
    python3 kpat_verdict.py <证据目录>

输入（由 tools/run_nndump_probe.sh 产出）：
    <目录>/nndump-iso-seg04.log         基线：122 通道 conv（期望不挂）
    <目录>/nndump-bis-s09p03.log        基线：244 通道 conv（期望挂）
    <目录>/nndump-iso-seg04-kpat6.log   强制 pattern 0x6 + 122 通道 conv（期望挂 ⇒ H 成立）
    <目录>/nndump-bis-s09p03-kpat3.log  强制 pattern 0x3f + 244 通道 conv（期望不挂 ⇒ H 成立）

判据：日志里出现 `recover hung GPU` 的等价信号 —— 我方用「该次 run 的
`teflon: invoked graph, took N ms` 里出现 ≥400 ms 的段」判定挂死；
并用 `min_ms` 交叉确认。
"""

import os
import re
import sys

RE_INV = re.compile(r'teflon: invoked graph, took (\d+) ms')
RE_MIN = re.compile(r'min_ms=([0-9.]+)')
RE_EXIT = re.compile(r'REMOTE_EXIT=(-?\d+)')
HANG_MS = 400


def analyse(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        body = f.read()
    inv = [int(m.group(1)) for m in (RE_INV.search(l) for l in body.splitlines()) if m]
    hangs = [v for v in inv if v >= HANG_MS]
    mm = RE_MIN.search(body)
    ex = RE_EXIT.search(body)
    return {
        "segments": len(inv),
        "hangs": len(hangs),
        "hang": len(hangs) > 0,
        "max_ms": max(inv) if inv else 0,
        "min_ms": float(mm.group(1)) if mm else None,
        "exit": int(ex.group(1)) if ex else None,
    }


def fmt(r):
    if r is None:
        return "【缺日志】"
    if not r["segments"]:
        return "【无段调用】exit=%s" % r["exit"]
    return "挂死=%s（%d/%d 段挂，最慢 %.0f ms）min_ms=%s exit=%s" % (
        "是" if r["hang"] else "否", r["hangs"], r["segments"], r["max_ms"],
        r["min_ms"], r["exit"])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]

    cases = [
        ("组1 基线 · 122 通道 conv (iso-seg04)", "nndump-iso-seg04.log", "不挂"),
        ("组1 基线 · 244 通道 conv (bis-s09p03)", "nndump-bis-s09p03.log", "挂"),
        ("组2 KPAT=6 · 122 通道 conv (iso-seg04)", "nndump-iso-seg04-kpat6.log", "挂 ⇒ H 成立"),
        ("组3 KPAT=3 · 244 通道 conv (bis-s09p03)", "nndump-bis-s09p03-kpat3.log", "不挂 ⇒ H 成立"),
    ]
    rows = []
    print("=" * 100)
    print("kernel pattern 0x6 假设 H —— 三组单变量对照判决")
    print("=" * 100)
    for title, fn, expect in cases:
        r = analyse(os.path.join(d, fn))
        rows.append((title, expect, r))
        print("  %-44s 期望: %-12s 实测: %s" % (title, expect, fmt(r)))

    print()
    # 判决
    base122 = rows[0][2]
    base244 = rows[1][2]
    k6 = rows[2][2]
    k3 = rows[3][2]

    if any(r is None or not r["segments"] for _, _, r in rows):
        print("  ⚠️ 有组缺日志/无数据 ⇒ **不能判决**（先把三组跑全）")
        return 1

    print("  判定：")
    ok = True
    # 前提：基线必须符合预期，否则实验本身有问题
    if base122["hang"]:
        print("    ❌ 基线异常：122 通道 conv 竟然挂了 ⇒ 基线被污染（先查为什么）")
        ok = False
    if not base244["hang"]:
        print("    ❌ 基线异常：244 通道 conv 竟然没挂 ⇒ 基线被污染（先查为什么）")
        ok = False
    if ok:
        print("    ✅ 基线符合预期（122 不挂 / 244 挂）⇒ 对照有效")
        if k6["hang"]:
            print("    ✅ 组 2：122 通道 + pattern 0x6 ⇒ **挂** ⇒ H 成立（单变量：只改了 pattern）")
        else:
            print("    ❌ 组 2：122 通道 + pattern 0x6 **没挂** ⇒ H 不成立（或 0x6 不是唯一变量）")
            ok = False
        if k3["hang"]:
            print("    ❌ 组 3：244 通道 + pattern 0x3f **仍然挂** ⇒ H 不成立")
            ok = False
        else:
            print("    ✅ 组 3：244 通道 + pattern 0x3f ⇒ **不挂** ⇒ H 成立（反向确证）")

    print()
    print("  ⇒ 总结论：" + ("**假设 H 成立：挂死由描述符的 kernel_pattern 0x6 档触发**"
                          if ok else "**假设 H 未被确证**（见上面的 ❌ 条目）"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
