#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_hang_positions.py —— 从 tf_load_check 的 run 日志里抽出「哪段子图挂死」

用法:  tools/analyze_hang_positions.py <run.log> [更多日志...]
输出:  每轮 34 段的耗时序列、挂死位置集合、跨轮一致性判定

判据来源（写死在此，避免"看图说话"）：
  * 「一段子图」 = 日志里一行 `teflon: invoked graph, took N ms`
      —— 出处 /root/clawtemp/mesa_src/tfl_device.c:709
  * 「NPU 作业跑了多久」 = 日志里一行 `Running the NN job took N ms.`
      —— 出处 /root/clawtemp/mesa_src/etnaviv_ml.c:1036
  * 挂死判据 = 该段子图耗时 >= 400 ms
      —— 实测依据：内核 `recover hung GPU!` 与本日志的慢任务**条数一一相等**
         （per-channel 折维那次：内核 36 条 vs 日志 36 段慢任务，位置完全对应）
  * 轮次 = 每 34 段一次（子图数由 `teflon: compiling graph` 行数决定，不写死）

退出码: 0 = 解析成功（无论是否检出挂死）
"""
import re
import sys

RE_INV = re.compile(r'teflon: invoked graph, took (\d+) ms')
RE_CMP = re.compile(r'teflon: compiling graph: (\d+) tensors (\d+) operations')
RE_JOB = re.compile(r'Running the NN job took (\d+) ms')
HANG_MS = 400


def parse(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    nsub = sum(1 for l in lines if RE_CMP.search(l))
    inv, job = [], []
    for l in lines:
        m = RE_INV.search(l)
        if m:
            inv.append(int(m.group(1)))
        m = RE_JOB.search(l)
        if m:
            job.append(int(m.group(1)))
    return lines, nsub, inv, job


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    all_rounds = []
    for path in argv[1:]:
        lines, nsub, inv, job = parse(path)
        print("=" * 100)
        print("日志: %s" % path)
        print("  子图数(compiling graph)=%d  段调用(invoked graph)=%d  NPU 作业行=%d"
              % (nsub, len(inv), len(job)))
        if nsub == 0 or len(inv) % nsub != 0:
            print("  ⚠️ 段数与轮次不整除，无法按轮切分；按整体处理")
            rounds = [inv]
        else:
            rounds = [inv[i:i + nsub] for i in range(0, len(inv), nsub)]
        print("  轮次数: %d" % len(rounds))
        bands = []
        for r, seg in enumerate(rounds, 1):
            bad = [k + 1 for k, v in enumerate(seg) if v >= HANG_MS]
            bands.append(tuple(bad))
            print("  第%d轮 挂死段: %s" % (r, bad if bad else "（无）"))
            print("         耗时: %s" % " ".join(str(v) for v in seg))
        same = len(set(bands)) == 1
        print("  各轮挂死位置一致: %s" % ("是 ✅" if same else "否 ❌"))
        if same and bands[0]:
            print("  挂死段集合(1-based): %s" % list(bands[0]))
        all_rounds.append((path, bands, same))
    if len(all_rounds) == 2:
        a, b = all_rounds[0], all_rounds[1]
        if a[2] and b[2] and a[1] == b[1]:
            print("=" * 100)
            print("★ 两个日志的挂死位置集合**完全相同**: %s" % list(a[1][0]))
        elif a[2] and b[2]:
            print("=" * 100)
            print("★ 两个日志的挂死位置**不同**:")
            print("    %s → %s" % (a[0], list(a[1][0])))
            print("    %s → %s" % (b[0], list(b[1][0])))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
