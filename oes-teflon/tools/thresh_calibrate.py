#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值校准工具 —— 用实测分数分布建议 confidence_threshold。

背景
----
厂商阈值表是按厂商照片分布标定的（min 0.50 / 中位 0.85 / max 2.00）。
我们的照片实测分数普遍偏低，导致「识别不出标签」。
本工具**只做统计和建议，不改任何东西**：不改模型、不改代码、不写数据库。

策略（可审计，三条规则）
----------------------
R1 保留优先：某 label 的实测 max >= 厂商阈值  ->  保留厂商阈值不动。
R2 下调    ：实测 max <  厂商阈值  ->  建议值 = clamp(max*0.95, FLOOR, 厂商阈值)
R3 下限    ：FLOOR 默认 0.25，防止噪声类被抬到极低阈值而误标。

阈值 > 1.0 的类**在数学上永不可达**（模型最后算子是 LOGISTIC，输出严格 (0,1)），
本工具单独列出，不做任何静默处理。

输入
----
TSV，三列，无表头： unit_id <TAB> label <TAB> confidence
（由 psql 从 concept_rawdata.result 这个 json 数组展开得到）

用法
----
    python3 thresh_calibrate.py scores.tsv [--floor 0.25] [--top 20]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

# ---- 厂商原始阈值（来自 thresholds.txt，已核实与 DB concept.confidence_threshold 逐值一致）
# 335 类。完整表见 repo 内 docs/thresholds-335.txt；这里只硬编码需要展示的关键值，
# 其余由输入 TSV 的 label 与 vendor_thresh.tsv 合并得到。
VENDOR_MIN = 0.50
VENDOR_MEDIAN = 0.85
VENDOR_MAX = 2.00


def load_scores(path):
    """读入 TSV -> (per_label={label:{unit:conf}}, per_unit={unit:{label:conf}})"""
    per_label = defaultdict(dict)
    per_unit = defaultdict(dict)
    n_bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                n_bad += 1
                continue
            uid, label, conf = parts
            uid, label = uid.strip(), label.strip()
            try:
                v = float(conf)
            except ValueError:
                n_bad += 1
                continue
            per_label[label][uid] = v
            per_unit[uid][label] = v
    return per_label, per_unit, n_bad


def suggest(mx, vendor, floor):
    """返回 (建议阈值, 动作标签)"""
    if vendor is not None and vendor > 1.0:
        # 数学不可达：不静默处理，单独报告，保留厂商值
        return vendor, "UNREACHABLE"
    if vendor is None:
        return None, "NO_VENDOR"
    if mx >= vendor:
        return vendor, "KEEP"          # R1
    if mx < floor:
        # R3：实测信号连下限都不到 —— 下调也打不出来，属样本不足，不是阈值问题
        return vendor, "NO_SIGNAL"
    newv = min(vendor, max(mx * 0.95, floor))   # R2 + R3
    return round(newv, 4), "LOWER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="TSV: unit <TAB> label <TAB> confidence")
    ap.add_argument("--vendor", help="TSV: label <TAB> vendor_threshold（缺省则全部按无厂商值处理）")
    ap.add_argument("--floor", type=float, default=0.25, help="下调下限（默认 0.25）")
    ap.add_argument("--top", type=int, default=25, help="展示 top N 个有信号的类")
    a = ap.parse_args()

    per_label, per_unit, n_bad = load_scores(a.scores)
    units = sorted(per_unit, key=lambda s: int(s) if s.isdigit() else 0)

    vendor = {}
    if a.vendor:
        with open(a.vendor, encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2:
                    try:
                        vendor[p[0]] = float(p[1])
                    except ValueError:
                        pass

    print("=" * 78)
    print("阈值校准报告（只读统计，不修改任何文件）")
    print("=" * 78)
    print(f"样本照片数 : {len(units)}   单位: {units}")
    print(f"类别数     : {len(per_label)}")
    if n_bad:
        print(f"[警告] 跳过 {n_bad} 行无法解析")
    if not vendor:
        print("[注意] 未提供厂商阈值表（--vendor），无法区分 KEEP/LOWER；"
              "本次只统计实测分布。")
    print()

    # ---- 1. 不可达类（阈值 > 1.0）
    unreachable = [(lb, vendor.get(lb)) for lb in sorted(per_label)
                   if vendor.get(lb) is not None and vendor[lb] > 1.0]
    if unreachable:
        print(f"--- [1] 数学上永不可达的类（厂商阈值 > 1.0，模型输出严格 (0,1)）: {len(unreachable)} 个 ---")
        print("    这几类厂商语义可能是「禁用」。要启用必须把阈值改到 1.0 以下 —— 属改厂商标定，需人工决策。")
        for lb, th in unreachable:
            mx = max(per_label[lb].values())
            need = round(mx * 0.95, 4)
            print(f"    {lb:<20} 厂商={th:<5}  实测 max={mx:.4f}  "
                  f"需降到 <= {need} 才可能命中")
        print()

    # ---- 2. 有信号的类（实测 max > 0.05）
    sig = sorted(((max(d.values()), lb) for lb, d in per_label.items()), reverse=True)
    sig = [(mx, lb) for mx, lb in sig if mx > 0.05]

    print(f"--- [2] 实测有信号的类（max > 0.05）: {len(sig)} / {len(per_label)} ---")
    hdr = f"    {'label':<20} {'max':>8} {'次高':>8} {'厂商阈值':>8}  {'建议':>8}  动作"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    lowered, nosignal = [], []
    for mx, lb in sig[: a.top]:
        vals = sorted(per_label[lb].values(), reverse=True)
        second = vals[1] if len(vals) > 1 else 0.0
        vth = vendor.get(lb)
        sv, act = suggest(mx, vth, a.floor)
        act_cn = {"KEEP": "保留", "LOWER": "下调", "UNREACHABLE": "不可达",
                  "NO_SIGNAL": "信号不足", "NO_VENDOR": "无厂商值"}[act]
        if act == "LOWER":
            lowered.append((lb, vth, sv, mx))
        elif act == "NO_SIGNAL":
            nosignal.append((lb, vth, mx))
        print(f"    {lb:<20} {mx:>8.4f} {second:>8.4f} "
              f"{(f'{vth:.2f}' if vth is not None else '-'):>8}  "
              f"{(f'{sv:.2f}' if sv is not None else '-'):>8}  {act_cn}")
    if len(sig) > a.top:
        print(f"    ... 另有 {len(sig) - a.top} 个有信号类未展示（用 --top 调整）")
    print()

    # ---- 3. 建议阈值下每图标签数 vs 厂商阈值下每图标签数
    print("--- [3] 每图标签数对比 ---")
    def count_tags(use_suggest):
        out = {}
        for u in units:
            n = 0
            for lb, c in per_unit[u].items():
                vth = vendor.get(lb)
                if vth is None or vth > 1.0:
                    continue
                th = suggest(max(per_label[lb].values()), vth, a.floor)[0] if use_suggest else vth
                if th is not None and c >= th:
                    n += 1
            out[u] = n
        return out

    if vendor:
        base = count_tags(False)
        sugg = count_tags(True)
        print(f"    {'unit':<8} {'厂商阈值':>8} {'建议阈值':>8}  变化")
        for u in units:
            d = sugg[u] - base[u]
            print(f"    {u:<8} {base[u]:>8} {sugg[u]:>8}  {('+' if d > 0 else '' if d == 0 else '')}{d if d else ''}")
        print()

    # ---- 4. 结论
    print("--- [4] 结论 ---")
    if len(units) < 20:
        print(f"    [样本不足] 只有 {len(units)} 张照片。低于 20 张时，任何阈值调整都是过拟合。")
        print("             建议继续上传到 20-30 张再定结论。")
    print(f"    实测下建议下调的类共 {len(lowered)} 个：")
    for lb, vth, sv, mx in sorted(lowered, key=lambda t: -t[3])[:15]:
        print(f"       {lb:<20} {vth:.2f} -> {sv:.2f}   (实测 max {mx:.4f})")
    if len(lowered) > 15:
        print(f"       ... 另有 {len(lowered) - 15} 个")
    if nosignal:
        print(f"    信号不足（下调也没用，样本不够）{len(nosignal)} 个：")
        for lb, vth, mx in sorted(nosignal, key=lambda t: -t[2])[:8]:
            print(f"       {lb:<20} 厂商 {vth:.2f}  实测 max {mx:.4f}  < floor {a.floor}")
    if unreachable:
        print(f"    永不可达类 {len(unreachable)} 个需单独决策（降阈值 or 保留厂商语义）"
              f"—— 详见上方 [1] 段给出的目标值。")
    print()
    print("    本工具不写库、不改文件。执行修改前请人工确认上表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
