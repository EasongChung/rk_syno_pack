#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
libteflon.so 产物独立校验器 —— 不信任构建方自述，产物自己说话。

设计要点（全部针对「假绿」）
--------------------------
1. **探针自证**：先用必然存在的 ELF 结构串验证 strings 探针有效。
   探针失效时报 PROBE_UNTRUSTED 并 exit 2，**绝不输出 PASS**。
   —— 「六个都不在」若探针坏了，就是假绿。
2. **正反对照**：补丁器的正向标记必须**在**产物里；
   诊断补丁的标记必须**不在**。两类结论互为对方的有效性证明。
3. **一条 PASS 必带证据行**：任何一条结论都打印它匹配到的原文。
   拿不出证据的结论视为未通过（规范 §2.4.2）。
4. **未知标记不放过**：未登记在册的 `och-` 标记一律列出，
   避免「新增了一个补丁但没告知」。

用法
----
    python3 verify_artifact.py <libteflon.so> [--json out.json]
退出码：0 全通过 / 1 有失败项 / 2 探针不可信（不给结论）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

# ---- 探针自证用：ELF 结构里必然出现的串（任何正常 .so 都有）
PROBE_SELFCHECK = [".rodata", "GLIBC", "__cxa_finalize"]

# ---- 正向：per-channel 补丁器被应用的标记
# 这些串由 patch-etnaviv-perchannel-quant.py 写进产物源码。
MUST_PRESENT = [
    "och-patch-perchannel-quant",
    # 布局契约修法（calc_quantization / requantize_weights / v7 写回）。
    # 由 build-teflon.sh 以 shell 追加方式写进 etnaviv_ml_nn.c，符号名 och_pcl_marker。
    # ★ 真实样本负控制：run 39229309524 的产物（b8681800）**没有**这个标记，
    #   跑本校验器应当报缺 —— 这是「这道门能红」的实证，不是假设。
    "och-patch-perchannel-layout",
]

# ---- 反向：诊断类补丁的标记，它们「不该」在本次构建的产物里
MUST_ABSENT = [
    "och-patch-alias-diag",
    "[och-alias]",
    "och-patch-creatediag",
    "[och-create]",
    "[och-wc]",
    "[och-ro]",
]

# ---- 已登记的**合法**标记（出现在产物里是正常的，不算「偷偷加补丁」）
REGISTERED_OTHER = [
    "och-fix",                      # OCFIX
    "och-patch-v7-outchannel-clamp",  # OCFIX 的尾部越界钳制
    "och-rofix",                    # ROFIX 的读回路径打印
    "och-patch-perchannel-quant",   # 已列在 MUST_PRESENT
]

STRINGS_RE = re.compile(rb"[ -~]{4,}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def strings_of(path):
    """提取可打印串（≥4 字符，7-bit）。仅用于「未登记标记扫描」。"""
    with open(path, "rb") as fh:
        data = fh.read()
    return STRINGS_RE.findall(data)


def find(needles, raw):
    """在**原始字节**上匹配 needle。

    ★ 必须字节级，不能用 strings 提取结果：
      GNU strings 默认 7-bit，含非 ASCII 的探针会在 UTF-8 字节处被切断并**静默漏检**。
      本工具曾用「提取串再搜」的做法，属同一陷阱，已修（见 test_verify_artifact.py
      的 unicode 负控制用例 —— 若退回 7-bit 提取，那个用例必然失败）。
    """
    hits = {}
    for n in needles:
        nb = n.encode("utf-8")
        if nb not in raw:
            continue
        ev, idx = [], 0
        while len(ev) < 3:
            p = raw.find(nb, idx)
            if p < 0:
                break
            seg = raw[max(0, p - 30): p + 130]
            ev.append(seg.decode("utf-8", "replace")
                      .replace("\n", "\\n").replace("\x00", ".").strip())
            idx = p + 1
        hits[n] = ev
    return hits


def elf_info(path):
    """用 readelf 取架构与节信息；失败返回 (None, err)。"""
    try:
        out = subprocess.run(["readelf", "-h", path], capture_output=True,
                             text=True, timeout=120).stdout
        machine = re.search(r"Machine:\s*(.+)", out)
        cls = re.search(r"Class:\s*(.+)", out)
        return {
            "machine": machine.group(1).strip() if machine else None,
            "class": cls.group(1).strip() if cls else None,
        }, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sofile")
    ap.add_argument("--json", help="把结果写成 JSON")
    ap.add_argument("--extra-present", action="append", default=[],
                    help="额外要求的正向标记（可多次）")
    ap.add_argument("--extra-absent", action="append", default=[],
                    help="额外要求的反向标记（可多次）—— 自测注入用")
    a = ap.parse_args()
    must_present = MUST_PRESENT + a.extra_present
    must_absent = MUST_ABSENT + a.extra_absent

    path = a.sofile
    print("=" * 74)
    print("libteflon.so 产物独立校验")
    print("=" * 74)

    if not os.path.isfile(path):
        print(f"[FATAL] 文件不存在: {path}")
        return 1

    size = os.path.getsize(path)
    digest = sha256(path)
    info, err = elf_info(path)
    print(f"文件   : {path}")
    print(f"大小   : {size:,} 字节")
    print(f"sha256 : {digest}")
    if info:
        print(f"架构   : {info.get('class')} / {info.get('machine')}")
    else:
        print(f"架构   : 读取失败（{err}）—— 若目标为 A311D 的 aarch64，此项需人工确认")
    print()

    # ---- 步骤 1：探针自证
    print("--- [1] 探针自证（防假绿的第一道闸）---")
    with open(path, "rb") as fh:
        raw = fh.read()
    blob = strings_of(path)
    print(f"  可打印串 {len(blob):,} 条（≥4 字符）；标记匹配走**字节级**（非 7-bit 提取）")
    selfhits = find(PROBE_SELFCHECK, raw)
    ok_probe = 0
    for s in PROBE_SELFCHECK:
        mark = "命中" if s in selfhits else "未命中"
        print(f"    {s:<20} {mark}")
        if s in selfhits:
            ok_probe += 1
    if ok_probe < 2:
        print()
        print("  [PROBE_UNTRUSTED] 探针无法在正常 ELF 里找到足够证据。")
        print("  ⇒ 后续所有「未找到」都不能证明「不存在」。**本工具拒绝给结论。**")
        print("  ⇒ 不输出 PASS / FAIL。exit 2。")
        if a.json:
            json.dump({"verdict": "PROBE_UNTRUSTED", "selfcheck": selfhits},
                      open(a.json, "w"), ensure_ascii=False, indent=2)
        return 2
    print(f"  ⇒ 探针有效（{ok_probe}/{len(PROBE_SELFCHECK)} 自证命中），可以下结论")
    print()

    results = {"file": path, "size": size, "sha256": digest, "elf": info,
               "probe_ok": ok_probe, "checks": []}

    # ---- 步骤 2：正向标记（补丁被应用的证据）
    print("--- [2] 正向标记：per-channel 补丁必须已应用 ---")
    pres = find(must_present, raw)
    n_pass = 0
    for n in must_present:
        if n in pres:
            n_pass += 1
            print(f"  [PASS] {n}")
            for e in pres[n]:
                print(f"         证据: {e[:110]}")
        else:
            print(f"  [FAIL] {n} —— 未在产物中找到")
            print("         含义：补丁**可能没被应用**（编译通过 ≠ 补丁生效）")
    print()

    # ---- 步骤 3：反向标记（诊断补丁必须缺席）
    print("--- [3] 反向标记：诊断补丁标记必须缺席 ---")
    abs_ = find(must_absent, raw)
    n_abs_ok = 0
    for n in must_absent:
        if n in abs_:
            print(f"  [FAIL] {n} —— 竟然出现了")
            for e in abs_[n]:
                print(f"         证据: {e[:110]}")
        else:
            n_abs_ok += 1
            print(f"  [PASS] {n} —— 缺席")
    print()
    if n_pass == 0:
        print("  ★ 注意：反向门此刻**缺乏有效性依据** —— 正向标记都没找到，")
        print("    无法证明「探针能在这个产物里找到东西」，所以上面的『缺席』不可采信。")
        print()

    # ---- 步骤 4：未登记的 och- 标记
    print("--- [4] 未登记的 och- 标记（防「偷偷加补丁」）---")
    joined = b"\n".join(blob)
    found_och = set(re.findall(rb"och-[a-z0-9_-]+", joined))
    known = (set(x.encode() for x in must_present)
             | set(x.encode() for x in must_absent)
             | set(x.encode() for x in REGISTERED_OTHER))
    unknown = sorted(x.decode() for x in found_och - known)
    if unknown:
        for u in unknown:
            print(f"  [注意] 未登记: {u}")
    else:
        print("  无未登记的 och- 标记")
    print()

    # ---- 结论
    total_fail = (len(must_present) - n_pass) + (len(must_absent) - n_abs_ok)
    print("--- [结论]---")
    if n_pass == len(must_present) and n_abs_ok == len(must_absent):
        print("  全部通过：per-channel 补丁**已应用**，诊断补丁**未应用**。")
        verdict = "PASS"
        rc = 0
    else:
        print(f"  失败项 {total_fail} 个。详见上方逐条。")
        verdict = "FAIL"
        rc = 1
    print()
    print("  ★ 本结论只回答「补丁有没有被应用」。")
    print("    **不回答**「接管数是否恢复」「输出是否与 CPU 逐字节一致」。")
    print("    后者是四级判据 ② 与 ④，须另行实测，不得由此推断。")

    results.update(verdict=verdict, unknown_och=unknown,
                   must_present_hits=pres, must_absent_hits=abs_)
    if a.json:
        json.dump(results, open(a.json, "w"), ensure_ascii=False, indent=2)
        print(f"\n  JSON 已写: {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
