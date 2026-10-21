#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_artifact.py 的回归自测 —— 正控制 + 负控制 + 三态退出码。

纪律要求：任何门/检查器都必须有「正控制 + 至少一个负控制 + 一次回归」三样，
且**负控制必须包含真实样本**，不能只有合成样本（合成样本容易和实现同源同错）。

样本来源
--------
真实负控制 : /root/clawtemp/o2tf-src/art/libteflon.so
             已知状态：aarch64 未 strip 构建，含全部 6 个诊断标记、
             **不含** per-channel 标记（它早于 per-channel 补丁）。
             这是唯一能证明「反向门不是空话」的样本 —— 合成样本做不到。
合成样本   : ideal / leak1 / allnoise / garbage，覆盖四个判定分支。

运行
----
    python3 tools/test_verify_artifact.py
    python3 tools/test_verify_artifact.py --real /path/to/real.so   # 换真实样本
退出码：0 全部符合预期
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "verify_artifact.py")
DEFAULT_REAL = "/root/clawtemp/o2tf-src/art/libteflon.so"

ELFISH = (b"\x7fELF\x02\x01\x01\x00"
          b".rodata\x00GLIBC\x00__cxa_finalize\x00" + b"\x00" * 64)
NOISE = ["och-patch-alias-diag", "[och-alias]", "och-patch-creatediag",
         "[och-create]", "[och-wc]", "[och-ro]"]


def make(tmp, name, payload):
    p = os.path.join(tmp, name)
    with open(p, "wb") as fh:
        fh.write(payload)
    return p


def run(path, extra=()):
    r = subprocess.run([sys.executable, TOOL, path, *extra],
                       capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout


def expect(label, path, want_rc, want_markers, forbid_markers=(), extra=()):
    rc, out = run(path, extra)
    problems = []
    if rc != want_rc:
        problems.append(f"退出码 {rc} != 期望 {want_rc}")
    for m in want_markers:
        if m not in out:
            problems.append(f"输出缺少标记 {m}")
    for m in forbid_markers:
        if m in out:
            problems.append(f"输出不应出现标记 {m}")
    status = "OK  " if not problems else "FAIL"
    print(f"  [{status}] {label:<34} rc={rc} (期望 {want_rc})")
    for p in problems:
        print(f"         · {p}")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=DEFAULT_REAL)
    a = ap.parse_args()

    print("=" * 70)
    print("verify_artifact.py 回归自测")
    print("=" * 70)
    ok = []

    with tempfile.TemporaryDirectory() as tmp:
        print("\n--- 正控制 ---")
        ideal = make(tmp, "ideal.so", ELFISH +
                    b"och-patch-perchannel-quant\x00"
                    b"och-patch-perchannel-layout\x00 note\n")
        ok.append(expect("理想产物（全符合）", ideal, 0,
                         ["全部通过"], ["PROBE_UNTRUSTED"]))

        # ★ 新增要求的负控制：有 quant 标记、缺 layout 标记 ⇒ 门必须能红
        nolayout = make(tmp, "nolayout.so", ELFISH +
                        b"och-patch-perchannel-quant\x00 note\n")
        ok.append(expect("缺 och-patch-perchannel-layout", nolayout, 1,
                         ["[FAIL] och-patch-perchannel-layout"], ["全部通过"]))

        print("\n--- 负控制：合成样本 ---")
        leak = make(tmp, "leak1.so", ELFISH +
                    b"och-patch-perchannel-quant\x00[och-ro] applied\n")
        ok.append(expect("泄漏 1 个诊断标记", leak, 1,
                         ["[FAIL] [och-ro]"], ["全部通过"]))

        noise = make(tmp, "allnoise.so", ELFISH +
                     b"".join(x.encode() + b"\x00" for x in NOISE))
        ok.append(expect("全是诊断标记、无 per-channel", noise, 1,
                         ["缺乏有效性依据"], ["全部通过"]))

        garbage = make(tmp, "garbage.bin", bytes(range(256)) * 4)
        ok.append(expect("非 ELF 垃圾 -> 拒绝给结论", garbage, 2,
                         ["PROBE_UNTRUSTED"], ["全部通过"]))

        # ★ 这条是本工具自身 bug 的专项回归。
        # 曾用「7-bit 提取可打印串、再在提取结果里搜」。GNU strings 默认 7-bit，
        # 含非 ASCII 的探针会在 UTF-8 字节处被切断并**静默漏检**。
        # 现已改为原始字节匹配。若有人退回旧实现，此用例必然失败。
        uni = "och-非ASCII标记-不一致"
        uniso = make(tmp, "unicode.so",
                     ELFISH + b"och-patch-perchannel-quant\x00" +
                     uni.encode("utf-8") + b"\x00\n")
        ok.append(expect("非 ASCII 标记必须能检出（字节级）", uniso, 1,
                         ["[FAIL] " + uni],
                         ["全部通过"], extra=["--extra-absent", uni]))

        print("\n--- 负控制：真实样本（最关键的一项）---")
        if os.path.isfile(a.real):
            ok.append(expect("真实老构建 77110f33", a.real, 1,
                             ["探针有效", "[FAIL] och-patch-perchannel-quant",
                              "竟然出现了"], ["全部通过"]))
        else:
            print(f"  [SKIP] 真实样本不存在: {a.real}")
            print("         该项是唯一能证伪「反向门是空话」的样本，建议补上。")

    n = len(ok)
    n_ok = sum(ok)
    print("\n" + "=" * 70)
    print(f"结果: {n_ok}/{n} 通过")
    if n_ok != n:
        print("★ 有用例不符预期 —— 门或实现有问题，不要放过。")
    print("=" * 70)
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
