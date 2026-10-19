#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check-patch-verify-order.py 的正/负控制。

正向：真实 build-teflon.sh 必须 PASS。
负向：把「读原始源码的检查」块与「打补丁」块对调，门必须转红并指出行号。

★ 门自己写错就会出现假绿 —— 这个项目栽过 6 次，所以这道门也必须有负控制。
"""

import io
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = os.path.join(HERE, "check-patch-verify-order.py")
SH = os.path.join(ROOT, "build-recipe", "build-teflon.sh")
NL = chr(10)

fails = []


def run(path):
    r = subprocess.run([sys.executable, GATE, "--file", path],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


rc, out = run(SH)
print("--- 正向 ---")
print(out.strip())
if rc != 0:
    fails.append("正向控制失败：真实脚本应 PASS")

print()
print("--- 负向：把 --verify / 结构自测 块挪到打补丁之后 ---")
src = io.open(SH, encoding="utf-8", newline="").read()
lines = src.split(NL)
vi = next(i for i, l in enumerate(lines) if "--verify \\" in l)
vi_end = next(j for j in range(vi, len(lines)) if lines[j].strip() == "fi")
pi = next(i for i, l in enumerate(lines) if 'python3 "$PATCHER14"' in l)
pi_end = next(j for j in range(pi, len(lines)) if lines[j].strip() == "}")
blk_verify = lines[vi:vi_end + 1]
blk_patch = lines[pi:pi_end + 1]
swapped = lines[:vi] + blk_patch + [""] + lines[vi_end + 1:pi] + blk_verify + lines[pi_end + 1:]
d = tempfile.mkdtemp(prefix="cvo-")
bad = os.path.join(d, "build-teflon.sh")
io.open(bad, "w", encoding="utf-8", newline="").write(NL.join(swapped))
rc2, out2 = run(bad)
print(out2.strip())
if rc2 == 0:
    fails.append("负向控制失败：顺序对调后门仍报 PASS —— 门没生效")
elif "打补丁之后" not in out2:
    fails.append("负向控制失败：门转红了但没指出「在打补丁之后」")

print()
print("ORDER-GATE SELFTEST: " + ("PASS" if not fails else "FAIL"))
for f in fails:
    print("  - " + f)
sys.exit(0 if not fails else 1)