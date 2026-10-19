#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check-tee-pipefail.py 的正/负控制。

正向：现有仓库必须 PASS。
负向：把一个 `set -euxo pipefail` 改回 `set -eux`（在有 `| tee` 的块里），
      门必须转红并指出该块。
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GATE = os.path.join(HERE, "check-tee-pipefail.py")
WF = os.path.join(ROOT, ".github", "workflows", "build-teflon-fix-rofix-perchannel.yml")
NL = chr(10)

fails = []

r = subprocess.run([sys.executable, GATE], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print("--- 正向 ---")
print(((r.stdout or "") + (r.stderr or "")).strip())
if r.returncode != 0:
    fails.append("正向控制失败：真实仓库应 PASS")

print()
print("--- 负向：把一个 pipefail 去掉 ---")
src = io.open(WF, encoding="utf-8").read()
pos = src.find("set -euxo pipefail")
assert pos >= 0, "找不到 pipefail"
bad = src[:pos] + "set -eux" + src[pos + len("set -euxo pipefail"):]
d = tempfile.mkdtemp(prefix="teepf-")
p = os.path.join(d, "build-teflon-fix-rofix-perchannel.yml")
io.open(p, "w", encoding="utf-8", newline="").write(bad)
r2 = subprocess.run([sys.executable, GATE, "--dir", os.path.dirname(p)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
out2 = (r2.stdout or "") + (r2.stderr or "")
print(out2.strip())
if r2.returncode == 0:
    fails.append("负向控制失败：去掉 pipefail 后门仍报 PASS")
elif "pipefail" not in out2:
    fails.append("负向控制失败：门转红了但没指出 pipefail")

print()
print("TEE-PIPEFAIL SELFTEST: " + ("PASS" if not fails else "FAIL"))
for f in fails:
    print("  - " + f)
sys.exit(0 if not fails else 1)