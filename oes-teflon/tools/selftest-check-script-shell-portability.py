# -*- coding: utf-8 -*-
"""check-script-shell-portability.py 的正/负控制。

正向：真实 build-recipe/*.sh 全部 PASS。
负向：注入 ${PIPESTATUS[0]}（bash-only）后必须转红并指出行号与原因。
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = os.path.join(HERE, "check-script-shell-portability.py")
SRC = os.path.join(ROOT, "build-recipe", "build-teflon.sh")
NL = chr(10)

fails = []
r = subprocess.run([sys.executable, GATE], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print("--- 正向 ---")
print(((r.stdout or "") + (r.stderr or "")).strip())
if r.returncode != 0:
    fails.append("正向控制失败：真实脚本应 PASS")

print()
print("--- 负向：注入 PIPESTATUS ---")
d = tempfile.mkdtemp(prefix="shport-")
neg = os.path.join(d, "neg.sh")
t = io.open(SRC, encoding="utf-8").read()
anchor = 'echo "==> [3/6] 获取 Mesa'
assert anchor in t, "找不到注入锚点"
t = t.replace(anchor, "_bad=${PIPESTATUS[0]}" + NL + anchor, 1)
io.open(neg, "w", encoding="utf-8", newline="").write(t)
r2 = subprocess.run([sys.executable, GATE, "--dir", d], capture_output=True, text=True,
                    encoding="utf-8", errors="replace")
out2 = (r2.stdout or "") + (r2.stderr or "")
print(out2.strip())
if r2.returncode == 0:
    fails.append("负向控制失败：注入 bash-only 构造后门仍报 PASS")
elif "PIPESTATUS" not in out2:
    fails.append("负向控制失败：门转红了但没指出 PIPESTATUS")

print()
print("SHELL-PORTABILITY SELFTEST: " + ("PASS" if not fails else "FAIL"))
for f in fails:
    print("  - " + f)
sys.exit(0 if not fails else 1)