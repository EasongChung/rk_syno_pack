#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态门：shell 脚本不得使用与它**自己 shebang** 不兼容的语法。

## 为什么

`build-teflon.sh` 的 shebang 是 `#!/bin/sh`。GitHub 的 `run:` 块默认是 `bash`，
所以 `check-run-syntax.py` 用 `bash -n` 检查**永远绿**；
而 CI 直接执行该脚本时是 **dash** 解释，bash 专有语法当场炸。

真实事故（run 39230736466）：

    ./oes-teflon/build-recipe/build-teflon.sh: 975: Bad substitution

那一行用了 `${PIPESTATUS[0]}` —— bash 的数组展开，dash 不认。
★ 补丁器因此**根本没被调用**，而后面那道「补丁是否落地」的源码级硬门
  全部报 0 ⇒ 症状离病因十万八千里。

⇒ 判据：shebang 指向 `/bin/sh` 或 `/usr/bin/sh` 的脚本，
**不得**出现下列 bash-only 构造；`bash -e` 允许。
另外用**与 shebang 相符**的解释器做一次 `-n` 语法检查。

## 用法

    python3 tools/check-script-shell-portability.py [--dir <目录>]...
"""

import argparse
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# bash-only 构造 → 说明（**正则必须能定位到行**，便于输出证据行）
BASH_ONLY = [
    (r"\$\{PIPESTATUS\[", "PIPESTATUS 数组（dash 不支持；改用 >log 2>&1 + $? + cat）"),
    (r"\[\[\s", "[[ ]]（dash 不支持）"),
    (r"^\s*source\s", "source（dash 用 `.`）"),
    (r"\bdeclare\s+-[aA]", "declare -A（bash 关联数组）"),
    (r"\blocal\s+-[aA]", "local -A（bash 关联数组）"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*\[", "数组展开 ${x[i]}（dash 不支持）"),
    (r"<<-", "heredoc <<- 的 tab 剥离语义在 dash 下不同（低风险，仅提示）"),
    (r"\bfunction\s+\w+\s*\(\)", "function 关键字（POSIX 风格请用 name() {）"),
]


def shebang(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    m = re.match(r"#!\s*(\S+)", first)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None)
    args = ap.parse_args()
    dirs = args.dir or [os.path.join(ROOT, "build-recipe")]

    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".sh"):
                files.append(os.path.join(d, fn))

    print("=== shell 可移植性门（按 shebang 判别解释器）===")
    print("  共扫 %d 个 .sh" % len(files))
    bad = []
    for p in files:
        sb = shebang(p)
        base = os.path.basename(p)
        if sb is None:
            bad.append("%s: 没有 shebang" % base)
            continue
        if "bash" in sb:
            continue                      # bash 脚本不受 sh 约束
        if "sh" not in sb:
            print("  跳过 %s（shebang=%s，不是 sh 系）" % (base, sb))
            continue
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        for i, l in enumerate(lines, 1):
            if l.lstrip().startswith("#"):
                continue                  # 注释里的示例不算
            for pat, why in BASH_ONLY:
                if re.search(pat, l, re.M):
                    bad.append("%s:%d  %s  |  %s" % (base, i, why, l.strip()[:80]))
        # 用与 shebang 相符的解释器做语法检查
        exe = sb.lstrip("/bin/").split("/")[-1] if sb.startswith("/") else sb
        try:
            r = subprocess.run([exe, "-n", p], capture_output=True, text=True)
            tag = "%s -n %s" % (exe, base)
            if r.returncode == 0:
                print("  OK  : %s" % tag)
            else:
                bad.append("%s: %s -n 失败: %s" % (base, exe,
                                                   (r.stderr or "").strip()[:160]))
        except FileNotFoundError:
            print("  跳过 %s（本机没有 %s，无法做 -n 语法检查；"
                  "静态判据仍然生效）" % (base, exe))

    for b in bad:
        print("  FAIL: " + b)
    if not bad:
        print("SCRIPT-SHELL-PORTABILITY: PASS")
        return 0
    print("SCRIPT-SHELL-PORTABILITY: FAIL（%d）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())