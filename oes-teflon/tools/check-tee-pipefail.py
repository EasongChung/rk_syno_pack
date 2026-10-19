#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态门：凡 `run:` 块里出现 `| tee` 的，必须开 `pipefail`。

## 为什么

`cmd 2>&1 | tee log` 的退出码取自 **`tee`**（恒为 0），不是 `cmd`。
没有 `pipefail` 时 `cmd` 的 `exit 1` 被完全吞掉，`set -e` 也不触发
（整条管道的状态是 0）。

⇒ **该步骤的 `success` 与它真正做的事无关。**

真实事故：run `39227152464` / `39228292468` 的 `Build libteflon.so` 都报
**success**，而 `build-fix.log` 里明明白白写着
`ERROR: per-channel 布局补丁结构自测未通过` —— 脚本在那里 `exit 1`，
`libteflon.so` **根本没产出**。两次红因此从两个看起来毫不相关的步骤漏出来
（一次像「产物有问题」，一次像「补丁器坏了」），把病因掩盖成两个症状。

★ 全仓扫描：这类 build 步共 **8 个 / 8 个 workflow**（不是 1 个）。

## 判据

每个 workflow 的每个 `run:` 块：含 `| tee` ⇒ 必须含 `pipefail`，
除非属于下面三种**声明过的**例外（不是 bug，但要显式）：

1. 该行自带 `|| true` —— 显式吞掉；
2. 管道出现在 `if` / `if !` 条件里 —— 退出状态由条件自己处理；
3. 该步 `continue-on-error: true` —— 成功与否本就不是判据
   （必须是显式声明，不能默默放过）。

用法：python3 tools/check-tee-pipefail.py [--dir <workflows 目录>]
"""

import argparse
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WF = os.path.join(ROOT, ".github", "workflows")

RE_TEE = re.compile(r"\|\s*tee\b")
RE_PIPEFAIL = re.compile(r"\bpipefail\b")
RE_OPT_OUT = re.compile(r"\|\|\s*true\b")
RE_IF = re.compile(r"^\s*(if|elif)\s.*\|\s*tee")
RE_CONT = re.compile(r"^\s*continue-on-error:\s*true")
RE_STEP = re.compile(r"^\s*-\s+name:")


def blocks(lines):
    """把 workflow 切成 (step 名, 行区间)。按 `- name:` 切。"""
    name, start = None, None
    for i, l in enumerate(lines):
        m = RE_STEP.match(l)
        if m:
            if name is not None:
                yield name, start, i
            name = l.strip()[len("- name:"):].strip()
            start = i
    if name is not None:
        yield name, start, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=WF)
    args = ap.parse_args()
    if not os.path.isdir(args.dir):
        sys.stderr.write("ERROR: 找不到 %s\n" % args.dir)
        return 2

    bad, n = [], 0
    for fn in sorted(os.listdir(args.dir)):
        if not fn.endswith(".yml"):
            continue
        p = os.path.join(args.dir, fn)
        lines = io.open(p, encoding="utf-8", errors="replace").read().split("\n")
        for name, lo, hi in blocks(lines):
            seg = lines[lo:hi]
            tee_lines = [l for l in seg if RE_TEE.search(l)]
            if not tee_lines:
                continue
            n += 1
            if any(RE_PIPEFAIL.search(l) for l in seg):
                continue
            if any(RE_OPT_OUT.search(l) or RE_IF.search(l) for l in tee_lines):
                continue
            if any(RE_CONT.match(l) for l in seg[:6]):
                continue
            bad.append("%s :: %s（第 %d 行起）—— 有 `| tee` 却没开 pipefail，"
                       "该步骤的 success 与命令的成败无关"
                       % (fn, name[:52], lo + 1))

    print("=== tee/pipefail 门 ===")
    print("  共扫 %d 个含 `| tee` 的 run 块" % n)
    for b in bad:
        print("  FAIL: " + b)
    if not bad:
        print("TEE-PIPEFAIL: PASS")
        return 0
    print("TEE-PIPEFAIL: FAIL（%d）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())