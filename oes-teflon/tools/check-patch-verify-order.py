#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态门：build-teflon.sh 里 `--verify` 必须在**打补丁之前**执行。

## 为什么要有这道门

`gen-*-patch.py --verify` 回答的是「已入库补丁器的 `OLD_*` 锚点，
是否与**未打补丁的**真源码逐字一致」。

打完补丁之后，目标函数已被整段替换，`OLD_*` 自然失配 —— 报出来的
「锚点不符」是**假警报**，而它会让 build-tflon.sh 直接 exit 1，
白白烧掉一整轮 CI。

真实事故（run 39226436430）：

    ERROR: calc_quantization 内 `if (poperation->conv.depthwise)` 实得 0（期望 1）
    ERROR: 布局补丁锚点与真实源码不符

同一段注释在 `.github/workflows/build-teflon-fix-rofix-perchannel.yml` 里
早就写过了（Preflight 不能放真源码锚点核对，因为那时 Mesa 还没下载；
post-build 又因为已打补丁而失配），本轮还是在脚本里重犯 ——
**说明「写下来」不够，必须让机器查。**

## 判据

对 build-teflon.sh 里每一处 `gen-<x>-patch.py ... --verify` 调用：
它引用的 `--out <补丁器>` 所对应的 `patch-<x>*.py` 的**调用点**
必须在文件里出现在**更靠后**的位置。

用法：python3 tools/check-patch-verify-order.py [--file <build-teflon.sh>]
"""

import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "build-recipe", "build-teflon.sh")

# `--verify` 调用行 + `--out` 指向的补丁器
RE_VERIFY = re.compile(r"gen-([\w-]+?)-patch\.py")
RE_PATCH = re.compile(r"patch-[^\s\"']*?-([\w-]+?)\.py")


def assignments(lines):
    """收集 `NAME="$SCRIPT_DIR/patch-....py"` 这类间接引用的变量名。"""
    out = {}
    for l in lines:
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)=\"\$SCRIPT_DIR/(patch-[\w-]+\.py)\"", l)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT)
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        sys.stderr.write("ERROR: 找不到 %s" % args.file)
        return 2

    src = open(args.file, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    lines = src.split("\n")
    var2patch = assignments(lines)
    inv_vars = {v: k for k, v in var2patch.items()}

    bad = []
    checked = set()
    for i, l in enumerate(lines):
        m = RE_VERIFY.search(l)
        if not m:
            continue
        if "--verify" not in "\n".join(lines[i:i + 4]):
            continue
        tag = m.group(1)
        if tag in checked:
            continue
        checked.add(tag)

        fname = None
        for fn in var2patch.values():
            if re.search(r"%s\.py$" % re.escape(tag), fn):
                fname = fn
                break
        if fname is None:
            bad.append("第 %d 行：gen-%s-patch.py --verify 找不到对应的 patcher 文件名"
                       % (i + 1, tag))
            continue

        hits = []
        guard = re.compile(r"\[\s*!?\s*-f\s+\"?\$%s" % re.escape(inv_vars[fname]))
        chatter = re.compile(r"^\s*(echo|printf|read|export)\b")
        for j, x in enumerate(lines):
            if re.search(r"^\s*[A-Z_][A-Z0-9_]*=", x):
                continue                       # 这是赋值，不是调用
            if guard.search(x):
                continue                       # 存在性守卫（if [ ! -f ... ]）不是调用
            if chatter.match(x):
                continue                       # echo/printf 里提到变量不算调用
            if fname in x or ("$" + inv_vars[fname]) in x:
                hits.append(j)
        if not hits:
            bad.append("第 %d 行：gen-%s-patch.py --verify 找不到 %s 的调用点"
                       % (i + 1, tag, fname))
            continue
        first = min(hits)
        if first < i:
            bad.append("第 %d 行的 gen-%s-patch.py --verify 在**打补丁之后**"
                       "（%s 的调用在第 %d 行）—— 打完补丁锚点必然失配，"
                       "会白烧一轮 CI" % (i + 1, tag, fname, first + 1))

    print("=== 锚点复核顺序门（--verify 必须在打补丁之前）===")
    if not checked:
        print("  WARNING: 没找到任何 --verify 调用点 —— 判据可能已失效，请复核本工具")
        return 2
    for b in bad:
        print("  FAIL: " + b)
    if not bad:
        print("  共核对 %d 处 --verify，顺序全部正确" % len(checked))
        print("PATCH-VERIFY-ORDER: PASS")
        return 0
    print("PATCH-VERIFY-ORDER: FAIL（%d）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())