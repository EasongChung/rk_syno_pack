#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态门：核对「未打补丁原文」的工具必须排在**打补丁之前**执行。

## 为什么要有这道门

有两类检查读的是**原始** Mesa 源码：

* `gen-<x>-patch.py ... --verify` —— 回答「已入库补丁器的 `OLD_*` 锚点
  是否与未打补丁的真源码逐字一致」；
* `selftest-perchannel-layout.py` —— 第一步就调 `--verify`。

打完补丁之后，目标函数已被整段替换，`OLD_*` 必然命中 0 次 ⇒ 报出来的
「锚点不符」是**假警报**，而 build-teflon.sh 会直接 exit 1 ⇒ 整轮 CI 白烧。

★ 同一个坑连栽两次：

    run 39226436430 —— --verify 排在打补丁之后
    run 39227152464 —— 结构自测排在打补丁之后（它内部也跑 --verify）

两处当时都有注释写着「真源码锚点核对不能放在打补丁之后」，
所以**写进注释的纪律不叫纪律，能让机器查的才叫**。

## 判据

对 build-teflon.sh 里每一处「读原始源码」的工具调用，
它所对应的补丁器调用点必须出现在文件里**更靠后**的位置。

## 正/负控制

    python3 tools/check-patch-verify-order.py            # 正向：真实脚本应 PASS
    python3 tools/selftest-check-patch-verify-order.py   # 正+负控制

用法：python3 tools/check-patch-verify-order.py [--file <build-teflon.sh>]
"""

import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "build-recipe", "build-teflon.sh")

# ① gen-<x>-patch.py ... --verify：直接核对 OLD_* 是否命中原文
# ② selftest-perchannel-layout.py：内部第一步就是 --verify，同样要求原文
RE_PRISTINE = re.compile(r"(gen-[\w-]+?-patch\.py|selftest-perchannel-layout\.py)")


def assignments(lines):
    """收集 `NAME="$SCRIPT_DIR/patch-....py"` 这类间接引用。"""
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
    guard_tpl = r"\[\s*!?\s*-f\s+\"?\$%s"
    chatter = re.compile(r"^\s*(echo|printf|read|export)")
    assign = re.compile(r"^\s*[A-Z_][A-Z0-9_]*=")
    argline = re.compile(r"^\s*--?[A-Za-z]")   # 续行里的命令行参数，不是调用

    # 补丁器调用点：文件名或 "$变量" 出现在**非守卫/非回显/非赋值/非参数**的行上
    patcher_hits = {}
    for j, x in enumerate(lines):
        if assign.match(x) or chatter.match(x) or argline.match(x):
            continue
        for fn, var in inv_vars.items():
            if fn not in x and ("$" + var) not in x:
                continue
            if re.search(guard_tpl % re.escape(var), x):
                continue          # 存在性守卫（if [ ! -f "$VAR" ]）不是调用
            patcher_hits.setdefault(fn, []).append(j)

    bad = []
    checked = set()
    for i, l in enumerate(lines):
        m = RE_PRISTINE.search(l)
        if not m:
            continue
        tool = m.group(1)
        if chatter.match(l) or assign.match(l):
            continue
        window = "\n".join(lines[i:i + 4])
        if tool.startswith("gen-") and "--verify" not in window:
            continue
        if tool in checked:
            continue
        checked.add(tool)
        tag = (tool[len("gen-"):-len("-patch.py")] if tool.startswith("gen-")
               else "perchannel-layout")

        fname = None
        for fn in var2patch.values():
            if re.search(r"%s\.py$" % re.escape(tag), fn):
                fname = fn
                break
        if fname is None:
            bad.append("第 %d 行的 %s 找不到对应的 patcher 文件名" % (i + 1, tool))
            continue
        hits = patcher_hits.get(fname) or []
        if not hits:
            bad.append("第 %d 行的 %s 找不到 %s 的调用点" % (i + 1, tool, fname))
            continue
        first = min(hits)
        if first < i:
            bad.append("第 %d 行的 %s 在**打补丁之后**（%s 的调用在第 %d 行）"
                       "—— 打完补丁锚点必然失配，会白烧一轮 CI"
                       % (i + 1, tool, fname, first + 1))

    print("=== 锚点复核顺序门（读原始源码的检查必须在打补丁之前）===")
    if not checked:
        print("  WARNING: 没找到任何受检调用点 —— 判据可能已失效，请复核本工具")
        return 2
    for b in bad:
        print("  FAIL: " + b)
    if not bad:
        print("  共核对 %d 处，顺序全部正确：%s"
              % (len(checked), ", ".join(sorted(checked))))
        print("PATCH-VERIFY-ORDER: PASS")
        return 0
    print("PATCH-VERIFY-ORDER: FAIL（%d）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())