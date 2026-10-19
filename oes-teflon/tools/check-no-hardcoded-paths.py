#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态门：脚本里**不许**出现写死的本地绝对路径。

## 为什么

写死 `D:\\...` / `C:\\...` 的脚本在 CI（Linux）上会以「文件不存在」失败，
而失败常常伪装成**别的问题**：

    run 39227152464 —— `--src-root` 用了相对路径，本地过、CI 上目录不存在
    run 39228292468 —— GEN/PATCH 写死了本地 Windows 路径，
                        子进程返回非 0，报出来的是「补丁打不上」

★ 同一个 bug 类，本项目连栽三轮 CI。**一条命令就能拦住的，不该用一轮 CI 去发现。**

判据：扫 `oes-teflon/tools/*.py` 与 `oes-teflon/build-recipe/*.py` 的
**字符串字面量**（AST Constant），命中即 FAIL：

| 模式 | 拦什么 |
|---|---|
| `(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_]` | `D:\\clawtemp`、`C:/Users` |
| `/(home\|Users)/[A-Za-z0-9_.-]+/` | `/home/xxx/`、`/Users/xxx/` |

★ **左邻断言是必须的**：没有它，`RESULT:\\s*ALL\\s*PASS` 里的 `T:\`、
`https://github...` 里的 `s://` 都会被误判（第一版就栽在这）。
★ 只扫 `.py`：`.sh` 不是 Python，`ast.parse` 会报语法错。
★ 只查字面量不查注释 —— 注释里写「本轮在 D:\\... 踩过坑」是正当的。

用法：python3 tools/check-no-hardcoded-paths.py
"""

import ast
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SELF = os.path.abspath(__file__)

TARGETS = [
    os.path.join(HERE),
    os.path.join(ROOT, "build-recipe"),
]

RE_WIN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_]")
RE_HOME = re.compile(r"/(home|Users)/[A-Za-z0-9_.-]+/")


def literals(path):
    """产出 (行号, 字符串字面量)。语法错误直接抛，不吞。"""
    src = open(path, encoding="utf-8-sig").read()
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def main():
    bad = []
    n_files = 0
    for d in TARGETS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(d, fn)
            # ★ 排除本文件：它的文档字符串里**必然**要举 `D:\\...` 作例子，
            #   否则就是自指误报。排除是正当的，但必须写明，不能默默跳过。
            if os.path.abspath(p) == SELF:
                continue
            n_files += 1
            try:
                for ln, s in literals(p):
                    if RE_WIN.search(s) or RE_HOME.search(s):
                        bad.append("%s:%d  %r" % (fn, ln, s[:90]))
            except SyntaxError as e:
                bad.append("%s  语法错误: %s" % (fn, e))

    print("=== 写死本地路径门 ===")
    print("  共扫 %d 个脚本" % n_files)
    for b in bad:
        print("  FAIL: " + b)
    if not bad:
        print("NO-HARDCODED-PATHS: PASS")
        return 0
    print("NO-HARDCODED-PATHS: FAIL（%d）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())