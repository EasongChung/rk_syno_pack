#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门：交付 workflow 里「构建日志断言」的每条字面量，都必须与源码里
**真正的 echo/print 输出**对得上。

## 为什么需要（两次真实失败）

- run `39207851035`：`Static check - PERCHANNEL docs consistency` 红（§2.4.1(c)）
- run `39209044465`：**第 23 步红在语法错误上**（`for pat in` 缺 `; do`），
  日志里一行 `OK:` 都没有

第 23 步的判据形式是 `grep -qF '<字面量>' build-fix.log`。
⇒ **字面量写错 = 这条断言永远命中不了 = CI 必红**，而且症状极难定位。

第一版只查「该字面量在源码文本里出现过」—— **不够**：
源码里有大段注释，注释里的字符串根本不会被打印。
⇒ 必须只认 `echo` / `printf` / `print(...)` 的**字符串实参**。

## 三类断言，判据不同

| 类别 | 判据 |
|---|---|
| **正向门**（补丁确实打上了） | 字面量必须出现在某个 `echo`/`print` 的实参里 |
| **调试门**（调试开关真的开了） | 同上 |
| **反向门**（作废/诊断方案不得出现） | 字面量**必须不**出现在任何 `echo`/`print` 实参里 |

分类靠脚本里的 `# 正向门` / `# 反向门` 注释切分，**不靠猜**。

## 用法

    python3 tools/check-log-asserts.py
    python3 tools/check-log-asserts.py --selftest
"""

import argparse
import glob
import os
import re
import sys

import yaml

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WF_DIR = os.path.join(REPO, ".github", "workflows")

FAILS = []


def rep(label, ok, detail=""):
    print("  %-62s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def strip_comments(src):
    out = []
    for ln in src.split("\n"):
        s = ln.lstrip()
        if s.startswith("#") and not s.startswith("#!"):
            continue
        res, q = [], None
        for i, ch in enumerate(ln):
            if q:
                res.append(ch)
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
                res.append(ch)
            elif ch == "#" and (not res or res[-1] in " \t"):
                break
            else:
                res.append(ch)
        out.append("".join(res))
    return "\n".join(out)


def _call_args(src, name):
    """提取 `name(...)` 的**括号配平**实参文本。

    ★ 第一版用 `print\\s*\\((.*)\\)` + re.S ⇒ **贪婪**匹配到文件里最后一个 `)`，
      于是从第一个 `print(` 到文件末尾**整段**都进了「语料」，
      任何字面量都「能找到」⇒ 门形同虚设（且它自己报的 84 条 PASS 是虚的）。
      ⇒ 改成手工扫描括号配平。
    """
    out = []
    for m in re.finditer(r"\b%s\s*\(" % re.escape(name), src):
        i = m.end() - 1
        depth = 0
        q = None
        while i < len(src):
            ch = src[i]
            if q:
                if ch == "\\":
                    i += 2
                    continue
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    out.append(src[m.end():i])
                    break
            i += 1
    return out


def printed_corpus():
    """仓库里所有 echo / printf / print(...) 的字符串实参拼接。"""
    parts = []
    files = (glob.glob(os.path.join(REPO, "oes-teflon", "build-recipe", "*"))
             + glob.glob(os.path.join(REPO, "oes-teflon", "tools", "*.py"))
             + glob.glob(os.path.join(REPO, "tools", "oes", "*.py")))
    for f in files:
        if os.path.isdir(f):
            continue
        src = strip_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in re.finditer(r"(?:echo|printf)\s+(.+)$", src, re.M):
            parts.append(m.group(1))
        parts.extend(_call_args(src, "print"))
    return "\n".join(parts)


def classify(run):
    """把脚本里的每个 `for pat in …; do` 块收集起来，按**前面最近的注释**分类。

    ★ 第一版的两个错：
      1) 只匹配 `for pat in` 那一行 ⇒ 多行列表只吃到第一行（13 条里漏 11 条）
      2) **反向门不该用语料判** —— 反向门里的串正是「被关掉的那个补丁器会打印的」，
         它出现在某个 print() 里是**正常**的；它「不该出现」指的是
         **在本次构建的日志里**不出现 —— 那是运行期事实，本地静态判不了。
         ⇒ 反向门一律列为「本地不可检」，既不假装通过也不假装失败。
    """
    fwd, dbg, rev = [], [], []
    cat = "fwd"
    block, in_block = [], False
    for ln in run.split("\n"):
        if "反向门" in ln:
            cat = "rev"
        elif "正向门" in ln:
            cat = "dbg"
        m = re.match(r"\s*for pat in (.*)$", ln)
        if m and not in_block:
            in_block, block = True, [m.group(1)]
            # ★ 单行写法 `for pat in 'A'; do` 的终止符就在**这一行**里，
            #   必须立刻检查；否则块永远不闭合 ⇒ 分类结果为空（第一版就这样，
            #   而它的 selftest 正好抓到了）。
            if "; do" not in block[0]:
                continue
            lits = re.findall(r"'([^']*)'", "\n".join(block))
            {"fwd": fwd, "dbg": dbg, "rev": rev}[cat].extend(lits)
            in_block, block = False, []
            continue
        if in_block:
            block.append(ln)
            if "; do" in ln:
                lits = re.findall(r"'([^']*)'", "\n".join(block))
                {"fwd": fwd, "dbg": dbg, "rev": rev}[cat].extend(lits)
                in_block, block = False, []
    return fwd, dbg, rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    corpus = printed_corpus()
    if args.selftest:
        print("=== 自测：分类与判据本身 ===")
        demo = ("x=1\n# 正向门\nfor pat in 'AAA'; do\n echo \"$pat\"; done\n"
                "# 反向门\nfor pat in 'BBB'; do\n echo \"$pat\"; done\n")
        f, d, r = classify(demo)
        # 注意：`# 正向门` 在 classify 里归到 **dbg** 桶（它就是「调试开关」那段）
        rep("分类：'正向门' 段的 AAA 进 dbg 桶", d == ["AAA"], repr((f, d, r)))
        rep("分类：'反向门' 段的 BBB 进 rev 桶", r == ["BBB"], repr(r))
        rep("分类：fwd 桶此时为空", f == [], repr(f))
        rep("语料非空（否则所有判据都会假 PASS）", len(corpus) > 1000,
            "corpus=%d 字符" % len(corpus))
        rep("★ 负控制：一个从未出现在 echo 里的串，正向判据必须判红",
            "ZZZ_NEVER_PRINTED_ZZZ" not in corpus)
        print()
        if FAILS:
            print("SELFTEST RESULT: %d FAIL" % len(FAILS))
            return 1
        print("SELFTEST RESULT: ALL PASS")
        return 0

    total = 0
    for f in sorted(glob.glob(os.path.join(WF_DIR, "*.yml"))):
        name = os.path.basename(f)
        d = yaml.safe_load(open(f, encoding="utf-8"))
        for job in (d.get("jobs") or {}).values():
            for st in (job.get("steps") or []):
                nm = st.get("name", "")
                run = st.get("run") or ""
                if "for pat in" not in run or "build-fix.log" not in run \
                        and "$L" not in run:
                    continue
                if "grep -qF" not in run:
                    continue
                fwd, dbg, rev = classify(run)
                if not (fwd or dbg or rev):
                    continue
                print("=== %s :: %s ===" % (name, nm))
                for p in fwd + dbg:
                    total += 1
                    rep("正向/调试门 必须会被打印: " + p[:44], p in corpus)
                if rev:
                    print("  · 反向门 %d 条（本地**不可检**，需运行期确认）：%s"
                          % (len(rev), ", ".join(x[:26] for x in rev)))
                print()

    print("共核对 %d 条日志断言" % total)
    if FAILS:
        print("RESULT: %d FAIL —— 对不上的断言在 CI 上必然红" % len(FAILS))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
