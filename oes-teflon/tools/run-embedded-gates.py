#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各 workflow 里 `python3 - <<'PY'` 的**内嵌静态门**抽出来，在本地逐个真跑。

## 为什么需要（一次真实失败换来的）

`build-teflon-fix-rofix-perchannel.yml` 的 `Static check - PERCHANNEL docs consistency`
在 **run 39207851035** 上红了 —— 原因是 `build-teflon.sh` 头部少写了 3 处字面量
（`patch-etnaviv-perchannel-quant.py` / `input/output 仍严格` / `默认关闭`）。

而我在 dispatch 前**只验了那个步骤「存在」**，没验它**能不能过**
⇒ 03_强制开发规范.md §2.4「结构性校验 ≠ 功能可用」的第三例。

这些内嵌门有一个共同性质：**它们只读仓库里的 `build-teflon.sh`**，
不需要 Mesa、不需要编译、不需要 TF ⇒ **完全可以常设地跑**。

⇒ 本工具把「验步骤存在」升级成「**真跑那个步骤**」，并接进 `anchor-check.yml`。
下次这类漏在 push 就红，不必等 dispatch。

## 用法

    python3 tools/run-embedded-gates.py            # 跑全部 workflow 的内嵌门
    python3 tools/run-embedded-gates.py --list     # 只列出有哪些门
    python3 tools/run-embedded-gates.py --only PERCHANNEL   # 只跑名字含该片段的
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WF_DIR = os.path.join(REPO, ".github", "workflows")

HEREDOC = re.compile(r"python3 - <<'PY'\n(.*?)\n[ \t]*PY\n", re.S)

# ★ 只跑**不需要构建产物**的内嵌门。
#   第一版按 job 名跳过整个 `build` job ⇒ 一个门都没跑到（exit 2）。
#   那些纯静态门恰恰**就在 build job 里**，只是排在 Build 步骤之前。
#   ⇒ 改成按**步骤名白名单**筛：docs consistency / undefined names 这类
#      只读仓库里的 build-teflon.sh 的门。其余（Verify artifact / Collect /
#      Layout fingerprint …）依赖构建产物，明确列为「不在此列」并打印出来，
#      不假装它们不存在。
RUNNABLE = re.compile(r"docs\s+consistency|undefined\s+names", re.I)


def collect():
    """返回 (可跑的门, 不可跑的门)。

    可跑 = 步骤名命中 RUNNABLE（只读仓库里的 build-teflon.sh，不需要构建产物）。
    不可跑 = 其余内嵌门，明确列出，不假装不存在。
    """
    import yaml
    ok_items, skipped = [], []
    if not os.path.isdir(WF_DIR):
        return ok_items, skipped
    for fn in sorted(os.listdir(WF_DIR)):
        if not fn.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(WF_DIR, fn)
        try:
            d = yaml.safe_load(open(path, encoding="utf-8"))
        except Exception as exc:                                  # noqa: BLE001
            skipped.append((fn, "?", "YAML 解析失败", ""))
            continue
        for jname, job in (d.get("jobs") or {}).items():
            for st in (job.get("steps") or []):
                run = st.get("run") or ""
                m = HEREDOC.search(run)
                if not m:
                    continue
                name = st.get("name", "?")
                item = (fn, jname, name, m.group(1))
                (ok_items if RUNNABLE.search(name) else skipped).append(item)
    return ok_items, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    items, skipped = collect()
    if args.only:
        items = [i for i in items if args.only in i[2]]

    print("=== 可在本地真跑的内嵌门：%d 个 ===" % len(items))
    for fn, j, name, _ in items:
        print("  %-52s [%s] %s" % (name[:52], j, fn))
    print()
    print("=== 明确**不在此列**的内嵌门（依赖构建产物/Mesa/TF）：%d 个 ==="
          % len(skipped))
    for fn, j, name, _ in skipped:
        print("  %-52s [%s] %s" % (name[:52], j, fn))
    if args.list:
        return 0
    if not items:
        print("\n没有可跑的内嵌门 —— 这本身可疑（workflow 结构变了？）")
        return 2

    fails = []
    for fn, j, name, code in items:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8", newline="\n") as f:
            f.write(code)
            p = f.name
        r = subprocess.run([sys.executable, p], cwd=REPO,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        os.unlink(p)
        ok = r.returncode == 0
        print()
        print("--- %s [%s] -> %s" % (name, j, "PASS" if ok else "FAIL"))
        tail = (r.stdout or "").strip().splitlines()[-3:]
        for ln in tail:
            print("    " + ln[:150])
        if not ok:
            err = (r.stderr or "").strip().splitlines()
            for ln in err[-3:]:
                print("    ! " + ln[:150])
            fails.append(name)

    print()
    if fails:
        print("RESULT: %d FAIL —— %s" % (len(fails), "; ".join(fails)))
        print("        这些门在 workflow 里是硬门，本地不过 = dispatch 必红。")
        return 1
    print("RESULT: ALL PASS（%d 个内嵌门本地全过）" % len(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
