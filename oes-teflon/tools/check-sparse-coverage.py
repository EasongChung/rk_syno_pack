#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门：`anchor-check.yml` 的稀疏检出**覆盖度**与 **tag/URL 一致性**。

## 为什么需要这道门（och 指定的硬约束）

sparse-checkout 最危险的失败模式是**静默**的：

    该检的文件没检出 ⇒ check-patches-vs-source.py 报 SKIP ⇒
    如果 SKIP 被当成通过，这道门就变成「什么都没查却显示绿」。

och 的硬约束有两条，本门逐条自动核对：

1. **tag 与 URL 必须与 `build-teflon.sh` 第 3 步逐字一致**
   —— 漂移了就核对的不是构建真正用的那份源码，整道门失效**且不报错**。
2. **文件缺失必须 exit 2 `INCOMPLETE`，不许 SKIP 降级**

⇒ 本门**不看 workflow 里的字符串就算完**，它把 workflow 的 sparse 目录清单
与 `check-patches-vs-source.py` 的 `PATCHES` 目标表**对撞**：
新增一个补丁而忘了加目录 ⇒ 这里当场红。

## 用法

    python3 tools/check-sparse-coverage.py
    python3 tools/check-sparse-coverage.py --selftest
"""

import argparse
import importlib.util
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WF = os.path.join(REPO, ".github", "workflows", "anchor-check.yml")
SH = os.path.join(REPO, "oes-teflon", "build-recipe", "build-teflon.sh")
GATE = os.path.join(HERE, "check-patches-vs-source.py")

FAILS = []


def rep(label, ok, detail=""):
    print("  %-58s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def load_patch_targets():
    spec = importlib.util.spec_from_file_location("cpvs", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PATCHES


def parse_sparse_dirs(wf_text):
    """逐行扫出 `git sparse-checkout set` 之后的目录。

    不用正则连着匹配：第一版用 `git sparse-checkout set \\\\\\n((?:\\s+src/\\S+\\n)+)`
    失败在**最后一行没有尾随换行**（它就是 run 块的末行）⇒ 整段匹配不上。
    行式扫描对末尾换行、缩进、续行反斜杠都免疫。
    """
    lines = wf_text.replace("\r\n", "\n").split("\n")
    out = []
    collecting = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("git sparse-checkout set"):
            # 同行带目录的情况（少见）。★ 续行反斜杠 replace 后会变成空串，
            #   必须跳过 —— 第一版没跳，于是清单里混进 ''，把**所有**目录判成未覆盖。
            rest = s.split("set", 1)[1].strip().replace("\\", "").strip()
            if rest:
                out.append(rest)
            collecting = True
            continue
        if collecting:
            if s.startswith("#") or s == "":
                if out:
                    break
                continue
            if s.startswith("src/"):
                out.append(s.replace("\\", "").strip())
                continue
            if s.endswith("\\") and not s.startswith("src/"):
                continue
            break
    return out or None


def parse_build_clone(sh_text):
    """从 build-teflon.sh 里抽出 clone 的 URL 与 --branch 表达式。"""
    m = re.search(r"git clone ([^\n]*?)\\\n\s*(https?://\S+)\s+(\S+)",
                  sh_text)
    if not m:
        m2 = re.search(r"git clone\s+((?:[^\n\\]*\\\n)*[^\n]*)", sh_text)
        if not m2:
            return None, None, None
        line = " ".join(x.strip() for x in m2.group(1).split("\n"))
        u = re.search(r"(https?://\S+)", line)
        b = re.search(r"--branch\s+(\S+)", line)
        return (u.group(1) if u else None,
                b.group(1) if b else None, line)
    url = m.group(2)
    br = re.search(r"--branch\s+(\S+)", m.group(1) + " ")
    return url, (br.group(1) if br else None), (m.group(1) + url)


def _norm_dirs(dirs):
    """归一化 sparse 目录：统一成**相对仓库根**的 `src/...`。

    ★ 第一版直接把 sparse 目录（`src/gallium/...`）和补丁目标
      （`gallium/...`，因为 `--src-root` 指向 `mesa/src`）对撞，
      两边基准不同 ⇒ **12 个补丁全部误判为未覆盖**。
      而且第一版的 selftest 因为没有「抽之前是全覆盖」的正控制，
      **照样 PASS** —— 恒真负控制的典型。
    """
    out = []
    for d in dirs or []:
        d = d.strip().rstrip("/")
        if not d:
            continue
        if not d.startswith("src/"):
            d = "src/" + d
        out.append(d)
    return out


def uncovered_targets(dirs, P):
    norm = _norm_dirs(dirs)
    res = []
    for name, (_patcher, rel) in sorted(P.items()):
        # 目标表是相对 mesa/src 的，补上前缀再比
        t = rel.replace("\\", "/")
        if not t.startswith("src/"):
            t = "src/" + t
        if not any(t.startswith(d + "/") for d in norm):
            res.append(name)
    return res


def main_checks():
    wf = open(WF, encoding="utf-8").read()
    sh = open(SH, encoding="utf-8").read()
    P = load_patch_targets()

    print("=== 1) tag / URL 与 build-teflon.sh 第 3 步逐字一致 ===")
    url_b, branch_b, line_b = parse_build_clone(sh)
    rep("能从 build-teflon.sh 解析出 clone URL", bool(url_b), repr(url_b))
    rep("能从 build-teflon.sh 解析出 --branch 表达式", bool(branch_b), repr(branch_b))
    rep("build-teflon.sh 用同一个 MESA_TAG 变量",
        'MESA_TAG="${MESA_TAG:-mesa-24.3.0}"' in sh or 'MESA_TAG="${MESA_TAG' in sh)

    url_w = re.search(r"(https?://gitlab\.freedesktop\.org/\S+)", wf)
    url_w = url_w.group(1) if url_w else None
    rep("workflow 的 clone URL 与 build-teflon.sh 逐字相同", url_w == url_b,
        "build=%r  wf=%r" % (url_b, url_w))
    rep("workflow 的 --branch 与 build-teflon.sh 逐字相同",
        ('--branch "$MESA_TAG"' in wf) == ('$MESA_TAG' in (branch_b or "")),
        "build=%r" % (branch_b,))
    rep("workflow 也用 $MESA_TAG 变量（不是写死 tag）",
        '--branch "$MESA_TAG"' in wf and "$MESA_TAG" in wf)

    print()
    print("=== 2) sparse 覆盖度：每个补丁目标都必须落在某个 sparse 目录下 ===")
    dirs = parse_sparse_dirs(wf)
    rep("能从 workflow 解析出 sparse 目录清单", dirs is not None, repr(dirs))
    if dirs:
        bad = uncovered_targets(dirs, P)
        for name in sorted(P):
            rel = P[name][1].replace("\\", "/")
            rep("%-11s -> %s" % (name, rel), name not in bad,
                "未被任何 sparse 目录覆盖 ⇒ 会静默 SKIP")
        norm = _norm_dirs(dirs)
        unused = [d for d in norm
                  if not any(("src/" + P[n][1].replace("\\", "/")).startswith(d + "/")
                             for n in P)]
        if unused:
            print("  (提示) 这些 sparse 目录当前没有补丁用到: %s" % unused)
        else:
            print("  (提示) 无多余 sparse 目录")

    print()
    print("=== 3) 缺文件必须 exit 2，不许 SKIP 降级 ===")
    rep("workflow 里有独立的 sparse 覆盖度硬门",
        "Sparse coverage hard gate" in wf)
    m = re.search(r"Sparse coverage hard gate.*?exit 2", wf, re.S)
    rep("缺文件时 exit 2", bool(m))
    rep("错误信息明写「这不是通过」", "这不是通过" in wf)
    rep("覆盖度硬门用 set -eu（不 -x，退出码语义干净）",
        bool(re.search(r"Sparse coverage hard gate.*?set -eu", wf, re.S)))
    rep("check-patches-vs-source.py 的 exit 2 语义仍在（未降级）",
        "INCOMPLETE" in open(GATE, encoding="utf-8").read())

    print()
    if FAILS:
        print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("RESULT: ALL PASS")
    return 0


def selftest():
    """负控制：拿一个**覆盖不全**的 sparse 清单，本门必须红。

    ★ 正控制必须断言「**抽之前**是覆盖的」——
      第一版只断言「抽掉后未覆盖」，结果**解析器整个坏掉时 selftest 照样 PASS**
      （因为此时**所有**目录都算未覆盖）。这正是 §2.2 说的「负控制要针对
      该门最容易犯的那个错」，这里最易犯的就是「负控制恒真」。
    """
    print("=== 正控制：真实清单必须覆盖全部补丁 ===")
    real = parse_sparse_dirs(open(WF, encoding="utf-8").read())
    rep("真实清单非空", bool(real), repr(real))
    rep("★ 清单里没有空串项（第一版的 bug：续行反斜杠 replace 出 ''）",
        real is not None and "" not in real, repr(real))
    P = load_patch_targets()

    def uncovered_of(ds):
        return uncovered_targets(ds, P)

    before = uncovered_of(real or [])
    rep("★ 抽之前：全部补丁都被覆盖（否则下面的负控制没有意义）",
        not before, "未覆盖: %s" % before)

    print()
    print("=== 负控制：抽掉一个目录，本门必须报「未覆盖」 ===")
    trimmed = [d for d in (real or []) if "etnaviv/drm" not in d]
    after = uncovered_of(trimmed)
    rep("抽掉 src/etnaviv/drm 后 npu-detect 变成未覆盖",
        "npu-detect" in after, str(after))
    rep("★ 负控制非恒真：抽掉之前是全覆盖、抽掉之后才不是",
        (not before) and bool(after))
    trimmed2 = [d for d in (real or []) if "frontends/teflon" not in d
                and "etnaviv/drm" not in d]
    after2 = uncovered_of(trimmed2)
    rep("再抽掉 teflon ⇒ 未覆盖数增加（单调）",
        len(after2) > len(after), "%d -> %d" % (len(after), len(after2)))

    print()
    if FAILS:
        print("SELFTEST RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("SELFTEST RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else main_checks())
