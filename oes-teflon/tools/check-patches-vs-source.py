#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用门：把一整组补丁打到**真实 Mesa 源码副本**上，逐个验证「真打进去了」。

## 为什么要有这道门（och 第 N 轮提升为项目通用规则）

> 「mock 证逻辑」必须配一道「真源码证锚点」。

`tools/selftest-*.py` 跑在**合成 mock** 上，而 mock 的锚点是测试自己逐字喂进去的
⇒ **锚点必然命中**，验不了「上游是否改过这段」。

真实事故：`patch-etnaviv-perchannel-quant.py` 的 `OLD_GATE` 里那条 URL 被手抄成
`#per_axis_vs_per_tensor`，真源码是 `#per-axis_vs_per-tensor`
⇒ `count == 0`，CI 上必然 fail-closed，而 mock 自测 12 项全绿。

⇒ **新补丁一律要过这道门。** 它不依赖任何合成 mock。

## 它做什么（对每个补丁器）

1. 把目标文件复制到临时目录（不动真源码）
2. 以子进程跑该补丁器，捕获 rc 与 stderr
3. **落地证明（通用、不依赖人工写标记）**：
   导入补丁器模块，取其所有模块级 `str` 常量；对每个常量比较它在
   「打补丁前的目标文件」与「打补丁后的目标文件」里的出现次数。
   要求**至少有一个常量从 0 次变成 ≥1 次** ——
   即「该补丁器独有的文本确实被插进去了」。
   （OLD 类常量会反向从 ≥1 变 0，方向相反，不参与判定。）
4. 打完检查括号配平

任何一步不满足 ⇒ 该补丁器判 FAIL ⇒ 整门 exit 1。

## 为什么不用人工维护「标记清单」

12 个补丁器的标记名各不相同（`MARKER` / `COMPILE_MARK` / 各种字符串常量），
人工维护清单本身就会漂（这正是本门要防的那类漂移）。
改成「自动从模块里找证据」，新增补丁器**零维护**地纳入本门。

## 用法

    # 验证修复轮那一组（npu-detect / v7-biasfix / OCFIX / FSGUARD / ROFIX / PERCHANNEL）
    python3 tools/check-patches-vs-source.py --src-root <mesa 源码目录> --set fix

    # 验证取证轮那一组（DIAG / WCDIAG / RODIAG / ALIASDIAG / CREATEDIAG）
    python3 tools/check-patches-vs-source.py --src-root <mesa 源码目录> --set diag

    # 负控制自测（扰动一个锚点，本门必须拒绝）
    python3 tools/check-patches-vs-source.py --src-root <mesa 源码目录> --selftest

`--src-root` 只需包含补丁要打的那些 .c 文件（CI 上是
`oes-teflon/third_party/mesa-src-26.2.5/src/...`）。
**缺文件会显式报 SKIP 并计入「未覆盖」，绝不静默当通过。**
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_RECIPE = os.path.join(os.path.dirname(HERE), "build-recipe")

# 每个补丁器 -> 目标 .c 相对路径（与 build-teflon.sh 里的调用一一对应）
PATCHES = {
    "npu-detect":   ("patch-etnaviv-npu-detect.py",
                     "etnaviv/drm/etnaviv_gpu.c"),
    "conv-only":    ("patch-teflon-conv-only.py",
                     "gallium/frontends/teflon/tfl_device.c"),
    "v7-biasfix":   ("patch-etnaviv-v7-biasfix.py",
                     "gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"),
    "ocfix":        ("patch-etnaviv-outchannel-clamp.py",
                     "gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"),
    "fsguard":      ("patch-etnaviv-filter-shape-guard.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "padgate":      ("patch-etnaviv-padding-gate.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "rofix":        ("patch-etnaviv-readout-offset-fix.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "perchannel":   ("patch-etnaviv-perchannel-quant.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "diag-bias":    ("patch-etnaviv-diag-bias.py",
                     "gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"),
    "wcdiag":       ("patch-etnaviv-write-core-diag.py",
                     "gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"),
    "rodiag":       ("patch-etnaviv-readout-diag.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "aliasdiag":    ("patch-etnaviv-alias-diag.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
    "creatediag":   ("patch-etnaviv-create-diag.py",
                     "gallium/drivers/etnaviv/etnaviv_ml.c"),
}

# ★ 互斥组（来自 build-teflon.sh 里的冲突检查，**不是**我猜的）：
#     TEFLON_DIAG  与 TEFLON_FIX_V7 冲突（脚本里直接 exit 1，只能二选一）
#     TEFLON_ROFIX 与 TEFLON_RODIAG 冲突（两者替换同一个循环体）
#   把互斥的两个补丁放进同一个组合 ⇒ 得到的是**组合定义错误**，
#   表现为「某个补丁锚点失配」，极具误导性 —— 必须在这里挡住。
EXCLUSIVE = [
    ("diag-bias", "v7-biasfix", "TEFLON_DIAG 与 TEFLON_FIX_V7 只能二选一"),
    ("rofix", "rodiag", "TEFLON_ROFIX 与 TEFLON_RODIAG 互斥（替换同一循环体）"),
]

# ★ 组合：**顺序必须与 build-teflon.sh 的 3b..3m 步完全一致**。
#   踩过的坑：曾按「修复/取证」分组，diag 组排在 pristine 文件上跑，
#   于是 DIAG（WCDIAG/FIX_V7 之前那一步）与 ALIASDIAG（ROFIX 之后）
#   因前置补丁没打而锚点失配 ⇒ 报了两个**假 FAIL**。
#   ⇒ 这里只维护**一条**顺序（全量），子集按该顺序过滤。
BUILD_ORDER = [
    "npu-detect",      # 3b
    "conv-only",       # 3c
    "diag-bias",       # 3d  TEFLON_DIAG      （与 FIX_V7 互斥）
    "v7-biasfix",      # 3e  TEFLON_FIX_V7
    "wcdiag",          # 3f  TEFLON_WCDIAG
    "ocfix",           # 3g  TEFLON_OCFIX
    "fsguard",         # 3h  TEFLON_FSGUARD
    "padgate",         # 3h3 TEFLON_PADDING_GATE（报告 56）
    "rodiag",          # 3i  TEFLON_RODIAG    （与 ROFIX 互斥）
    "rofix",           # 3j  TEFLON_ROFIX
    "aliasdiag",       # 3k  TEFLON_ALIASDIAG
    "creatediag",      # 3l  TEFLON_CREATEDIAG
    "perchannel",      # 3m  TEFLON_PERCHANNEL
]

# 各组合都是**真实构建里跑过的开关组合**（按 BUILD_ORDER 过滤后应用）
SETS = {
    # 交付候选轮：四处修复 + per-channel，无任何取证打印
    "fix": ["npu-detect", "v7-biasfix", "ocfix", "fsguard", "padgate", "rofix", "perchannel"],
    # 取证轮：FIX_V7+OCFIX+FSGUARD+ROFIX+ALIASDIAG+CREATEDIAG（run 39146418954 那一组）
    "diag": ["npu-detect", "v7-biasfix", "ocfix", "fsguard", "rofix",
             "aliasdiag", "creatediag"],
    "wcdiag": ["npu-detect", "v7-biasfix", "wcdiag", "ocfix", "fsguard"],
    "rodiag": ["npu-detect", "v7-biasfix", "ocfix", "fsguard", "rodiag"],
    "convonly": ["npu-detect", "conv-only"],
    # 「能同时开」的最大限度组合（去掉两个互斥方各自的一个）
    "maximal": ["npu-detect", "conv-only", "v7-biasfix", "wcdiag", "ocfix", "fsguard",
                "padgate", "rofix", "aliasdiag", "creatediag", "perchannel"],
}

# 用来证明「补丁文本真的插进去了」的最小长度，短的常量（如 "1"）没有判别力。
# ★ 不能定得太高：`och-patch-alias-diag` 只有 20 字符，
#   阈值 24 会把 ALIASDIAG 的证据全滤掉 ⇒ 报出一个**假的 FAIL**。
MIN_MARKER_LEN = 12


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def marker_strings(mod):
    """取模块里所有「有判别力的」字符串常量。"""
    out = []
    for k in dir(mod):
        if k.startswith("_"):
            continue
        v = getattr(mod, k)
        if isinstance(v, str) and len(v.strip()) >= MIN_MARKER_LEN:
            out.append((k, v))
    return out


def find_source(src_root, rel):
    """在 src_root 下找 rel 指向的 .c。

    支持两种布局（本地那份 mesa_src 是**平铺**的，只有文件名，没有 src/ 层级）：
      1) 镜像布局：<src_root>/<rel>            （CI 上 third_party/mesa-src-26.2.5/src/…）
      2) 平铺布局：<src_root>/<basename(rel)>   （本地人工挑出来的几个 .c）
    返回 (路径, 布局名)；找不到返回 (None, None)。
    """
    p1 = os.path.join(src_root, rel.replace("/", os.sep))
    if os.path.exists(p1):
        return p1, "mirror"
    p2 = os.path.join(src_root, os.path.basename(rel))
    if os.path.exists(p2):
        return p2, "flat"
    return None, None


def stage_sources(src_root, names, workdir):
    """把需要的 .c 复制到临时目录。返回 (映射, 缺失列表, 布局统计)。"""
    staged, missing = {}, []
    layouts = {}
    for nm in names:
        rel = PATCHES[nm][1]
        src, layout = find_source(src_root, rel)
        if src is None:
            missing.append((nm, rel))
            continue
        layouts[layout] = layouts.get(layout, 0) + 1
        dst = os.path.join(workdir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        staged[nm] = dst
    return staged, missing, layouts


def check_one(name, target, verbose=True):
    """对单个补丁器跑一遍，返回 (ok, 原因列表)。"""
    bad = []
    patcher = os.path.join(BUILD_RECIPE, PATCHES[name][0])
    if not os.path.exists(patcher):
        return False, ["补丁器不存在: " + patcher]

    # ★ 必须用 universal newlines 读（newline=None）：
    #   本地那份源码是 CRLF，而补丁器里的锚点常量是 LF；
    #   若用 newline="" 保留 CRLF，**任何含换行的常量都永远匹配不上**
    #   ⇒ 落地证明失效，会报出假 FAIL。这个坑踩过一次（ALIASDIAG）。
    with open(target, encoding="utf-8") as f:
        before = f.read()

    try:
        mod = load_module(patcher, "chk_%s" % name.replace("-", "_"))
        marks = marker_strings(mod)
    except Exception as exc:                              # noqa: BLE001
        marks = []
        if verbose:
            print("      (无法导入补丁器取常量：%r)" % (exc,))

    r = subprocess.run([sys.executable, patcher, target],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        # 统一返回三元组：rc!=0 分支也要带上「已落地证据」（此处为空）
        return False, ["补丁器 rc=%d：%s"
                       % (r.returncode, err[-1][:200] if err else "(无输出)")], []

    with open(target, encoding="utf-8") as f:
        after = f.read()

    if after == before:
        bad.append("打完补丁内容完全没变（rc=0 但什么都没插进去）")

    # 落地证明：至少一个该补丁器独有的常量从 0 次变成 ≥1 次
    landed = []
    if marks:
        for k, v in marks:
            nb = before.count(v)
            na = after.count(v)
            if nb == 0 and na >= 1:
                landed.append("%s(%d 字符)" % (k, len(v)))
    if marks and not landed:
        # 兜底：至少要有**新增行**（补丁确实动了代码），
        # 但证据强度弱于「补丁器独有常量落地」，报告里必须标明。
        blines = set(before.split("\n"))
        newl = [x for x in set(after.split("\n")) - blines if x.strip()]
        if newl:
            bad.append("常量级证据缺失（只有新增行 %d 行）—— "
                       "补丁动了代码，但无法证明插入的是**该补丁器**的文本" % len(newl))
        else:
            bad.append("没有任何补丁器独有的字符串常量被插入，且没有新增行 —— "
                       "锚点可能命中了但没真正改到代码")
    elif not marks:
        bad.append("补丁器模块里没有可用的字符串常量，无法证明落地")

    if after.count("{") != after.count("}"):
        bad.append("括号不配平 { %d / } %d" % (after.count("{"), after.count("}")))

    # 幂等：再打一次必须被拒绝（防「同一个补丁被打两遍」）
    r2 = subprocess.run([sys.executable, patcher, target],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if r2.returncode == 0:
        bad.append("同一补丁打第二遍居然成功 —— 会重复注入")

    return (not bad), bad, landed


def run_set(src_root, names, verbose=True):
    """按 BUILD_ORDER 的顺序跑 names 里的补丁（子集也继承全量顺序）。

    ★ 踩过的坑：曾按「修复组 / 取证组」各自有序地跑，取证组因此被排在
      pristine 文件上 —— DIAG 在 FIX_V7 **之前**、ALIASDIAG 在 ROFIX **之后**，
      前置补丁没打 ⇒ 锚点失配 ⇒ 报了两个**假 FAIL**。
      ⇒ 只维护一条全量顺序，子集一律按它过滤。
    """
    want = set(names)
    conflicts = [(a, b, why) for a, b, why in EXCLUSIVE
                 if a in want and b in want]
    if conflicts:
        for a, b, why in conflicts:
            sys.stderr.write(
                "ERROR: 组合定义错误 —— 同时包含 %s 与 %s：%s\n"
                "       这会让其中一个补丁锚点失配，报出来的是**误导性**的假 FAIL。\n"
                "       请改用 SETS 里真实跑过的组合。\n" % (a, b, why))
        return None
    ordered = [nm for nm in BUILD_ORDER if nm in want]
    dropped = sorted(want - set(BUILD_ORDER))
    if dropped and verbose:
        print("  ⚠️ 以下名字不在 BUILD_ORDER 里，已忽略: %s" % dropped)
    workdir = tempfile.mkdtemp(prefix="chk_patches_")
    try:
        staged, missing, layouts = stage_sources(src_root, ordered, workdir)
        if verbose and layouts:
            print("  源码布局: %s" % ", ".join("%s×%d" % (k, v)
                                          for k, v in sorted(layouts.items())))
        missing_names = {nm for nm, _rel in missing}
        results = []
        for nm in ordered:
            if nm in missing_names:
                rel = dict(missing)[nm]
                results.append((nm, "SKIP", ["源文件不在 --src-root: %s" % rel]))
                continue
            out = check_one(nm, staged[nm], verbose=verbose)
            ok, bad, _landed = out
            results.append((nm, "PASS" if ok else "FAIL", bad))
        return results
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def report(results):
    print()
    n_fail = sum(1 for _n, s, _b in results if s == "FAIL")
    n_skip = sum(1 for _n, s, _b in results if s == "SKIP")
    n_pass = sum(1 for _n, s, _b in results if s == "PASS")
    for nm, st, bad in results:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[st]
        print("  %-12s %s" % (nm, mark))
        for b in bad:
            print("       - %s" % b)
    print()
    print("  PASS=%d  FAIL=%d  SKIP=%d（共 %d）" % (n_pass, n_fail, n_skip, len(results)))
    if n_skip:
        print("  ⚠️ SKIP 的补丁**未覆盖**：它们的目标文件不在 --src-root 里。")
        print("     SKIP 不等于通过 —— CI 上必须让 --src-root 覆盖全部目标文件。")
    return n_fail, n_skip


def selftest(src_root):
    print("=== 正控制：fix 组在真实源码上必须全过 ===")
    res = run_set(src_root, SETS["fix"])
    f, s = report(res)
    if f:
        print("\nSELFTEST: 正控制就 FAIL 了 —— 本门自身有问题，先修门")
        return 1

    print()
    print("=== 负控制：扰动一个锚点后，本门必须拒绝 ===")
    work = tempfile.mkdtemp(prefix="chk_neg_")
    try:
        rel = PATCHES["perchannel"][1]
        src, _layout = find_source(src_root, rel)
        if src is None:
            print("  负控制无法构造：--src-root 里找不到 %s" % rel)
            return 1
        dst = os.path.join(work, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        txt = open(src, encoding="utf-8").read()
        # 扰动：把 per-axis 那句注释改一个字符 —— 锚点必然失配
        bad = txt.replace("Per-axis quantization not supported",
                          "Per-axis quantization not supportedX", 1)
        if bad == txt:
            print("  负控制无法构造：真实源码里找不到那句注释（源码已变？）")
            return 1
        open(dst, "w", encoding="utf-8", newline="").write(bad)
        ok, bads, _landed = check_one("perchannel", dst)
        print("  扰动后 perchannel: %s" % ("PASS（★本门失效！）" if ok else "FAIL（正确拒绝）"))
        for b in bads:
            print("       - %s" % b)
        if ok:
            print("\nSELFTEST: FAIL —— 负控制没被拒，本门形同虚设")
            return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    print("SELFTEST RESULT: ALL PASS（正控制全过 + 负控制被正确拒绝）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="Mesa 源码根目录（需含补丁要打的那些 .c）")
    ap.add_argument("--set", default="fix", choices=sorted(SETS))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.src_root):
        sys.stderr.write("ERROR: --src-root 不是目录: %s\n" % args.src_root)
        return 1

    if args.selftest:
        return selftest(args.src_root)

    print("=== 真源码锚点核对：set=%s  src-root=%s ==="
          % (args.set, args.src_root))
    res = run_set(args.src_root, SETS[args.set])
    if res is None:
        return 3          # 组合定义错误（互斥）
    n_fail, n_skip = report(res)
    if n_fail:
        print("\nVERDICT: FAIL —— %d 个补丁在真实源码上没打进去。" % n_fail)
        print("         请用对应的 tools/gen-*-patch.py 从真实源码重新生成锚点。")
        return 1
    if n_skip:
        print("\nVERDICT: INCOMPLETE —— 有补丁被跳过（源文件缺失），不能算全过。")
        return 2
    print("\nVERDICT: PASS —— 全部补丁在真实源码上落地且形态正确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
