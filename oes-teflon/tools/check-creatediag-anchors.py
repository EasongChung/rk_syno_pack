#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CREATEDIAG 锚点 vs **真实 Mesa 源码** 的一致性检查（构建后运行，真源码在盘上）。

用法：
    python3 oes-teflon/tools/check-creatediag-anchors.py <etnaviv_ml.c 路径>

传入的文件此刻已被 3h(FSGUARD) + 3j(ROFIX) + 3k(ALIASDIAG) + 3l(CREATEDIAG) 改过。
本脚本：

  0) 留档：打印三个注入点、两个目标函数、以及 native `created resource` 行的
     **构建后行号**（stripped 产物没有 DWARF；若真机再崩，这些坐标是「同构建
     重跑」时的起点，不能沿用旧地址 —— ROFIX 已经作废过一次）。
  1) 三口径体检 + 保真逆变换：把 CREATEDIAG 的三处替换各倒回去 1 处。
     ⚠️ 三处口径不同：create 是段中间插入（OLD 应消失 0 处）；destroy / mark
        是只追加型（OLD 应**保留** 1 处）。replace() 两种都对，但要**分别断言**。
  2) **打印纪律门（build 后复检）**：在**真实源码**上把补丁后的 create_tensor
     与逆变换后的 create_tensor 都去注释、去 ML_DBG 语句后逐行比对，要求
     **完全相同**。这一条不是循环论证：它比的是「补丁后的真源码」与「把补丁
     倒回去的真源码」，任何重发射的代码差异都会被抓出来（当前版本会在此判红）。
  3) 让 `tools/gen-creatediag-patch.py` 在逆变换后的**真实源码**上重抽锚点，
     产出与 `build-recipe/patch-etnaviv-create-diag.py` 的 12 个语义载荷
     **逐字节**比对（硬门：真源码漂移即失败）。
  4) 用生成器产物在逆变换副本上跑正例（行为一致性；任何错误都按硬失败处理）。

为什么必须"构建后"跑：Preflight 阶段 mesa 还没下载，只能拿补丁常量拼 mock
（selftest-creatediag.py 干那个）；真实锚点是否逐字一致只有源码在盘上才能验。
"""

import contextlib
import difflib
import importlib.util
import io
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-create-diag.py")
GEN_PATH = os.path.join(HERE, "gen-creatediag-patch.py")
PAYLOAD = ("MARKER", "COMPILE_MARK", "FN_CREATE", "FN_DESTROY", "STRUCT_CREATE",
           "STRUCT_DESTROY", "OLD_CREATE", "NEW_CREATE", "OLD_DESTROY",
           "NEW_DESTROY", "ANCHOR_MARK", "NEW_MARK")

NEEDLES = [
    ("create_tensor 定义行", "etna_ml_create_tensor(struct etna_ml_subgraph"),
    ("destroy_tensor 定义行", "etna_ml_destroy_tensor(struct etna_ml_subgraph"),
    ("create 注入点", "och patch: create_tensor 分配取证"),
    ("destroy 注入点", "och patch: create_tensor 分配取证 打到 destroy 侧"),
    ("编译期标记", 'och_creatediag_marker[] = "och-patch-creatediag"'),
    ("native created resource 行", 'ML_DBG("created resource %p for tensor %d with size %d'),
    ("reference_tensor_with_offset 定义行", "reference_tensor_with_offset(struct etna_ml_subgraph"),
    ("read_outputs 拷贝行", "((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;"),
]

FAILS = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(label, ok, detail=""):
    print("  %-56s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append("%s (%s)" % (label, detail))


def line_of(text, needle):
    for i, l in enumerate(text.split("\n"), 1):
        if needle in l:
            return i
    return -1


def strip_code(text):
    """去注释 + 去 ML_DBG 语句 + 去空行，只留可执行代码（打印纪律门用）。"""
    t = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r"ML_DBG\s*\(.*?\)\s*;", "", t, flags=re.S)
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python3 check-creatediag-anchors.py <etnaviv_ml.c>\n")
        return 2
    ml = sys.argv[1]
    if not os.path.isfile(ml):
        sys.stderr.write("ERROR: 文件不存在: %s\n" % ml)
        return 1

    PATCH = load(PATCH_PATH, "creatediag_patch_chk")
    with open(ml, encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")

    print("=== 0. 现场行号（构建后；留档用）===")
    for label, needle in NEEDLES:
        print("  %-28s %d" % (label, line_of(src, needle)))
    print("  （stripped 产物无 DWARF：偏移→行号必须靠**同构建**未 strip 副本重跑取）")

    print()
    print("=== 1. 三口径体检 + 逆变换 ===")
    report("MARKER 在场", PATCH.MARKER in src)
    for tag, new in (("CREATE", PATCH.NEW_CREATE), ("DESTROY", PATCH.NEW_DESTROY),
                     ("MARK", PATCH.NEW_MARK)):
        n = src.count(new)
        report("NEW_%s 命中 %d 处（须恰好 1）" % (tag, n), n == 1)
    if FAILS:
        print("\nRESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1

    # ★ 在打补丁后的源码上直接验「三口径」这件事本身：
    #   create  —— 段中间插入 ⇒ 旧锚点**必须已被打断**（0 处）
    #   destroy / mark —— 只追加型 ⇒ 旧锚点**必须仍存在**（1 处，嵌在新代码里）
    report("【口径 create】打补丁后 OLD_CREATE 已被打断（0）",
           src.count(PATCH.OLD_CREATE) == 0, "n=%d" % src.count(PATCH.OLD_CREATE))
    for tag, old in (("DESTROY", PATCH.OLD_DESTROY), ("MARK", PATCH.ANCHOR_MARK)):
        report("【口径 %s】打补丁后 OLD_%s 仍在（只追加型，1）" % (tag, tag),
               src.count(old) == 1, "n=%d" % src.count(old))

    inv = src.replace(PATCH.NEW_CREATE, PATCH.OLD_CREATE, 1)
    inv = inv.replace(PATCH.NEW_DESTROY, PATCH.OLD_DESTROY, 1)
    inv = inv.replace(PATCH.NEW_MARK, PATCH.ANCHOR_MARK, 1)
    for tag, old in (("CREATE", PATCH.OLD_CREATE), ("DESTROY", PATCH.OLD_DESTROY),
                     ("MARK", PATCH.ANCHOR_MARK)):
        report("逆变换后 OLD_%s 恰好回到 1" % tag, inv.count(old) == 1,
               "n=%d" % inv.count(old))
    for tag, new in (("CREATE", PATCH.NEW_CREATE), ("DESTROY", PATCH.NEW_DESTROY),
                     ("MARK", PATCH.NEW_MARK)):
        report("逆变换后 NEW_%s 已消失" % tag, inv.count(new) == 0,
               "n=%d" % inv.count(new))
    if FAILS:
        print("\nRESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1

    print()
    print("=== 2. 打印纪律门（在真实源码上复检）===")
    for fn in (PATCH.FN_CREATE, PATCH.FN_DESTROY):
        a = strip_code(PATCH._body(src, fn) or "")
        b = strip_code(PATCH._body(inv, fn) or "")
        ok = bool(a) and a == b
        report("%s 注入是纯打印（真源码复检）" % fn.split("(")[0][-22:], ok,
               "" if ok else "patched %d 行 / 逆向 %d 行" % (len(a), len(b)))
        if not ok:
            print("      —— 差异（- 逆变换后=真源码原样 / + 打补丁后）:")
            for l in difflib.unified_diff(b, a, "ORIG", "PATCHED", n=1, lineterm=""):
                print("        " + l)

    d = tempfile.mkdtemp(prefix="creatediag_anchor_")
    invp = os.path.join(d, "etnaviv_ml.c")
    with open(invp, "w", encoding="utf-8", newline="\n") as f:
        f.write(inv)

    print()
    print("=== 3. 生成器在真实源码（逆变换后）上重抽锚点 ===")
    GEN = load(GEN_PATH, "creatediag_gen_chk")
    GEN.SRC = invp
    out = os.path.join(d, "regen", "patch-etnaviv-create-diag.py")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    GEN.OUT = out
    gen_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(gen_err):
            GEN.main()
    except SystemExit as exc:
        report("生成器在真源码上跑通（含 STRUCT 计数门）", False,
               "die()，exit=%s；stderr=%r" % (exc.code, gen_err.getvalue().strip()[-200:]))
    else:
        report("生成器在真源码上跑通（含 STRUCT 计数门）", True, "")
        report("生成器产出补丁文件", os.path.exists(out), out)
        if os.path.exists(out):
            REGEN = load(out, "creatediag_regen_chk")
            diff = [k for k in PAYLOAD if getattr(REGEN, k) != getattr(PATCH, k)]
            report("12 个语义载荷与仓库补丁逐字节一致（真源码锚点未漂移）",
                   not diff, "不一致: %s" % diff)

            print()
            print("=== 4. 生成器产物在逆变换副本上跑正例 ===")
            d2 = tempfile.mkdtemp(prefix="creatediag_anchor_run_")
            p2 = os.path.join(d2, "etnaviv_ml.c")
            with open(p2, "w", encoding="utf-8", newline="\n") as f:
                f.write(inv)
            regen_mod = load(out, "creatediag_regen_run")
            saved = sys.argv
            sys.argv = ["patch-etnaviv-create-diag.py", p2]
            o, e = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    rc = regen_mod.main()
            except Exception as exc:  # noqa: BLE001
                rc = -1
                e.write("EXCEPTION: %r\n" % (exc,))
            finally:
                sys.argv = saved
            report("生成器产物正例通过（生成器与补丁行为一致）", rc == 0,
                   "rc=%s err=%r" % (rc, e.getvalue().strip()[:200]))

    print()
    if FAILS:
        print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("RESULT: PASS（真源码锚点与仓库补丁一致，且注入为纯打印）")
    return 0


if __name__ == "__main__":
    sys.exit(main())