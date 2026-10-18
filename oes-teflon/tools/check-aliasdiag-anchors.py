#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIASDIAG 锚点 vs **真实 Mesa 源码** 的一致性检查（构建后运行，真源码在盘上）。

用法：
    python3 oes-teflon/tools/check-aliasdiag-anchors.py <etnaviv_ml.c 路径>

传入的文件此刻已被 3h(FSGUARD) + 3j(ROFIX) + 3k(ALIASDIAG) 改过。本脚本：

 0) 留档：打印两个注入点、两个目标函数、以及 read_outputs 拷贝行的**构建后行号**
    （stripped 产物没有 DWARF；若真机再崩，这些坐标是「同构建重跑」时的起点，
     不能沿用旧地址 —— ROFIX 已经作废过一次。）
 1) 保真逆变换：按补丁常量把 ALIASDIAG 的三处替换倒回去（各恰好 1 处）。
    ⚠️ 三处口径不同：ref 是段中间插入（OLD 应消失 0 处）；rec/mark 是只追加型
       （OLD 应**保留** 1 处）。逆变换用 replace() 两种都对，但要**分别断言**。
 2) 让 `tools/gen-aliasdiag-patch.py` 在逆变换后的**真实源码**上跑（含 STRUCT
    计数门 + 提取门），产出与 `build-recipe/patch-etnaviv-alias-diag.py` 的 12 个
    语义载荷**逐字节**比对（硬门：真源码漂移即失败）。
 3) 用生成器产物在逆变换副本上跑正例（行为一致性；任何错误都按硬失败处理）。

为什么必须"构建后"跑：Preflight 阶段 mesa 还没下载，只能拿补丁常量拼 mock
（selftest-aliasdiag.py 干那个）；真实锚点是否逐字一致只有源码在盘上才能验。
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-alias-diag.py")
GEN_PATH = os.path.join(HERE, "gen-aliasdiag-patch.py")
PAYLOAD = ("MARKER", "COMPILE_MARK", "TARGET_FN_REF", "TARGET_FN_REC",
           "STRUCT_REF", "STRUCT_REC", "OLD_REF", "NEW_REF", "OLD_REC",
           "NEW_REC", "ANCHOR_MARK", "NEW_MARK")

NEEDLES = [
    ("reference_tensor_with_offset 定义行", "reference_tensor_with_offset(struct etna_ml_subgraph"),
    ("recreate_tensor 定义行", "recreate_tensor(struct etna_ml_subgraph"),
    ("alias 注入点（ref）", "och patch: alias 迁移取证（reference_tensor_with_offset）"),
    ("alias 注入点（rec）", "och patch: alias 迁移取证（recreate_tensor）"),
    ("编译期标记", 'och_aliasdiag_marker[] = "och-patch-alias-diag"'),
    ("read_outputs 入口行", "etna_ml_subgraph_read_outputs(struct pipe_context"),
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


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python3 check-aliasdiag-anchors.py <etnaviv_ml.c>\n")
        return 2
    ml = sys.argv[1]
    if not os.path.isfile(ml):
        sys.stderr.write("ERROR: 文件不存在: %s\n" % ml)
        return 1

    PATCH = load(PATCH_PATH, "aliasdiag_patch_chk")
    with open(ml, encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")

    print("=== 0. 现场行号（构建后；留档用）===")
    for label, needle in NEEDLES:
        print("  %-26s %d" % (label, line_of(src, needle)))
    print("  （stripped 产物无 DWARF：偏移→行号必须靠**同构建**未 strip 副本重跑取）")

    print()
    print("=== 1. 三口径体检 + 逆变换 ===")
    report("MARKER 在场", PATCH.MARKER in src)
    for tag, new in (("ref", PATCH.NEW_REF), ("rec", PATCH.NEW_REC),
                     ("mark", PATCH.NEW_MARK)):
        n = src.count(new)
        report("NEW_%s 命中 %d 处（须恰好 1）" % (tag, n), n == 1)
    if FAILS:
        print("\nRESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1

    # ★ 打补丁后的源码上直接验「三口径」这件事本身：
    #   ref  —— 段中间插入 ⇒ 旧锚点**必须已被打断**（0 处）
    #   rec  —— 只追加型   ⇒ 旧锚点**必须仍存在**（1 处，嵌在新代码里）
    #   mark —— 同上
    # （我第一版把这条断言错写在逆变换之后 ⇒ 正例被判红；本地真源码模拟抓到的。）
    report("【口径 ref】打补丁后 OLD_REF 已被打断（0）",
           src.count(PATCH.OLD_REF) == 0, "n=%d" % src.count(PATCH.OLD_REF))
    for tag, old in (("rec", PATCH.OLD_REC), ("mark", PATCH.ANCHOR_MARK)):
        report("【口径 %s】打补丁后 OLD_%s 仍在（只追加型，1）" % (tag, tag),
               src.count(old) == 1, "n=%d" % src.count(old))

    inv = src.replace(PATCH.NEW_REF, PATCH.OLD_REF, 1)
    inv = inv.replace(PATCH.NEW_REC, PATCH.OLD_REC, 1)
    inv = inv.replace(PATCH.NEW_MARK, PATCH.ANCHOR_MARK, 1)
    # 逆变换之后：三处旧锚点都应**回到 1 处**（连续文本被还原）
    for tag, old in (("ref", PATCH.OLD_REF), ("rec", PATCH.OLD_REC),
                     ("mark", PATCH.ANCHOR_MARK)):
        report("逆变换后 OLD_%s 恰好回到 1" % tag, inv.count(old) == 1,
               "n=%d" % inv.count(old))
    for tag, new in (("ref", PATCH.NEW_REF), ("rec", PATCH.NEW_REC),
                     ("mark", PATCH.NEW_MARK)):
        report("逆变换后 NEW_%s 已消失" % tag, inv.count(new) == 0,
               "n=%d" % inv.count(new))
    if FAILS:
        print("\nRESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1

    d = tempfile.mkdtemp(prefix="aliasdiag_anchor_")
    invp = os.path.join(d, "etnaviv_ml.c")
    with open(invp, "w", encoding="utf-8", newline="\n") as f:
        f.write(inv)

    print()
    print("=== 2. 生成器在真实源码（逆变换后）上重抽锚点 ===")
    GEN = load(GEN_PATH, "aliasdiag_gen_chk")
    GEN.SRC = invp
    out = os.path.join(d, "regen", "patch-etnaviv-alias-diag.py")
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
            REGEN = load(out, "aliasdiag_regen_chk")
            diff = [k for k in PAYLOAD if getattr(REGEN, k) != getattr(PATCH, k)]
            report("12 个语义载荷与仓库补丁逐字节一致（真源码锚点未漂移）",
                   not diff, "不一致: %s" % diff)

            print()
            print("=== 3. 生成器产物在逆变换副本上跑正例 ===")
            d2 = tempfile.mkdtemp(prefix="aliasdiag_anchor_run_")
            p2 = os.path.join(d2, "etnaviv_ml.c")
            with open(p2, "w", encoding="utf-8", newline="\n") as f:
                f.write(inv)
            regen_mod = load(out, "aliasdiag_regen_run")
            saved = sys.argv
            sys.argv = ["patch-etnaviv-alias-diag.py", p2]
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
    print("RESULT: PASS（真源码锚点与仓库补丁一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
