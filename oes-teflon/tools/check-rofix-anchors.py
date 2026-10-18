#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROFIX 锚点 vs **真实 Mesa 源码** 的一致性检查（构建后运行，此时真源码在盘上）。

用法：
    python3 oes-teflon/tools/check-rofix-anchors.py <etnaviv_ml.c 路径>

传入的文件此刻已被 3h(FSGUARD) + 3j(ROFIX) 改过（可能还叠加了别的步）。本脚本：

 0) 留档：打印 read_outputs 入口行 / 拷贝行的**构建后行号**。
    （stripped 产物没有 DWARF，这些行号只对**与本次构建同源**的未 strip 副本有意义；
     它们的作用是让「若真机再崩」时的排查有一个确定的起点，而不是猜。）
 1) 保真逆变换：按补丁自己的常量把 ROFIX 的整块替换倒回去（恰好 1 处）——
    回到「clean + FSGUARD」形态；不逆变换，生成器就抽不出锚点。
 2) 让 `tools/gen-rofix-patch.py` 在这份**真实源码**上跑（含 STRUCT 计数门 +
    提取门），产出与 `build-recipe/patch-etnaviv-readout-offset-fix.py` 的 5 个语义载荷
    （MARKER / DIAG_MARKER / STRUCT / OLD / NEW）**逐字节**比对（硬门：真源码漂移即失败）。
 3) 用生成器产物在逆变换副本上跑正例（行为一致性）。
    本轮生成器模板自带 comment-strip ⇒ RODIAG 那轮的已知 WARN 不复存在，
    所以「正例被拒」一律按**硬失败**处理，不再降级放行。

为什么必须"构建后"跑：Preflight 阶段 mesa 还没下载，只能拿补丁常量拼 mock 自测
（selftest-rofix.py 干那个）；「真实源码里的锚点是否仍逐字一致」只有等
3h/3j 跑完、源码在盘上才能验证。
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")
GEN_PATH = os.path.join(HERE, "gen-rofix-patch.py")
PAYLOAD = ("MARKER", "DIAG_MARKER", "STRUCT", "OLD", "NEW")
ENTRY = "etna_ml_subgraph_read_outputs(struct pipe_context"
COPY = "((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;"

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
        sys.stderr.write("用法: python3 check-rofix-anchors.py <etnaviv_ml.c>\n")
        return 2
    ml = sys.argv[1]
    if not os.path.isfile(ml):
        sys.stderr.write("ERROR: 文件不存在: %s\n" % ml)
        return 1

    PATCH = load(PATCH_PATH, "rofix_patch_chk")
    with open(ml, encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")

    print("=== 0. 现场行号（构建后；留档用）===")
    print("  read_outputs 入口行: %d" % line_of(src, ENTRY))
    print("  拷贝行:            %d" % line_of(src, COPY))
    print("  （stripped 产物无 DWARF：若真机再崩，偏移→行号的映射必须靠**同构建**"
          "的未 strip 副本重跑取，不能沿用 675a1091 的旧地址）")

    print()
    print("=== 1. 逆变换（ROFIX 替换倒回去）===")
    n_new = src.count(PATCH.NEW)
    print("  NEW 命中 %d 处（须恰好 1）" % n_new)
    report("ROFIX 注入确实存在于该文件（否则逆变换无意义）", n_new == 1,
           "n=%d" % n_new)
    report("补丁标记 MARKER 在场", PATCH.MARKER in src)
    if FAILS:
        print()
        print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    inv = src.replace(PATCH.NEW, PATCH.OLD, 1)
    report("逆变换后 OLD 恰好回到 1 处", inv.count(PATCH.OLD) == 1,
           "n=%d" % inv.count(PATCH.OLD))
    report("逆变换后 NEW 已消失", inv.count(PATCH.NEW) == 0,
           "n=%d" % inv.count(PATCH.NEW))
    if FAILS:
        print()
        print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1

    d = tempfile.mkdtemp(prefix="rofix_anchor_")
    invp = os.path.join(d, "etnaviv_ml.c")
    with open(invp, "w", encoding="utf-8", newline="\n") as f:
        f.write(inv)

    print()
    print("=== 2. 生成器在真实源码（逆变换后）上重抽锚点 ===")
    GEN = load(GEN_PATH, "rofix_gen_chk")
    GEN.SRC = invp
    out = os.path.join(d, "regen", "patch-etnaviv-readout-offset-fix.py")
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
            REGEN = load(out, "rofix_regen_chk")
            diff = [k for k in PAYLOAD if getattr(REGEN, k) != getattr(PATCH, k)]
            report("5 个语义载荷与仓库补丁逐字节一致（真源码锚点未漂移）",
                   not diff, "不一致: %s" % diff)

            print()
            print("=== 3. 生成器产物在逆变换副本上跑正例 ===")
            d2 = tempfile.mkdtemp(prefix="rofix_anchor_run_")
            p2 = os.path.join(d2, "etnaviv_ml.c")
            with open(p2, "w", encoding="utf-8", newline="\n") as f:
                f.write(inv)
            regen_mod = load(out, "rofix_regen_run")
            saved = sys.argv
            sys.argv = ["patch-etnaviv-readout-offset-fix.py", p2]
            o, e = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    rc = regen_mod.main()
            except Exception as exc:  # noqa: BLE001
                rc = -1
                e.write("EXCEPTION: %r\n" % (exc,))
            finally:
                sys.argv = saved
            # 不降级：本轮生成器模板自带 comment-strip，任何错误都是硬失败
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
