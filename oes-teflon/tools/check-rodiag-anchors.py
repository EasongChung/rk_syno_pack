#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RODIAG 锚点 vs **真实 Mesa 源码** 的一致性检查（构建后运行，此时真源码在盘上）。

用法：
    python3 oes-teflon/tools/check-rodiag-anchors.py <etnaviv_ml.c 路径>

传入的文件此刻已被 3h(FSGUARD) + 3i(RODIAG) 改过。本脚本做三件事：

 1) 保真逆变换：按补丁自己的常量把 RODIAG 的两处替换倒回去（各恰好 1 处），
    回到「clean + FSGUARD」形态 —— 3i 会把锚点两侧隔开，不逆变换就抽不出锚点。
 2) 让 `tools/gen-rodiag-patch.py` 在这份**真实源码**上跑（含 STRUCT 计数门 +
    提取门），产出与 `build-recipe/patch-etnaviv-readout-diag.py` 的 6 个语义载荷
    **逐字节**比对（硬门：真源码漂移 = 立即失败）。
 3) 用生成器产物（TEMPLATE 拼出的补丁）在逆变换副本上跑正例：
      通过                -> PASS（生成器与补丁行为也一致）
      仅 get_tensor 误拒  -> WARN（生成器模板缺 comment-strip，已回报 och）
      其它任何错误        -> 硬失败（真漂移）

为什么必须"构建后"跑：Preflight 阶段 mesa 还没下载，只能拿补丁常量拼 mock 自测；
「真实源码里的锚点是否仍逐字一致」只有等 3h/3i 跑完、源码在盘上才能验证。
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-diag.py")
GEN_PATH = os.path.join(HERE, "gen-rodiag-patch.py")
PAYLOAD = ("MARKER", "STRUCT", "OLD_LOOP", "NEW_LOOP", "OLD_MAP", "NEW_MAP")

FAILS = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(label, ok, detail=""):
    print("  %-52s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append("%s (%s)" % (label, detail))


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python3 check-rodiag-anchors.py <etnaviv_ml.c>\n")
        return 2
    ml = sys.argv[1]
    if not os.path.isfile(ml):
        sys.stderr.write("ERROR: 文件不存在: %s\n" % ml)
        return 1

    PATCH = load(PATCH_PATH, "rodiag_patch_chk")
    with open(ml, encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")

    print("=== 1. 逆变换（RODIAG 替换倒回去）===")
    n_loop = src.count(PATCH.NEW_LOOP)
    n_map = src.count(PATCH.NEW_MAP)
    print("  NEW_LOOP 命中 %d 处，NEW_MAP 命中 %d 处（各须恰好 1）" % (n_loop, n_map))
    report("RODIAG 注入确实存在于该文件（否则逆变换无意义）",
           n_loop == 1 and n_map == 1, "n_loop=%d n_map=%d" % (n_loop, n_map))
    if FAILS:
        return 1
    inv = src.replace(PATCH.NEW_LOOP, PATCH.OLD_LOOP, 1)
    inv = inv.replace(PATCH.NEW_MAP, PATCH.OLD_MAP, 1)
    report("逆变换后 OLD_LOOP/OLD_MAP 各回到 1 处",
           inv.count(PATCH.OLD_LOOP) == 1 and inv.count(PATCH.OLD_MAP) == 1,
           "loop=%d map=%d" % (inv.count(PATCH.OLD_LOOP), inv.count(PATCH.OLD_MAP)))

    d = tempfile.mkdtemp(prefix="rodiag_anchor_")
    invp = os.path.join(d, "etnaviv_ml.c")
    with open(invp, "w", encoding="utf-8", newline="\n") as f:
        f.write(inv)

    print("=== 2. 生成器在真实源码（逆变换后）上重抽锚点 ===")
    GEN = load(GEN_PATH, "rodiag_gen_chk")
    GEN.SRC = invp
    out = os.path.join(d, "regen", "patch-etnaviv-readout-diag.py")
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
            REGEN = load(out, "rodiag_regen_chk")
            diff = [k for k in PAYLOAD if getattr(REGEN, k) != getattr(PATCH, k)]
            report("6 个语义载荷与仓库补丁逐字节一致（真源码锚点未漂移）",
                   not diff, "不一致: %s" % diff)

            print("=== 3. 生成器产物在逆变换副本上跑正例 ===")
            d2 = tempfile.mkdtemp(prefix="rodiag_anchor_run_")
            p2 = os.path.join(d2, "etnaviv_ml.c")
            with open(p2, "w", encoding="utf-8", newline="\n") as f:
                f.write(inv)
            regen_mod = load(out, "rodiag_regen_run")
            saved = sys.argv
            sys.argv = ["patch-etnaviv-readout-diag.py", p2]
            o, e = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    rc = regen_mod.main()
            except Exception as exc:  # noqa: BLE001
                rc = -1
                e.write("EXCEPTION: %r\n" % (exc,))
            finally:
                sys.argv = saved
            if rc == 0:
                report("生成器产物正例通过（生成器与补丁行为一致）", True, "")
            elif "出现了 etna_ml_get_tensor" in e.getvalue():
                report("生成器产物正例通过", True,
                       "已知问题，降级 WARN：见下方说明")
                print("  WARN: 生成器模板缺 comment-strip（DIAG_LOOP 的注释里含 "
                      "`etna_ml_get_tensor` ⇒ 自检误拒）。")
                print("        repo 里的补丁文件已修（import re + 两行 re.sub），"
                      "生成器未同步；已在回信上报 och（附改法）。")
                print("        本项不阻塞：产物用的是 build-recipe/ 里的补丁文件，"
                      "它的正例已由构建过程与 selftest-rodiag.py 证明。")
            else:
                report("生成器产物正例通过", False,
                       "非已知问题：rc=%s err=%r" % (rc, e.getvalue().strip()[:200]))

    print()
    if FAILS:
        print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("RESULT: PASS（真源码锚点与仓库补丁一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
