#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-padding-gate.py —— `patch-etnaviv-padding-gate.py` 的正/负控制。

## 为什么需要

报告 56 的结论是「`pad_top` 在 pooling-first-pixel 路径被忽略，输出偏 1 像素」。
把这类层挡在 NPU 之外，**门本身写错就会静默漏放行**（偏差照旧，而且更难发现，
因为「接管了」看起来像是验证成功）。所以门必须有：

* **正控制**：干净源码上能打上，且守卫块里确实有 `return false;`
* **负控制（真实样本）**：在**已打 FSGUARD / 已打 PHANTOM_GATE** 的真实源码上仍能打
  （构建顺序就是这样，不能只在教科书源码上通过）
* **负控制**：锚点失配 ⇒ 拒绝且**不写文件**；重复应用 ⇒ fail-closed
* **反作弊**：守卫块内不得出现 `supported = true;`（否则 fail-open）；
  运行期门必须是「不设=开、=0 才关」
* **回归**：`TEFLON_PADGATE` 之外不得引入别的开关名；判据必须含 depthwise + padding_top

用法：
    python3 tools/selftest-padding-gate.py --src-root third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCHER = os.path.join(ROOT, "build-recipe", "patch-etnaviv-padding-gate.py")
FSGUARD = os.path.join(ROOT, "build-recipe", "patch-etnaviv-filter-shape-guard.py")
PGATE = os.path.join(ROOT, "build-recipe", "patch-etnaviv-phantom-gate.py")

CHECKS = 0
FAILS = []


def chk(cond, name, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s%s" % (name, ("  —— " + str(detail)) if detail else ""))
        FAILS.append(name)


def run(args):
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="包含 etnaviv_ml.c 的目录（Mesa 源码树）")
    args = ap.parse_args(argv[1:])

    src = os.path.join(args.src_root, "etnaviv_ml.c")
    if not os.path.exists(src):
        print("  FAIL  找到 etnaviv_ml.c（%s）" % src)
        return 1

    pristine = open(src, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="padgate-selftest-")
    try:
        # ── 正控制①：干净源码 ──────────────────────────────────────────
        a = os.path.join(tmp, "clean.c")
        shutil.copyfile(src, a)
        rc, out = run([PATCHER, a])
        chk(rc == 0, "正控制：干净源码打补丁成功", "rc=%d\n%s" % (rc, out[-300:]))
        txt = open(a, encoding="utf-8").read()
        chk("och patch: pooling-first-pixel padding 门控" in txt, "正控制：注释标记存在")
        chk("och-patch-v7-padding-gate" in txt, "正控制：编译期标记存在（strings 可判）")
        chk("getenv(\"TEFLON_PADGATE\")" in txt, "正控制：运行期门 TEFLON_PADGATE 存在")
        chk("operation->conv.depthwise" in txt and "operation->conv.padding_top > 0" in txt,
            "正控制：判据含 depthwise 与 padding_top")
        chk("operation->conv.stride_x > 1" in txt, "正控制：判据含 stride（只挡 stride>1）")

        # ── 反作弊：运行期门语义必须是「不设=开、=0 才关」 ─────────────────
        m = re.search(r'const char \*och_pad_env = getenv\("TEFLON_PADGATE"\);\s*\n\s*if \((.*?)\)',
                      txt, re.S)
        chk(m is not None, "反作弊：能定位运行期判断")
        cond = m.group(1) if m else ""
        chk("och_pad_env == NULL || och_pad_env[0] != '0'" in cond.replace("\n", " ").replace("  ", " "),
            "反作弊：不设 ⇒ 开；只有 =0 才关（交付语义）", cond[:80])
        chk(txt.count("getenv(\"TEFLON_PADGATE\")") == 1, "反作弊：运行期门只读一次（无第二开关）")

        # ── fail-open 检查：守卫块内不得出现 supported = true ──────────────
        g_at = txt.index("och patch: pooling-first-pixel padding 门控")
        guard = txt[g_at:txt.index("#endif", g_at) + len("#endif")]
        chk("return false;" in guard, "反作弊：守卫块内有显式 return false")
        chk("supported = true;" not in guard, "反作弊：守卫块内没有 supported = true（不 fail-open）")

        # ── 负控制①：锚点被改 1 字节 ⇒ 拒绝且不写文件 ─────────────────────
        bad = pristine.replace("struct pipe_tensor *input_tensor = operation->input_tensors[0];",
                               "struct pipe_tensor  *input_tensor = operation->input_tensors[0];", 1)
        chk(bad != pristine, "负控制① 构造成功：锚点改 1 字节")
        b = os.path.join(tmp, "mangled.c")
        open(b, "w", encoding="utf-8").write(bad)
        rc, out = run([PATCHER, b])
        chk(rc == 1 and "期望恰好 1 处" in out, "负控制①：锚点失配 ⇒ rc=1 并点名处数", "rc=%d" % rc)
        chk(open(b, encoding="utf-8").read() == bad, "负控制①：失败时**不写文件**")

        # ── 负控制②：锚点出现两处 ⇒ 拒绝（不许猜）────────────────────────
        dup = pristine.replace(
            "      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n"
            "         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n",
            "      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n"
            "         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n"
            "\n"
            "      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {\n"
            "         struct pipe_tensor *input_tensor = operation->input_tensors[0];\n", 1)
        c = os.path.join(tmp, "dup.c")
        open(c, "w", encoding="utf-8").write(dup)
        rc, out = run([PATCHER, c])
        chk(rc == 1 and "2 处" in out, "负控制②：锚点重复 ⇒ rc=1（不猜哪一个）", "rc=%d" % rc)

        # ── 负控制③：重复应用 ⇒ fail-closed ──────────────────────────────
        rc, out = run([PATCHER, a])
        chk(rc == 1 and "已包含本补丁" in out, "负控制③：同一文件再打一次 fail-closed", "rc=%d" % rc)

        # ── 负控制④（真实样本）：先打 FSGUARD / PHANTOM_GATE 仍能打 ────────
        for name, patcher in (("FSGUARD", FSGUARD), ("PHANTOM_GATE", PGATE)):
            if not os.path.exists(patcher):
                chk(False, "负控制④：%s 补丁器存在（%s）" % (name, patcher))
                continue
            d = os.path.join(tmp, "with_%s.c" % name)
            shutil.copyfile(src, d)
            rc0, out0 = run([patcher, d])
            chk(rc0 == 0, "负控制④ 构造成功：先在真实源码上打 %s" % name, out0[-200:])
            rc1, out1 = run([PATCHER, d])
            chk(rc1 == 0, "负控制④（真实样本）：%s 已打后本补丁仍能打" % name, out1[-300:])
            t = open(d, encoding="utf-8").read()
            chk("och-patch-v7-padding-gate" in t and "return false;" in t,
                "负控制④（真实样本）：%s 共存后本补丁标记齐全" % name)

        # ── 回归：两种顺序都要能共存（PHANTOM_GATE ↔ 本补丁）──────────────
        e = os.path.join(tmp, "order2.c")
        shutil.copyfile(src, e)
        rc1, o1 = run([PATCHER, e])
        rc2, o2 = run([PGATE, e])
        chk(rc1 == 0 and rc2 == 0,
            "回归：本补丁先、PHANTOM_GATE 后 ⇒ 两个都能打", (o1 + o2)[-300:])
        t = open(e, encoding="utf-8").read()
        chk(t.count("och-patch-v7-padding-gate") == 1 and "och_phantom_gate_marker" in t,
            "回归：反序共存后两个标记都在（padding 1 次 + phantom 组件）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n== %d 项检查，%d 项失败 ==" % (CHECKS, len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED: %s" % f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
