#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-v7-biasfix.py —— `patch-etnaviv-v7-biasfix.py` 的正/负控制。

## 为什么需要这个自测（2026-10-22，CI run 39764385698 编译失败）

该补丁器原先用 `NEW % MARKER` 渲染模板，模板里因此写着 `%%d`（格式化后变成 `%d`）。
报告 56 把渲染改成 `replace()` 后忘了还原 `%%` ⇒ C 侧得到字面量 `%%d` + 多余实参
⇒ `-Werror=format-extra-args` 编译失败。**这类错误本地可查**，所以补这道自测。

## 覆盖

| 组 | 内容 |
|---|---|
| 正控制 | layout 形态产出 v8 对齐的 in_c=1 特判；dense 形态去掉特判 |
| 正控制 | 两种形态的日志字段 dense_depthwise=0/1 正确 |
| 负控制 | 上游 OLD 被改 1 字节 ⇒ 拒绝打补丁且**不写文件** |
| 负控制 | 重复应用 ⇒ fail-closed（标记检测） |
| 反作弊 | 两种形态产出**不同**文件（开关不是空开） |
| 反作弊 | 输出中不得残留 `%%` 转义（本次事故的直接回归项） |
| 回归 | 越界告警（OOB guard）与扫描循环在两种形态下都保留 |

用法：
    python3 tools/selftest-v7-biasfix.py --src-root third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCHER = os.path.join(ROOT, "build-recipe", "patch-etnaviv-v7-biasfix.py")

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
    p = subprocess.run([sys.executable, PATCHER] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="包含 etnaviv_ml_nn_v7.c 的目录（Mesa 源码树）")
    args = ap.parse_args(argv[1:])

    src = os.path.join(args.src_root, "etnaviv_ml_nn_v7.c")
    if not os.path.exists(src):
        print("  FAIL  找到 etnaviv_ml_nn_v7.c（%s）" % src)
        return 1

    pristine = open(src, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="v7bf-selftest-")
    try:
        # ── 正控制①：layout 形态 ───────────────────────────────────────────
        a = os.path.join(tmp, "layout.c")
        shutil.copyfile(src, a)
        rc, out = run([a])
        chk(rc == 0, "正控制：layout 形态打补丁成功", "rc=%d\n%s" % (rc, out[-300:]))
        lay = open(a, encoding="utf-8").read()
        chk("if (operation->depthwise)" in lay, "正控制：layout 形态保留 v8 的 in_c=1 特判")
        chk("dense_depthwise=0" in lay, "正控制：layout 形态日志字段 dense_depthwise=0")
        chk("%%d" not in lay, "回归（本次事故）：layout 形态无 %% 残留转义")
        chk("depthwise=%d" in lay, "正控制：printf 格式是 %d 而不是 %%d")

        # ── 正控制②：dense 形态（fold-early）──────────────────────────────
        b = os.path.join(tmp, "dense.c")
        shutil.copyfile(src, b)
        rc, out = run([b, "--dense-depthwise"])
        chk(rc == 0, "正控制：dense 形态打补丁成功", "rc=%d\n%s" % (rc, out[-300:]))
        den = open(b, encoding="utf-8").read()
        chk("if (operation->depthwise)\n      input_channels = 1;" not in den,
            "正控制：dense 形态**去掉** in_c=1 特判")
        chk("input_channels = operation->input_channels;" in den,
            "正控制：dense 形态回到 input_channels（扫描整块 W*H*C）")
        chk("dense_depthwise=1" in den, "正控制：dense 形态日志字段 dense_depthwise=1")
        chk("%%d" not in den, "回归（本次事故）：dense 形态无 %% 残留转义")
        chk(lay != den, "反作弊：两种形态产出**不同**文件（开关不是空开）")

        # ── 回归：扫描循环与越界告警两种形态都在 ───────────────────────────
        for tag, txt in (("layout", lay), ("dense", den)):
            chk("for (unsigned i = 0; i < och_bound; i++)" in txt,
                "回归：%s 形态的扫描循环保留" % tag)
            chk("och_bound > operation->weight_tensor_size" in txt,
                "回归：%s 形态的越界告警保留" % tag)
            chk("(weights[i] - operation->weight_zero_point) * operation->input_zero_point" in txt,
                "回归：%s 形态的补偿公式逐字保留" % tag)

        # ── 负控制①：上游 OLD 被改 1 字节 ⇒ 拒绝且不写文件 ─────────────────
        bad = pristine.replace("uint8_t *weights,", "uint8_t  *weights,", 1)
        chk(bad != pristine, "负控制① 构造成功：上游 OLD 改 1 字节")
        c = os.path.join(tmp, "mangled.c")
        open(c, "w", encoding="utf-8").write(bad)
        rc, out = run([c])
        chk(rc == 1 and "期望恰好 1 处" in out,
            "负控制①：锚点失配 ⇒ rc=1 并点名处数", "rc=%d" % rc)
        chk(open(c, encoding="utf-8").read() == bad,
            "负控制①：失败时**不写文件**")

        # ── 负控制②：重复应用 ⇒ fail-closed ────────────────────────────────
        rc, out = run([a])
        chk(rc == 1 and "已包含本补丁" in out,
            "负控制②：同一文件再打一次必须 fail-closed", "rc=%d" % rc)
        rc, out = run([a, "--dense-depthwise"])
        chk(rc == 1 and "已包含本补丁" in out,
            "负控制②：layout 已打过再按 dense 打也必须 fail-closed", "rc=%d" % rc)
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
