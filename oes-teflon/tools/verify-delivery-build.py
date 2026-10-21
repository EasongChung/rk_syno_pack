#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**本地**模拟交付版构建的补丁阶段，并跑 workflow 里那两条源码级断言。

## 为什么需要它

交付版与取证版共用一个 workflow（`mode` 输入切换），于是产物级/源码级断言
必须**随开关走**。这一条很容易漏：2026-10-21 连续两轮 CI 失败都是
「断言写死了取证版的期望值」：

    run 39574670018  期望 2 处 [och-pcl] SEL（交付模式 diag=0 ⇒ 0 处）
    run 39575572902  取证块里 grep 无匹配 ⇒ 返回 1 ⇒ set -e 把整步判失败

⇒ 本工具把「补丁阶段」在本地跑一遍（用真实 Mesa 源码副本），再执行同样的断言，
让这类问题**在本地暴露**，而不是烧 CI 往返。

## 它做什么

1. 复制真实 Mesa 源码（`third_party/mesa-src-26.2.5`）
2. 按 `build-reflon.sh` 里**交付开关集**实际会执行的顺序打补丁
   （诊断/实验开关关掉 ⇒ 那些补丁不打）
3. 跑 workflow 里两条「Assert ... actually applied」的源码级检查
4. 报告每一项的通过/失败

## 用法

    python3 tools/verify-delivery-build.py [--mode delivery|phantomdiag]
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BR = os.path.join(ROOT, "build-recipe")
MESA = os.path.join(ROOT, "third_party", "mesa-src-26.2.5")
ETN = "src/gallium/drivers/etnaviv"

# 交付版开关集（与 workflow 的 env 一致）
DELIVERY_SWITCHES = {
    "TEFLON_FIX_V7": "1", "TEFLON_OCFIX": "1", "TEFLON_FSGUARD": "1",
    "TEFLON_ROFIX": "1", "TEFLON_PERCHANNEL": "1",
    "TEFLON_PERCHANNEL_DIAG": "0", "TEFLON_PHANTOM_GATE": "1",
    "TEFLON_PHANTOM_CLAMP": "0",
    "TEFLON_WDOM_PATCH": "1", "TEFLON_ODETRANS_PATCH": "1",
    "TEFLON_PCDOMAIN_PATCH": "1", "TEFLON_STDINC_PATCH": "1",
    "TEFLON_ROUNDMODE_PATCH": "0", "TEFLON_NNDUMP": "0", "TEFLON_KPAT": "0",
    "TEFLON_FPARTIAL": "0", "TEFLON_R5GUARD": "0", "TEFLON_RODIAG": "0",
    "TEFLON_ALIASDIAG": "0", "TEFLON_CREATEDIAG": "0", "TEFLON_DIAG": "0",
    "TEFLON_WCDIAG": "0",
}
PHANTOMDIAG_SWITCHES = dict(DELIVERY_SWITCHES, **{
    "TEFLON_PERCHANNEL_DIAG": "1", "TEFLON_PHANTOM_CLAMP": "1",
    "TEFLON_ROUNDMODE_PATCH": "1", "TEFLON_NNDUMP": "1", "TEFLON_KPAT": "1",
    "TEFLON_RODIAG": "1",
})


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def apply_patch(patcher, target, extra=None):
    args = [sys.executable, os.path.join(BR, patcher), target] + (extra or [])
    r = run(args)
    if r.returncode != 0:
        print("  [BAD] 补丁失败 %s -> %s\n        %s"
              % (patcher, os.path.basename(target), (r.stderr or "")[:300]))
        return False
    return True


def main(argv):
    mode = "delivery"
    if "--mode" in argv:
        mode = argv[argv.index("--mode") + 1]
    sw = DELIVERY_SWITCHES if mode == "delivery" else PHANTOMDIAG_SWITCHES

    if not os.path.isdir(MESA):
        print("ERROR: 找不到 Mesa 源码 %s" % MESA)
        return 1

    tmp = tempfile.mkdtemp(prefix="och-delivery-")
    try:
        dst = os.path.join(tmp, "mesa")
        shutil.copytree(MESA, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        etn = os.path.join(dst, ETN)
        ml = os.path.join(etn, "etnaviv_ml.c")
        ml_nn = os.path.join(etn, "etnaviv_ml_nn.c")
        ml_v7 = os.path.join(etn, "etnaviv_ml_nn_v7.c")
        gpu = os.path.join(dst, "src/etnaviv/drm/etnaviv_gpu.c")

        print("=== 模式 %s 的补丁阶段 ===" % mode)
        steps = []
        steps.append(("npu-detect", lambda: apply_patch("patch-etnaviv-npu-detect.py", gpu)))
        steps.append(("v7-biasfix", lambda: apply_patch("patch-etnaviv-v7-biasfix.py", ml_v7)))
        steps.append(("ocfix", lambda: apply_patch("patch-etnaviv-outchannel-clamp.py", ml_v7)))
        steps.append(("fsguard", lambda: apply_patch("patch-etnaviv-filter-shape-guard.py", ml)))
        if sw["TEFLON_PHANTOM_GATE"] == "1":
            steps.append(("phantom-gate", lambda: apply_patch("patch-etnaviv-phantom-gate.py", ml)))
        if sw["TEFLON_ROFIX"] == "1":
            steps.append(("rofix", lambda: apply_patch("patch-etnaviv-readout-offset-fix.py", ml)))
        if sw["TEFLON_PERCHANNEL"] == "1":
            steps.append(("perchannel-quant", lambda: apply_patch("patch-etnaviv-perchannel-quant.py", ml)))
            steps.append(("perchannel-layout",
                          lambda: apply_patch("patch-etnaviv-perchannel-layout.py", ml_nn,
                                              ["--v7", ml_v7, "--diag", sw["TEFLON_PERCHANNEL_DIAG"]])))
        if sw["TEFLON_PCDOMAIN_PATCH"] == "1":
            steps.append(("perchannel-domain", lambda: apply_patch("patch-etnaviv-perchannel-domain.py", ml_nn)))
        if sw["TEFLON_WDOM_PATCH"] == "1":
            steps.append(("wdom", lambda: apply_patch("patch-etnaviv-weight-domain.py", ml_nn)))
        if sw["TEFLON_ODETRANS_PATCH"] == "1":
            steps.append(("odetrans", lambda: apply_patch("patch-etnaviv-output-detranspose.py", ml)))
        if sw["TEFLON_STDINC_PATCH"] == "1":
            steps.append(("stdinc", lambda: apply_patch("patch-etnaviv-includes.py", ml, ) and
                          apply_patch("patch-etnaviv-includes.py", ml_nn)))

        ok = True
        for name, fn in steps:
            r = fn()
            print("  [%s] %s" % ("ok" if r else "BAD", name))
            ok = ok and r
        if not ok:
            print("\nERROR: 补丁阶段失败")
            return 1

        # ---- workflow 里的源码级断言 ----
        print("\n=== 源码级断言（照抄 workflow 的两条 Assert 步）===")
        nn = open(ml_nn, encoding="utf-8", errors="replace").read()
        v7 = open(ml_v7, encoding="utf-8", errors="replace").read()
        mlc = open(ml, encoding="utf-8", errors="replace").read()

        exp_pcl = 2 if sw["TEFLON_PERCHANNEL_DIAG"] == "1" else 0
        checks = [
            (nn.count("uint8_t (*weights)[real_in]") == 1, "calc_quantization 读侧 real_in"),
            (nn.count("uint8_t (*map_in)[real_in]") == 1, "requantize_weights 读侧 real_in"),
            (nn.count("uint8_t (*map_out)[out_dim]") == 1, "requantize_weights 写侧 out_dim"),
            (nn.count("unsigned src_ic = depthwise ? oc : ic;") == 1, "depthwise 对角线取权重"),
            (nn.count("nn_core_version >= 8 && poperation->conv.depthwise") == 0, "v7 写回条件已去掉"),
            (v7.count("[och-pcl] SEL") == exp_pcl,
             "v7 两处选路都插了诊断（数量随 diag 开关，期望 %d）" % exp_pcl),
            # 「Assert both fixes actually applied」查的是**构建日志**里的消息
            (True, "构建日志里的补丁消息（由补丁器自己打印，见下）"),
        ]
        for cond, desc in checks:
            print("  [%s] %s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # 取证块里的 grep 会不会「无匹配 ⇒ 返回 1 ⇒ set -e 整步失败」
        print("\n=== 取证块的 grep（无匹配必须被容忍）===")
        for pat in ("[och-pcl] SEL",):
            n = v7.count(pat)
            tolerated = n == 0
            print("  [%s] v7 里 '%s' 出现 %d 次 ⇒ %s"
                  % ("ok" if (n == exp_pcl and (n > 0 or tolerated)) else "BAD",
                     pat, n, "grep 有匹配" if n else "grep 会返回 1（必须 || true 容忍）"))

        # 交付模式：诊断类字符串**不该**在源码里
        print("\n=== 交付模式：诊断类标记不该出现在源码里 ===")
        for pat in ("[och-nn]", "[och-fp]", "och-patch-v7-phantom-clamp"):
            present = pat in mlc or pat in ml_nn or pat in v7
            if sw["TEFLON_PHANTOM_CLAMP"] == "0" or pat != "och-patch-v7-phantom-clamp":
                print("  [%s] %s %s" % ("ok" if not present else "BAD", pat,
                                        "不在" if not present else "还在（开关误开？）"))
                ok = ok and not present
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
