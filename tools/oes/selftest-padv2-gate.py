#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""padv2_fill_gate 的正/反例自测（可本地跑，只需 `tflite` 读 API，不需 TF）。

## 覆盖
  正例
    P1 良构且反量化后全 0（INT8, zp=127, scale=0.01, 值=127）  -> ZERO
    P2 float32 良构全 0                                        -> ZERO
    P3 真实产物 concept_int8_pertensor.tflite                  -> DEGENERATE 且不阻断
    P4 已改写完的 padfix 产物                                    -> 无 PADV2
  反例（★ 每一条都必须拦住 / 必须不被误判成 ZERO）
    N1 良构非零（float32 值 1.0）                             -> NONZERO（阻断）
    N2 良构非零（INT8, zp=0, scale=0.01, 值=100 -> 1.0）       -> NONZERO（阻断）
    N3 量化域恰好落在 zp 上（值=127, zp=127）**但 scale 正常**  -> ZERO
       ★ 这条防的是「把 zp 误当 0」：zp≠0 时值必须按 (v-zp)*scale 算
    N4 形状不匹配（值非零但 shape 与 pads 不同）               -> DEGENERATE（不阻断）
    N5 NaN                                                       -> DEGENERATE（不阻断）
    N6 非有限前的短路：None（非常量）                           -> UNKNOWN（阻断）

★ 纪律：门必须有负控制。N1/N2/N6 若不被拦，这道门形同虚设。
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from padv2_fill_gate import DEGENERATE_ABS, classify, scan_model  # noqa: E402

FAILS = []


def rep(label, ok, detail=""):
    print("  %-58s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def v(x, dt=np.float32):
    return np.array(x, dtype=dt).astype(np.float64)


print("=== P/N 纯逻辑用例 ===")
# P1 INT8 量化域：值 127, zp 127, scale 0.01 ⇒ 真实 0
val, note = v([127], np.int8), "已按 scale=0.01 zp=127 反量化"
verdict, _ = classify((val - 127) * 0.01, note, [4, 2])
rep("P1 量化域落在 zp 上 ⇒ ZERO", verdict == "ZERO", "实得 " + verdict)

# P2 float32 全 0
verdict, _ = classify(v([0.0, 0.0, 0.0, 0.0]), "无量化参数，按字面值", [4, 2])
rep("P2 float32 全 0 ⇒ ZERO", verdict == "ZERO", "实得 " + verdict)

# N3 zp≠0 时必须按 (v-zp)*scale 算，不能把 zp 当 0
verdict, _ = classify((v([127], np.int8) - 127) * 0.01, "zp=127", [4, 2])
rep("N3 zp=127 时 127 不被当成 1.27 ⇒ ZERO", verdict == "ZERO", "实得 " + verdict)

# N1 良构非零 float32
verdict, _ = classify(v([[0., 0.], [1., 1.], [1., 1.], [0., 0.]]),
                      "无量化参数，按字面值", [4, 2])
rep("★N1 良构非零（fill=1）⇒ 必须 NONZERO", verdict == "NONZERO", "实得 " + verdict)

# N2 良构非零（量化域），形状必须与 pads 的 [4,2] 一致，
#    否则会落进「形状不匹配 ⇒ DEGENERATE」分支（那是另一条用例 N4b）。
ones_i8 = v([[100, 100], [100, 100], [100, 100], [100, 100]], np.int8)
verdict, _ = classify((ones_i8 - 0) * 0.01, "scale=0.01 zp=0", [4, 2])
rep("★N2 量化域非零（1.0，形状[4,2]）⇒ 必须 NONZERO",
    verdict == "NONZERO", "实得 " + verdict)

# N2b 同一个值但形状与 pads 不符 ⇒ 走 DEGENERATE（形状不合法，本就不是可信常量）
verdict, _ = classify((v([100], np.int8) - 0) * 0.01, "scale=0.01 zp=0", [4, 2])
rep("N2b 同值但形状[1]≠pads[4,2] ⇒ DEGENERATE", verdict == "DEGENERATE",
    "实得 " + verdict)

# N4 形状不匹配 + 非零 ⇒ DEGENERATE
verdict, _ = classify(v([3.40282e38]), "FLT_MAX", [4, 2])
rep("N4 形状/量级都不合法 ⇒ DEGENERATE", verdict == "DEGENERATE", "实得 " + verdict)

verdict, _ = classify(v([5.0]), "字面值", [4, 2])
rep("N4b 形状不符的良构非零 ⇒ DEGENERATE", verdict == "DEGENERATE", "实得 " + verdict)

# N5 NaN
verdict, _ = classify(v([np.nan]), "字面值", [4, 2])
rep("N5 NaN ⇒ DEGENERATE", verdict == "DEGENERATE", "实得 " + verdict)

# N6 非常量
verdict, _ = classify(None, "无 buffer 引用（非常量）", [4, 2])
rep("★N6 非常量 ⇒ 必须 UNKNOWN", verdict == "UNKNOWN", "实得 " + verdict)

# 边界：恰好等于退化阈值
verdict, _ = classify(v([DEGENERATE_ABS]), "边界", [4, 2])
rep("边界 |v|==DEGENERATE_ABS ⇒ DEGENERATE", verdict == "DEGENERATE", "实得 " + verdict)

print()
print("=== 真实产物用例 ===")
ART = r"D:\clawtemp\o2tf-src\art13\dl\out"
pre = os.path.join(ART, "concept_int8_pertensor.tflite")
post = os.path.join(ART, "concept_int8_pertensor_padfix.tflite")
if not os.path.exists(pre):
    print("  (跳过：找不到 %s)" % pre)
else:
    r1 = scan_model(pre)
    rep("P3 真实修前产物：1 个 PADV2 判为 DEGENERATE 且不阻断",
        r1["n_padv2"] == 1 and r1["counts"].get("DEGENERATE") == 1
        and not r1["blocking"], str(r1["counts"]))
    rep("P3b 真实修前产物：不误判成 ZERO",
        r1["counts"].get("ZERO", 0) == 0, str(r1["counts"]))
if os.path.exists(post):
    r2 = scan_model(post)
    rep("P4 真实修后产物：0 个 PADV2", r2["n_padv2"] == 0, str(r2["n_padv2"]))

print()
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("RESULT: ALL PASS")
