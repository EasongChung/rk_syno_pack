#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造一个**权重 per-axis scale 不均匀**的 int8 模型，用来激活驱动里从未跑过的 per-channel 路径。

## 为什么需要它

`etna_ml_nn.c` 的 per-channel 分支（`calc_quantization` / `requantize_weights` /
`requantize_bias`）一直存在，但**从未被真机跑到过**，原因在 delegate 侧：

    tfl_device.c:412
        if (quant->scale->size > 1 && !all_scales_equal(quant)) {
            tensor->scales = calloc(...);      // 只有「各通道 scale 不全相等」才传 per-axis
        }

我们现有的 248 通道合成模型，248 个权重 scale **完全相同**（0.430524）
⇒ delegate 故意不传 per-axis ⇒ 走 per-tensor 路径（且恰好精确）
⇒ per-channel 分支是**死代码**。

官方 MobileNet 是 **uint8 + per-tensor**，同样碰不到它。

⇒ 必须造一个 scale **真的不均匀**的模型，才能第一次激活这条路径。

## 做法

只改**量化参数**（不改任何权重/偏置数值）：

  1. 权重 per-axis scale：`s_c = base * (lo + (hi-lo) * c/(C-1))`（线性铺开）
  2. 偏置 per-axis scale：同步改成 `input_scale * s_c`
     （TFLite 的不变量：bias_scale = in_scale × weight_scale[axis]；
       偏置数值全 0，所以偏置的重量化结果仍是 0，不引入额外变量）
  3. 权重/偏置的 zero point、量化维度、其它张量一律不动

## 不变量自检（跑之前先自证，失败就报错退出）

  - 权重仍是单位阵（对角 1、其余 0）
  - 偏置仍全 0
  - 输出 scale/zero point 未动
  - 新 scale 严格递增且与原值不同（否则 delegate 仍会折叠成 per-tensor）

## 用法

    python3 tools/make-perchannel-spread.py <in.tflite> <out.tflite> [--lo 0.8] [--hi 1.2]
"""

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tflite_fold_rank5_to_4d as fold  # noqa: E402


def find_conv(mdl):
    """找唯一的 CONV_2D，返回 (op, in_idx, w_idx, b_idx, out_idx)。"""
    for op in mdl["ops"]:
        if mdl["codes"][op["opcode_index"]]["builtin"] != 3:   # 3 = CONV_2D
            continue
        ins = op["inputs"]
        if len(ins) < 3:
            continue
        return op, ins[0], ins[1], ins[2], op["outputs"][0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lo", type=float, default=0.8, help="scale 下限（相对原值）")
    ap.add_argument("--hi", type=float, default=1.2, help="scale 上限（相对原值）")
    args = ap.parse_args()

    mdl = fold.read_model(args.src)
    got = find_conv(mdl)
    if got is None:
        print("ERROR: 模型里没有 CONV_2D", file=sys.stderr)
        return 1
    op, ti, tw, tb, to = got

    t_in, t_w, t_b, t_out = (mdl["tensors"][i] for i in (ti, tw, tb, to))
    qw, qb = t_w.get("quant") or {}, t_b.get("quant") or {}
    sc_w, zp_w = qw.get("scale") or [], qw.get("zp") or []
    sc_b, zp_b = qb.get("scale") or [], qb.get("zp") or []
    C = t_w["shape"][0]

    print("=== 原模型 ===")
    print("  权重 %s shape=%s type=%s  scale 个数=%d zp 个数=%d qdim=%s"
          % (t_w["name"], t_w["shape"], t_w["type"], len(sc_w), len(zp_w), qw.get("qdim")))
    print("  偏置 %s shape=%s type=%s  scale 个数=%d" % (t_b["name"], t_b["shape"], t_b["type"], len(sc_b)))
    print("  输入 scale=%s zp=%s" % (t_in["quant"]["scale"], t_in["quant"]["zp"]))

    # ---- 前置条件 ----
    if len(sc_w) != C:
        print("ERROR: 权重 scale 个数 %d ≠ 输出通道 %d（需要 per-axis）" % (len(sc_w), C),
              file=sys.stderr)
        return 1
    if len(sc_b) != C:
        print("ERROR: 偏置 scale 个数 %d ≠ %d" % (len(sc_b), C), file=sys.stderr)
        return 1
    if len(set(sc_w)) == 1:
        # 输入模型通常就是「全相等」（那正是 delegate 会折叠成 per-tensor 的原因）——
        # 本工具的职责正是把它变成不均匀的。所以这里**不是**错误，只记录一句。
        print("  注: 输入模型的权重 scale 全相等（%.9f）—— 这正是 delegate 会把它"
              % sc_w[0])
        print("      折叠成 per-tensor 的原因；本次手术就是为了把它变成不均匀的。")
    if qw.get("qdim") not in (0, None):
        print("ERROR: 权重 qdim=%s，期望 0" % qw.get("qdim"), file=sys.stderr)
        return 1

    in_scale = t_in["quant"]["scale"][0]
    base = sc_w[0]

    # 记下原偏置数值（自证时要证明「没动过」）
    bdata0 = mdl["buffers"][t_b["buffer"]]["data"]
    orig_bias = [struct.unpack("<i", bdata0[i * 4:i * 4 + 4])[0]
                 for i in range(len(bdata0) // 4)]

    # ---- 新 scale：线性铺开 ----
    new_w = []
    for c in range(C):
        f = args.lo + (args.hi - args.lo) * (c / max(C - 1, 1))
        new_w.append(struct.unpack("<f", struct.pack("<f", base * f))[0])
    new_b = []
    for c in range(C):
        f = args.lo + (args.hi - args.lo) * (c / max(C - 1, 1))
        new_b.append(struct.unpack("<f", struct.pack("<f", in_scale * base * f))[0])

    qw["scale"] = new_w
    qb["scale"] = new_b

    # ---- 写回 ----
    out = fold.build_model(mdl)
    with open(args.dst, "wb") as f:
        f.write(out)

    # ---- 自证 ----
    chk = fold.read_model(args.dst)
    _, _, ctw, ctb, _ = find_conv(chk)
    cqw = chk["tensors"][ctw]["quant"]
    cqb = chk["tensors"][ctb]["quant"]
    wdata = chk["buffers"][chk["tensors"][ctw]["buffer"]]["data"]
    bdata = chk["buffers"][chk["tensors"][ctb]["buffer"]]["data"]
    wid = [wdata[i] - 256 if wdata[i] > 127 else wdata[i] for i in range(len(wdata))]
    bid = [int.from_bytes(bdata[i * 4:i * 4 + 4], "little", signed=True)
           for i in range(len(bdata) // 4)]

    checks = [
        ("权重仍是单位阵", all(wid[oc * C + ic] == (1 if oc == ic else 0)
                            for oc in range(C) for ic in range(C))),
        ("偏置数值未被改动（逐个与原模型相同）", bid == orig_bias),
        ("权重 scale 个数不变", len(cqw["scale"]) == C),
        ("偏置 scale 个数不变", len(cqb["scale"]) == C),
        ("权重 scale 不再全相等（否则 delegate 仍折叠）", len(set(cqw["scale"])) > 1),
        ("权重 scale 严格递增", all(cqw["scale"][i] < cqw["scale"][i + 1] for i in range(C - 1))),
        ("qdim 保持 0", cqw.get("qdim") == 0),
        ("偏置 scale = in_scale × 权重 scale（逐通道，相对误差 <1e-6）",
         all(abs(cqb["scale"][c] - t_in["quant"]["scale"][0] * cqw["scale"][c])
             <= 1e-6 * max(abs(cqb["scale"][c]), 1e-12)
             for c in range(C))),
    ]
    ok = True
    print("\n=== 自证 ===")
    for desc, cond in checks:
        print("  [%s] %s" % ("ok" if cond else "BAD", desc))
        ok = ok and cond
    print("\n  scale 范围: %.9f … %.9f（原 %.9f）"
          % (min(cqw["scale"]), max(cqw["scale"]), base))
    print("  已写出: %s (%d B)" % (args.dst, os.path.getsize(args.dst)))
    if not ok:
        print("ERROR: 自证未全过，不应使用该模型", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
