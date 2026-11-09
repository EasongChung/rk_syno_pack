#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f4-exact-reference.py —— F4 判决模型的「精确整数数学」参考输出（零设备）

AGENTS.md 四级判据④的权威基准 = 精确整数数学（单次精确舍入），不是 TFLite CPU
（TFLite 两步重量化有 ≤1 LSB 已知偏差，报告43）。本脚本对判决模型逐元素计算：

    acc[o,p] = sum_i in_q[p,i]*w_q[o,i] + bias_q[o]      （zp 全 0，对称）
    out_q[o,p] = round_half_away(acc * S_IN*S_W/S_OUT) 截断到 int8

输入 = manifest 对应 input.bin；权重/偏置从模型读回（fold.read_model）。
产出：f4-decision-c{N}-exact.bin（NHWC [1,50,50,N] int8）+ 统计 JSON。

用法:
    python3 tools/f4-exact-reference.py [--dir model-surgery/f4]
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))
import tflite_fold_rank5_to_4d as fold  # noqa: E402

S_IN, S_W, S_OUT = 0.02, 0.01, 0.5
# 用整数比例做精确舍入：conv_scale = S_IN*S_W/S_OUT = 1/2500（精确有理数）
NUM, DEN = 1, 2500


def round_half_away(x_num, x_den):
    """round(x_num/x_den) half-away-from-zero，纯整数 arithmetic。"""
    sign = -1 if x_num < 0 else 1
    a = abs(x_num)
    q, r = divmod(a, x_den)
    if 2 * r >= x_den:
        q += 1
    return sign * q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(HERE), "model-surgery", "f4"))
    args = ap.parse_args()
    stats = {}
    for c in (122, 120):
        mp = os.path.join(args.dir, "f4-decision-c%d.tflite" % c)
        ip = os.path.join(args.dir, "f4-decision-c%d-input.bin" % c)
        mdl = fold.read_model(mp)
        T = mdl["tensors"]
        w_raw = bytes(mdl["buffers"][T[1]["buffer"]]["data"])
        b_raw = bytes(mdl["buffers"][T[2]["buffer"]]["data"])
        assert len(w_raw) == c * c, "权重字节数 %d != c*c=%d" % (len(w_raw), c * c)
        w = [struct.unpack_from("b", w_raw, i)[0] for i in range(c * c)]
        bias = list(struct.unpack("<%di" % c, b_raw))
        inp = open(ip, "rb").read()
        assert len(inp) == 50 * 50 * c
        # 输入按像素重排：pix[p] = list of c int8
        pix = []
        for p in range(2500):
            row = inp[p * c:(p + 1) * c]
            pix.append([v if v < 128 else v - 256 for v in row])
        wrows = [w[o * c:(o + 1) * c] for o in range(c)]
        out = bytearray(2500 * c)
        sat = 0
        ch121_absmax = 0
        for o in range(c):
            wr = wrows[o]
            bo = bias[o]
            for p in range(2500):
                acc = sum(a * b for a, b in zip(pix[p], wr)) + bo
                q = round_half_away(acc * NUM, DEN)
                if q > 127:
                    q = 127
                    sat += 1
                elif q < -128:
                    q = -128
                    sat += 1
                out[p * c + o] = q & 0xFF
                if o == c - 1:
                    v = q if q < 128 else q - 256
                    if abs(v) > ch121_absmax:
                        ch121_absmax = abs(v)
        op = os.path.join(args.dir, "f4-decision-c%d-exact.bin" % c)
        with open(op, "wb") as f:
            f.write(bytes(out))
        stats["c%d" % c] = {"file": os.path.basename(op), "bytes": len(out),
                            "saturated": sat, "last_channel_absmax": ch121_absmax}
        print("OK: c=%d → %s（饱和 %d / %d，末通道 |max|=%d）"
              % (c, op, sat, 2500 * c, ch121_absmax))
    with open(os.path.join(args.dir, "exact-reference-stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
