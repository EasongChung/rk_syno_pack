#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_1x1_channels.py —— 合成「几何固定、只改通道数」的 1×1 conv 诊断模型

## 目的（承接报告 28/30/31/32）

真机实测：挂死的最小单元 = **单个 1×1 CONV_2D，244→244 通道**；122→122 同型不挂。
独立复核（oca 报告 31）确认：**Mesa 描述符侧没有任何单个量能挑出 244**
⇒ 判别量要么是**量级阈值**，要么**不在描述符里**。

⇒ 本工具造一组**只改通道数**的模型（几何固定 50×50，1×1，relu=1，int8，per-channel 权重），
用来扫描触发边界：
    196 / 220 / 240 / 244 / 260 / 280 / 287 …

## 模型结构（严格照抄真实作业，只改通道数）

    input  [1, N/2, 2, 2500]  int8            ← 图输入（激活，无数据）
      └ RESHAPE  (shape=[1, N, 50, 50])
          └ TRANSPOSE (perm=[0,2,1,3])  ⇒ [1, 50, 50, N]
              └ CONV_2D  weights[N,1,1,N]  bias[N]  relu=1  ⇒ [1, 50, 50, N]

  · 元素数自洽：输入 N/2·2·2500 = N·2500 = 输出 N·50·50 ✓
  · **权重/偏置裁剪或按通道复制扩展**，scale/zp 沿用真实 244 通道那一份的值
    ⇒ conv_scale / shift 与真实作业同量级（不会因 shift 越界引入假象）

## 合格标准（先立，达不到就不用它）

  · N=122 ⇒ 必须**不挂**（与真实 iso-seg04 同表现）
  · N=244 ⇒ 必须**挂**（与真实 iso-seg09 同表现）
  ⇒ 这两点不过，扫描结果一律不作数。

## 用法

    python3 sweep_1x1_channels.py <折维模型.tflite> <输出目录> [N1 N2 ...]
    python3 sweep_1x1_channels.py <折维模型.tflite> <输出目录> 244 --nondg

默认扫描 196 220 240 244 260 280 287。

`--nondg`：重设计量化参数（对角权重 + 每通道唯一偏置 + 输出 scale = 输入×权重），
使输出**不再饱和到 -128** ⇒ 恢复「与 CPU 逐字节一致」判据的信息量
（见 build() 内的说明与报告 34）。
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_fold_rank5_to_4d as fold  # noqa: E402

# 真实模型里的目标作业：段 #9 的前 3 算子（RESHAPE, TRANSPOSE, CONV）
#   op#51 RESHAPE(t186 → t187[1,244,50,50])
#   op#52 TRANSPOSE(t187 → t188[1,50,50,244])
#   op#53 CONV_2D(t188, t104[244,1,1,244], t103[244]) → t189
OPS = (51, 52, 53)
CONV_OP = 53
TPL_INPUT = 186          # RESHAPE 的输入（被折维后的中间张量）
CONV_W = 104             # 权重
CONV_B = 103             # 偏置
CONV_IN = 188            # CONV 的输入（TRANSPOSE 的输出）
H_W = 50
W_W = 50


def crop_extend(arr, n):
    """把长度 244 的通道轴裁剪/扩展到 n（沿用真实值；不足按通道重复）"""
    a = list(arr)
    if len(a) == n:
        return a
    out = []
    for i in range(n):
        out.append(a[i] if i < len(a) else a[i % len(a)])
    return out


def build(mdl, n, out_path, nondg=False):
    T = mdl["tensors"]
    ops = mdl["ops"]

    # ---- 取真实作业的量化参数 ----
    wq = T[CONV_W]["quant"]
    bq = T[CONV_B]["quant"]
    iq = T[CONV_IN]["quant"]
    w_scale = crop_extend(wq["scale"], n)
    w_zp = crop_extend(wq["zp"], n) if wq.get("zp") else [0] * n
    b_scale = crop_extend(bq["scale"], n)
    b_zp = crop_extend(bq["zp"], n) if (bq.get("zp") and len(bq["zp"]) == len(bq["scale"])) \
        else (list(bq["zp"]) if bq.get("zp") else [0])
    in_scale = list(iq["scale"])
    in_zp = list(iq["zp"]) if iq.get("zp") else [0]

    # ======================================================================
    # --nondg：重设计量化参数，让输出**不再饱和**（Muse §7 的「非退化数值」层）
    #
    # 真实模型沿用的量化参数里，输出 scale = 0x1.0101p+120（≈2^127 × 输入 scale）
    # ⇒ 任何累加值 requantize 后都饱和到 int8 下界 ⇒ 输出恒为 -128 ⇒
    #    「与 CPU 逐字节一致」这条判据**零信息量**（一个什么都不算、只填 -128 的驱动也能通过）。
    #
    # 非退化设计（1×1、输入为全零 ⇒ 输出只由偏置决定）：
    #   · 权重：对角阵 w[oc][oc] = 1（其余 0）⇒ 通道之间**完全独立**
    #   · 偏置：bias[oc] = (oc mod 127) - 63（int32 累加域）⇒ 每个通道一个**唯一可辨**的值
    #   · 输出 scale = 输入 scale × 权重 scale（⇒ multiplier ≈ 1，累加值不被压缩）
    #   · 输出 zp = 0 ⇒ 输出 = bias[oc] ∈ [-63, 63] ⇒ **绝不饱和**
    # ⇒ 于是：① ④ 判据恢复信息量；② 任何通道错位/错序都能被逐字节比对抓到。
    # ======================================================================
    if nondg:
        w_scale = [in_scale[0]] * n
        w_zp = [0] * n
        b_scale = [in_scale[0] * in_scale[0]] * n     # = 输入 scale × 权重 scale
        b_zp = [0] * n

    # ---- 权重/偏置数据：裁剪/扩展 ----
    wdata = T[CONV_W]["buffer"]
    bdata = T[CONV_B]["buffer"]
    wbuf = mdl["buffers"][wdata]["data"]
    bbuf = mdl["buffers"][bdata]["data"]
    w_elems = n * n                      # [N,1,1,N] 的元素数
    new_w = bytearray()
    for oc in range(n):
        if nondg:
            # 对角阵：w[oc][oc] = 1（int8），其余 0 ⇒ 通道完全独立
            for ic in range(n):
                new_w.append(1 if ic == oc else 0)
            continue
        src_oc = oc if oc < 244 else oc % 244
        row = wbuf[src_oc * 244: (src_oc + 1) * 244]     # 该输出通道的 244 个输入通道权重
        for ic in range(n):
            src_ic = ic if ic < 244 else ic % 244
            new_w.append(row[src_ic])
    assert len(new_w) == w_elems
    # ★ 偏置是 **int32**（4 字节/元素）！第一版按 int8 写 1 字节/元素 ⇒
    #   模型字节数不符 ⇒ 解释器报 "Tensor 7 is invalidly specified (976 != 244)"
    #   ⇒ 8 个扫描模型**一个都没跑起来**（差点被当成「都不挂」的假绿）。
    #   ⇒ 现在按 int32 重写，并加**字节数断言**（本地就能抓这类错）。
    new_b = bytearray()
    for oc in range(n):
        if nondg:
            # 每个通道一个唯一可辨的偏置值（int32 累加域，范围 [-63,63] ⇒ 绝不饱和）
            new_b += struct.pack("<i", (oc % 127) - 63)
            continue
        src_oc = oc if oc < 244 else oc % 244
        new_b += bbuf[src_oc * 4: (src_oc + 1) * 4]
    assert len(new_b) == n * 4, "偏置字节数应为 N*4（int32），实得 %d" % len(new_b)

    # ---- 新模型的张量表（重排，索引从 0 开始）----
    def mk(shape, ttype, buf_idx, quant, name):
        return {"shape": list(shape), "type": ttype, "buffer": buf_idx, "name": name,
                "quant": quant, "is_variable": False, "has_rank": True}

    def q(scales, zps, qdim=0):
        return {"min": [], "max": [], "scale": list(scales), "zp": list(zps), "qdim": qdim}

    bufs = [{"data": None, "offset": 0, "size": 0}]        # buffer 0 = 空
    def add_buf(data):
        bufs.append({"data": bytes(data), "offset": 0, "size": len(data)})
        return len(bufs) - 1

    assert len(new_w) == n * n, "权重字节数应为 N*N，实得 %d" % len(new_w)
    b_w = add_buf(new_w)
    b_b = add_buf(new_b)
    b_shape = add_buf(struct.pack("<%di" % 4, 1, n, H_W, W_W))
    # ★ perm 必须是 (0,2,3,1)：真实模型里 op#52（1×1 块的那个 TRANSPOSE）用的就是 t1=(0,2,3,1)，
    #   委派**只认这个 perm** ⇒ 我第一版用 (0,2,1,3) ⇒ 委派判该 TRANSPOSE unsupported
    #   ⇒ 回退 CPU ⇒ CPU 端 conv 参数不自洽（filter_input_channel=0）⇒ 模型根本没跑起来。
    b_perm = add_buf(struct.pack("<%di" % 4, 0, 2, 3, 1))

    tensors = [
        mk([1, n // 2, 2, W_W * H_W], T[TPL_INPUT]["type"], 0,
           q(in_scale, in_zp), "och.sweep.input"),
        mk([1, n, H_W, W_W], T[186 + 1]["type"], 0, q(in_scale, in_zp), "och.sweep.flat"),
        mk([1, H_W, W_W, n], T[CONV_IN]["type"], 0, q(in_scale, in_zp), "och.sweep.nhwc"),
        mk([1, H_W, W_W, n], T[189]["type"], 0, q(in_scale, in_zp), "och.sweep.out"),
        mk([4], 2, b_shape, None, "och.sweep.shape"),
        mk([4], 2, b_perm, None, "och.sweep.perm"),
        mk([n, 1, 1, n], T[CONV_W]["type"], b_w, q(w_scale, w_zp, wq.get("qdim", 0)), "och.sweep.w"),
        mk([n], T[CONV_B]["type"], b_b, q(b_scale, b_zp, bq.get("qdim", 0)), "och.sweep.b"),
    ]
    I_IN, I_FLAT, I_NHWC, I_OUT, I_SHAPE, I_PERM, I_W, I_B = range(8)

    # ---- 算子：照抄真实作业的选项，只改引用的张量 ----
    def clone_op(src, inputs, outputs):
        o = dict(src)
        o["inputs"] = list(inputs)
        o["outputs"] = list(outputs)
        opts = src.get("opts")
        if opts:
            opts = dict(opts)
            if "new_shape" in opts:
                opts["new_shape"] = [1, n, H_W, W_W]
            o["opts"] = opts
        return o

    new_ops = [
        clone_op(ops[OPS[0]], [I_IN, I_SHAPE], [I_FLAT]),
        clone_op(ops[OPS[1]], [I_FLAT, I_PERM], [I_NHWC]),
        clone_op(ops[OPS[2]], [I_NHWC, I_W, I_B], [I_OUT]),
    ]

    out_model = {
        "version": mdl["version"],
        "description": "och sweep 1x1 conv C=%d" % n,
        "metadata_buffer": mdl["metadata_buffer"],
        "codes": mdl["codes"],
        "buffers": bufs,
        "tensors": tensors,
        "ops": new_ops,
        "subgraph": {"inputs": [I_IN], "outputs": [I_OUT], "name": "och.sweep", "debug_md": 0},
        "sigs": [],
        "metas": [],
    }
    blob = fold.build_model(out_model)
    with open(out_path, "wb") as f:
        f.write(blob)
    return len(blob), len(new_w), len(new_b)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src = sys.argv[1]
    outdir = sys.argv[2]
    args = sys.argv[3:]
    nondg = False
    if "--nondg" in args:
        nondg = True
        args = [a for a in args if a != "--nondg"]
    ns = [int(x) for x in args] or [196, 220, 240, 244, 260, 280, 287]
    os.makedirs(outdir, exist_ok=True)
    mdl = fold.read_model(src)
    suffix = "-nondg" if nondg else ""
    for n in ns:
        p = os.path.join(outdir, "sweep1x1-c%04d%s.tflite" % (n, suffix))
        sz, nw, nb = build(mdl, n, p, nondg=nondg)
        print("  C=%-5d → %s  (%d B; 权重 %d B, 偏置 %d B)%s"
              % (n, os.path.basename(p), sz, nw, nb, "  [非退化量化]" if nondg else ""))
    return 0


if __name__ == '__main__':
    sys.exit(main())
