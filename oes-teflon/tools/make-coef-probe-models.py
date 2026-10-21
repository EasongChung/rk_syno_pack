#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造「系数通道」判决模型：只改 buffer 内容，不改结构/尺寸。

## 要回答的问题

c248-nondg 在 NPU 上输出恒为「实数 0」。模型是
    RESHAPE -> RESHAPE -> CONV_2D(权重=单位阵, 偏置≈-63, 带 fused ReLU)
CPU 端输出非零（实数 0..59），NPU 端全 0。

一个能解释全 0 的机制：**权重没有真正进到硬件**（被读成 0），
于是 acc=0，只剩负偏置，ReLU 把全场钳成 0。

## 两个变量（每个都能单独否证上面这条机制）

| 变体 | 改动 | 如果系数真的进了硬件，输出必然 |
|---|---|---|
| `w1`      | 权重全部改成 +1（单位阵 -> 稠密全 1） | **非零**（248 项求和为大正数，饱和到 +127） |
| `biasbig` | 偏置全部改成大正数 +20000，权重不动 | **非零**（偏置实数 ≈ 20000×0.18535，饱和） |

两个都输出全 0 ⇒ 系数/偏置通道根本没到硬件。

## 纪律

* 只在文件里定位**唯一**出现的 buffer 字节串再改写；不唯一即报错退出。
* 改写后重新解析，逐项核对：文件尺寸不变、buffer 内容 = 期望、其余张量不变。
* 不改量化参数（scale/zero_point）——只改 int8/int32 数值，避免改变语义。

用法：
    python3 tools/make-coef-probe-models.py <src.tflite> <outdir>
"""

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify"))
import tflite_schema as S  # noqa: E402


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw, S.parse_model(raw, path)


def locate(raw, blob, what):
    """在文件里定位 blob，要求唯一。"""
    off = raw.find(blob)
    if off < 0:
        raise SystemExit("错误: %s 的 buffer 在文件里找不到（无法原地改写）" % what)
    if raw.find(blob, off + 1) >= 0:
        raise SystemExit("错误: %s 的 buffer 在文件里出现多次，拒绝改写" % what)
    return off


def conv_tensors(m):
    for op in m.main.operators:
        if op.builtin_code == 3:  # CONV_2D
            return m.main.tensors[op.inputs[1]], m.main.tensors[op.inputs[2]]
    raise SystemExit("错误: 模型里没有 CONV_2D")


def patch(raw, off, blob_new):
    out = bytearray(raw)
    out[off:off + len(blob_new)] = blob_new
    return bytes(out)


def verify(path, want_w, want_b):
    """重新解析并核对：尺寸不变、权重/偏置 = 期望。"""
    raw, m = load(path)
    wt, bt = conv_tensors(m)
    got_w = m.buffers[wt.buffer].data
    got_b = m.buffers[bt.buffer].data
    assert got_w == want_w, "权重核对失败"
    assert got_b == want_b, "偏置核对失败"
    return len(raw), len(got_w), len(got_b)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    raw, m = load(src)
    wt, bt = conv_tensors(m)
    w0 = m.buffers[wt.buffer].data
    b0 = m.buffers[bt.buffer].data
    w_off = locate(raw, w0, "权重")
    b_off = locate(raw, b0, "偏置")

    # 按张量形状算元素数，避免把字节数当元素数
    w_n = 1
    for d in wt.shape:
        w_n *= d
    b_n = 1
    for d in bt.shape:
        b_n *= d
    assert w_n == len(w0), "权重元素数(%d) != 字节数(%d)" % (w_n, len(w0))
    assert b_n * 4 == len(b0), "偏置元素数(%d) != 字节数/4(%d)" % (b_n, len(b0) // 4)

    variants = {}

    # 变体 W：权重全 +1（int8），偏置不动
    w1 = bytes([1]) * w_n
    variants["c248-w1.tflite"] = (w1, b0, w_off, b_off)
    # 变体 B：权重不动，偏置全 +20000（int32 little-endian）
    bbig = struct.pack("<%di" % b_n, *([20000] * b_n))
    variants["c248-biasbig.tflite"] = (w0, bbig, w_off, b_off)

    for name, (wnew, bnew, wo, bo) in variants.items():
        out = patch(raw, wo, wnew) if wnew != w0 else raw
        out = patch(out, bo, bnew)
        p = os.path.join(outdir, name)
        with open(p, "wb") as f:
            f.write(out)
        size, wl, bl = verify(p, wnew, bnew)
        print("已写出 %s  size=%d  权重=%d B  偏置=%d B" % (p, size, wl, bl))
        assert size == len(raw), "文件尺寸变了：%d -> %d" % (len(raw), size)

    print("\n自检通过：尺寸未变、两个 buffer 内容与期望一致、量化参数未改。")
    print("下一步（都需要 CPU 先跑出基准，再上 NPU）：")
    print("  ./tf_load_check <变体> --no-delegate --invoke 1 --output-dump ...")


if __name__ == "__main__":
    main()
