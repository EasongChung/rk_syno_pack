#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f4-make-decision-models.py —— F4 store/add 语义判决实验载体（零设备）

## 判决定什么（交接包§4 / oca 复核四件套，2026-11-09）

F4 钳位把幻影核变成「重算通道 C-1」。NN core 输出若是 store 语义 ⇒ 幂等无害；
若是累加/偏置重算 ⇒ 末通道数值被写坏。本载体在**已知不挂点 C=122** 上判决：

  · f4-decision-c122.tflite  —— 判决件：m=6，钳位触发（去重幻影原值 {122..127}）
  · f4-decision-c120.tflite  —— 控制对：%8==0，m=0，钳位不触发，确立 ±1 LSB 基线

## 设计（oca 四件套 + 我方修正）

  · 单 CONV_2D 1×1，NHWC [1,50,50,C] → [1,50,50,C]，无融合激活（保留负值信号）；
  · per-tensor 对称 int8（zp 全 0）：绕开 per-channel 折叠损失与 zp 对称门伴生假设
    （c248 per-tensor 曾达精确整数数学 100%，报告43）；
  · 权重 = 确定性伪随机满阵（非对角），输入 = 固定种子随机 int8（随模型给 .bin）；
  · 偏置逐通道不同且非零，**ch121 偏置贡献特意放大**（≈10 LSB）：
      仅 ch121 出现 Δ≈偏置贡献 ⇒ 偏置重算形态；平方级放大 ⇒ 累加形态；
      ch121 max|Δ|≤1 且其余通道与控制对同量级 ⇒ store 幂等；
  · 输出 scale 留足裕量（累加器 std≈24 LSB），避免饱和把判据吃掉；
  · 钳位触发判据 = och-f4 日志**去重幻影原值基数 = m**（不是日志行数：
    write_core_6 站点2 每 block 重复打印，in_c=122 时预测 132 行）。

## 自检

生成后用独立解析器 verify/tflite_schema.py 读回，逐项核对 shape/量化/拓扑/缓冲字节；
不过不产出文件。另做负控制：故意改坏一个量化参数，解析核对必须报不一致。

用法:
    python3 tools/f4-make-decision-models.py [-o model-surgery/f4] [--seed 20261109]
"""

import argparse
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))   # vendored flatbuffers + tflite 生成代码
sys.path.insert(0, os.path.join(ROOT, "verify"))

TENSOR_TYPE_INT8 = 9
TENSOR_TYPE_INT32 = 2
BUILTIN_CONV_2D = 3
PADDING_VALID = 1
ACTIVATION_NONE = 0

H = W = 50
S_IN = 0.02
S_W = 0.01
S_OUT = 0.5
S_B = S_IN * S_W          # TFLite 不变量：bias_scale = in_scale × weight_scale
Z_ALL = 0                 # 对称量化


def xorshift_bytes(seed, n):
    out = bytearray()
    s = seed & 0xFFFFFFFF or 1
    for _ in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out.append(s & 0xFF)
    return bytes(out)


def rand_i8(seed, n):
    """确定性伪随机 int8（-128..127 全域）"""
    return bytes((b if b < 128 else b - 256) & 0xFF for b in xorshift_bytes(seed, n))


def design_bias(c):
    """逐通道偏置实数值：ch=c-1 放大到 5.0（输出贡献≈10 LSB），其余小幅交错。"""
    out = []
    for i in range(c):
        if i == c - 1:
            out.append(5.0)
        else:
            out.append((((i * 37) % 13) - 6) * 0.5)
    return out


def build(c, seed):
    """返回 (model_bytes, input_bytes, weights_bytes, bias_int32_list, bias_real_list)

    ★ 写入路径 = tflite_fold_rank5_to_4d.build_model（vendored 官方 flatbuffers +
      生成代码）——sweep 系列模型用它生成并被真机 TFLite 加载过（报告35），实证可用。
      不用 tools/tflite_write.py 的自研 Builder：本会话实测它与当前 tflite_schema
      解析器不匹配（make_tiny_model 产物也解析失败），不在 F4 范围内修它，挂账。
    """
    import tflite_fold_rank5_to_4d as fold

    w = rand_i8(seed + 1, c * 1 * 1 * c)          # [c,1,1,c]
    bias_real = design_bias(c)
    bias_q = [int(round(v / S_B)) for v in bias_real]
    for v in bias_q:
        if not (-(2**31) <= v < 2**31):
            raise ValueError("bias int32 溢出")
    bdata = struct.pack("<%di" % c, *bias_q)
    inp = rand_i8(seed + 2, 1 * H * W * c)

    def q(scale):
        # zp 必须显式给 [0]：对称量化也是仿射量化（is_affine 需要 zp 向量存在）
        return {"min": [], "max": [], "scale": [scale], "zp": [Z_ALL], "qdim": 0}

    mdl = {
        "version": 3,
        "description": "F4 store/add decision c%d (och, 2026-11-09)" % c,
        "metadata_buffer": [], "sigs": [], "metas": [],
        "codes": [{"dep": BUILTIN_CONV_2D, "custom": None, "ver": 6,
                   "builtin": BUILTIN_CONV_2D}],
        "buffers": [{"data": None, "offset": 0, "size": 0},
                    {"data": w, "offset": 0, "size": 0},
                    {"data": bdata, "offset": 0, "size": 0}],
        "tensors": [
            {"shape": [1, H, W, c], "type": TENSOR_TYPE_INT8, "buffer": 0,
             "name": "in", "quant": q(S_IN), "is_variable": False, "has_rank": True},
            {"shape": [c, 1, 1, c], "type": TENSOR_TYPE_INT8, "buffer": 1,
             "name": "w", "quant": q(S_W), "is_variable": False, "has_rank": True},
            {"shape": [c], "type": TENSOR_TYPE_INT32, "buffer": 2,
             "name": "bias", "quant": q(S_B), "is_variable": False, "has_rank": True},
            {"shape": [1, H, W, c], "type": TENSOR_TYPE_INT8, "buffer": 0,
             "name": "out", "quant": q(S_OUT), "is_variable": False, "has_rank": True},
        ],
        "ops": [{"opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3],
                 "opts_type": 1,   # BuiltinOptions.Conv2DOptions（vendored schema 实测=1）
                 "opts": {"__type__": "Conv2DOptions", "Padding": PADDING_VALID,
                          "StrideW": 1, "StrideH": 1, "FusedActivationFunction": ACTIVATION_NONE,
                          "DilationWFactor": 1, "DilationHFactor": 1, "QuantizedBiasType": 0}}],
        "subgraph": {"inputs": [0], "outputs": [3],
                     "name": "f4-decision-c%d" % c, "debug_md": 0},
    }
    return fold.build_model(mdl), inp, w, bias_q, bias_real


def verify_readback(path, c, w, bias_q):
    """读回核对（不过即失败，不认产出）。

    主核对 = vendored 官方 flatbuffers 生成代码读取器（fold.read_model，与写入器
    不同层的独立实现：写入走 Add* 构造、读取走 GetRootAs 访问器）；
    交叉核对 = 自研 tflite_schema 的 shape/量化/拓扑（其 legacy union options 口径
    与生成代码不一致，options 一项不以它为准 —— 本会话实测记录在案）。
    """
    import tflite_fold_rank5_to_4d as fold
    mdl = fold.read_model(path)
    T, ops = mdl["tensors"], mdl["ops"]
    assert len(T) == 4, "张量数 %d != 4" % len(T)
    assert T[0]["shape"] == [1, H, W, c]
    assert T[1]["shape"] == [c, 1, 1, c]
    assert T[2]["shape"] == [c]
    assert T[3]["shape"] == [1, H, W, c]
    assert T[0]["type"] == TENSOR_TYPE_INT8 and T[2]["type"] == TENSOR_TYPE_INT32
    assert len(ops) == 1
    assert mdl["codes"][ops[0]["opcode_index"]]["builtin"] == BUILTIN_CONV_2D
    assert ops[0]["inputs"] == [0, 1, 2] and ops[0]["outputs"] == [3]
    assert mdl["subgraph"]["inputs"] == [0] and mdl["subgraph"]["outputs"] == [3]
    for idx, s in ((0, S_IN), (1, S_W), (2, S_B), (3, S_OUT)):
        q = T[idx]["quant"]
        assert len(q["scale"]) == 1 and abs(q["scale"][0] - s) < 1e-9, "tensor%d scale 不符" % idx
        assert q["zp"] == [Z_ALL], "tensor%d zp 不符" % idx
    # 权重/偏置字节逐字核对
    assert bytes(mdl["buffers"][T[1]["buffer"]]["data"]) == w, "权重缓冲字节不一致"
    assert bytes(mdl["buffers"][T[2]["buffer"]]["data"]) == struct.pack("<%di" % c, *bias_q), \
        "偏置缓冲字节不一致"
    # conv options 核对（VALID / stride1 / 无激活 / dilation1）
    o = ops[0]["opts"]
    assert ops[0]["opts_type"] == 1 and o["__type__"] == "Conv2DOptions"
    assert o["Padding"] == PADDING_VALID
    assert o["StrideW"] == 1 and o["StrideH"] == 1
    assert o["FusedActivationFunction"] == ACTIVATION_NONE
    assert o["DilationWFactor"] == 1 and o["DilationHFactor"] == 1
    # 交叉核对：自研解析器的 shape/量化/拓扑（options 口径差异已记录，不查 options）
    import tflite_schema as S
    with open(path, "rb") as f:
        m2 = S.parse_model(f.read(), path)
    sg2 = m2.main
    assert len(sg2.tensors) == 4 and len(sg2.operators) == 1
    assert list(sg2.tensors[0].shape) == [1, H, W, c]
    assert list(sg2.tensors[3].shape) == [1, H, W, c]
    assert sg2.operators[0].builtin_code == BUILTIN_CONV_2D
    assert abs(sg2.tensors[0].quantization.scale[0] - S_IN) < 1e-9
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=os.path.join(ROOT, "model-surgery", "f4"))
    ap.add_argument("--seed", type=int, default=20261109)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    manifest = {"purpose": "F4 store/add 语义判决（C=122 判决件 + C=120 控制对）",
                "seed": args.seed, "geometry": [1, H, W], "kernel": [1, 1],
                "quant": {"in": [S_IN, Z_ALL], "w": [S_W, Z_ALL],
                          "bias": [S_B, Z_ALL], "out": [S_OUT, Z_ALL]},
                "models": []}
    for c in (122, 120):
        data, inp, w, bias_q, bias_real = build(c, args.seed)
        mp = os.path.join(args.outdir, "f4-decision-c%d.tflite" % c)
        ip = os.path.join(args.outdir, "f4-decision-c%d-input.bin" % c)
        with open(mp, "wb") as f:
            f.write(data)
        with open(ip, "wb") as f:
            f.write(inp)
        verify_readback(mp, c, w, bias_q)
        m = (8 - c % 8) % 8
        entry = {
            "file": os.path.basename(mp), "bytes": len(data),
            "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "input_file": os.path.basename(ip),
            "input_md5": hashlib.md5(inp).hexdigest(),
            "out_c": c, "phantom_m": m,
            "clamp_criterion": "och-f4 日志去重幻影原值基数 == %d" % m,
            "bias_last_channel_real": bias_real[c - 1],
            "bias_last_channel_out_lsb": round(bias_real[c - 1] / S_OUT, 2),
        }
        manifest["models"].append(entry)
        print("OK: %s (%d B, md5=%s, m=%d, 读回核对通过)"
              % (mp, len(data), entry["md5"], m))
    # 负控制：改坏量化参数后读回核对必须失败（判据能失败）
    data, inp, w, bias_q, _ = build(122, args.seed)
    bad = os.path.join(args.outdir, "_negctl_bad.tflite")
    with open(bad, "wb") as f:
        f.write(data)
    # 直接改坏文件里的一个 scale 浮点字节（S_IN 的 float32 表示必在文件中）
    raw = bytearray(data)
    needle = struct.pack("<f", S_IN)
    at = raw.find(needle)
    if at < 0:
        raise RuntimeError("负控制 setup 失败：找不到 S_IN 浮点字节")
    raw[at] ^= 0x01
    with open(bad, "wb") as f:
        f.write(bytes(raw))
    try:
        verify_readback(bad, 122, w, bias_q)
    except AssertionError:
        print("OK: 负控制通过（改坏 scale 后读回核对报不一致 ⇒ 判据能失败）")
        os.remove(bad)
    else:
        raise RuntimeError("负控制失效：改坏 scale 后读回核对仍通过")
    mf = os.path.join(args.outdir, "manifest.json")
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("OK: manifest → %s" % mf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
