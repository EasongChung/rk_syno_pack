#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_grouped_dw.py -- 修复 ShuffleNetV2 下采样单元中被误标为 Conv2D 的深度卷积

背景：
  ShuffleNetV2 下采样单元 Branch 2 的 3x3 卷积是深度卷积 (group=C)。
  onnx2tf 在转换时将 op#58 和 op#153 误标为了普通 CONV_2D (code 3)，
  但其权重形状依然是深度卷积格式 [C, 3, 3, 1]。
  这导致 TFLite 的标准 Conv2D 读越界，pad-to-8.py 也会因为 in维!=输入通道 而拒绝。

功能：
  1. 识别并更正所有此类伪装成 Conv2D 的深度卷积。
  2. 将 opcode 设为 DEPTHWISE_CONV_2D (code 4)。
  3. 将权重形状规整为 TFLite 标准的 [1, 3, 3, C]。
  4. 物理转置权重数据：W_dw[ky, kx, c] = W_conv[c, ky, kx]。
  5. 导出自洽的 TFLite 模型。
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tflite_fold_rank5_to_4d as fold

def fix_model(src_path, dst_path):
    print(f"=== 读取模型: {src_path} ===")
    mdl = fold.read_model(src_path)
    ops = mdl["ops"]
    T = mdl["tensors"]
    
    # 查找 DEPTHWISE_CONV_2D 的 opcode_index
    dw_code_idx = None
    for idx, code in enumerate(mdl["codes"]):
        if code["builtin"] == 4:
            dw_code_idx = idx
            break
            
    if dw_code_idx is None:
        dw_code_idx = len(mdl["codes"])
        mdl["codes"].append({"builtin": 4, "version": 1})
        print(f"  在代码表新增 DEPTHWISE_CONV_2D: index={dw_code_idx}")

    fixed_count = 0
    for i, op in enumerate(ops):
        b = mdl["codes"][op["opcode_index"]]["builtin"]
        if b == 3 and len(op["inputs"]) >= 2: # CONV_2D
            w_idx = op["inputs"][1]
            wt = T[w_idx]
            wsh = wt["shape"]
            in_idx = op["inputs"][0]
            in_sh = T[in_idx]["shape"]
            # 检测特征: 权重是 [C, kh, kw, 1] 且 输入是 [1, h, w, C]，kh > 1
            if len(wsh) == 4 and wsh[3] == 1 and len(in_sh) == 4 and in_sh[3] == wsh[0] and wsh[1] > 1:
                c, kh, kw, _ = wsh
                print(f"  发现结构性伪装深度卷积: op#{i} in={in_sh} weight={wsh} -> 修复为 DEPTHWISE_CONV_2D")
                op["opcode_index"] = dw_code_idx
                wt["shape"] = [1, kh, kw, c]
                # 字节物理转置
                b_idx = wt["buffer"]
                raw_w = mdl["buffers"][b_idx]["data"]
                wsz = 1 # int8 = 1 byte
                w_new = bytearray(len(raw_w))
                for ci in range(c):
                    for k in range(kh * kw):
                        # 原数据: [c, kh*kw] -> 新数据: [kh*kw, c]
                        w_new[(k * c + ci) * wsz : (k * c + ci + 1) * wsz] = raw_w[(ci * kh * kw + k) * wsz : (ci * kh * kw + k + 1) * wsz]
                mdl["buffers"][b_idx]["data"] = bytes(w_new)
                fixed_count += 1
                
    print(f"共修复 {fixed_count} 个结构畸形深度卷积。")
    blob = fold.build_model(mdl)
    with open(dst_path, "wb") as f:
        f.write(blob)
    print(f"=== 成功写出修复后模型: {dst_path} ({len(blob)} 字节) ===")
    return fixed_count

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python3 fix_grouped_dw.py <in.tflite> <out.tflite>")
        sys.exit(1)
    fix_model(sys.argv[1], sys.argv[2])
