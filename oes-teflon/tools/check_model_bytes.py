#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_model_bytes.py —— 静态检查 TFLite 模型的「常量 buffer 字节数 vs 张量声明」是否一致

## 为什么需要这道门

合成诊断模型时，我方踩了两次同类的坑：
  ① 偏置是 **int32**（4 字节/元素），却按 int8 写了 1 字节/元素
     ⇒ 解释器直接拒绝：`Tensor 7 is invalidly specified in schema (976 != 244)`
  ② 校验脚本自己把 INT8(type 9) 的元素大小映成了 32 字节 ⇒ **假警报**
     ⇒ 「检查器自己会错」这条纪律，不是口号，是本项目的日常。

⇒ 本工具把「字节数必须与声明一致」变成**本地可跑的门**，
   免得再出现「模型没跑起来却被当成实验结果」。

## 用法

    python3 check_model_bytes.py <model.tflite> [更多...]

退出码: 0 = 全部一致；非 0 = 有不一致
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_fold_rank5_to_4d as fold  # noqa: E402

# TFLite TensorType → 元素字节数
ESZ = {
    0: 4,    # FLOAT32
    1: 2,    # FLOAT16
    2: 4,    # INT32
    3: 1,    # UINT8
    4: 8,    # INT64
    5: 0,    # STRING（变长，不校验）
    6: 1,    # BOOL
    7: 2,    # INT16
    9: 1,    # INT8
}


def check(path):
    m = fold.read_model(path)
    T, B = m["tensors"], m["buffers"]
    bad = []
    for i, t in enumerate(T):
        esz = ESZ.get(t["type"])
        if esz is None or esz == 0:
            continue
        n = 1
        for d in t["shape"]:
            n *= d
        data = B[t["buffer"]]["data"] if t["buffer"] < len(B) else None
        if not data:
            continue                      # 激活张量：空 buffer，正常
        need = n * esz
        if len(data) != need:
            bad.append((i, t["name"], t["type"], t["shape"], need, len(data)))
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print("  [缺] %s" % path)
            rc = 1
            continue
        bad = check(path)
        name = os.path.basename(path)
        if bad:
            rc = 1
            print("  [✗] %-32s 字节数不符 %d 处：" % (name, len(bad)))
            for i, tn, ty, shape, need, have in bad:
                print("        t%-3d %-22s type=%-2s shape=%-22s 需要 %d B，实得 %d B"
                      % (i, tn, ty, "x".join(map(str, shape)), need, have))
        else:
            print("  [✓] %-32s 常量字节数全部与声明一致" % name)
    return rc


if __name__ == '__main__':
    sys.exit(main())
