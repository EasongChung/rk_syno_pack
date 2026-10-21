#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tflite_write 的已知缺口登记表（**故意失败的测试**）。

`test_tflite_write.py` 覆盖 table/vtable/标量/字符串/各类标量向量/嵌套 table，
全部通过。但有一类组合**尚未修复**，在此登记为**显式失败**，
这样任何人改动 Builder 后跑一遍就知道有没有把它弄坏、也一目了然哪里还缺。

## 缺口：`create_offset_vector` 在 `minalign == 8` 时元素偏移错误

`minalign` 是「历史最大字段宽度」。一旦用过 8 字节字段（典型就是
`QuantizationParameters.zero_point` 的 int64 向量），`minalign` 升到 8。

此时 `create_offset_vector` 的元素 uoffset 值算错：

| minalign | 向量长度 | 期望元素值 | 实测 |
|---|---|---|---|
| 4 | ✅ 2 | `[len-t2, len-t1]` | ✅ |
| 8 | ✅ 2 | `[len-t2, len-t1]` | ❌ 第一个元素为 0 |

### 现象

字节流里长度字段与元素之间多出 padding 间隔，导致 `真实位置 = 写入位置 + 值`
指不到子表（解析报 `vtable 长度非法` / `vtable_len=0`）。

### 已排除的可能

* ❌ 不是 `prepend_uoffset` 的公式本身 —— `minalign=4` 时同一段代码完全正确；
* ❌ 不是「子对象必须在 start_object 之前建好」那条规则 —— 已按规则重排
  （`make_tiny_model.py` 的顶层与 `_tensor`/`_operator` 全部改过），
  且 `create_offset_vector` 单独构造的极简用例在 `minalign=4` 下正确；
* ❌ 不是回填索引 `len(buf) - head` —— 逐步 dump 验证过该索引恒为 0（正确）。

### 尚未查明

`minalign=8` 时 `prep(4, 0)` 引入的 padding 与「逆序 prepend 元素」叠加后，
长度字段到第一个元素之间多出 8 字节。根因未定。

### 影响范围

| 用途 | 是否受影响 |
|---|---|
| `test_tflite_write.py` 覆盖的 15 项（table/标量/字符串/标量向量/嵌套） | ❌ 不受影响 |
| `tools/make_tiny_model.py` 产出的小模型 | ⚠️ **受影响**，当前**不可用** |
| 已有 `models/pertensor.tflite`（外部产出的真模型） | 不受影响（本文件只参与生成） |

### 处置

**不猜测、不糊弄**：在修好之前，`make_tiny_model.py` **必须**在自己的自检里
失败退出（它已经这样做了 —— 独立解析器 `verify/tflite_schema.py` 解析不过就不落盘）。
绝不让一个编码可疑的 `.tflite` 上设备 —— 那会浪费一整轮真机验证。

下一步若要继续，正确的切入点是**逐字节 dump `minalign=8` 下
`create_offset_vector` 的构建过程**（本文件末尾的 `__main__` 会打印），
比对 `minalign=4` 的同一步，找出 padding 多出来的来源。
"""

import struct
import sys

sys.path.insert(0, "/root/workspace/oes-teflon/tools")
from tflite_write import Builder  # noqa: E402


def read_vec(b: bytes, pos: int):
    p = pos + struct.unpack_from("<I", b, pos)[0]
    n = struct.unpack_from("<I", b, p)[0]
    return n, [struct.unpack_from("<I", b, p + 4 + 4 * i)[0] for i in range(n)]


def build(minalign8: bool):
    b = Builder()
    if minalign8:
        b.create_long_vector([0, -128])      # 把 minalign 抬到 8
    b.start_object(0)
    t1 = b.end_object()
    b.start_object(0)
    t2 = b.end_object()
    v = b.create_offset_vector([t1, t2])
    b.start_object(2)
    b.add_uoffset(0, v)
    root = b.end_object()
    return b, b.finish(root), t1, t2, v


def check(minalign8: bool):
    _, d, t1, t2, _ = build(minalign8)
    root = struct.unpack_from("<I", d, 0)[0]
    vt = root - struct.unpack_from("<i", d, root)[0]
    o0 = struct.unpack_from("<H", d, vt + 4)[0]
    p = root + o0
    if p + 4 > len(d):
        return None, None
    try:
        n, elems = read_vec(d, p)
    except struct.error:
        return None, None
    return n, elems


def main() -> int:
    print("=== 已知缺口检查（预期：minalign=4 过、minalign=8 不过）===")
    ok4 = True
    ok8 = False
    for m8 in (False, True):
        tag = "minalign=8" if m8 else "minalign=4"
        n, elems = check(m8)
        if n is None:
            print(f"  {'✅' if not m8 else '★'} {tag:11s} 元素偏移读不出来（解析失败）")
        else:
            good = n == 2 and len(elems) == 2 and elems[0] > 0 and elems[1] > 0
            print(f"  {'✅' if good else '★'} {tag:11s} len={n} elems={elems}")
            if m8:
                ok8 = good
            else:
                ok4 = good
    print()
    if ok4 and not ok8:
        print("状态与登记一致：minalign=4 正常、minalign=8 仍缺。")
        print("→ make_tiny_model.py 产出的模型**不可用**，它会在自检阶段失败退出。")
        return 0
    if ok4 and ok8:
        print("★ minalign=8 现在通过了 —— 请把本登记删掉，并把")
        print("  tools/make_tiny_model.py 跑通后生成的模型投到设备验证。")
        return 1
    print("★ minalign=4 回归了 —— Builder 的基本路径被弄坏了，先修它。")
    return 1


if __name__ == "__main__":
    sys.exit(main())