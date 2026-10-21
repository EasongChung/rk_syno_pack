#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比对 NPU 输出与 CPU 参考，自动判定**布局**与**舍入**两类偏差。

## 为什么需要

C=248 实测发现：NPU 输出「按 HWC 读只有 1.19% 一致」，但这不是算错——
把它按 **CHW** 重排后 76.51% 逐字节一致、其余恰好差 −1、|差|>1 为 0。
若不做这个判定，很容易把「布局错」误判成「数值算错」而继续乱改驱动。

本工具把判据固化，输出三件事：

1. **布局**：HWC 直接比 vs CHW 重排比，谁的完全一致率高
2. **数值**：完全一致率、差值分布、`|差| > 1` 的像素数（>0 说明不只是舍入）
3. **方向**：`npu < cpu` / `npu > cpu` 的计数（单向偏小 = 截断而非四舍五入）

## 用法

  python3 tools/compare-npu-vs-cpu.py <npu.bin> <cpu.bin> [--shape 50,50,248] [--selftest]

`--shape H,W,C`（默认 50,50,248）。两个文件都必须恰好 H*W*C 字节。

## 自测

  python3 tools/compare-npu-vs-cpu.py --selftest

用 `verify/evidence/coef-probe-20261021/` 与 `verify/artifacts/xfer/` 里的真实样本：
正控制：已知 wdom-on.bin 在 CHW 下应判为「一致率 > 70% 且 |差|>1 == 0」；
负控制：把 CPU 参考人为扰动 1（应检出 |差|>1）、把布局换回 HWC（一致率应崩）。
"""

import collections
import sys

DEFAULT_SHAPE = (50, 50, 248)


def read_pcm(path, shape):
    """按 TFLite 输出语义读：原始字节 → int8。"""
    raw = open(path, "rb").read()
    h, w, c = shape
    n = h * w * c
    if len(raw) != n:
        raise SystemExit("尺寸不符：%s 有 %d 字节，%dx%dx%d 需要 %d"
                         % (path, len(raw), h, w, c, n))
    return [b - 256 if b > 127 else b for b in raw]


def chw_to_hwc(vals, shape):
    h, w, c = shape
    p = h * w
    out = [0] * (p * c)
    for ch in range(c):
        base = ch * p
        for pix in range(p):
            out[pix * c + ch] = vals[base + pix]
    return out


def compare(npu, cpu):
    diff = [a - b for a, b in zip(npu, cpu)]
    h = collections.Counter(diff)
    exact = h[0]
    lt = sum(1 for d in diff if d < 0)
    gt = sum(1 for d in diff if d > 0)
    return {
        "exact": exact,
        "exact_pct": 100.0 * exact / len(diff),
        "diff_hist": h,
        "max_abs": max(abs(d) for d in diff) if diff else 0,
        "gt1": sum(1 for d in diff if abs(d) > 1),
        "lt": lt,
        "gt": gt,
        "mean_npu": sum(npu) / len(npu),
        "mean_cpu": sum(cpu) / len(cpu),
    }


def report(tag, r):
    top = [(d, c) for d, c in sorted(r["diff_hist"].items(), key=lambda kv: -kv[1])[:5]]
    print("  [%s] 完全一致 %.2f%%  |差|>1 = %d  差值top5=%s" %
          (tag, r["exact_pct"], r["gt1"], top))
    print("        npu<cpu: %d   npu>cpu: %d   均值 %.4f vs %.4f" %
          (r["lt"], r["gt"], r["mean_npu"], r["mean_cpu"]))


def analyse(npu_path, cpu_path, shape):
    npu = read_pcm(npu_path, shape)
    cpu = read_pcm(cpu_path, shape)

    print("== HWC（原样）==")
    r_hwc = compare(npu, cpu)
    report("HWC", r_hwc)

    print("== CHW→HWC 重排 ==")
    r_chw = compare(chw_to_hwc(npu, shape), cpu)
    report("CHW", r_chw)

    print("\n== 判定 ==")
    if r_chw["exact_pct"] > r_hwc["exact_pct"] + 20:
        print("  布局：NPU 输出为 **CHW**（图输出未做 detranspose）")
    elif r_hwc["exact_pct"] > r_chw["exact_pct"] + 20:
        print("  布局：**HWC**（正常）")
    else:
        print("  布局：无法判定（两种读法都不高，需先查数值）")

    best = r_chw if r_chw["exact_pct"] > r_hwc["exact_pct"] else r_hwc
    if best["gt1"] == 0 and best["gt"] == 0 and best["lt"] > 0:
        print("  舍入：恒 **向下**（截断），与四舍五入差 −1  ⇒ 需 rounding 补偿")
    elif best["gt1"] == 0 and best["exact_pct"] > 99.99:
        print("  舍入：与 CPU **逐字节一致** ✅")
    elif best["gt1"] > 0:
        print("  舍入：存在 |差|>1 ⇒ **不是纯舍入**，需回查数值路径")
    return r_hwc, r_chw


def selftest():
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npu = os.path.join(base, "verify/artifacts/xfer/wdom-on.bin")
    cpu = os.path.join(base, "verify/evidence/coef-probe-20261021/cpu-248.bin")
    if not (os.path.isfile(npu) and os.path.isfile(cpu)):
        print("  [SKIP] 缺少真实样本，跳过自测")
        return True

    ok = True
    shape = DEFAULT_SHAPE
    n = read_pcm(npu, shape)
    c = read_pcm(cpu, shape)

    # 正控制：CHW 判定成立
    r = compare(chw_to_hwc(n, shape), c)
    cond = r["exact_pct"] > 70.0 and r["gt1"] == 0 and r["gt"] == 0 and r["lt"] > 0
    print("  [%s] 正控制：真实样本判为 CHW + 恒向下（一致 %.2f%%，|差|>1=%d，npu>cpu=%d）"
          % ("ok" if cond else "BAD", r["exact_pct"], r["gt1"], r["gt"]))
    ok = ok and cond

    # 负控制 1：人为扰动 2 ⇒ 必须检出 |差|>1
    c2 = list(c)
    c2[100] = (c2[100] + 2 + 128) % 256 - 128
    r2 = compare(chw_to_hwc(n, shape), c2)
    cond = r2["gt1"] >= 1
    print("  [%s] 负控制 1：+2 扰动被检出为 |差|>1（%d 处）" % ("ok" if cond else "BAD", r2["gt1"]))
    ok = ok and cond

    # 负控制 2：按 HWC 读同一份数据 ⇒ 一致率必须崩
    r3 = compare(n, c)
    cond = r3["exact_pct"] < 10.0
    print("  [%s] 负控制 2：HWC 误读被识别（一致率仅 %.2f%%）" % ("ok" if cond else "BAD", r3["exact_pct"]))
    ok = ok and cond

    return ok


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        ok = selftest()
        print("自测%s" % ("通过" if ok else "失败"))
        return 0 if ok else 1
    if len(argv) < 3:
        print(__doc__)
        return 2
    shape = DEFAULT_SHAPE
    if "--shape" in argv:
        i = argv.index("--shape")
        shape = tuple(int(x) for x in argv[i + 1].split(","))
    analyse(argv[1], argv[2], shape)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
