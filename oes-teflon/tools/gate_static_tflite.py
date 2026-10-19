#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tflite 静态门检查器（复现门 a / b / c，外加一项它们查不出来的证据）。

用法：
    python3 tools/gate_static_tflite.py <model.tflite> [more.tflite ...]

门（按 och 的定义）：
  a  退化张量：|scale| >= DEGENERATE（默认 1e30）。1.334440e36 = FLT_MAX/255
     是它的典型值 ⇒ 量化器把**溢出范围**写进了 scale。
  b  空形状：rank == 0 的张量（注意：TFLite 本身允许标量常量，
     所以这里只**报数**不判死，由使用方决定它是否致命）。
  c  per-tensor：所有量化张量都必须 len(scale)==1；len>1 即 per-channel。
     ⚠️ Mesa 的量化门按设计只收 per-tensor（scale->size==1 && zp->size==1）。

★ 本工具比门 a/b/c 多做一件关键的事（这决定了"该修导出还是修量化器"）：
  **把每个张量的常量 buffer 解码出来，看真实数值范围。**
  · 若**浮点源模型**里已经有 |v|>=1e30 的常量/权重 ⇒ 污染发生在**量化之前**
    （ONNX 往返 / 原始 float16 模型 / 权重转置），改量化器没用。
  · 若浮点源干净、只有 int8 模型的 **scale** 是 1.3e36 ⇒ 污染在**量化/校准**那一步。
  这两种情况的修法完全不同，必须先分开。
"""

import os
import sys

import numpy as np
import tflite

sys.stdout.reconfigure(encoding="utf-8")

DEGENERATE = 1e30
TYPES = {0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
         5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
         10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
         14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4"}
DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64,
          6: np.bool_, 7: np.int16, 9: np.int8, 15: np.uint32, 16: np.uint16}


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[4:8] != b"TFL3":
        raise SystemExit("不是 tflite（缺 TFL3 标识）: %s" % path)
    return tflite.Model.GetRootAs(bytearray(data), 0), len(data)


def const_of(model, t, dtype):
    """返回该张量的常量 numpy 数组；非常量返回 None。"""
    bi = t.Buffer()
    if bi is None:
        return None
    b = model.Buffers(bi)
    if b is None or b.DataLength() == 0:
        return None
    raw = b.DataAsNumpy()
    if raw is None:
        return None
    arr = np.frombuffer(raw.tobytes(), dtype=dtype)
    return arr


def analyse(path, names=None):
    model, nbytes = load(path)
    names = names or []
    name = os.path.basename(path)
    print("=" * 100)
    print("模型: %s  (%d B)" % (name, nbytes))
    desc = model.Description()
    if desc:
        print("  description: %s" % desc.decode("utf-8", "replace"))
    sg = model.Subgraphs(0)
    print("  subgraphs=%d  buffers=%d  opcodes=%d"
          % (model.SubgraphsLength(), model.BuffersLength(), model.OperatorCodesLength()))
    print("  subgraph0: tensors=%d operators=%d inputs=%s outputs=%s"
          % (sg.TensorsLength(), sg.OperatorsLength(),
             [sg.Inputs(i) for i in range(sg.InputsLength())],
             [sg.Outputs(i) for i in range(sg.OutputsLength())]))

    degen, empty_shape, per_channel, huge_const, unquantized = [], [], [], [], []
    maxabs = 0.0
    worst = None
    all_float_consts = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        tname = t.Name().decode("utf-8", "replace") if t.Name() else "t%d" % i
        shape = [t.Shape(k) for k in range(t.ShapeLength())]
        ttype = TYPES.get(t.Type(), "T%d" % t.Type())
        q = t.Quantization()
        scales, zps, qdim = [], [], None
        if q is not None:
            scales = [q.Scale(k) for k in range(q.ScaleLength())]
            zps = [q.ZeroPoint(k) for k in range(q.ZeroPointLength())]
            if hasattr(q, "QuantizedDimension"):
                qdim = q.QuantizedDimension()
        if scales:
            mx = max(abs(s) for s in scales)
            qmin = [q.Min(k) for k in range(q.MinLength())] if q.MinLength() else []
            qmax = [q.Max(k) for k in range(q.MaxLength())] if q.MaxLength() else []
            if mx >= DEGENERATE:
                # ★ 区分「实测范围 = FLT_MAX」与「量化器没采到数据 ⇒ 兜底默认值」
                #   scale 有值但 min/max 数组为空 ⇒ **兜底**（fallback）
                #   min/max 有值且 = ±FLT_MAX ⇒ 实测（真有张量爆了）
                kind = "兜底(无 min/max)" if not qmin else (
                    "实测 min=%s max=%s" % (qmin[:2], qmax[:2]))
                degen.append((i, tname, ttype, shape, mx, len(scales), kind))
            if len(scales) > 1:
                per_channel.append((i, tname, ttype, shape, len(scales), qdim))
        if len(shape) == 0:
            empty_shape.append((i, tname, ttype))
        if ttype in ("FLOAT32", "FLOAT16", "INT8", "UINT8", "INT32") and not scales:
            # 有常量数据却没量化参数（float 模型里正常；int8 模型里可疑）
            arr = const_of(model, t, DTYPES.get(t.Type(), np.float32))
            if arr is not None and arr.size:
                unquantized.append((i, tname, ttype, shape, arr.size))
        if ttype in ("FLOAT32", "FLOAT16"):
            arr = const_of(model, t, DTYPES[t.Type()])
            if arr is not None and arr.size:
                a = float(np.max(np.abs(arr.astype(np.float64))))
                all_float_consts.append((a, i, tname, ttype, tuple(shape)))
                if a > maxabs:
                    maxabs, worst = a, (i, tname, ttype, shape)
                if a >= DEGENERATE:
                    huge_const.append((i, tname, ttype, shape, a))

    all_float_consts.sort(reverse=True)
    top_consts = all_float_consts[:8]

    print()
    print("--- 门 a：退化 scale（|scale| >= %.0e）---" % DEGENERATE)
    if not degen:
        print("  PASS：0 个")
    for i, tn, tt, sh, mx, n, kind in degen:
        print("  t%-4d %-46s %-8s max|scale|=%.6e  scale 个数=%-4d %s"
              % (i, tn[:46], tt, mx, n, kind))
    print()
    print("--- 门 b：rank-0（空形状）张量 ---")
    if not empty_shape:
        print("  PASS：0 个")
    for i, tn, tt in empty_shape:
        print("  t%-4d %-28s %-8s shape=[]" % (i, tn, tt))
    print()
    print("--- 门 c：per-channel（len(scale)>1）---")
    if not per_channel:
        print("  PASS：全部 per-tensor")
    for i, tn, tt, sh, n, qd in per_channel:
        print("  t%-4d %-28s %-8s shape=%-20s scale 个数=%d quantized_dimension=%s"
              % (i, tn, tt, sh, n, qd))
    print()
    print("--- ★ 额外证据：常量数据里有没有巨大数值 ---")
    if not huge_const:
        print("  干净：没有任何浮点常量 |v| >= %.0e" % DEGENERATE)
    for i, tn, tt, sh, a in huge_const:
        print("  t%-4d %-28s %-8s shape=%-20s max|v|=%.6e" % (i, tn, tt, sh, a))
    if worst:
        print("  全体浮点常量的最大 |v| = %.6e  （出现在 t%d %s %s）"
              % (maxabs, worst[0], worst[1], worst[3]))
    # 最大的几个常量：饱和值往往就是「本该很小却很大」的那个 bug
    all_float_consts.sort(reverse=True)
    print()
    print("--- ★ 指定名字的 scale 明细（--names 子串匹配；跨变体按**名字**对齐，索引会漂移）---")
    if not names:
        print("  （未指定 --names）")
    hit = 0
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        tname = t.Name().decode("utf-8", "replace") if t.Name() else "t%d" % i
        if not any(nm in tname for nm in names):
            continue
        hit += 1
        q = t.Quantization()
        sc = [q.Scale(k) for k in range(q.ScaleLength())] if q else []
        zp = [q.ZeroPoint(k) for k in range(q.ZeroPointLength())] if q else []
        qmin = [q.Min(k) for k in range(q.MinLength())] if q else []
        qmax = [q.Max(k) for k in range(q.MaxLength())] if q else []
        shape = [t.Shape(k) for k in range(t.ShapeLength())]
        print("  t%-4d %-50s %-8s shape=%-16s nscale=%-4d zp0=%-5s min=%-11s max=%-11s scale0=%.6e"
              % (i, tname[:50], TYPES.get(t.Type(), "?"), str(shape)[:16], len(sc),
                 zp[0] if zp else "-", ("%.4g" % qmin[0]) if qmin else "(空)",
                 ("%.4g" % qmax[0]) if qmax else "(空)",
                 sc[0] if sc else float("nan")))
    if names and not hit:
        print("  没有匹配到任何张量（名字在两个变体间不一致时常见）")
    print()
    print("--- 附：最大的 8 个浮点常量（饱和值常是「本该很小却很大」的 bug）---")
    if not top_consts:
        print("  无浮点常量")
    for a, i, tn, tt, sh in top_consts:
        print("   t%-4d %-40s %-8s shape=%-18s max|v|=%.6e"
              % (i, tn[:40], tt, str(sh)[:18], a))
    print()
    print("--- 附：有常量但无量化参数的张量（前 10）---")
    if not unquantized:
        print("  无")
    for i, tn, tt, sh, sz in unquantized[:10]:
        print("  t%-4d %-28s %-8s shape=%-20s 元素=%d" % (i, tn, tt, sh, sz))
    if len(unquantized) > 10:
        print("  … 共 %d 个" % len(unquantized))

    ok = not degen and not per_channel
    print()
    print("判定：%s（门 a=%s 门 b=%d个 门 c=%s）"
          % ("PASS" if ok else "FAIL",
             "PASS" if not degen else "FAIL", len(empty_shape),
             "PASS" if not per_channel else "FAIL"))
    return ok


def main():
    argv = sys.argv[1:]
    names = []
    if "--names" in argv:
        i = argv.index("--names")
        names = [x for x in argv[i + 1].split(",") if x]
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        sys.stderr.write(__doc__)
        return 2
    allok = True
    for p in args:
        if not os.path.isfile(p):
            print("跳过（不存在）: %s" % p)
            continue
        allok &= analyse(p, names)
        print()
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())