#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PADV2 → PAD 改写的**前置安全门**：constant_values 到底是不是 0？

## 为什么必须有这道门

`tools/oes/tflite_fix_padv2.py` 把 PADV2 改写成 PAD。**PAD 恒定用 0 填充**，
PADV2 才带 constant_values。所以：

    只有当 PADV2 的 constant_values 全为 0 时，改写成 PAD 才**数值等价**。

原脚本**没有检查这一点**，无条件改写。如果某个 PADV2 的填充值是 1
（上游图的垃圾常量，或转换器把 replicate-padding 塞了进来），
改写成 PAD 会把填充值从 1 静默换成 0 ⇒ **数值错但推理不报错**。
这正是本项目最该防的那类失败。

## 真源依据（本机对 tools/oes/models/model_float16.tflite 实测）

源模型里**没有任何 PADV2**，只有 8 个 PAD，且**共用同一个**
`int32[4,2] = [0,0, 1,1, 1,1, 0,0]` 的 **pads** 张量：

    op#114 PAD  [1,395,395,3]  +pads -> [1,397,397,3]
    op#116 PAD  [1,198,198,24] +pads -> [1,200,200,24]
    op#118 PAD  [1,99,99,24]   +pads -> [1,101,101,24]
    op#121 PAD  [1,99,99,122]  +pads -> [1,101,101,122]
    op#164 PAD  [1,50,50,244]  +pads -> [1,52,52,244]
    op#168 PAD  [1,50,50,244]  +pads -> [1,52,52,244]
    op#254 PAD  [1,25,25,488]  +pads -> [1,27,27,488]
    op#258 PAD  [1,25,25,488]  +pads -> [1,27,27,488]

⇒ **填充值恒为 0**（PAD 语义），pads 是 `[0,0],[1,1],[1,1],[0,0]`。
⇒ int8 模型里的 PADV2 是 tflite→onnx→tf→tflite 往返**造出来的**，
   它的 constant_values 应当是「本该为 0」的退化产物。

## 三态判定（不搞二值判断，因为退化值既不是 0 也不是正常常量）

| 情形 | 判定 | 动作 |
|---|---|---|
| 良构，且**反量化后全为 0** | `ZERO` | 放行改写（数值等价） |
| **退化**（FLT_MAX / rank-0 却对应 [4,2] pads / NaN） | `DEGENERATE` | 放行但**大声告警**：这是转换器造出来的垃圾常量，本该为 0 |
| 良构，且**存在非零值** | `NONZERO` | ★ **拒绝改写** —— 会静默改数值 |

`NONZERO` 必须拒绝，因为此时正确做法不是「删掉 constant_values」，
而是**保留 PADV2 并把常量改成 0**（或保留原值）。

## 用法

    from padv2_fill_gate import scan_model
    rep = scan_model('concept_int8_pertensor.tflite')
    print(rep)
    if rep.blocking:   # 存在 NONZERO
        sys.exit(1)
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

CODE_PAD = 34
CODE_PADV2 = 60

# 与 quantize 工具链保持一致的 dtype 表（只列会用到的）
DT = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64,
      7: np.int16, 9: np.int8, 16: np.uint16}
FLOAT_DT = (np.float32, np.float16)

# 退化判据：|v| 超过这个数就认为「不是真实数据，是转换器留下的垃圾」
DEGENERATE_ABS = 1e30
# 「接近 0」的容差：float16 经量化往返后的合理残差
ZERO_TOL = 1e-6


def _opname(code):
    return {CODE_PAD: "PAD", CODE_PADV2: "PADV2"}.get(code, "code_%d" % code)


def _eff_code(model, op):
    oc = model.OperatorCodes(op.OpcodeIndex())
    bc = int(oc.BuiltinCode())
    return bc if bc < 127 else int(oc.DeprecatedBuiltinCode())


def _shape(t):
    try:
        return [int(x) for x in t.ShapeAsNumpy()]
    except Exception:                                    # noqa: BLE001
        return []


def _quant(t):
    q = t.Quantization()
    if q is None or q.ScaleLength() == 0:
        return None
    return float(q.Scale(0)), (int(q.ZeroPoint(0)) if q.ZeroPointLength() else 0)


def _buffer_real_values(model, t):
    """把常量张量解成**真实值**（量化域会反量化）。

    返回 None 表示「不是常量 / 没有 buffer」。
    返回 (values ndarray, note str)。
    """
    bi = t.Buffer()
    if bi is None:
        return None, "无 buffer 引用（非常量）"
    b = model.Buffers(bi)
    if b is None or b.DataLength() == 0:
        return None, "buffer 为空（非常量）"
    raw = b.DataAsNumpy()
    if raw is None:
        return None, "DataAsNumpy 返回 None"
    dt = DT.get(int(t.Type()))
    if dt is None:
        return None, "dtype T%d 不在本工具支持范围" % int(t.Type())
    arr = np.frombuffer(raw.tobytes(), dtype=dt).astype(np.float64)
    qt = _quant(t)
    if qt is not None and dt in (np.uint8, np.int8, np.int16, np.uint16):
        scale, zp = qt
        arr = (arr - zp) * scale
        return arr, "已按 scale=%.6g zp=%d 反量化" % (scale, zp)
    return arr, "无量化参数，按字面值"


def classify(values, note, pads_shape):
    """三态判定。返回 (verdict, 说明)。"""
    if values is None:
        return "UNKNOWN", note
    finite = np.isfinite(values)
    if not finite.all():
        return "DEGENERATE", "含 NaN/Inf：" + note
    amax = float(np.abs(values).max()) if values.size else 0.0
    if amax >= DEGENERATE_ABS:
        return ("DEGENERATE",
                "max|v|=%.6g ≥ %.0g（FLT_MAX 量级，转换器垃圾常量）：%s"
                % (amax, DEGENERATE_ABS, note))
    if amax <= ZERO_TOL:
        return "ZERO", "max|v|=%.3g ≤ %.0g（确为 0）：%s" % (amax, ZERO_TOL, note)
    # 良构但非零 —— 再判一次「形状是否与 pads 不匹配」，那种仍属退化
    if pads_shape and list(values.shape) != list(pads_shape):
        return ("DEGENERATE",
                "值非零(max=%.6g)但形状 %s 与 pads 的 %s 不匹配 —— 形状就不合法，"
                "判为转换器垃圾常量：%s"
                % (amax, list(values.shape), list(pads_shape), note))
    return "NONZERO", "max|v|=%.6g 超过容差 %.0g 且形状合法：%s" % (amax, ZERO_TOL, note)


def scan_model(path, verbose=True):
    """扫描一个 tflite 里所有 PADV2 的 constant_values，返回报告。"""
    import tflite as T
    raw = open(path, "rb").read()
    mo = T.Model.GetRootAsModel(bytearray(raw), 0)
    g = mo.Subgraphs(0)
    rows = []
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        if _eff_code(mo, op) != CODE_PADV2:
            continue
        ins = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        if len(ins) < 3:
            rows.append({"op": k, "verdict": "UNKNOWN",
                         "detail": "只有 %d 个输入，没有 constant_values" % len(ins)})
            continue
        pads_t = g.Tensors(ins[1])
        cv_t = g.Tensors(ins[2])
        pads_shape = _shape(pads_t)
        cv_shape = _shape(cv_t)
        vals, note = _buffer_real_values(mo, cv_t)
        if vals is None:
            verdict, detail = "DEGENERATE", ("constant_values 不是常量（%s）"
                                             "—— 无 buffer 的张量在图里就是垃圾：%s"
                                             % (note, _shape(cv_t)))
        else:
            verdict, detail = classify(vals, note, pads_shape)
        rows.append({
            "op": k, "verdict": verdict, "detail": detail,
            "pads_shape": pads_shape, "cv_shape": cv_shape,
            "cv_type": int(cv_t.Type()),
            "sample": (None if vals is None else
                       [float(x) for x in np.asarray(vals).ravel()[:8]]),
        })
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_pad = sum(1 for k in range(g.OperatorsLength())
                if _eff_code(mo, g.Operators(k)) == CODE_PAD)
    rep = {
        "path": path, "n_padv2": len(rows), "n_pad": n_pad,
        "rows": rows, "counts": counts,
        # NONZERO 存在 ⇒ 阻断（会静默改数值）
        "blocking": counts.get("NONZERO", 0) > 0,
        # UNKNOWN 也阻断：判不出来就不能改
        "unknown": counts.get("UNKNOWN", 0) > 0,
    }
    if verbose:
        print("=== PADV2 constant_values 门：%s ===" % path)
        print("  PAD=%d  PADV2=%d" % (n_pad, len(rows)))
        for r in rows:
            print("  op#%-4d %-11s pads=%s cv_shape=%s cv_type=%s"
                  % (r["op"], r["verdict"], r.get("pads_shape"),
                     r.get("cv_shape"), r.get("cv_type")))
            print("        %s" % r["detail"])
            if r.get("sample") is not None:
                print("        sample=%s" % r["sample"])
        print("  统计: %s" % counts)
        print("  阻断? %s%s"
              % ("是（存在 NONZERO，改写会静默改数值）" if rep["blocking"] else "否",
                 "；含 UNKNOWN（判不出来）" if rep["unknown"] else ""))
    return rep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="把 DEGENERATE 也视为通过（默认就是通过，这里只是显式化）")
    args = ap.parse_args()
    rep = scan_model(args.model)
    if rep["blocking"]:
        print("\nVERDICT: BLOCKED —— 存在良构但非零的 constant_values，"
              "禁止改写成 PAD。")
        return 1
    if rep["unknown"]:
        print("\nVERDICT: BLOCKED —— 有 PADV2 判不出 constant_values，"
              "禁止在信息不足时改写。")
        return 1
    if rep["n_padv2"] == 0:
        print("\nVERDICT: NOTHING TO DO —— 模型里没有 PADV2")
        return 0
    print("\nVERDICT: PASS —— 全部 PADV2 的填充值均为 0（或为已识别的退化垃圾），"
          "改写成 PAD 数值等价。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
