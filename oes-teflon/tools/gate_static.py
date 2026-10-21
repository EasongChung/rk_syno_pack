#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INT8 交付门（静态部分 a/b/c）批量体检。

依据 `docs/README.md`「六条交付门」中的静态三条：
  a. 退化张量 = 0（`abs(scale)>1e10` 或 `scale<=0`）
  b. 无空形状张量（`shape=[]`）
  c. 全部 conv/dwconv 权重 per-tensor（`scale->size==1 && zp->size==1`）

a/b/c 是**零成本**的（不需设备、不需 CI、不需推理），可以在换模型之前先把候选筛一遍。
运行期三门（d 输入敏感性 / e vs float 参照 / f 饱和率）不在本工具范围内。

用法：
    python3 tools/gate_static.py <model.tflite> [...]
    python3 tools/gate_static.py --dir <目录>      # 扫描目录下所有 .tflite
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verify"))
import tflite_schema as ts  # noqa: E402

DEGENERATE_SCALE = 1e10


def gates(path: Path) -> dict:
    m = ts.parse_model(path.read_bytes(), path.name)
    out = {"file": str(path), "bytes": path.stat().st_size, "subgraphs": len(m.subgraphs)}
    for si, sg in enumerate(m.subgraphs):
        prefix = "" if si == 0 else f"[sg{si}]"
        degen, empty, non_pt = [], [], []

        for i, t in enumerate(sg.tensors):
            q = getattr(t, "quantization", None)
            sc = list(getattr(q, "scale", []) or []) if q else []
            zp = list(getattr(q, "zero_point", []) or []) if q else []
            if sc and (abs(sc[0]) > DEGENERATE_SCALE or sc[0] <= 0):
                degen.append((i, sc[0], list(t.shape)))
            if len(t.shape) == 0:
                empty.append((i, t.name))

        # 门 c：卷积族算子的**权重**输入（第 2 个输入）必须 per-tensor
        for op in sg.operators:
            if op.builtin_name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D_TRANSPOSE"):
                continue
            if len(op.inputs) < 2:
                continue
            w = op.inputs[1]
            if w < 0 or w >= len(sg.tensors):
                continue
            q = getattr(sg.tensors[w], "quantization", None)
            sc = list(getattr(q, "scale", []) or []) if q else []
            zp = list(getattr(q, "zero_point", []) or []) if q else []
            if not (len(sc) == 1 and len(zp) == 1):
                non_pt.append((op.opcode_index, w, len(sc), len(zp)))

        # 浮点模型（float16/float32）本就没有量化参数，门 c 对其**不适用**，
        # 否则会把「scale.size=0」误判成「非 per-tensor」。先判定是否为量化模型。
        is_quant = any(
            getattr(getattr(t, "quantization", None), "scale", None)
            for t in sg.tensors
        )

        out[prefix + "tensors"] = len(sg.tensors)
        out[prefix + "ops"] = len(sg.operators)
        out[prefix + "rank5"] = sum(1 for t in sg.tensors if len(t.shape) >= 5)
        out[prefix + "is_quant"] = is_quant
        out[prefix + "a_degenerate"] = degen
        out[prefix + "b_empty"] = empty
        out[prefix + "c_nonpertensor"] = non_pt
    return out


def report(r: dict) -> bool:
    ok = True
    print(f"\n── {r['file']}  ({r['bytes']:,} B)")
    for si in range(r["subgraphs"]):
        p = "" if si == 0 else f"[sg{si}]"
        a, b, c = r[p + "a_degenerate"], r[p + "b_empty"], r[p + "c_nonpertensor"]
        is_quant = r[p + "is_quant"]
        n5 = r[p + "rank5"]
        print(f"   张量 {r[p+'tensors']}  算子 {r[p+'ops']}  rank≥5 {n5}")
        ga = "✅ a 通过" if not a else f"❌ a 失败（退化 {len(a)}）"
        gb = "✅ b 通过" if not b else f"❌ b 失败（空形状 {len(b)}）"
        if not is_quant:
            gc = "—  c 不适用（浮点模型无量化参数）"
        elif c:
            gc = f"⚠️  c 未过（{len(c)} 个权重非 per-tensor）"
        else:
            gc = "✅ c 通过"
        print(f"   {ga}   {gb}   {gc}")
        if a:
            ok = False
            for i, s, shp in a[:8]:
                print(f"        t{i:<4} scale={s:.4g}  shape={shp}")
            if len(a) > 8:
                print(f"        …另有 {len(a)-8} 个")
        if b:
            ok = False
            for i, nm in b[:6]:
                print(f"        t{i:<4} 空形状 name={nm[:40]}")
        if c and is_quant:
            for n, w, ns, nz in c[:4]:
                print(f"        node{n} 权重 t{w}: scale.size={ns} zp.size={nz}")
            if len(c) > 4:
                print(f"        …另有 {len(c)-4} 个")
    print(f"   ⇒ 静态门 a/b: {'全部通过 ✅' if ok else '未通过 ❌'}")
    return ok


def main() -> int:
    args = sys.argv[1:]
    files: list[Path] = []
    if args and args[0] == "--dir":
        files = sorted(Path(args[1]).rglob("*.tflite"))
    else:
        files = [Path(a) for a in args]
    if not files:
        print(__doc__)
        return 2
    allok = True
    for f in files:
        try:
            allok &= report(gates(f))
        except Exception as e:  # noqa: BLE001
            print(f"\n── {f}  解析失败: {e}")
            allok = False
    print("\n" + "=" * 64)
    print("  总结：以上模型中，静态门 a/b 全过的才值得进入设备验证。")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())