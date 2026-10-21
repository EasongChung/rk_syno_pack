#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isolate_subgraph.py —— 把 TFLite 图里的「某一段被委派的子图」单独抠出来，做成可独立运行的模型

为什么需要：
  * 定位 NPU 挂死时，我们需要回答一个二选一的问题：
      (a) 挂死是**那一份作业自身**的性质（per-job）？
      (b) 挂死是**作业序列/驱动状态**的性质（positional / stateful）？
    只有把子图隔离出来单独跑，才能把 (a)/(b) 分开。
  * 本工具的产物不是交付物，是**诊断模型**（输入被换成该子图的外部输入张量）。

用法:
  tools/isolate_subgraph.py --list <run.log> <model.tflite>
      打印 34 段子图 → 模型算子下标的映射（依据：run 日志里的逐段算子表 + 全模型支持表）

  tools/isolate_subgraph.py --isolate <run.log> <model.tflite> <段号> <输出.tflite>
      抠出该段（1-based，与 run 日志的段号一致）成独立模型

  tools/isolate_subgraph.py --truncate <run.log> <model.tflite> <段号> <输出.tflite>
      保留第 1..段号 段（上游照常跑），用于二分法找挂死起点

实现要点：
  * 子图→算子下标 的映射由 run 日志推导：全模型算子表给出每个算子 supported/unsupported，
    逐段编译日志给出每段的算子类型序列；把 supported 算子按顺序切段，并与逐段类型序列**逐项校验**。
  * 剪枝：只保留被引用到的张量与 buffer，模型从 ~6MB 缩到 ~几百 KB（推送快）。
  * signature_def 同步改写（指向新的输入/输出），避免解释器因签名不一致报错。

退出码: 0 成功 / 非 0 失败
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_fold_rank5_to_4d as fold   # noqa: E402

RE_MODEL_OP = re.compile(r'^\s*(\d+)\s+(\S+)\s+v(\d+)\s+(supported|unsupported)\s+in:')
RE_CMP = re.compile(r'teflon: compiling graph: (\d+) tensors (\d+) operations')
RE_HDR = re.compile(r'^\s*idx\s+type\s+inputs')
RE_ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+([\d,]+)\s+([\d,]+)\s*$')


def parse_log(path):
    """返回 (全模型算子类型与支持情况, 每段的算子类型序列)"""
    L = open(path, encoding='utf-8', errors='replace').read().splitlines()
    model_ops = []          # [(idx, type, supported)]
    for l in L:
        m = RE_MODEL_OP.match(l)
        if m:
            model_ops.append((int(m.group(1)), m.group(2), m.group(4) == 'supported'))
    cidx = [i for i, l in enumerate(L) if RE_CMP.search(l)]
    cidx.append(len(L))
    segs = []
    for k in range(len(cidx) - 1):
        hs = [i for i in range(cidx[k], cidx[k + 1]) if RE_HDR.match(L[i])]
        types = []
        if hs:
            j = hs[-1] + 2
            while j < cidx[k + 1]:
                m = RE_ROW.match(L[j])
                if not m:
                    break
                types.append(m.group(2))
                j += 1
        segs.append(types)
    return model_ops, segs


def map_segments(model_ops, segs):
    """supported 算子按顺序切段，并与逐段类型序列逐项校验 → 每段的模型算子下标列表"""
    sup = [(i, t) for (i, t, s) in model_ops if s]
    out, p = [], 0
    for types in segs:
        idxs = []
        for t in types:
            if p >= len(sup):
                raise SystemExit("ERROR: supported 算子不够用，日志与模型对不上")
            if sup[p][1] != t:
                raise SystemExit("ERROR: 段内算子类型对不上：日志 %s vs 模型 %s（第 %d 个）"
                                 % (t, sup[p][1], p))
            idxs.append(sup[p][0])
            p += 1
        out.append(idxs)
    if p != len(sup):
        raise SystemExit("ERROR: supported 算子数不匹配：用了 %d / 共 %d" % (p, len(sup)))
    return out


def tensor_has_data(mdl, ti):
    bi = mdl["tensors"][ti]["buffer"]
    if bi <= 0 or bi >= len(mdl["buffers"]):
        return False
    return bool(mdl["buffers"][bi]["data"])


def build_isolated(mdl, op_idxs, out_path):
    """保留 op_idxs 这些算子；外部激活输入变成图输入；剪枝未引用的张量/buffer"""
    kept = [mdl["ops"][i] for i in op_idxs]
    produced = set()
    for o in kept:
        produced.update(o["outputs"])

    # 外部输入：被消费但不是本段产出、且不是常量（常量仍留在算子输入里）
    ext_inputs = []
    for o in kept:
        for t in o["inputs"]:
            if t in produced or t in ext_inputs:
                continue
            if tensor_has_data(mdl, t):
                continue
            ext_inputs.append(t)
    ext_inputs.sort()

    # 输出：本段产出中、且不被本段其它算子消费
    consumed = set()
    for o in kept:
        consumed.update(o["inputs"])
    outs = [t for o in kept for t in o["outputs"] if t not in consumed]

    # ---- 剪枝：收集被引用的张量 ----
    used = set(ext_inputs) | set(outs)
    for o in kept:
        used.update(o["inputs"])
        used.update(o["outputs"])
    used = sorted(used)
    remap = {t: i for i, t in enumerate(used)}

    new_tensors = [dict(mdl["tensors"][t]) for t in used]
    new_ops = []
    for o in kept:
        new_ops.append({
            "opcode_index": o["opcode_index"],
            "inputs": [remap[t] for t in o["inputs"]],
            "outputs": [remap[t] for t in o["outputs"]],
            "opts_type": o["opts_type"],
            "opts": o["opts"],
        })

    # buffer 剪枝
    # ★ 必须保留原 buffer 0：TFLite 硬性要求「model 的第 0 个 buffer 必须是空 buffer」。
    #   本模型的张量**一个都不引用 buffer 0**（每个张量各有自己的 buffer），
    #   剪枝后若不补 0，新 buffer 0 会变成带数据的权重 ⇒ 解释器直接拒绝加载。
    used_bufs = sorted({0} | {t["buffer"] for t in new_tensors})
    bremap = {b: i for i, b in enumerate(used_bufs)}
    new_buffers = []
    for b in used_bufs:
        bb = dict(mdl["buffers"][b])
        bb["offset"] = 0
        new_buffers.append(bb)
    for t in new_tensors:
        t["buffer"] = bremap[t["buffer"]]

    new_sg = {"inputs": [remap[t] for t in ext_inputs],
              "outputs": [remap[t] for t in outs],
              "name": mdl["subgraph"]["name"],
              "debug_md": 0}

    # signature_def 同步改写
    new_sigs = []
    for s in mdl["sigs"]:
        new_sigs.append({
            "key": s["key"], "subgraph_index": s["subgraph_index"],
            "inputs": [{"name": "input_%d" % i, "tensor_index": remap[t]}
                       for i, t in enumerate(ext_inputs)],
            "outputs": [{"name": "output_%d" % i, "tensor_index": remap[t]}
                        for i, t in enumerate(outs)],
        })

    out = {
        "version": mdl["version"], "description": mdl["description"],
        "metadata_buffer": mdl["metadata_buffer"],
        "codes": mdl["codes"], "buffers": new_buffers, "tensors": new_tensors,
        "ops": new_ops, "subgraph": new_sg, "sigs": new_sigs, "metas": mdl["metas"],
    }
    blob = fold.build_model(out)
    with open(out_path, "wb") as f:
        f.write(blob)
    return {"ops": len(new_ops), "tensors": len(new_tensors), "buffers": len(new_buffers),
            "inputs": [mdl["tensors"][t]["shape"] for t in ext_inputs],
            "outputs": [mdl["tensors"][t]["shape"] for t in outs],
            "bytes": len(blob)}


def build_truncated(mdl, op_idxs, out_path):
    """保留从第 0 个算子到 op_idxs 末尾（含）的所有算子"""
    last = op_idxs[-1]
    keep = list(range(0, last + 1))
    kept = [mdl["ops"][i] for i in keep]
    produced = set()
    for o in kept:
        produced.update(o["outputs"])
    consumed = set()
    for o in kept:
        consumed.update(o["inputs"])
    outs = [t for o in kept for t in o["outputs"] if t not in consumed]
    used = set(outs)
    for o in kept:
        used.update(o["inputs"])
        used.update(o["outputs"])
    used = sorted(used)
    remap = {t: i for i, t in enumerate(used)}
    new_tensors = [dict(mdl["tensors"][t]) for t in used]
    new_ops = [{"opcode_index": o["opcode_index"],
                "inputs": [remap[t] for t in o["inputs"]],
                "outputs": [remap[t] for t in o["outputs"]],
                "opts_type": o["opts_type"], "opts": o["opts"]} for o in kept]
    used_bufs = sorted({0} | {t["buffer"] for t in new_tensors})
    bremap = {b: i for i, b in enumerate(used_bufs)}
    new_buffers = []
    for b in used_bufs:
        bb = dict(mdl["buffers"][b])
        bb["offset"] = 0
        new_buffers.append(bb)
    for t in new_tensors:
        t["buffer"] = bremap[t["buffer"]]
    new_sg = {"inputs": [remap[t] for t in mdl["subgraph"]["inputs"]],
              "outputs": [remap[t] for t in outs],
              "name": mdl["subgraph"]["name"], "debug_md": 0}
    new_sigs = []
    for s in mdl["sigs"]:
        new_sigs.append({"key": s["key"], "subgraph_index": s["subgraph_index"],
                         "inputs": [{"name": "in", "tensor_index": remap[t]}
                                    for t in mdl["subgraph"]["inputs"]],
                         "outputs": [{"name": "out", "tensor_index": remap[t]} for t in outs]})
    out = {"version": mdl["version"], "description": mdl["description"],
           "metadata_buffer": mdl["metadata_buffer"], "codes": mdl["codes"],
           "buffers": new_buffers, "tensors": new_tensors, "ops": new_ops,
           "subgraph": new_sg, "sigs": new_sigs, "metas": mdl["metas"]}
    blob = fold.build_model(out)
    with open(out_path, "wb") as f:
        f.write(blob)
    return {"ops": len(new_ops), "tensors": len(new_tensors), "buffers": len(new_buffers),
            "bytes": len(blob)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("model")
    ap.add_argument("--out", help="输出模型路径")
    ap.add_argument("--seg", type=int, help="段号（1-based）")
    ap.add_argument("--prefix", type=int, help="只取该段的前 K 个算子（二分用）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--truncate", action="store_true")
    a = ap.parse_args()

    model_ops, segs = parse_log(a.log)
    segmap = map_segments(model_ops, segs)

    if a.list:
        print("段号 → 模型算子下标（1-based 段号）")
        for k, idxs in enumerate(segmap, 1):
            print("  #%-3d ops=%-2d  %s" % (k, len(idxs), idxs))
        return 0

    if not (a.isolate or a.truncate) or not a.out or not a.seg:
        print(__doc__)
        return 2
    if a.seg < 1 or a.seg > len(segmap):
        raise SystemExit("ERROR: 段号越界（1..%d）" % len(segmap))

    mdl = fold.read_model(a.model)
    idxs = segmap[a.seg - 1]
    if a.prefix:
        # 二分：只取该段的前 K 个算子（段内算子序 = 模型算子序，按 segmap 顺序）
        if a.prefix < 1 or a.prefix > len(idxs):
            raise SystemExit("ERROR: prefix 越界（1..%d）" % len(idxs))
        idxs = idxs[:a.prefix]
    if a.isolate:
        info = build_isolated(mdl, idxs, a.out)
    else:
        info = build_truncated(mdl, idxs, a.out)
    print("段 #%d → %s" % (a.seg, a.out))
    print("  算子 %d  张量 %d  buffer %d  大小 %d B" %
          (info["ops"], info["tensors"], info["buffers"], info["bytes"]))
    if "inputs" in info:
        print("  图输入形状: %s" % info["inputs"])
        print("  图输出形状: %s" % info["outputs"])
    return 0


if __name__ == '__main__':
    sys.exit(main())
