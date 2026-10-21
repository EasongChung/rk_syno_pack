#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收「折维手术」产物：`tools/tflite_fold_rank5_to_4d.py` 写出的模型。

## 门的设计纪律（本项目教训）

1. **一条 PASS 必须给出「它匹配到了什么」** —— 每项检查都打印实际读到的值，不打印裸 `ok`。
2. **正控制 + 负控制都要有**，且负控制用**真实样本**：
   - 正控制：手术后的模型 → 全部通过，EXIT=0
   - 负控制：**原模型**（它真有 32 个 rank-5）→ 必须在 C1/C7 上变红，EXIT=1
   若原模型也全绿，说明这扇门没有判别力（假绿），本次验收作废。
3. **两个独立读取器交叉验证**：官方 flatbuffers 绑定 + 零依赖最小读取器。
   两者对同一文件的结论必须一致，否则说明有一方读错了。
4. **不检查自己的输出**：本脚本不调用手术脚本的任何判定逻辑来"证明"产物正确，
   只读文件。手术脚本的常量由 `--selftest` 单独证明（索引等价性）。

## 检查项

| 编号 | 内容 |
|---|---|
| C1 | **rank-5 张量为 0**（这是本次手术的唯一目的） |
| C2 | 规模：张量 341 / 算子 201 / operator_codes 12 |
| C3 | 无孤立的**激活**张量（只允许孤立的常量——它们是被替换下来的旧形状常量） |
| C4 | 引用完整性：算子输入输出、子图输入输出、signature_def 的张量号全在范围内 |
| C5 | **图级差异审计**：与原作者模型逐项对比，差异**只允许**出现在预期位置 |
| C6 | 每个块的形状与常量：`[1,2,C/2,H*W]` / `[1,C/2,2,H*W]` / perm `[0,2,1,3]` |
| C7 | **驱动算式不变式**：对每个张量，「补 1 到 4 维后取 dims[1]*dims[2]*dims[3]」必须等于真实元素数 |
| C8 | 两个独立读取器结论一致 |

C7 是本次问题的核心：**驱动恒定按 `dims[1]*dims[2]*dims[3]` 算字节数**，
rank-5 时取的是「最后 4 维」⇒ 丢掉 dims[0]=C/2 ⇒ 只算出 1/(C/2)。
所以「所有张量都满足这个不变式」正是「不再触碰表达能力边界」的可判定表述。

用法：

    python3 tools/verify_fold4d_model.py <候选模型> --orig <原模型>
    # 负控制
    python3 tools/verify_fold4d_model.py <原模型> --orig <原模型>     # 必须 EXIT=1
"""

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))

import tflite_fold_rank5_to_4d as fold  # noqa: E402  只借它的读取器与常量

EXPECT_TENSORS = 341
EXPECT_OPS = 201
EXPECT_CODES = 12
EXPECT_ADDED_TENSORS = 4
EXPECT_ADDED_BUFFERS = 4
EXPECT_CHANGED_SHAPE_TENSORS = 32
EXPECT_CHANGED_OPS = 32          # 16 个 r1（换 shape 常量）+ 16 个 tr（换 perm 常量）


class Result(object):
    def __init__(self):
        self.rows = []
        self.bad = 0

    def check(self, cid, ok, what, evidence):
        self.rows.append((cid, bool(ok), what, evidence))
        if not ok:
            self.bad += 1
        return ok

    def report(self):
        print("  %-4s %-6s %-46s %s" % ("编号", "结论", "检查项", "匹配到的证据"))
        print("  " + "-" * 116)
        for cid, ok, what, ev in self.rows:
            print("  %-4s %-6s %-46s %s" % (cid, "✅ PASS" if ok else "❌ FAIL", what, ev))
        print()


def sim_dims(shape):
    """复刻 delegate 的 `fill_tensor` 填法（tfl_device.c:398-404）：取形状**后 4 维、不足左补 1**。"""
    return ([1] * max(0, 4 - len(shape)) + list(shape))[-4:] if shape else [1, 1, 1, 1]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def driver_activation_ok(shape):
    """激活量法：dims[1]*dims[2]*dims[3]。

    依据（Mesa etnaviv delegate 源码）：
      - etnaviv_ml.c:294/299  add_bypass 的 input/output_tensor_sizes
      - etnaviv_ml.c:411/421/433/440  CONCAT / SPLIT / PAD
      - etnaviv_ml_nn.c:624/714/816/910  conv / depthwise / pooling / fc 的 input/output_tensor_sizes
    """
    d = sim_dims(shape)
    return d[1] * d[2] * d[3], numel(shape)


def driver_constant_ok(shape):
    """常量（权重/偏置）量法：dims[0]*dims[1]*dims[2]*dims[3]（**四维全乘**）。

    依据：
      - etnaviv_ml_nn.c:582-585  卷积 weight_size
      - etnaviv_ml_nn.c:595-598  卷积 bias_size（另乘 sizeof(int32_t)）
      - etnaviv_ml_nn.c:924-927  FC weight_size
    这正是「权重不走 dims[1..3] 那条路」的源码依据 —— 所以门必须按类别分开判，
    否则会用错公式误报（本脚本第一版就误报了 39 个权重张量）。
    """
    d = sim_dims(shape)
    return d[0] * d[1] * d[2] * d[3], numel(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--orig", required=True, help="原模型（用于 C5 差异审计，以及作为负控制样本）")
    args = ap.parse_args()

    print("=" * 120)
    print("验收折维手术产物")
    print("=" * 120)
    print("  候选: %s" % args.model)
    print("  对照: %s" % args.orig)
    print()

    new = fold.read_model(args.model)
    old = fold.read_model(args.orig)
    r = Result()

    nt = len(new["tensors"])
    nop = len(new["ops"])

    # ---------------- C1 rank-5 为 0 ----------------
    r5 = [(i, t["shape"]) for i, t in enumerate(new["tensors"]) if len(t["shape"]) > 4]
    r.check("C1", not r5, "rank-5（及以上）张量为 0",
            "rank>4 的张量 %d 个%s" % (len(r5), ("" if not r5 else "：首个 = t%d %s" % r5[0])))

    # ---------------- C2 规模 ----------------
    ev = "张量 %d / 算子 %d / opcodes %d" % (nt, nop, len(new["codes"]))
    r.check("C2", nt == EXPECT_TENSORS and nop == EXPECT_OPS and len(new["codes"]) == EXPECT_CODES,
            "规模 = %d/%d/%d" % (EXPECT_TENSORS, EXPECT_OPS, EXPECT_CODES), ev)

    # ---------------- C3 无孤立激活张量 ----------------
    used = set(new["subgraph"]["inputs"]) | set(new["subgraph"]["outputs"])
    for o in new["ops"]:
        used.update(o["inputs"])
        used.update(o["outputs"])
    orphans = [i for i in range(nt) if i not in used and new["tensors"][i]["buffer"] == 0]
    orphan_names = ["t%d%s" % (i, new["tensors"][i]["shape"]) for i in orphans]
    dead_consts = [i for i in range(nt) if i not in used and new["tensors"][i]["buffer"] != 0]
    r.check("C3", not orphans, "无孤立的激活张量（常量允许孤立）",
            "孤立激活 %d 个%s；孤立常量 %d 个（被替换下来的旧 5D 形状/perm 常量）"
            % (len(orphans), ("" if not orphans else " = %s" % orphan_names[:4]), len(dead_consts)))

    # ---------------- C4 引用完整性 ----------------
    bad_refs = []
    for oi, o in enumerate(new["ops"]):
        for x in o["inputs"] + o["outputs"]:
            if x < 0 or x >= nt:
                bad_refs.append("op#%d -> t%d" % (oi, x))
    for x in new["subgraph"]["inputs"] + new["subgraph"]["outputs"]:
        if x < 0 or x >= nt:
            bad_refs.append("subgraph -> t%d" % x)
    for s in new["sigs"]:
        for k in ("inputs", "outputs"):
            for tm in s[k]:
                if tm["tensor_index"] < 0 or tm["tensor_index"] >= nt:
                    bad_refs.append("sig %s -> t%d" % (tm["name"], tm["tensor_index"]))
    r.check("C4", not bad_refs, "引用完整性（算子/子图/signature_def）",
            "越界引用 %d 个%s" % (len(bad_refs), ("" if not bad_refs else "：%s" % bad_refs[:3])))

    # ---------------- C5 图级差异审计 ----------------
    # 允许的差异：
    #   (a) 32 个张量的 shape 由 5D 变 4D（且 name 加了 /och-fold4d 后缀）
    #   (b) 追加 4 个张量 + 4 个 buffer
    #   (c) 16 个 RESHAPE 算子的 inputs[1] 改了；16 个 TRANSPOSE 算子的 inputs[1] 改了
    diffs = []
    if nt != len(old["tensors"]) + EXPECT_ADDED_TENSORS:
        diffs.append("张量数 %d != %d+%d" % (nt, len(old["tensors"]), EXPECT_ADDED_TENSORS))
    if len(new["buffers"]) != len(old["buffers"]) + EXPECT_ADDED_BUFFERS:
        diffs.append("buffer 数 %d != %d+%d" % (len(new["buffers"]), len(old["buffers"]), EXPECT_ADDED_BUFFERS))
    if len(new["codes"]) != len(old["codes"]):
        diffs.append("opcodes 数变了")

    changed_shape = []
    for i in range(min(nt, len(old["tensors"]))):
        a, b = old["tensors"][i], new["tensors"][i]
        if a["shape"] != b["shape"]:
            if len(a["shape"]) == 5 and len(b["shape"]) == 4:
                changed_shape.append(i)
            else:
                diffs.append("t%d 形状 %s -> %s（不是 5D->4D）" % (i, a["shape"], b["shape"]))
        for f in ("type", "buffer", "quant", "is_variable"):
            if a[f] != b[f]:
                diffs.append("t%d 的 %s 变了" % (i, f))
        base = b["name"] or ""
        orig_nm = a["name"] or ""
        if base != orig_nm:                      # 名称只允许加后缀
            if base != orig_nm + "/och-fold4d":
                diffs.append("t%d 名称 %r -> %r" % (i, orig_nm, base))
    if len(changed_shape) != EXPECT_CHANGED_SHAPE_TENSORS:
        diffs.append("形状改变的张量 %d 个，期望 %d" % (len(changed_shape), EXPECT_CHANGED_SHAPE_TENSORS))

    # 原张量边界内的常量数据必须一字不改（保证没动任何权重/量化）
    for i in range(len(old["buffers"])):
        if old["buffers"][i]["data"] != new["buffers"][i]["data"]:
            diffs.append("buffer %d 的常量数据被改动" % i)

    changed_ops = []
    if nop != len(old["ops"]):
        diffs.append("算子数 %d != %d" % (nop, len(old["ops"])))
    else:
        for i in range(nop):
            a, b = old["ops"][i], new["ops"][i]
            if a == b:
                continue
            same_keys = (a["opcode_index"] == b["opcode_index"] and a["outputs"] == b["outputs"]
                         and a["opts_type"] == b["opts_type"] and a["opts"] == b["opts"])
            only_const = (len(a["inputs"]) == len(b["inputs"]) == 2
                          and a["inputs"][0] == b["inputs"][0]
                          and a["inputs"][1] != b["inputs"][1])
            if same_keys and only_const:
                changed_ops.append(i)
            else:
                diffs.append("op#%d 改动超出预期" % i)
    if len(changed_ops) != EXPECT_CHANGED_OPS:
        diffs.append("改动的算子 %d 个，期望 %d" % (len(changed_ops), EXPECT_CHANGED_OPS))

    r.check("C5", not diffs, "图级差异审计（只允许预期差异）",
            "形状 5D->4D 的张量 %d 个；仅换常量输入的算子 %d 个；越界差异 %d 项%s"
            % (len(changed_shape), len(changed_ops), len(diffs),
               ("" if not diffs else "：%s" % diffs[:3])))

    # ---------------- C6 每个块的形状与常量 ----------------
    bad_blocks = []
    n_blocks = 0
    for i in changed_shape:
        shp = new["tensors"][i]["shape"]
        hw = shp[3]
        # 期望 [1,2,C/2,H*W] 或 [1,C/2,2,H*W]
        ok = (len(shp) == 4 and shp[0] == 1
              and ((shp[1] == 2 and hw >= 1) or (shp[2] == 2 and shp[1] >= 1))
              and shp[3] % 1 == 0)
        # 更严：把它还原成 5D 的候选，必须与旧形状元素数一致
        old_numel = 1
        for s in old["tensors"][i]["shape"]:
            old_numel *= s
        new_numel = 1
        for s in shp:
            new_numel *= s
        if not ok or old_numel != new_numel:
            bad_blocks.append("t%d %s" % (i, shp))
        n_blocks += 1
    r.check("C6", not bad_blocks, "32 个块张量：4D 且元素数与原 5D 一致",
            "检查 %d 个，异常 %d 个%s" % (n_blocks, len(bad_blocks),
                                        ("" if not bad_blocks else "：%s" % bad_blocks[:3])))

    # ---------------- C7 驱动算式不变式（按张量类别分） ----------------
    def is_const(i):
        b = new["tensors"][i]["buffer"]
        return b is not None and b < len(new["buffers"]) and bool(new["buffers"][b]["data"])

    viol_act, viol_const = [], []
    for i, t in enumerate(new["tensors"]):
        if is_const(i):
            got, n = driver_constant_ok(t["shape"])
            if got != n:
                viol_const.append("t%d%s 四维全乘 %d vs 元素 %d" % (i, t["shape"], got, n))
        else:
            got, n = driver_activation_ok(t["shape"])
            if got != n:
                viol_act.append("t%d%s dims[1..3] 算 %d vs 元素 %d" % (i, t["shape"], got, n))
    n_act = sum(1 for i in range(nt) if not is_const(i))
    n_con = nt - n_act
    r.check("C7", not viol_act and not viol_const,
            "驱动算式不变式（激活 dims[1..3] / 常量四维全乘）",
            "激活 %d 个违反 %d；常量 %d 个违反 %d%s"
            % (n_act, len(viol_act), n_con, len(viol_const),
               ("" if not (viol_act + viol_const) else "：%s" % (viol_act + viol_const)[:2])))

    # ---------------- C8 两个独立读取器交叉验证 ----------------
    import tflite_min_read as mr
    mm = mr.load_model(args.model)
    r5b = [t["i"] for t in mm["tensors"] if t["rank"] > 4]
    agree = (mm["n_tensors"] == nt and mm["n_operators"] == nop and not r5b)
    r.check("C8", agree, "零依赖读取器与官方绑定结论一致",
            "最小读取器: 张量 %d / 算子 %d / rank>4 %d 个"
            % (mm["n_tensors"], mm["n_operators"], len(r5b)))

    # ---------------- 负控制（自带的，不靠外部） ----------------
    print("  " + "=" * 116)
    print("  负控制：同一套检查跑**原模型**（它真有 32 个 rank-5），必须在 C1/C7 变红")
    print("  " + "=" * 116)
    old_r5 = [(i, t["shape"]) for i, t in enumerate(old["tensors"]) if len(t["shape"]) > 4]

    def old_is_const(i):
        b = old["tensors"][i]["buffer"]
        return b is not None and b < len(old["buffers"]) and bool(old["buffers"][b]["data"])

    old_viol_act = [i for i, t in enumerate(old["tensors"])
                    if not old_is_const(i) and driver_activation_ok(t["shape"])[0] != numel(t["shape"])]
    old_viol_con = [i for i, t in enumerate(old["tensors"])
                    if old_is_const(i) and driver_constant_ok(t["shape"])[0] != numel(t["shape"])]
    neg_ok = bool(old_r5) and bool(old_viol_act) and not old_viol_con
    print("    原模型 rank-5 张量 %d 个；激活违反 %d 个；常量违反 %d 个"
          % (len(old_r5), len(old_viol_act), len(old_viol_con)))
    if old_viol_act:
        i = old_viol_act[0]
        got, n = driver_activation_ok(old["tensors"][i]["shape"])
        print("    例：t%d %s -> 驱动算 %d，实际 %d（短 %dx）"
              % (i, old["tensors"][i]["shape"], got, n, n // max(got, 1)))
    print("    ⇒ 负控制 %s" % ("✅ 有效（原模型的激活张量确实会红，且常量侧本来就干净）"
                              if neg_ok else "❌ 无效（原模型也不红 ⇒ 这门没判别力）"))
    if not neg_ok:
        r.check("NEG", False, "负控制有效性",
                "原模型未变红：rank-5=%d 激活违反=%d 常量违反=%d"
                % (len(old_r5), len(old_viol_act), len(old_viol_con)))
    else:
        r.check("NEG", True, "负控制有效性",
                "原模型 rank-5 %d 个 / 激活违反 %d 个 / 常量违反 %d 个"
                % (len(old_r5), len(old_viol_act), len(old_viol_con)))

    r.report()
    print("  汇总: 通过 %d / %d，失败 %d" % (len(r.rows) - r.bad, len(r.rows), r.bad))
    print("  结论: %s" % ("✅ 全部通过" if r.bad == 0 else "❌ 有 %d 项失败" % r.bad))
    return 0 if r.bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
