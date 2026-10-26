#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pad-to-8.py —— 模型手术：把 conv 算子的输出通道补齐到 8 的倍数（消除幻影通道）

## 为什么做这个（零驱动改动路线）

Mesa `etnaviv_ml_nn_v7.c` 三个写入器都按
    kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)      (v7:193)
分配通道 ⇒ 通道总数上取整到 8 的倍数 ⇒ 末批多出

    m = 8*ceil(C/8) - C        (0..7)

个**幻影通道**，被真实执行 ⇒ 权重读 / 偏置读 / 输出写 三处越界。

真机实测：1×1 conv 在 C=244 / 284 挂死；122/196/220/240/248/252/260/280 不挂。
⇒ `m>0` 是**必要条件**，但**不充分**（≥3 个判据同时 10/10 拟合，判别律仍欠定）。

把 C 补到 8 的倍数 ⇒ **m=0** ⇒ 该机制**不可能触发**，且：

  ★ 补齐后 (kpc0, sb, d, rem) **与已实测不挂的 248 行完全相同**
    （kpc0 = ceil(C/8) 不变 ⇒ sb/d/rem 不变 ⇒ 只把 m 归零）
  ★ **零驱动改动、零 CI**：只改模型文件
  ★ 语义**可证**：补出来的通道写入的是 zp（权重）/ 0（偏置）
    ⇒ (w-zp)=0 ⇒ 累加为 0 ⇒ 输出恒为 zp_out ⇒ 下游 (x-zp_out)=0 ⇒ 贡献 0

## 补零的精确规则（别写错，会静默改数值）

  · **权重补 `weight_zero_point`，不是 0**：
    TFLite 的累加是 sum((w - w_zp) * (x - x_zp))，只有 w == w_zp 才贡献 0。
    非对称量化（zp≠0）下补 0 会引入巨大误差。
  · **偏置补 0**（int32 累加域，0 不贡献）。
  · **per-axis 权重量化的 scale/zp 数组要同步延长**（长度必须等于输出通道数）。
  · **深度卷积（DEPTHWISE_CONV_2D）**：其输出通道数 == 输入通道数（1:1 绑定），
    所以**必须把它的输入也补齐**，否则张量形状对不上 ⇒ 本工具会自动级联处理。
  · **消费者**：任何读到被改张量的算子，其权重**输入轴**必须同步补 zp。

## 边界与失败策略（fail-closed）

  · 若某个被改张量的消费者**不是** conv/FC（例如 ADD / CONCAT / RESHAPE），
    本工具**直接报错退出**并列出该算子 —— 绝不产出半改的模型。
  · 若被改张量是**图的输出**，输出形状会变 ⇒ 默认报错退出；
    `--allow-output-pad` 可放行（调用方必须自己裁剪）。

## 用法

    python3 pad-to-8.py <in.tflite> <out.tflite> [--dry-run]
    python3 pad-to-8.py <in.tflite> <out.tflite> --ops 53,55,56
    python3 pad-to-8.py <in.tflite> <out.tflite> --only-high-risk
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in [os.path.join(REPO_ROOT, "oes-teflon", "tools"), os.path.join(REPO_ROOT, "tools"), HERE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import tflite_fold_rank5_to_4d as fold  # noqa: E402
import importlib.util  # noqa: E402

_sim_file = os.path.join(HERE, "sim-write-core-index.py")
if not os.path.exists(_sim_file):
    _sim_file = os.path.join(REPO_ROOT, "oes-teflon", "tools", "sim-write-core-index.py")
_spec = importlib.util.spec_from_file_location("sim_write_core_index", _sim_file)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

CONV_2D, DEPTHWISE_CONV_2D, ADD, FULLY_CONNECTED = 3, 4, 0, 9
TYPE_SIZE = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 7: 2, 9: 1}   # float32/float16/int32/uint8/int64/int16/int8


def builtin(mdl, op):
    return mdl["codes"][op["opcode_index"]]["builtin"]


def tensor_bytes(mdl, ti):
    t = mdl["tensors"][ti]
    return mdl["buffers"][t["buffer"]]["data"]


def q_of(mdl, ti):
    q = mdl["tensors"][ti].get("quant")
    return q or {}


def zp_of(mdl, ti, axis_len=1):
    """取张量的 zero point：per-axis 取 axis_len-1，per-tensor 取 0。"""
    q = q_of(mdl, ti)
    zp = q.get("zp") or []
    if not zp:
        return 0
    if len(zp) > 1 and axis_len:
        return zp[min(axis_len - 1, len(zp) - 1)]
    return zp[0]


def scale_of(mdl, ti, axis_len=1):
    q = q_of(mdl, ti)
    sc = q.get("scale") or []
    if not sc:
        return 1.0
    if len(sc) > 1 and axis_len:
        return sc[min(axis_len - 1, len(sc) - 1)]
    return sc[0]


def out_ch(t):
    sh = list(t["shape"])
    if len(sh) == 4:
        return sh[0] if sh[0] > 1 else sh[3]
    if len(sh) == 2:
        return sh[0]
    return None


def ch_axis(t):
    """张量通道所在的维度下标（NHWC ⇒ 最后一维）。"""
    return len(t["shape"]) - 1


def is_conv_like(mdl, op):
    # 幻影通道仅存在于 2D 卷积/深度卷积的 NN 核心写入器中，FC 不走卷积核，不破坏分类输出契约
    return builtin(mdl, op) in (CONV_2D, DEPTHWISE_CONV_2D)



def const_i32(mdl, ti):
    """读一个 int32 常量张量的值列表（STRIDED_SLICE 的 begin/end、TRANSPOSE 的 perm…）。"""
    t = mdl["tensors"][ti]
    if t.get("type") != 2:          # 2 = INT32
        return None
    data = mdl["buffers"][t["buffer"]]["data"]
    n = len(data) // 4
    return [int.from_bytes(data[i * 4:i * 4 + 4], "little", signed=True) for i in range(n)]


def set_const_i32(mdl, ti, vals):
    """改写一个 int32 常量张量的内容（长度必须一致）。"""
    t = mdl["tensors"][ti]
    data = mdl["buffers"][t["buffer"]]["data"]
    if len(data) != len(vals) * 4:
        raise SystemExit("ERROR: 常量张量 %d 长度不符（%d B vs 期望 %d B）"
                         % (ti, len(data), len(vals) * 4))
    out = bytearray()
    for v in vals:
        out += int(v).to_bytes(4, "little", signed=True)
    mdl["buffers"][t["buffer"]]["data"] = bytes(out)


def pad_to(c, a=8):
    return ((c + a - 1) // a) * a


class Track(object):
    """通道轴跟踪（移植自 oca 的 plan-pad-to-8.py 的传播规则）。

    每个张量记 (axis, count, target)：
      axis   = 通道所在的维度下标（NHWC 在 3；TRANSPOSE 后会移动！）
      count  = 当前通道数
      target = 补齐后的目标通道数
    ★ 关键：**通道轴会移动** —— NHWC 在 dim3；TRANSPOSE(0,3,1,2) 后在 dim1；
      RESHAPE [1,2,X,H,W] 后在 dim2 ⇒ STRIDED_SLICE 切的是 begin/end 的**第 2 个分量**。
    """

    def __init__(self, mdl):
        self.mdl = mdl
        self.info = {}      # tensor_index -> (axis, count, target)
        self.rewrites = []  # (const_tensor_index, old_values, new_values, why)
        self.barriers = []  # (op_index, 原因) —— 无法安全传播的补齐请求


def build_track(mdl, ops, T):
    tr = Track(mdl)
    # 图输入：通道在最后一维
    for ti in mdl["subgraph"]["inputs"]:
        sh = list(T[ti]["shape"])
        if len(sh) >= 3:
            # ★ 图输入的通道数**不可补**（它是模型的输入契约）⇒ target = count
            tr.info[ti] = (len(sh) - 1, sh[-1], sh[-1])

    for i, op in enumerate(ops):
        b = builtin(mdl, op)
        ins = [tr.info.get(t) for t in op["inputs"] if t in tr.info]
        outs = op["outputs"]

        if b in (CONV_2D, DEPTHWISE_CONV_2D) and len(op["inputs"]) >= 2:
            wt = T[op["inputs"][1]]
            sh = list(wt["shape"])
            out_c = sh[3] if b == DEPTHWISE_CONV_2D else (sh[0] if sh[0] > 1 else sh[3])
            in_info = tr.info.get(op["inputs"][0])
            # depthwise 输出通道必须跟随输入通道
            if b == DEPTHWISE_CONV_2D and in_info and in_info[2]:
                out_tgt = in_info[2]
            else:
                out_tgt = pad_to(out_c)
            for o in outs:
                tr.info[o] = (len(T[o]["shape"]) - 1, out_c, out_tgt)
            if b == DEPTHWISE_CONV_2D and op["inputs"][0] in tr.info:
                iti = op["inputs"][0]
                iax, icnt, existing_tgt = tr.info[iti]
                if iti in mdl["subgraph"]["inputs"]:
                    tr.barriers.append(
                        (len(tr.info), "op#%d(depthwise) 要求输入补齐，但输入是图输入（契约不可改）" % i))
                else:
                    needed = max(existing_tgt, pad_to(icnt))
                    tr.info[iti] = (iax, icnt, needed)
        elif b == FULLY_CONNECTED and len(op["inputs"]) >= 2:
            # FC 保持原输出维度（例如 335 类别），绝不向外补齐破坏调用方契约
            wt = T[op["inputs"][1]]
            out_c = wt["shape"][0]
            for o in outs:
                tr.info[o] = (len(T[o]["shape"]) - 1, out_c, out_c)

        elif b == 2 and outs:                     # CONCATENATION
            axis = (op.get("opts") or {}).get("Axis", 3)
            cnt = sum(x[1] for x in ins) if ins else 0
            tgt = sum(x[2] for x in ins) if ins else 0
            for o in outs:
                ax = 3 if axis in (3, -1) else (ins[0][0] if ins else len(T[o]["shape"]) - 1)
                tr.info[o] = (ax, cnt, tgt if axis in (3, -1) else (ins[0][2] if ins else cnt))

        elif b == 22 and outs and len(op["inputs"]) > 1:   # RESHAPE
            shp = const_i32(mdl, op["inputs"][1])
            in_info = tr.info.get(op["inputs"][0])
            new = list(shp) if shp else None
            if new and len(new) == 5 and new[1] == 2:
                # [1, 2, X, H, W]：通道轴 dim2，X' = concat 目标的一半
                ax, cnt, tgt = (2, new[2], new[2])
                if in_info and in_info[2]:
                    half = in_info[2] // 2
                    ax, cnt, tgt = (2, new[2], half)
                    new[2] = half
            elif new and len(new) == 4:
                # [1, C, H, W]：通道轴 dim1。
                # ★ 只有「输入是 rank-5 的 SE 拆分布局 [1,2,X,h,w]」时，C_new 才能由
                #   半通道目标推出（C_new = 2 × X_target）。
                #   **输入是 rank-4 时没有任何通道依据** ⇒ 常量保持原值。
                #   实测反例：合成模型 op#0 输入 [1,122,2,2500]，最后一维 2500 被误当成
                #   通道数 ⇒ 常量改写成 [1,2500,50,50] = 6,250,000（元素数守恒被破坏）。
                ax, cnt, tgt = (1, new[1], new[1])
                if in_info and in_info[2] and len(T[op["inputs"][0]]["shape"]) == 5:
                    cnt, tgt = in_info[1] * 2, in_info[2] * 2
                    new[1] = tgt
            else:
                ax, cnt, tgt = in_info if in_info else (None, 0, 0)
            for o in outs:
                tr.info[o] = (ax, cnt, tgt)
            if new and shp and new != list(shp):
                tr.rewrites.append((op["inputs"][1], list(shp), new, "RESHAPE shape 常量"))

        elif b == 45 and outs:                   # STRIDED_SLICE
            beg = const_i32(mdl, op["inputs"][1]) if len(op["inputs"]) > 1 else None
            end = const_i32(mdl, op["inputs"][2]) if len(op["inputs"]) > 2 else None
            in_info = tr.info.get(op["inputs"][0])
            if in_info and beg and end:
                ax, cnt, tgt = in_info
                ocnt = (end[ax] - beg[ax]) if (ax is not None and ax < len(end)) else cnt
                nb, ne = list(beg), list(end)
                if ax is not None and ax < len(nb) and tgt and cnt and tgt != cnt:
                    half = tgt // 2
                    if beg[ax] == 0:
                        nb[ax] = 0
                        ne[ax] = half
                    else:
                        nb[ax] = half
                        ne[ax] = tgt
                    for o in outs:
                        tr.info[o] = (ax, ocnt, half)
                    if nb != list(beg):
                        tr.rewrites.append((op["inputs"][1], list(beg), nb, "STRIDED_SLICE begin 常量"))
                    if ne != list(end):
                        tr.rewrites.append((op["inputs"][2], list(end), ne, "STRIDED_SLICE end 常量"))
                else:
                    ocnt = (end[ax] - beg[ax]) if (ax is not None and ax < len(end)) else cnt
                    for o in outs:
                        tr.info[o] = (ax, ocnt, ocnt)

        elif b == 39 and outs and len(op["inputs"]) > 1:   # TRANSPOSE
            perm = const_i32(mdl, op["inputs"][1])
            in_info = tr.info.get(op["inputs"][0])
            if perm and in_info and in_info[0] is not None:
                ax = perm.index(in_info[0]) if in_info[0] in perm else in_info[0]
                for o in outs:
                    tr.info[o] = (ax, in_info[1], in_info[2])
            else:
                for o in outs:
                    tr.info[o] = in_info if in_info else (None, 0, 0)

        else:
            for o in outs:
                tr.info[o] = ins[0] if ins else (None, 0, 0)
    return tr


def preflight_conv_in_axis(mdl, ops, T):
    """**改写前**检查：conv 权重 in 维必须等于输入张量的通道数（NHWC ⇒ 最后一维）。

    为什么必须在改写前：放到位后写回阶段才报「权重字节数 ≠ 期望」已经晚了 ——
    那时半改的模型可能已落盘，而且真实根因会被字节数报错**掩盖**。
    实测（oca 模型）：op#58 权重 [244,3,3,1]、op#153 权重 [488,3,3,1]，
    对应输入通道 244 / 488 ⇒ 越界读 533,628 B / 2,138,904 B。
    """
    bad = []
    for i, op in enumerate(ops):
        if not is_conv_like(mdl, op):
            continue
        wsh = list(T[op["inputs"][1]]["shape"])
        ishp = list(T[op["inputs"][0]]["shape"])
        if not ishp or not wsh:
            continue
        b = builtin(mdl, op)
        if b == FULLY_CONNECTED:
            w_in = wsh[1] if len(wsh) == 2 else None
        elif b == DEPTHWISE_CONV_2D:
            w_in = wsh[3] if len(wsh) == 4 else None
        else:
            w_in = wsh[3] if len(wsh) == 4 else None
        in_c = ishp[-1]
        if w_in is not None and w_in != in_c:
            bad.append("op#%d 权重 %s 的 in 维 = %d，但输入张量 %d 的通道数 = %d"
                       % (i, wsh, w_in, op["inputs"][0], in_c))
    if bad:
        sys.stderr.write(
            "ERROR: %d 个 conv 的「权重 in 维」与输入通道数不一致 ⇒ 模型结构性畸形，"
            "拒绝手术：\n  " % len(bad) + "\n  ".join(bad) + "\n"
            "       ⇒ 这类算子会让驱动越界读权重（不是 pad-to-8 能修的）——\n"
            "         修法在模型侧（ONNX 侧溯源：是不是 depthwise 被降级成了 conv）。\n")
        return 1
    return 0


def preflight_shapes(mdl, ops, T):
    """**改写前**的形状自洽检查：RESHAPE / CONCAT 的元素数必须守恒。

    为什么必须在改写前：改写会把「坏常量」悄悄修成好值（实测：把 SE 块的
    [1,2,122,50,50] 改成 121，工具会自己改回 128 ⇒ 自相掩盖）。
    ⇒ 坏模型必须在**入口**就被拒。
    """
    def prod(shape):
        n = 1
        for x in shape:
            n *= x
        return n
    for i, op in enumerate(ops):
        b = builtin(mdl, op)
        if b == 22 and len(op["inputs"]) > 1:
            ain = prod(T[op["inputs"][0]]["shape"])
            aout = prod(T[op["outputs"][0]]["shape"])
            # ★ 必须比**常量**（reshape 的目标 shape），而不只是张量的声明形状 ——
            #   声明形状是「旧的正确值」，常量才是驱动真正用的东西；
            #   实测：把常量改坏（122→121）而声明形状不变 ⇒ 只比声明形状会漏过。
            shp = const_i32(mdl, op["inputs"][1])
            aconst = prod(shp) if shp else None
            if ain != aout or (aconst is not None and ain != aconst):
                sys.stderr.write(
                    "ERROR: op#%d RESHAPE 元素数不自洽（改写前）：输入 %d ≠ 输出声明 %d"
                    "（常量 %s）\n"
                    "       ⇒ 模型本身就是坏的（或 SE 块常量被改过）—— 拒绝手术。\n"
                    % (i, ain, aout, shp))
                return 1
        elif b == 2 and len(op["inputs"]) > 1:
            ain = sum(prod(T[t]["shape"]) for t in op["inputs"])
            aout = prod(T[op["outputs"][0]]["shape"])
            if ain != aout:
                sys.stderr.write(
                    "ERROR: op#%d CONCAT 元素数不自洽（改写前）：输入和 %d ≠ 输出 %d\n"
                    % (i, ain, aout))
                return 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ops", help="只处理这些算子下标（逗号分隔）；默认自动找 m>0 的 conv")
    ap.add_argument("--only-high-risk", action="store_true",
                    help="只处理命中高危形态（rem==1 或 全幻影核>=3 或 受影响核>=3）的 conv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-output-pad", action="store_true")
    args = ap.parse_args()

    mdl = fold.read_model(args.src)
    T = mdl["tensors"]
    ops = mdl["ops"]

    # ---------- 消费者表 ----------
    consumers = {}
    for i, op in enumerate(ops):
        for ti in op["inputs"]:
            consumers.setdefault(ti, []).append(i)

    # ---------- pre-flight：模型本身必须自洽 ----------
    # 顺序有讲究：先查**形状自洽**（更基础的约束），再查**权重/输入通道一致**。
    # 这样「SE 块常量被改坏」与「结构性畸形 conv」两类坏模型各自命中自己的理由，
    # 不会互相掩盖（自测的两个负控制分别依赖这两条）。
    if preflight_shapes(mdl, ops, T) != 0:
        return 1
    if preflight_conv_in_axis(mdl, ops, T) != 0:
        return 1

    # ---------- 选目标 ----------
    explicit = None
    if args.ops:
        explicit = set(int(x) for x in args.ops.replace(" ", "").split(",") if x)

    targets = []   # (op_index, new_out_c)
    for i, op in enumerate(ops):
        if not is_conv_like(mdl, op):
            continue
        if explicit is not None and i not in explicit:
            continue
        wti = op["inputs"][1]
        c = out_ch(T[wti])
        if not c or c % 8 == 0:
            continue
        oti = op["outputs"][0]
        ot = T[oti]
        h = ot["shape"][1] if len(ot["shape"]) == 4 else 1
        w = ot["shape"][2] if len(ot["shape"]) == 4 else 1
        kh = T[wti]["shape"][1] if len(T[wti]["shape"]) == 4 else 1
        kw = T[wti]["shape"][2] if len(T[wti]["shape"]) == 4 else 1
        geom = sim.Geometry(c, w, h, max(kh, 1), max(kw, 1), in_c=c, name="dense")
        a = sim.analyse(geom)
        high = (a["rem"] == 1 or a["fp_cores"] >= 3 or a["aff_cores"] >= 3)
        if explicit is None:
            if args.only_high_risk and not high:
                continue
            if a["m"] == 0:
                continue
        targets.append((i, ((c + 7) // 8) * 8, c, a))

    if not targets:
        sys.stderr.write("没有需要补齐的 conv 算子（目标: %s）\n" % args.src)
        return 1

    # ---------- 通道轴跟踪 + 目标传播 ----------
    # ★ 移植自 oca 的 plan-pad-to-8.py（该工具依赖 numpy，本机跑不了 ⇒ 取其规则、
    #   用本仓的纯标准库实现，并在这里做等价验证）。
    tr = build_track(mdl, ops, T)
    if tr.barriers:
        sys.stderr.write("ERROR: 有 %d 处补齐请求无法安全传播 ⇒ 拒绝手术（fail-closed）：\n"
                         % len(tr.barriers))
        for _, why in tr.barriers:
            sys.stderr.write("  · %s\n" % why)
        sys.stderr.write("      建议：改用 TEFLON_PHANTOM_GATE=1（驱动侧门控回落 CPU），"
                         "或用 --ops 排除这些算子。\n")
        return 1
    new_ch = {}
    for ti, (ax, cnt, tgt) in tr.info.items():
        if tgt and cnt and tgt != cnt:
            new_ch[ti] = tgt
    # 算子级目标：conv 的权重 out/in 维
    pad_ops = {}
    in_pad = {}
    for i, op in enumerate(ops):
        if not is_conv_like(mdl, op):
            continue
        oti = op["outputs"][0]
        tgt = tr.info.get(oti)
        if tgt and tgt[1] and tgt[2] != tgt[1]:
            pad_ops[i] = tgt[2]
        # 权重 in 维：跟随输入张量的**实际**新通道数（new_ch 里没有 ⇒ 不补）
        if op["inputs"][0] in tr.info:
            ax, cnt, itgt = tr.info[op["inputs"][0]]
            actual = new_ch.get(op["inputs"][0], cnt)
            if actual and actual != cnt:
                in_pad[i] = actual

    # 图输出检查
    graph_out = set(mdl["subgraph"]["outputs"])
    bad_out = sorted(graph_out & set(new_ch))
    if bad_out and not args.allow_output_pad:
        sys.stderr.write(
            "ERROR: 张量 %s 是图的输出且通道数被改 ⇒ 模型输出形状会变。\n"
            "       默认拒绝（--allow-output-pad 可放行，调用方须自行裁剪）。\n" % (bad_out,))
        return 1

    # ---------- 计划 ----------
    print("=== pad-to-8 计划 (%s) ===" % os.path.basename(args.src))
    print("%5s %-18s %8s %8s %8s" % ("op", "类型", "C_old", "C_new", "输入轴补到"))
    for i in sorted(pad_ops):
        op = ops[i]
        wti = op["inputs"][1]
        c = out_ch(T[wti])
        print("%5d %-18s %8d %8d %8d" % (i, op.get("builtin_name") or builtin(mdl, op),
                                          c, pad_ops[i], in_pad.get(i, 0)))
    print("被改的张量 %d 个；算子 %d 个" % (len(new_ch), len(set(list(pad_ops) + list(in_pad)))))
    if bad_out:
        print("⚠️ 图输出 %s 的通道数被改（--allow-output-pad 已放行）" % bad_out)
    if args.dry_run:
        print("（--dry-run：未写文件）")
        return 0

    # ---------- 应用 ----------
    # 1) 权重/偏置数据
    for i in sorted(set(list(pad_ops) + list(in_pad))):
        op = ops[i]
        b = builtin(mdl, op)
        wti, bti = op["inputs"][1], op["inputs"][2]
        wt, bt = T[wti], T[bti]
        wdata = bytearray(tensor_bytes(mdl, wti))
        bdata = bytearray(tensor_bytes(mdl, bti))
        wshape = list(wt["shape"])
        wsz = TYPE_SIZE.get(wt["type"], 1)
        bsz = TYPE_SIZE.get(bt["type"], 4)

        out_pad = pad_ops.get(i, 0)      # 输出轴要补到的通道数
        in_pad_to = in_pad.get(i, 0)     # 输入轴要补到的通道数

        if b == FULLY_CONNECTED:
            # FC 权重 [out_c, in_c]
            oc, ic = wshape
            new_oc = out_pad or oc
            new_ic = in_pad_to or ic
        elif b == DEPTHWISE_CONV_2D:
            # depthwise 权重 [1, kh, kw, out_c(=in_c)]
            _, kh, kw, oc = wshape
            new_oc = in_pad_to if in_pad_to else (out_pad or oc)
            new_ic = new_oc
        else:
            oc, kh, kw, ic = wshape
            new_oc = out_pad or oc
            new_ic = in_pad_to or ic

        wzp = zp_of(mdl, wti, axis_len=max(1, out_ch(wt) or 1))
        bzp = zp_of(mdl, bti, axis_len=max(1, out_ch(bt) or 1))

        # ---- 输出轴补齐 ----
        if new_oc > oc:
            elems_per_out = (kh * kw * ic) if b != DEPTHWISE_CONV_2D else (kh * kw)
            new_w = bytearray()
            for oc_i in range(oc):
                row = wdata[oc_i * elems_per_out * wsz: (oc_i + 1) * elems_per_out * wsz]
                new_w += row
            # 补的行：权重 = weight zp ⇒ (w - w_zp) = 0 ⇒ 贡献 0
            pad_row = bytes([wzp & 0xFF]) * (elems_per_out * wsz)
            for _ in range(new_oc - oc):
                new_w += pad_row
            wdata = new_w
            wshape = ([new_oc, kh, kw, ic] if b == CONV_2D else
                      [1, kh, kw, new_oc] if b == DEPTHWISE_CONV_2D else
                      [new_oc, ic])
            # 偏置补 0
            if len(bdata) < new_oc * bsz:
                new_b = bytearray(bdata[:oc * bsz])
                new_b += bytes(new_oc - oc) * bsz
                bdata = new_b
            # 偏置张量的形状也要跟着改（否则字节数与形状不自洽）
            bshape = list(bt["shape"])
            if len(bshape) == 1:
                bt["shape"] = [new_oc]
            elif len(bshape) == 2 and bshape[0] == 1:
                bt["shape"] = [1, new_oc]
            else:
                sys.stderr.write("ERROR: op#%d 偏置形状 %s 异常，无法安全补齐\n" % (i, bshape))
                return 1
            # per-axis 权重量化：scale/zp 数组延长
            q = q_of(mdl, wti)
            if q.get("scale") and len(q["scale"]) == oc:
                q["scale"] = list(q["scale"]) + [q["scale"][-1]] * (new_oc - oc)
                q["zp"] = list(q.get("zp") or [0] * oc) + [(q.get("zp") or [0] * oc)[-1]] * (new_oc - oc)
            # ★ 偏置的 per-axis 量化数组**也必须**延长。漏了这条 TFLite 会直接拒绝加载：
            #   "num_scales must be 1 for per-layer quantization, or 248 for per-axis
            #    quantization, but got 244. Tensor 7 has invalid quantization parameters."
            #   （实测：C2 上机时解释器创建失败 ⇒ 模型根本没跑起来。字节补齐不够。）
            bq = q_of(mdl, bti)
            if bq.get("scale") and len(bq["scale"]) == oc:
                bq["scale"] = list(bq["scale"]) + [bq["scale"][-1]] * (new_oc - oc)
                bq["zp"] = list(bq.get("zp") or [0] * oc) + [(bq.get("zp") or [0] * oc)[-1]] * (new_oc - oc)
            # 输出张量的 per-axis 量化
            oti = op["outputs"][0]
            oq = q_of(mdl, oti)
            if oq.get("scale") and len(oq["scale"]) == oc:
                oq["scale"] = list(oq["scale"]) + [oq["scale"][-1]] * (new_oc - oc)
                oq["zp"] = list(oq.get("zp") or [0] * oc) + [(oq.get("zp") or [0] * oc)[-1]] * (new_oc - oc)

        # ---- 输入轴补齐（只影响 conv 的输入通道轴）----
        if b != DEPTHWISE_CONV_2D and new_ic > ic:
            elems_per_out = kh * kw * ic
            new_w = bytearray()
            for oc_i in range(wshape[0]):
                row = wdata[oc_i * elems_per_out * wsz: (oc_i + 1) * elems_per_out * wsz]
                newrow = bytearray()
                for k in range(kh * kw):
                    newrow += row[k * ic * wsz:(k + 1) * ic * wsz]
                    newrow += bytes([wzp & 0xFF]) * (new_ic - ic) * wsz
                new_w += newrow
            wdata = new_w
            wshape = [wshape[0], kh, kw, new_ic]

        # 写回
        wt["shape"] = wshape
        mdl["buffers"][wt["buffer"]]["data"] = bytes(wdata)
        mdl["buffers"][wt["buffer"]]["size"] = len(wdata)
        if len(bdata) != (wshape[0] if b != DEPTHWISE_CONV_2D else wshape[3]) * bsz:
            # depthwise 的偏置长度 = out_c
            exp = (wshape[3] if b == DEPTHWISE_CONV_2D else wshape[0]) * bsz
            if len(bdata) != exp:
                sys.stderr.write("ERROR: op#%d 偏置字节数 %d ≠ 期望 %d\n" % (i, len(bdata), exp))
                return 1
        mdl["buffers"][bt["buffer"]]["data"] = bytes(bdata)
        mdl["buffers"][bt["buffer"]]["size"] = len(bdata)

    # 2) 输出张量形状 —— 用**跟踪到的通道轴**（TRANSPOSE/RESHAPE 后通道轴会移动，
    #    最后一维假设在这里是错的）
    for ti, new_c in new_ch.items():
        t = T[ti]
        ax = tr.info.get(ti, (None,))[0]
        if ax is None:
            ax = ch_axis(t)
        if ax >= len(t["shape"]):
            sys.stderr.write("ERROR: 张量 %d 的通道轴 %d 越界（形状 %s）\n"
                             % (ti, ax, t["shape"]))
            return 1
        t["shape"] = list(t["shape"])
        t["shape"][ax] = new_c

    # 2b) 常量改写（RESHAPE 的 shape、STRIDED_SLICE 的 begin/end）
    for ti, old_vals, new_vals, why in tr.rewrites:
        try:
            set_const_i32(mdl, ti, new_vals)
        except SystemExit as e:
            sys.stderr.write("ERROR: %s 改写失败: %s\n" % (why, e))
            return 1
        print("  [常量改写] %-28s tensor %-4d %s → %s"
              % (why, ti, old_vals, new_vals))

    # 3) 写模型
    mdl["description"] = (mdl.get("description") or "") + " [pad-to-8]"
    blob = fold.build_model(mdl)
    with open(args.dst, "wb") as f:
        f.write(blob)

    # 4) 复核：重新读回，检查权重字节数与形状自洽
    m2 = fold.read_model(args.dst)
    bad = []
    for i in sorted(set(list(pad_ops) + list(in_pad))):
        op = m2["ops"][i]
        wt = m2["tensors"][op["inputs"][1]]
        bt = m2["tensors"][op["inputs"][2]]
        n = 1
        for x in wt["shape"]:
            n *= x
        wb = m2["buffers"][wt["buffer"]]["data"]
        if len(wb) != n * TYPE_SIZE.get(wt["type"], 1):
            bad.append("op#%d 权重字节 %d ≠ 期望 %d" % (i, len(wb), n * TYPE_SIZE.get(wt["type"], 1)))
        nb = 1
        for x in bt["shape"]:
            nb *= x
        bbuf = m2["buffers"][bt["buffer"]]["data"]
        if len(bbuf) != nb * TYPE_SIZE.get(bt["type"], 4):
            bad.append("op#%d 偏置字节 %d ≠ 期望 %d" % (i, len(bbuf), nb * TYPE_SIZE.get(bt["type"], 4)))
    # ★ 不变量：per-axis 量化数组的长度必须是 1（per-tensor）或该张量**量化维**上的大小。
    #   实测漏检过：偏置的 scale 还是 244 而通道已补到 248 ⇒ TFLite 拒绝加载
    #   （"num_scales must be 1 ... or 248 ... but got 244"，Tensor 7 = bias）。
    for ti, t in enumerate(m2["tensors"]):
        qq = t.get("quant") or {}
        sc = qq.get("scale") or []
        if len(sc) <= 1:
            continue
        shp = list(t["shape"] or [])
        if not shp:
            continue
        qdim = qq.get("qdim", 0)
        if qdim is None or qdim >= len(shp):
            qdim = len(shp) - 1
        if len(sc) != shp[qdim]:
            bad.append("张量 %d 的 per-axis scale 长度 %d ≠ 量化维 %d 的大小 %d"
                       % (ti, len(sc), qdim, shp[qdim]))
    if bad:
        sys.stderr.write("ERROR: 写回复核失败:\n  " + "\n  ".join(bad) + "\n")
        return 1

    # ★ 不变量（形状守恒）：RESHAPE 的输入元素数必须等于输出元素数；
    #   CONCAT 的输入元素数之和必须等于输出元素数。
    #   ⇒ 这条能抓住「SE 块常量被改坏」这类不自洽（工具曾把 121 悄悄修回 128）。
    def _prod(shape):
        n = 1
        for x in shape:
            n *= x
        return n
    for i, op in enumerate(m2["ops"]):
        b2 = builtin(m2, op)
        if b2 == 22 and len(op["inputs"]) > 1:        # RESHAPE
            ain = _prod(m2["tensors"][op["inputs"][0]]["shape"])
            aout = _prod(m2["tensors"][op["outputs"][0]]["shape"])
            if ain != aout:
                sys.stderr.write(
                    "ERROR: op#%d RESHAPE 元素数不自洽：输入 %d ≠ 输出 %d\n"
                    "       （SE 块常量与张量形状对不上 —— 拒绝产出半改的模型）\n"
                    % (i, ain, aout))
                return 1
        elif b2 == 2 and len(op["inputs"]) > 1:      # CONCATENATION
            ain = sum(_prod(m2["tensors"][t]["shape"]) for t in op["inputs"])
            aout = _prod(m2["tensors"][op["outputs"][0]]["shape"])
            if ain != aout:
                sys.stderr.write(
                    "ERROR: op#%d CONCAT 元素数不自洽：输入和 %d ≠ 输出 %d\n"
                    % (i, ain, aout))
                return 1

    # ★ 不变量：每个 conv 的「权重 in 维」必须等于「输入张量在跟踪轴上的通道数」。
    #   （自测曾抓到「权重补到 248 而图输入仍是 244」⇒ 这个检查必须有。）
    tr2 = build_track(m2, m2["ops"], m2["tensors"])
    for i, op in enumerate(m2["ops"]):
        if not is_conv_like(m2, op):
            continue
        wsh = list(m2["tensors"][op["inputs"][1]]["shape"])
        in_w = wsh[3] if len(wsh) == 4 else (wsh[1] if len(wsh) == 2 else None)
        info = tr2.info.get(op["inputs"][0])
        if in_w is None or info is None or info[0] is None:
            continue
        ax, cnt, _ = info
        ishp = m2["tensors"][op["inputs"][0]]["shape"]
        if ax >= len(ishp):
            continue
        actual = ishp[ax]
        if in_w != actual:
            sys.stderr.write("ERROR: op#%d 权重 in 维 %d != 输入张量(轴 %d)通道数 %d\n"
                             % (i, in_w, ax, actual))
            return 1

    print()
    print("已写出: %s (%d B)" % (args.dst, len(blob)))
    print("写回复核：权重/偏置字节数与形状自洽 ✔")
    print("⚠️ 语义等价性是**推演**：补出的通道写 zp/0 ⇒ 贡献 0；")
    print("   但必须用四级判据第 ④ 级（与 CPU 参考逐字节一致）实测确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
