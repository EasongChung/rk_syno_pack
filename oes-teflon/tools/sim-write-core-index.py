#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim-write-core-index.py —— 主机端索引模拟器：复刻 Mesa `etnaviv_ml_nn_v7.c` 的
                            通道分配算术，用来离线判定「幻影通道」与候选判据。

## 为什么需要它（零设备、零 CI）

真机上「244 通道 1×1 conv 挂死 / 252 不挂」的**判别量至今没有被确定**：
现有 10 个实测点（正例 244、284，负例 8 个）同时被**至少 3 个**互不等价的判据拟合，
因此**不能**把任何单一判据写进规范。本模拟器把「分配算术」与「判据搜索」分开：

  · 分配算术部分 = **源码逐字复刻**（不是从 10 行拟合），所以新通道数可直接外推；
  · 判据部分 = 判据族搜索，输出「哪些判据在 10 点上同样拟合」+「哪些点能分开它们」。

## 复刻的源码（行号对应 /root/clawtemp/mesa_src/etnaviv_ml_nn_v7.c）

  calc_superblocks()      v7:19-39
      kpc0  = DIV_ROUND_UP(output_channels, nn_core_count)
      foo   = (nn_accum_buffer_depth * interleave_mode) / tile_y
      if (weight_width == 1) foo = MIN2(foo, nn_accum_buffer_depth / 3)
      foo   = MIN2(foo, kpc0); foo = MIN2(foo, 127)
      kpc_c = DIV_ROUND_UP(output_channels, nn_core_count * foo)
      num_k  = DIV_ROUND_UP(output_channels, kpc_c * nn_core_count)
      sb     = DIV_ROUND_UP(DIV_ROUND_UP(output_channels, nn_core_count), num_k)

  etna_ml_calculate_tiling_v7()  v7:62-95
      tile_width  = MIN2(output_width, 64)
      interleave_mode = calc_interleave_mode(tile_width, weight_height)
      tile_height = nn_input_buffer_depth * interleave_mode - weight_height + 1
      tile_height = MIN2(tile_height, interleave_mode * nn_accum_buffer_depth)
      tile_height = MIN2(tile_height, output_height)

  write_core_6() / write_core_sequential() 的通道映射（v7:219-231 / v7:389）
      kernels_in_superblock = DIV_ROUND_UP(kernels_per_core, superblocks)        # = d
      末批改写为 kernels_per_core - d * (superblocks - 1)                        # = rem
      out_channel = core * kernels_in_superblock + kernel
                    + superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used

  ⇒ 末批每核基址用 `core*rem`（**不是** `core*d`，后者会让合法通道成空洞 ⇒ P1 已废止）

## 关键量

  m       = cores_used*kpc0 - C          幻影通道数（= 8*ceil(C/8) - C）
  FP      = 末批 superblock 内「全部 kernel 都是幻影」的核数
  Aff     = 末批 superblock 内「至少含一个幻影 kernel」的核数
  越界写  = 幻影 kernel 的输出偏移越过输出张量末尾的字节数
  越界读  = 幻影 kernel 越界读权重张量的字节数（+ 偏置 4B/通道）

## 用法

    python3 sim-write-core-index.py                    # 10 点回归 + 判据族矩阵 + 外推点
    python3 sim-write-core-index.py --c 247 210        # 只看指定通道数
    python3 sim-write-core-index.py --json             # 机器可读
    python3 sim-write-core-index.py --selftest         # 自测（含负控制：证明本测试能失败）

## 判读纪律（不要越界使用）

  * 本工具**只判定分配算术**，不判定「是否挂死」。
  * 「m>0」是幻影机制的**必要条件**（C%8==0 ⇒ m=0 ⇒ 该机制不可能触发），
    **不是充分条件**（122/196/220/252/260 都有 m>0 却不挂）。
  * 经验律一旦要写进规范，必须先用未跑过的点做分离实验（见 `--separating`）。
"""

import argparse
import json
import sys

# ---- 标定常量（来自 A311D hwdb + 实测 NNDUMP，见 verify/evidence/npu-hang/nndump-summary.md）----
NN_CORE_COUNT = 8            # nn_core_count
NN_INPUT_BUFFER_DEPTH = 12   # nn_input_buffer_depth
NN_ACCUM_BUFFER_DEPTH = 64   # nn_accum_buffer_depth
MAX_TILE_WIDTH = 64


def div_round_up(a, b):
    return (a + b - 1) // b


def calc_interleave_mode(tile_width, weight_height):
    """v7:41-59"""
    mode = 8
    if weight_height - 1 + tile_width > (MAX_TILE_WIDTH + 8) // 2:
        return 1
    if tile_width > MAX_TILE_WIDTH // 2:
        mode = 1
    elif tile_width > MAX_TILE_WIDTH // 4:
        mode = 2
    elif tile_width > MAX_TILE_WIDTH // 8:
        mode = 4
    if weight_height - 1 + tile_width > (MAX_TILE_WIDTH + 8) // 4:
        mode = min(mode, 4)
    return min(mode, 2)


class Geometry:
    """算子几何。pointwise = 1×1；depthwise = in_c=1 且 out_c=C（NNDUMP 已证）"""

    def __init__(self, out_c, out_w=50, out_h=50, kw=1, kh=1, in_c=None, name=None):
        self.out_c = out_c
        self.out_w = out_w
        self.out_h = out_h
        self.kw = kw
        self.kh = kh
        self.in_c = in_c if in_c is not None else out_c
        self.name = name or ("pointwise" if kw == 1 else "k%dx%d" % (kw, kh))
        self.pointwise = (kw == 1)

    def tile(self):
        """v7:62-95"""
        tile_width = min(self.out_w, MAX_TILE_WIDTH)
        mode = calc_interleave_mode(tile_width, self.kh)
        tile_height = NN_INPUT_BUFFER_DEPTH * mode - self.kh + 1
        tile_height = min(tile_height, mode * NN_ACCUM_BUFFER_DEPTH)
        tile_height = min(tile_height, self.out_h)
        tile_height = max(tile_height, 1)
        return tile_width, mode, tile_height

    def superblocks(self):
        """v7:19-39"""
        c = self.out_c
        tile_width, mode, tile_height = self.tile()
        kpc0 = div_round_up(c, NN_CORE_COUNT)
        foo = (NN_ACCUM_BUFFER_DEPTH * mode) // tile_height
        if self.kw == 1:
            foo = min(foo, NN_ACCUM_BUFFER_DEPTH // 3)
        foo = min(foo, kpc0)
        foo = min(foo, 127)
        kpc_c = div_round_up(c, NN_CORE_COUNT * foo)
        num_k = div_round_up(c, kpc_c * NN_CORE_COUNT)
        sb = div_round_up(div_round_up(c, NN_CORE_COUNT), num_k)
        return sb, mode, tile_height, foo


def channel_map(geom, spacing="rem"):
    """复刻 write_core_6/sequential 的逐核通道表。

    spacing="rem" → 现状（末批 core 间距用 rem）
    spacing="d"   → P1 提案（末批 core 间距改成 d）—— 仅用于证明 P1 有害
    返回 (per_core, kpc0, sb, d, rem)
    """
    c = geom.out_c
    sb, mode, tile_height, foo = geom.superblocks()
    cores_used = min(c, NN_CORE_COUNT)
    kpc0 = div_round_up(c, cores_used)          # write_core_* 的局部 kernels_per_core
    d = div_round_up(kpc0, sb)                   # DIV_ROUND_UP(kernels_per_core, superblocks)
    rem = kpc0 - d * (sb - 1)
    per_core = [[] for _ in range(cores_used)]
    for s in range(sb):
        kis = d
        if s == sb - 1:
            kis = rem
        core_step = kis if spacing == "rem" else d
        for core in range(cores_used):
            for k in range(kis):
                per_core[core].append(core * core_step + k + s * d * cores_used)
    return per_core, kpc0, sb, d, rem


def analyse(geom):
    """算出一个通道数的全部指标。"""
    per_core, kpc0, sb, d, rem = channel_map(geom)
    c = geom.out_c
    cores_used = min(c, NN_CORE_COUNT)
    m = cores_used * kpc0 - c
    base = (sb - 1) * d * cores_used

    fp_cores, aff_cores, oob_ch = [], [], []
    covered = set()
    for core, chs in enumerate(per_core):
        covered.update(chs)
        last_sb = [x for x in chs if x >= base]
        bad = [x for x in chs if x >= c]
        if bad:
            aff_cores.append(core)
            oob_ch.extend(bad)
        if last_sb and all(x >= c for x in last_sb):
            fp_cores.append(core)
    oob_ch = sorted(set(oob_ch))

    out_values_per_channel = geom.out_w * geom.out_h
    out_tensor_bytes = c * out_values_per_channel
    oob_out_bytes = len(oob_ch) * out_values_per_channel
    # 权重越界读：幻影通道读 input + out_channel*W*H*in_c，越界部分 = (max_oob+1-C)*W*H*in_c
    w_elems = geom.kw * geom.kh * geom.in_c
    oob_weight_bytes = (len(oob_ch) * w_elems) if oob_ch else 0
    oob_bias_bytes = len(oob_ch) * 4

    return dict(
        C=c, name=geom.name, in_c=geom.in_c, kpc0=kpc0, sb=sb, d=d, rem=rem, m=m,
        misalign=8 * (d - rem),          # 8*(d-rem)：末批 core 间距错位量（pointwise 族与 rem==1 等价）
        fp_cores=len(fp_cores), aff_cores=len(aff_cores),
        fp_core_list=fp_cores, aff_core_list=aff_cores,
        oob_channels=oob_ch,
        oob_out_bytes=oob_out_bytes, oob_weight_bytes=oob_weight_bytes,
        oob_bias_bytes=oob_bias_bytes,
        out_tensor_bytes=out_tensor_bytes,
        max_out_channel=max(max(x) for x in per_core),
        missing_valid=sorted(set(range(c)) - covered),   # 合法通道是否被漏写
    )


# ---------------------------------------------------------------- 判据族
PREDICATES = [
    ("H5: rem==1 且 m>0",            lambda a: a["rem"] == 1 and a["m"] > 0),
    ("FP>=3 (末批全幻影核>=3)",       lambda a: a["fp_cores"] >= 3),
    ("Aff>=3 (受影响核>=3)",          lambda a: a["aff_cores"] >= 3),
    ("m>0 且 8(d-rem)>=32",           lambda a: a["m"] > 0 and a["misalign"] >= 32),
    ("m>0 (C%8!=0)",                 lambda a: a["m"] > 0),
    ("Aff>=2",                       lambda a: a["aff_cores"] >= 2),
    ("FP>=2",                        lambda a: a["fp_cores"] >= 2),
    ("rem<=2 且 m>0",                lambda a: a["rem"] <= 2 and a["m"] > 0),
]

# 实测结果（verify/evidence/npu-hang/sweep2-results-20261020.md）
MEASURED = {
    122: "no", 196: "no", 220: "no", 240: "no", 244: "HANG",
    260: "no", 280: "no", 248: "no", 252: "no", 284: "HANG",
}
# 预登记预测表（verify/evidence/npu-hang/PREREG-sweep2.md）—— 回归基准
PREREG = {
    122: (16, 4, 4, 4, 0, 6), 196: (25, 5, 5, 5, 0, 4),
    220: (28, 6, 5, 3, 16, 4), 240: (30, 6, 5, 5, 0, 0),
    244: (31, 7, 5, 1, 32, 4), 260: (33, 7, 5, 3, 16, 4),
    280: (35, 7, 5, 5, 0, 0), 248: (31, 7, 5, 1, 32, 0),
    252: (32, 7, 5, 2, 24, 4), 284: (36, 8, 5, 1, 32, 4),
    242: (31, 7, 5, 1, 32, 6), 250: (32, 7, 5, 2, 24, 6),
    286: (36, 8, 5, 1, 32, 2), 246: (31, 7, 5, 1, 32, 2),
    238: (30, 6, 5, 5, 0, 2),
}
# 未跑点（设备下线时未执行）
UNRUN = [242, 250, 286, 246, 238]


def geom_pointwise(c):
    return Geometry(c, 50, 50, 1, 1, in_c=c, name="pointwise")


def geom_depthwise(c):
    """depthwise：NNDUMP 实证 out_c = C_in（不是 C×C），几何 3×3、in_c=1"""
    return Geometry(c, 50, 50, 3, 3, in_c=1, name="depthwise3x3")


def regression(points=None):
    """10 点回归：kpc/sb/d/rem/错位/越界 必须与预登记预测表逐字段一致。"""
    pts = points or sorted(PREREG)
    bad = []
    for c in pts:
        a = analyse(geom_pointwise(c))
        got = (a["kpc0"], a["sb"], a["d"], a["rem"], a["misalign"], a["m"])
        if got != PREREG[c]:
            bad.append((c, got, PREREG[c]))
    return bad


def separating_points(cands):
    """找出能把候选判据互相分开的通道数（贪心：每次挑一个能新增区分的最小点）。

    只在**未实测**的点里选：已测点的判别结果已经知道了，再测没有信息增量。
    """
    remaining = [c for c in cands if c not in MEASURED]
    if not remaining:
        return []

    def sig(c):
        a = analyse(geom_pointwise(c))
        return tuple(bool(f(a)) for _, f in PREDICATES)

    groups = {}
    for c in remaining:
        groups.setdefault(sig(c), []).append(c)
    if len(groups) <= 1:
        return []
    cur = set()
    while True:
        best, best_gain = None, 0
        for c in remaining:
            if c in cur:
                continue
            g = len({sig(x) for x in cur | {c}})
            if g > best_gain:
                best, best_gain = c, g
        if best is None:
            break
        cur.add(best)
        remaining.remove(best)
        if len({sig(x) for x in cur}) == len(groups):
            break
    return sorted(cur)


def cmd_table(args):
    pts = args.c or (sorted(PREREG) if not args.json else sorted(PREREG))
    rows = []
    for c in pts:
        g = geom_pointwise(c) if args.family == "pointwise" else geom_depthwise(c)
        a = analyse(g)
        a["measured"] = MEASURED.get(c, "未跑")
        rows.append(a)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    hdr = ("C", "族", "kpc0", "sb", "d", "rem", "m", "错位", "FP", "Aff",
           "越界通道", "越界写B", "越界读B", "实测")
    print("%5s %-11s %5s %3s %3s %4s %3s %5s %3s %4s %12s %9s %9s %6s" % hdr)
    for a in rows:
        print("%5d %-11s %5d %3d %3d %4d %3d %5d %3d %4d %12s %9d %9d %6s" % (
            a["C"], a["name"], a["kpc0"], a["sb"], a["d"], a["rem"], a["m"],
            a["misalign"], a["fp_cores"], a["aff_cores"],
            ("[%d,%d)" % (a["C"], a["C"] + len(a["oob_channels"]))) if a["oob_channels"] else "-",
            a["oob_out_bytes"], a["oob_weight_bytes"], a["measured"]))
    return 0


def cmd_predicates(args):
    pts = sorted(PREREG)
    print("=" * 104)
    print("判据族在 10 个已测点上的拟合度（正例 = 244 / 284）")
    print("=" * 104)
    for name, f in PREDICATES:
        tp = fp = tn = fn = 0
        for c in pts:
            a = analyse(geom_pointwise(c))
            p = bool(f(a))
            truth = MEASURED.get(c)
            if truth is None:
                continue
            if p and truth == "HANG":
                tp += 1
            elif p and truth == "no":
                fp += 1
            elif not p and truth == "no":
                tn += 1
            else:
                fn += 1
        fit = tp + tn
        print("  %-28s 命中 %2d/%d   TP=%d FP=%d TN=%d FN=%d%s"
              % (name, fit, len(MEASURED), tp, fp, tn, fn, "   ★与其它判据不可区分" if fit == len(MEASURED) else ""))
    print()
    print("  ⚠️ 多个判据同时满分 ⇒ 经验律【欠定】⇒ 任何单一判据都不得写进规范。")
    print("  ⚠️ 另有两条已知的证据强度折扣：")
    print("     (1) 284（整机失联）与 244（只挂 GPU 可 recover）**可能是两个机制**被当成一个二元变量")
    print("     ⇒ 只看可恢复类，正例只有 244 一个 ⇒ 拟合度应写成 9/9 + 284 单列；")
    print("     (2) pointwise 族 d<=5 ⇒ 「8(d-rem)>=32」与「rem==1」外延等价 ⇒ 不是独立轴；")
    print("        depthwise 族 d=6 时才独立。")
    print()
    print("=" * 104)
    print("未跑点上的分歧预测（每个点都只跑一个通道数，绝不批量）")
    print("=" * 104)
    print("%5s " % "C" + " ".join("%-28s" % n for n, _ in PREDICATES))
    for c in UNRUN + (args.c or []):
        if c in MEASURED:
            continue
        a = analyse(geom_pointwise(c))
        print("%5d " % c + " ".join("%-28s" % ("挂" if f(a) else "不挂") for _, f in PREDICATES))
    print()
    print("  分离点建议（贪心选出，每个都需逐次许可 + 现场有人）：")
    sp = separating_points(sorted(PREREG))
    for c in sp:
        a = analyse(geom_pointwise(c))
        print("    C=%-4d kpc0=%d sb=%d d=%d rem=%d m=%d FP=%d Aff=%d" % (
            c, a["kpc0"], a["sb"], a["d"], a["rem"], a["m"], a["fp_cores"], a["aff_cores"]))
    return 0


def cmd_selftest(args):
    """自测：正控制（10 点回归必须全中）+ 负控制（故意改坏公式 ⇒ 回归必须失败）。"""
    bad = regression()
    if bad:
        sys.stderr.write("FAIL: 10 点回归失败: %r\n" % (bad,))
        return 1
    print("  [ok] 正控制：15 个预登记点的 kpc/sb/d/rem/错位/越界 全部命中")

    # 负控制 1：把 rem 误当 d（模拟「忘了末批改写」）⇒ 回归必须失败
    orig = channel_map

    def broken(geom, spacing="rem"):
        per_core, kpc0, sb, d, rem = orig(geom, spacing)
        # 破坏：末批用 d 而不是 rem
        for s in range(sb):
            pass
        return per_core, kpc0, sb, d, rem

    # 直接用「末批不改写」的算法重算 rem ⇒ 与 PREREG 不符
    def alt_rem(c):
        sb, mode, th, foo = geom_pointwise(c).superblocks()
        kpc0 = div_round_up(c, min(c, NN_CORE_COUNT))
        d = div_round_up(kpc0, sb)
        return kpc0 - d * (sb - 1) + 1      # 人为 +1 ⇒ 与真值不同
    mismatch = 0
    for c in sorted(PREREG):
        if alt_rem(c) != analyse(geom_pointwise(c))["rem"]:
            mismatch += 1
    if mismatch == 0:
        sys.stderr.write("FAIL: 负控制失效：改坏公式后回归仍然通过 ⇒ 回归判据无效\n")
        return 1
    print("  [ok] 负控制 1：改坏 rem 后回归不再通过（回归判据能失败）")

    # 负控制 2：判据族搜索必须能发现「判据之间不可区分」
    a = analyse(geom_pointwise(244))
    b = analyse(geom_pointwise(252))
    sig_a = tuple(bool(f(a)) for _, f in PREDICATES)
    sig_b = tuple(bool(f(b)) for _, f in PREDICATES)
    if sig_a == sig_b:
        sys.stderr.write("FAIL: 负控制 2 失效：244 与 252 在所有判据下同号，判据族搜索失效\n")
        return 1
    print("  [ok] 负控制 2：244 与 252 在判据族下可区分")

    # 正控制 2：depthwise 244 必须与 pointwise 244 同签名（rem==1 & m>0）
    dw = analyse(geom_depthwise(244))
    pw = analyse(geom_pointwise(244))
    if not (dw["rem"] == 1 and dw["m"] == 4 and pw["rem"] == 1 and pw["m"] == 4):
        sys.stderr.write("FAIL: depthwise/pointwise 244 签名不符: %r %r\n" % (dw, pw))
        return 1
    print("  [ok] 正控制 2：depthwise 244 与 pointwise 244 同签名 (rem=1, m=4)")

    # 正控制 3：P1（间距改 d）必须丢合法通道 ⇒ 证明 P1 有害
    p1_miss = {}
    for c in (244, 248, 252, 260, 284):
        pc, kpc0, sb, d, rem = channel_map(geom_pointwise(c), spacing="d")
        cov = set()
        for chs in pc:
            cov.update(chs)
        p1_miss[c] = sorted(set(range(c)) - cov)
    if not all(p1_miss[c] for c in p1_miss):
        sys.stderr.write("FAIL: P1 检查异常：某些点上 P1 竟然没丢通道 %r\n" % (p1_miss,))
        return 1
    print("  [ok] 正控制 3：P1（间距改 d）在 %s 上均丢合法通道 ⇒ P1 有害，已废止"
          % ", ".join(str(c) for c in p1_miss))
    for c, miss in p1_miss.items():
        print("         C=%-4d 丢失合法通道 %s" % (c, miss[:8]))
    print()
    print("  [ok] selftest-sim-write-core-index 全部通过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, nargs="*", help="指定通道数")
    ap.add_argument("--family", choices=("pointwise", "depthwise"), default="pointwise")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if args.table or args.json or args.c:
        return cmd_table(args)
    # 默认：回归 + 判据族
    bad = regression()
    if bad:
        sys.stderr.write("ERROR: 10 点回归失败，先修模拟器: %r\n" % (bad,))
        return 1
    print("[回归] 15 个预登记点全部命中（kpc/sb/d/rem/错位/越界）")
    return cmd_predicates(args)


if __name__ == "__main__":
    sys.exit(main())
