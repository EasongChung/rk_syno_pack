# -*- coding: utf-8 -*-
"""
主机端索引模拟器：复刻 Mesa etnaviv_ml_nn_v7.c 的 write_core_6 通道/权重/bias/输出偏移
索引与系数流落位，用于离线判定「幻影通道」与候选判据。

对应源码:
  v7:193 kernels_per_core = DRoU(out_c, cores_used)        cores_used = MIN2(out_c, ncc)
  v7:198 superblocks = calc_superblocks(...)
  v7:220-222 kernels_in_superblock = d ; 末批 = rem
  v7:225/231 out_channel = core*kis + kernel + sb*d*cores_used
  v7:226 weights_maps[kernel] = input + out_channel*W*H*in_c
  v7:238 biases[out_channel]
  v7:251 append_bits(out_values_per_channel*out_channel, 32)
  v7:254 路径选择 pointwise&&out_c>8 -> write_core_6
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

NCC, IBD, ABD = 8, 12, 64
OW = OH = 50


def align(x, a):
    return (x + a - 1) & ~(a - 1)


def dru(a, b):
    return (a + b - 1) // b


def calc_superblocks(C, w, tile_y, im):
    """v7:19-39 逐行"""
    kpc0 = dru(C, NCC)                              # v7:25
    foo = (ABD * im) // tile_y                      # v7:26
    if w == 1:
        foo = min(foo, ABD // 3)                    # v7:28-29
    foo = min(foo, kpc0)                            # v7:31
    foo = min(foo, 127)                             # v7:32
    kpc = dru(C, NCC * foo)                         # v7:34
    nk = dru(C, kpc * NCC)                          # v7:35
    sb = dru(dru(C, NCC), nk)                       # v7:36
    return sb, foo, kpc, nk, kpc0


def sim(C, w=1, h=1, in_c=None, pointwise=True):
    """返回 write_core_6 的完整索引表（单核视角 + 全核汇总）"""
    in_c = C if in_c is None else in_c
    tw = min(OW, 64)
    im = 1
    th = max(min(IBD * im - h + 1, im * ABD, OH), 1)
    sb, foo, kpc, nk, kpc0 = calc_superblocks(C, w, th, im)
    d = dru(kpc0, sb)                               # v7:220
    rem = kpc0 - d * (sb - 1)                       # v7:222
    stride = min(in_c, 6)                           # v7:197
    nblocks = dru(in_c, stride)                     # v7:229

    # 每个 (core, kernel) 的 out_channel（v7:225）
    table = []
    for core in range(NCC):
        for sbx in range(sb):
            kis = d if sbx < sb - 1 else rem
            for k in range(kis):
                oc = core * kis + k + sbx * d * NCC
                table.append((core, sbx, k, kis, oc))

    valid = [t for t in table if t[4] < C]
    phantom = [t for t in table if t[4] >= C]

    # 每核被幻影污染的程度（末批）
    phantom_cores = set()
    fully_phantom_cores = set()
    for core in range(NCC):
        rows = [t for t in table if t[0] == core and t[1] == sb - 1]
        bad = [t for t in rows if t[4] >= C]
        if bad:
            phantom_cores.add(core)
        if rows and len(bad) == len(rows):
            fully_phantom_cores.add(core)

    ovpc = OW * OH                                   # v7:196
    w_tensor = C * in_c * w * h                      # 权重张量字节
    b_tensor = C * 4                                 # bias 张量字节
    o_tensor = C * ovpc                              # 输出图像字节

    oob = []
    if phantom:
        ocs = sorted(set(t[4] for t in phantom))
        oob.append(("weights", min(ocs) * in_c, max(ocs) * in_c + in_c - 1, w_tensor))
        oob.append(("biases", min(ocs) * 4, max(ocs) * 4 + 3, b_tensor))
        oob.append(("output", min(ocs) * ovpc, max(ocs) * ovpc + ovpc - 1, o_tensor))

    # 系数流每核字节（v7:215-262）
    bits = 8 + 16
    for _ in range(kpc0):
        bits += 8 * in_c + 32 + 32
    per_core_actual = align((bits + 7) // 8, 4)
    per_core_slot = align(3 + kpc0 * (in_c + 8), 64)   # v7:454-455 分配槽
    total_slot = align(NCC * 4, 64) + per_core_slot * NCC  # v7:443/456

    return dict(C=C, in_c=in_c, sb=sb, foo=foo, kpc=kpc, nk=nk, kpc0=kpc0, d=d, rem=rem,
                nblocks=nblocks, stride=stride,
                phantom_n=len(set(t[4] for t in phantom)),
                phantom_ocs=sorted(set(t[4] for t in phantom)),
                phantom_cores=sorted(phantom_cores),
                fully_phantom_cores=sorted(fully_phantom_cores),
                oob=oob, per_core_actual=per_core_actual, per_core_slot=per_core_slot,
                coef_total=total_slot, w_tensor=w_tensor, o_tensor=o_tensor,
                misalign_per_core=NCC * (d - rem))


LABELS = {122: 0, 196: 0, 220: 0, 240: 0, 244: 1, 260: 0, 280: 0,
          248: 0, 252: 0, 284: 1}


def features(s):
    """候选判据族：每个返回 (名称, 值)"""
    f = {}
    f["rem==1 & m>0"] = int(s["rem"] == 1 and s["phantom_n"] > 0)
    f["m>0 (C%8!=0)"] = int(s["phantom_n"] > 0)
    f["m>=3"] = int(s["phantom_n"] >= 3)
    f["phantom_cores>=3"] = int(len(s["phantom_cores"]) >= 3)
    f["phantom_cores>=4"] = int(len(s["phantom_cores"]) >= 4)
    f["fully_phantom_cores>=3"] = int(len(s["fully_phantom_cores"]) >= 3)
    f["fully_phantom_cores>=4"] = int(len(s["fully_phantom_cores"]) >= 4)
    f["fully_phantom_cores>=2"] = int(len(s["fully_phantom_cores"]) >= 2)
    f["rem==1"] = int(s["rem"] == 1)
    f["misalign>=32"] = int(s["misalign_per_core"] >= 32)
    f["misalign>=24"] = int(s["misalign_per_core"] >= 24)
    f["C%8==4"] = int(s["C"] % 8 == 4)
    f["C%4==0"] = int(s["C"] % 4 == 0)
    f["kpc0%2==1"] = int(s["kpc0"] % 2 == 1)
    f["C==244"] = int(s["C"] == 244)
    f["sb>=7 & rem==1"] = int(s["sb"] >= 7 and s["rem"] == 1)
    f["m>0 & fully_phantom>=2"] = int(s["phantom_n"] > 0 and len(s["fully_phantom_cores"]) >= 2)
    f["m>0 & misalign>=16"] = int(s["phantom_n"] > 0 and s["misalign_per_core"] >= 16)
    f["m>0 & misalign>=32"] = int(s["phantom_n"] > 0 and s["misalign_per_core"] >= 32)
    return f


def main():
    print("=" * 100)
    print("(1) 索引模拟器复刻结果（几何 50x50，1x1 conv，int8）")
    print("=" * 100)
    hdr = ("C", "kpc0", "sb", "d", "rem", "m(幻影)", "幻影通道", "受影响核", "全幻影核",
           "错位8(d-rem)", "权重越界B", "输出越界B", "coef/core槽", "标签")
    print("%-6s%-6s%-4s%-4s%-5s%-8s%-14s%-12s%-10s%-12s%-11s%-11s%-12s%s"
          % hdr)
    sims = {}
    for C in sorted(LABELS):
        s = sim(C)
        sims[C] = s
        wl = [o for o in s["oob"] if o[0] == "weights"]
        ol = [o for o in s["oob"] if o[0] == "output"]
        print("%-6d%-6d%-4d%-4d%-5d%-8d%-14s%-12s%-10d%-12d%-11s%-11s%-12d%s"
              % (C, s["kpc0"], s["sb"], s["d"], s["rem"], s["phantom_n"],
                 str(s["phantom_ocs"]) if s["phantom_ocs"] else "-",
                 str(s["phantom_cores"]) if s["phantom_cores"] else "-",
                 len(s["fully_phantom_cores"]), s["misalign_per_core"],
                 ("+%d" % (wl[0][2] - wl[0][3] + 1)) if wl else "0",
                 ("+%d" % (ol[0][2] - ol[0][3] + 1)) if ol else "0",
                 s["per_core_slot"],
                 "挂" if LABELS[C] else "-"))

    print()
    print("=" * 100)
    print("(2) 候选判据族对 10 个实测点的拟合（须 10/10 才是候选）")
    print("=" * 100)
    names = list(features(sims[122]).keys())
    fit = []
    for nm in names:
        ok = all(features(sims[C])[nm] == LABELS[C] for C in LABELS)
        pred = {C: features(sims[C])[nm] for C in sorted(LABELS)}
        fit.append((ok, nm, pred))
    fit.sort(key=lambda t: (not t[0], t[1]))
    for ok, nm, pred in fit:
        hits = "".join("X" if pred[C] == LABELS[C] else "." for C in sorted(LABELS))
        print("  %-28s %s  %s" % (nm, "10/10 FIT" if ok else "        ", hits))
    print("  （X=命中 .=错） 点序: %s" % sorted(LABELS))

    print()
    print("=" * 100)
    print("(3) 判决点：找出能把「10/10 拟合的判据」互相分开的最小 C 集合")
    print("=" * 100)
    fitted = [nm for ok, nm, _ in fit if ok]
    print("  10/10 的判据: %s" % fitted)
    dis = []
    for C in range(101, 401):
        s = sim(C)
        vals = {nm: features(s)[nm] for nm in fitted}
        if len(set(vals.values())) > 1:
            dis.append((C, s, vals))
    print("  能分开这些判据的 C（共 %d 个）:" % len(dis))
    shown = 0
    for C, s, vals in dis:
        groups = {}
        for nm, v in vals.items():
            groups.setdefault(v, []).append(nm)
        if len(groups) > 1 and shown < 24:
            print("    C=%-4d kpc0=%-3d sb=%-2d d=%d rem=%-2d m=%-2d 受影响核=%-2d 全幻影核=%-2d  预测: %s"
                  % (C, s["kpc0"], s["sb"], s["d"], s["rem"], s["phantom_n"],
                     len(s["phantom_cores"]), len(s["fully_phantom_cores"]),
                     "; ".join("挂组{%s}" % ",".join(g) for g in groups.values() if g)))
            shown += 1

    print()
    print("=" * 100)
    print("(4) 离线风险规则：C%%8!=0 的 pointwise conv 即「幻影通道」风险点")
    print("=" * 100)
    risky = [C for C in range(101, 401) if C % 8 != 0]
    print("  1x1 conv 且 out_c%%8!=0 的 C 个数(101..400): %d / 300" % len(risky))
    print("  其中 rem==1（按现有经验律会挂）的 C: %s"
          % [C for C in risky if sim(C)["rem"] == 1][:40])
    print()
    print("  ★ 规则 R（我方主张）：m = 8*ceil(C/8) - C = (8 - C%%8)%%8  ⇒ C%%8!=0 即有幻影通道")
    print("  ★ C%%8==0 的 C（0 幻影，该机制不可能触发）: %s ..."
          % [C for C in range(101, 401) if C % 8 == 0][:20])

    print()
    print("=" * 100)
    print("(5) 系数流字节账（v7:215-262 实际写 vs v7:454 分配槽）")
    print("=" * 100)
    for C in sorted(LABELS):
        s = sim(C)
        print("  C=%-4d 实际每核 %5d B (align4)  分配槽 %5d B (align64)  差 %3d B  BO=%d B  实测coef_cache=%s"
              % (C, s["per_core_actual"], s["per_core_slot"],
                 s["per_core_slot"] - s["per_core_actual"], s["coef_total"],
                 {122: 16896, 196: 40960, 220: 51200, 240: 59904, 244: 62976,
                  260: 71168, 280: 80896}.get(C, "-")))


if __name__ == "__main__":
    main()
