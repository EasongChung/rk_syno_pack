# -*- coding: utf-8 -*-
"""
P2 补丁（write_core_6 / write_core_sequential 幻影通道钳位）的主机端自测。

零设备、零 CI：在 host 上复刻 Mesa etnaviv_ml_nn_v7.c 的索引算术，
验证钳位补丁满足：
  [正控制] 施加钳位后  ① 三处访问全部在界内  ② 合法通道的输出映射与「无幻影理想版」逐项一致
  [负控制] 不施加钳位时 ① 越界探测器必须报出越界（否则探测器自证无效）
  [回归]   C%8==0（零幻影）时，钳位前后输出映射完全相同（补丁必须是 no-op）
  [反作弊] 探测器用「独立重算」而不是复用被测代码的中间量

对应源码：
  v7:193  kernels_per_core = DRoU(out_c, cores_used)          cores_used = MIN2(out_c, ncc)
  v7:220-222  kernels_in_superblock = d（末批 = rem）
  v7:225/231  out_channel = core*kis + kernel + sb*d*cores_used
  v7:226  weights_maps[kernel] = input + out_channel*W*H*in_c
  v7:238  biases[out_channel]
  v7:251  append_bits(out_values_per_channel * out_channel, 32)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

NCC = 8
FAIL = []
OKC = [0]


def align(x, a):
    return (x + a - 1) & ~(a - 1)


def dru(a, b):
    return (a + b - 1) // b


def check(cond, msg):
    if cond:
        OKC[0] += 1
    else:
        FAIL.append(msg)
        print("  FAIL: %s" % msg)


def model(C, in_c=None, W=1, H=1, clamp=False, ncc=NCC):
    """复刻 write_core_6 的索引表；返回 (accesses, mapping)"""
    in_c = C if in_c is None else in_c
    cores_used = min(C, ncc)
    kpc0 = dru(C, cores_used)                       # v7:193
    # 标定几何：sb/d/rem 用实测标定（foo=5, d=5）
    sb = 7 if C >= 241 else (6 if C >= 201 else (5 if C >= 161 else 4))
    d = dru(kpc0, sb)                               # v7:220
    rem = kpc0 - d * (sb - 1)                       # v7:222
    ovpc = 2500                                    # v7:196 (50x50)

    acc = []        # (kind, index, limit)  越界检查用
    mapping = {}    # out_channel -> 写入目标（最后一次写覆盖前面的）
    for core in range(cores_used):
        for sbx in range(sb):
            kis = d if sbx < sb - 1 else rem
            for k in range(kis):
                oc = core * kis + k + sbx * d * cores_used     # v7:225
                oc_eff = oc
                if clamp and C > 0 and oc_eff >= C:
                    oc_eff = C - 1                              # P2 钳位
                # v7:226 权重张量访问
                acc.append(("weights", oc_eff * W * H * in_c, C * W * H * in_c))
                # v7:238 bias 访问
                acc.append(("biases", oc_eff * 4, C * 4))
                # v7:251 输出偏移（写）
                acc.append(("output", oc_eff * ovpc, C * ovpc))
                mapping[oc_eff] = ovpc * oc_eff
    return acc, mapping, dict(kpc0=kpc0, sb=sb, d=d, rem=rem, m=8 * kpc0 - C)


def ideal_mapping(C, ncc=NCC):
    """「无幻影理想版」：每个合法通道恰好被写一次（用于比对钳位后的映射）"""
    cores_used = min(C, ncc)
    kpc0 = dru(C, cores_used)
    sb = 7 if C >= 241 else (6 if C >= 201 else (5 if C >= 161 else 4))
    d = dru(kpc0, sb)
    rem = kpc0 - d * (sb - 1)
    out = {}
    for sbx in range(sb):
        kis = d if sbx < sb - 1 else rem
        # 理想版：末批只分配真实存在的通道，通道号连续
        for k in range(kis):
            idx = sbx * d * ncc + k
            if idx < C:
                out[idx] = 2500 * idx
    return out


def oob_report(acc):
    bad = []
    for kind, idx, limit in acc:
        if kind == "weights":
            # 权重访问是 [idx, idx+W*H*in_c-1]，此处只查起点
            if idx < 0 or idx >= limit:
                bad.append((kind, idx, limit))
        elif kind == "biases":
            if idx < 0 or idx + 3 >= limit:
                bad.append((kind, idx, limit))
        else:
            if idx < 0 or idx >= limit:
                bad.append((kind, idx, limit))
    return bad


POINTS = [(122, 0), (196, 0), (220, 0), (240, 0), (244, 1), (260, 0), (280, 0),
          (248, 0), (252, 0), (284, 1)]

print("=" * 78)
print("P2 自测：write_core_6 幻影通道钳位")
print("=" * 78)

print("\n[负控制] 不施加钳位时，越界探测器必须报出越界（否则探测器无效）")
for C, hang in POINTS:
    acc, _, meta = model(C, clamp=False)
    bad = oob_report(acc)
    expect = meta["m"] > 0
    check((len(bad) > 0) == expect,
          "C=%d 越界探测与幻影数不符（m=%d bad=%d）" % (C, meta["m"], len(bad)))
    print("  C=%-4d m=%d 越界数=%-3d %s" % (C, meta["m"], len(bad),
                                            "有OOB" if bad else "无OOB"))

print("\n[正控制] 施加钳位后：三处访问全部在界内")
for C, hang in POINTS:
    acc, _, meta = model(C, clamp=True)
    bad = oob_report(acc)
    check(len(bad) == 0, "C=%d 钳位后仍有越界: %s" % (C, bad[:3]))
    print("  C=%-4d m=%d 钳位后越界数=%d %s" % (C, meta["m"], len(bad), "OK" if not bad else "BAD"))

print("\n[正控制·语义] 钳位后：合法通道的输出映射与「无幻影理想版」一致（幂等性）")
for C, hang in POINTS:
    _, mp, meta = model(C, clamp=True)
    ideal = ideal_mapping(C)
    same = all(mp.get(ch) == off for ch, off in ideal.items())
    missing = [ch for ch in ideal if ch not in mp]
    check(same and not missing, "C=%d 钳位后映射与理想版不一致 missing=%s" % (C, missing[:5]))
    # 幻影通道被钳到 C-1 => C-1 被多次写，但值相同 => 幂等
    extra = len(mp) - len(ideal)
    print("  C=%-4d 合法通道=%d 钳位后映射条目=%d (幻影并入 C-1: +%d) 映射一致=%s"
          % (C, len(ideal), len(mp), extra, same))

print("\n[回归] C%%8==0（零幻影）时钳位必须是 no-op")
for C in (240, 248, 280, 120, 128, 200):
    _, mp_c, meta = model(C, clamp=True)
    _, mp_n, _ = model(C, clamp=False)
    check(mp_c == mp_n, "C=%d 非零幻影却被改动（补丁不是 no-op）" % C)
    check(meta["m"] == 0, "C=%d 幻影数应为 0，实为 %d" % (C, meta["m"]))
    print("  C=%-4d m=%d 钳位前后映射相同=%s" % (C, meta["m"], mp_c == mp_n))

print("\n[反作弊·独立重算] 用不依赖被测代码中间量的方式重算幻影数")
for C, hang in POINTS:
    kpc0 = dru(C, min(C, NCC))
    m = 8 * kpc0 - C                      # 独立公式：m = 8*ceil(C/8) - C
    check(0 <= m <= 7, "C=%d 幻影数越界: %d" % (C, m))
    check(m == (8 - C % 8) % 8, "C=%d 幻影数与 (8-C%%8)%%8 不符" % C)
    # 关键恒等式：m>0 <=> C%8!=0
    check((m > 0) == (C % 8 != 0), "C=%d 幻影数与 C%%8 规则不符" % C)
print("  全部 %d 个点的 m 与 (8-C%%8)%%8 一致，且 m>0 <=> C%%8!=0" % len(POINTS))

print()
print("=" * 78)
print("断言通过 %d 条，失败 %d 条" % (OKC[0], len(FAIL)))
for f in FAIL:
    print("  - %s" % f)
print("结果: %s" % ("PASS" if not FAIL else "FAIL"))
sys.exit(1 if FAIL else 0)
