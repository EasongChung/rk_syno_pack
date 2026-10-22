#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`patch-etnaviv-perchannel-foldearly.py` 的结构与数值自测。

两部分：

  A. **结构控制**（打补丁器本身）
     正控制：在真实源码副本上打补丁 ⇒ rc=0，且四条硬门都成立
     负控制①：同一份再打一次 ⇒ 必须 fail-closed（md5 失配）
     负控制②：三个钉死函数的源码各改 1 字节（真实样本，不是构造串）⇒ 各自必须 rc=2
     负控制③：插入锚点消失 ⇒ 必须 rc=2
     反作弊：`--verify` 对**已打补丁**的文件必须红（证明它真的在看未打补丁的原文）

  B. **数值镜像**（不变式 I1 / I2 / I7）
     I1  原地折叠 ⇒ weight_tensor_size 不变；
         上游对 depthwise 会缩成 1/C，对非 depthwise 不缩（两者都要断言）
     I2  非 depthwise：新实现与上游**逐元素**相同（按 (oc,ic,w,h) 对齐）
     I7  expand_depthwise 之后：对角 = 折叠值，非对角 = **新** zero_point（signed 时 -128）

跑法：python3 tools/selftest-perchannel-foldearly.py --src-root <包含 etnaviv_ml_nn.c 的目录>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PATCHER = os.path.join(REPO, "build-recipe", "patch-etnaviv-perchannel-foldearly.py")

FAILS = []
CHECKS = 0


def chk(cond, name, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILS.append(name)


def run(args):
    p = subprocess.run([sys.executable, PATCHER] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ── B. 数值镜像 ───────────────────────────────────────────────────────────────

def quant_params(lo, hi):
    """上游 calc_quant_params 的镜像。注意 max 在调用方已按上游 quirk 从 0 起。"""
    if hi < lo:
        lo, hi = hi, lo
    diff = hi - min(lo, 0.0)
    scale = diff / 255.0
    zp = round(-(lo / scale)) if scale else 0
    return scale, int(zp) & 0xFF


def fold_element(val_int, scale_c, zp_c, new_scale, new_zp, signed):
    if signed:
        q = round(scale_c * (val_int - zp_c) / new_scale)
        return (q + new_zp - 0x80) & 0xFF
    q = round(scale_c * (val_int - zp_c) / new_scale)
    return (q + new_zp) & 0xFF


def mirror_impls(oc, k, ic, depthwise, signed, buf, scales, zps):
    """返回 (new_vals_by_identity, new_scale, new_zp, n, pre)。

    identity = (oc, ic, w, h) —— 与内存布局无关的元素身份。

    **布局（必须与源码一致，这是本镜像最容易错的地方）**：
      新实现（折叠时）：depthwise `[1][kh][kw][C]` ⇒ idx = (w·k + h)·C + c
                        普通卷积 `[oc][kh][kw][ic]` ⇒ idx = ((o·k + w)·k + h)·ic + i
      上游（折叠时）：已 expand+transpose 成 `[oc][ic][W][H]`，尺寸 = oc·k²·ic_real
    其中 depthwise 的 pre（expand 之后）= oc·k²·C，而折叠时的缓冲只有 oc·k²。
    """
    pre = oc * k * k * (oc if depthwise else ic)
    n = oc * k * k * 1 if depthwise else oc * k * k * ic

    def nidx(o, i, w, h):
        if depthwise:
            return (w * k + h) * oc + o   # [1][kh][kw][C]
        return ((o * k + w) * k + h) * ic + i

    def as_int(raw):
        return (raw - 0x100) if (signed and raw >= 0x80) else raw

    # ── 新实现：逐元素，通道 = i % C（depthwise）或 i // (k·k·ic)（普通）
    maxv, minv = 0.0, float("inf")
    for i in range(n):
        c = (i % oc) if depthwise else (i // (k * k * ic))
        v = scales[c] * (as_int(buf[i]) - zps[c])
        maxv = max(maxv, v)
        minv = min(minv, v)
    new_scale, new_zp = quant_params(minv, maxv)

    new_vals = {}
    for o in range(oc):
        for w in range(k):
            for h in range(k):
                for i in range(1 if depthwise else ic):
                    raw = buf[nidx(o, i, w, h)]
                    new_vals[(o, i, w, h)] = fold_element(as_int(raw), scales[o], zps[o],
                                                          new_scale, new_zp, signed)
    return new_vals, new_scale, new_zp, n, pre


def upstream_new_size(oc, k, ic_real, depthwise):
    """上游 requantize_weights 的 new_size 公式（nn.c:242）：depthwise 时 in=1。"""
    return oc * k * k * (1 if depthwise else ic_real)


def upstream_fold_non_depthwise(oc, k, ic, buf, scales, zps, signed):
    """上游 requantize_weights 在**非 depthwise** 下的逐元素镜像（nn.c:242-269）。

    它把缓冲 cast 成 `[oc][W][H][ic]` 并按该步长顺序遍历；非 depthwise 时
    物理布局就是 `[oc][kh][kw][ic]` ⇒ 顺序读 == 正确读（这正是「非 depthwise 无害」的原因）。
    """
    def as_int(raw):
        return (raw - 0x100) if (signed and raw >= 0x80) else raw

    n = oc * k * k * ic
    maxv, minv = 0.0, float("inf")
    for j in range(n):
        c = j // (k * k * ic)
        v = scales[c] * (as_int(buf[j]) - zps[c])
        maxv = max(maxv, v)
        minv = min(minv, v)
    scale, zp = quant_params(minv, maxv)
    return [fold_element(as_int(buf[j]), scales[j // (k * k * ic)], zps[j // (k * k * ic)],
                         scale, zp, signed) for j in range(n)], scale, zp


def expand_mirror(buf, oc, k, ic, new_zp, signed):
    """expand_depthwise 的镜像（源码 nn.c:335-350 逐行对照）。"""
    out = [0] * (oc * k * k * ic)
    bg = ((new_zp - 128) & 0xFF) if signed else (new_zp & 0xFF)
    for channel in range(oc):
        in_channel = channel // oc  # 恒 0
        in_depth = channel % oc
        base_in = in_channel * k * k * ic
        base_out = channel * k * k * ic
        for i in range(k * k * ic):
            out[base_out + i] = buf[base_in + i] if (i % ic == in_depth) else bg
    return out


def part_b():
    print("\n=== B. 数值镜像（I1 / I2 / I7）===")
    # 场景 1：普通卷积（非 depthwise）—— I2 的核心
    oc, k, ic = 5, 3, 4
    buf = [((i * 37 + 11) & 0xFF) for i in range(oc * k * k * ic)]
    scales = [0.001 + 0.0001 * (i % 7) for i in range(oc)]
    zps = [0] * oc
    nv, ns, nzp, nsize, pre1 = mirror_impls(oc, k, ic, False, True, buf, scales, zps)

    # I2：逐元素相同。新实现的遍历顺序 (o,w,h,i) 正是上游 cast 的线性顺序 ⇒ 可直接比序列。
    seq_new = [nv[(o, i, w, h)] for o in range(oc) for w in range(k) for h in range(k) for i in range(ic)]
    seq_up, us, uzp = upstream_fold_non_depthwise(oc, k, ic, buf, scales, zps, True)
    chk(seq_new == seq_up, "I2 非 depthwise：新实现与上游逐元素相同",
        "len(new)=%d len(up)=%d" % (len(seq_new), len(seq_up)))
    if seq_new != seq_up:
        for idx, (a, b) in enumerate(zip(seq_new, seq_up)):
            if a != b:
                print("        首个差异 idx=%d new=%d upstream=%d" % (idx, a, b))
                break
    chk(nsize == pre1 == oc * k * k * ic, "I1 非 depthwise：折叠前后尺寸都是 oc·k²·ic",
        "n=%d pre=%d" % (nsize, pre1))
    chk(upstream_new_size(oc, k, ic, False) == nsize, "I1 非 depthwise：上游也不缩容")
    chk((ns, nzp) == (us, uzp), "I2 推论：折叠参数（scale/zp）也相同",
        "new=(%r,%d) up=(%r,%d)" % (ns, nzp, us, uzp))

    # 场景 2：depthwise per-channel —— I1 的关键（上游缩容、新实现不缩）
    C, k2 = 24, 3
    buf2 = [((i * 37 + 11) & 0xFF) for i in range(C * k2 * k2)]   # 折叠时：[1][k][k][C]
    scales2 = [0.001 + 0.0001 * (i % 7) for i in range(C)]
    zps2 = [0] * C
    nv2, ns2, nzp2, nsize2, pre2 = mirror_impls(C, k2, 1, True, True, buf2, scales2, zps2)
    chk(nsize2 == C * k2 * k2, "I1 depthwise：折叠时缓冲 = C·k²（原样，不缩）", "得到 %d" % nsize2)
    chk(pre2 == C * k2 * k2 * C, "I1 depthwise：expand 之后缓冲 = C·k²·C", "得到 %d" % pre2)
    up_size = upstream_new_size(C, k2, C, True)
    chk(up_size == C * k2 * k2, "I1 负对照：上游 new_size = oc·k²·1 = C·k²", "得到 %d" % up_size)
    chk(pre2 == up_size * C, "I1 复现根因：上游把它从 C·k²·C 缩到 C·k²，恰好 1/C",
        "%d vs %d" % (pre2, up_size))

    # I7：折叠后 expand ⇒ 对角=折叠值、非对角=新 zp
    folded = [nv2[(o, 0, w, h)] for w in range(k2) for h in range(k2) for o in range(C)]
    expanded = expand_mirror(folded, C, k2, C, nzp2, True)
    bg = (nzp2 - 128) & 0xFF
    bad = 0
    for channel in range(C):
        for i in range(k2 * k2 * C):
            v = expanded[channel * k2 * k2 * C + i]
            if i % C == channel:
                want = folded[(i // C) * C + channel]
            else:
                want = bg
            if v != want:
                bad += 1
    chk(bad == 0, "I7 expand 后：对角=折叠值、非对角=新 zp（signed 减 128）", "%d 个元素不符" % bad)
    chk(len(expanded) == C * k2 * k2 * C, "I7 附带：expand 后尺寸 = C·k²·C（与 input_channels=C 自洽）")
    # 反作弊：若 expand 返回的尺寸不是 C·k²·C，上面的 I7 断言就可能是空转
    chk(any(expanded[i] == bg for i in range(len(expanded))), "I7 反作弊：确实存在背景元素（非全对角）")


# ── A. 结构控制 ───────────────────────────────────────────────────────────────

def part_a(src_root):
    print("\n=== A. 结构控制（补丁器）===")
    src = os.path.join(src_root, "etnaviv_ml_nn.c")
    if not os.path.exists(src):
        chk(False, "找到 etnaviv_ml_nn.c", src)
        return
    pristine = open(src, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="fe-selftest-")

    work = os.path.join(tmp, "nn.c")
    shutil.copyfile(src, work)
    rc, out = run([work, "--verify"])
    chk(rc == 0, "正控制：pristine 上 --verify 通过", "rc=%d" % rc)

    rc, out = run([work])
    chk(rc == 0, "正控制：pristine 上打补丁成功", "rc=%d\n%s" % (rc, out[-400:]))
    patched = open(work, encoding="utf-8").read()
    chk("och-patch-perchannel-foldearly" in patched, "正控制：二进制标记已写入")
    chk("och_fold_channel(" in patched, "正控制：helper 已注入")
    fn = patched[patched.index("requantize_weights(struct etna_ml_subgraph"):]
    fn = fn[:fn.index("\n}\n") + 3]
    for tok in ("malloc(", "free(", "weight_tensor_size ="):
        chk(tok not in fn, "正控制：requantize_weights 不含 %r" % tok)
    chk(patched.index("per-channel 折叠提前（fold-early）") <
        patched.index("pointwise_to_2x2(subgraph", patched.index("etna_ml_lower_convolution")),
        "正控制：折叠块在 pointwise_to_2x2 之前")
    chk(patched.count("nn_core_version >= 8 && poperation->conv.depthwise") == 1,
        "正控制：v8 收口恰好 1 处")

    # ── 域修复并行形态（PCDOMAIN 并入 fold-early）──────────────────────────
    # 背景：fold-early 整函数重写了 requantize_weights ⇒ patch-etnaviv-perchannel-domain.py
    # 的锚点在它上面必然失配（CI run 39754283314 实测）。因此同一语义由 --pcdomain
    # 在本地实现里再生一遍。这一组检查证明「开关真的在起作用」，而不是文案。
    w_pc = os.path.join(tmp, "nn_pcdomain.c")
    shutil.copyfile(src, w_pc)
    rc_pc, out_pc = run([w_pc, "--pcdomain"])
    chk(rc_pc == 0, "正控制：--pcdomain 打补丁成功", "rc=%d\n%s" % (rc_pc, out_pc[-300:]))
    pc = open(w_pc, encoding="utf-8").read()
    for tok, name in (("int32_t och_requant", "int32 中间量（不截断）"),
                      ("CLAMP(och_qu, 0, 255)", "夹到 uint8 域"),
                      ('getenv("TEFLON_PCDOMAIN")', "运行期门 TEFLON_PCDOMAIN"),
                      ("[och-pcdomain]", "编译期标记 [och-pcdomain]"),
                      ("och-patch-perchannel-foldearly", "fold-early 标记仍在")):
        chk(tok in pc, "正控制：--pcdomain 含 %s" % name)
    # 负控制：不带开关时必须回到**上游公式**（int8 截断 + 减 0x80），且没有域修复标记
    up = open(work, encoding="utf-8").read()
    chk("[och-pcdomain]" not in up, "负控制：不带 --pcdomain 时没有 [och-pcdomain]")
    chk("int8_t requantized" in up and "0x80" in up,
        "负控制：不带 --pcdomain 时 signed 分支仍是上游公式")
    chk(up != pc, "反作弊：两个开关产出**不同**文件（否则开关是空开）")
    # 反作弊：--pcdomain 不得顺手改掉 unsigned 分支（只动 signed）
    def _fn(t):
        i = t.index("requantize_weights(struct etna_ml_subgraph")
        return t[i:t.index("\n}\n", i) + 3]
    us_up, us_pc = _fn(up), _fn(pc)
    chk("weights[i] = requantized + operation->weight_zero_point;" in us_pc,
        "反作弊：unsigned 分支在 --pcdomain 下逐字未变")

    rc2, out2 = run([work])
    chk(rc2 != 0, "负控制①：同一文件再打一次必须 fail-closed", "rc=%d" % rc2)
    chk("MISMATCH" in out2 or "md5" in out2, "负控制①：报错原因是 md5 失配")

    rc3, out3 = run([work, "--verify"])
    chk(rc3 != 0, "反作弊：--verify 对**已打补丁**的文件必须红（证明它看的是原文）", "rc=%d" % rc3)

    # 负控制②：三个钉死函数各改 1 字节（真实样本）
    for token, label in ((("float max = .0, min = FLT_MAX;"), "calc_quantization"),
                         (("unsigned new_size = operation->output_channels"), "requantize_weights"),
                         (("      if (nn_core_version >= 8 && poperation->conv.depthwise)"), "折叠块")):
        variant = pristine.replace(token, token.replace(" ", "  ", 1), 1)
        if variant == pristine:
            chk(False, "负控制② 构造成功：%s" % label, "token 未命中")
            continue
        w2 = os.path.join(tmp, "neg_%s.c" % label)
        open(w2, "w", encoding="utf-8").write(variant)
        rc4, out4 = run([w2, "--verify"])
        chk(rc4 == 2 and "MISMATCH" in out4, "负控制②：%s 改 1 字节 ⇒ rc=2 且报 MISMATCH" % label,
            "rc=%d" % rc4)
        chk(open(w2, encoding="utf-8").read() == variant, "负控制②：%s 失败时**不写文件**" % label)

    # 负控制③：插入锚点消失
    variant = pristine.replace("   operation->bias_tensor_size = bias_size;\n", "", 1)
    w3 = os.path.join(tmp, "neg_anchor.c")
    open(w3, "w", encoding="utf-8").write(variant)
    rc5, out5 = run([w3, "--verify"])
    chk(rc5 == 2 and "锚点" in out5, "负控制③：插入锚点消失 ⇒ rc=2 并点名锚点", "rc=%d" % rc5)

    shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="包含 etnaviv_ml_nn.c 的目录（Mesa 源码树或 build 目录）")
    args = ap.parse_args(argv[1:])

    part_a(args.src_root)
    part_b()

    print("\n== %d 项检查，%d 项失败 ==" % (CHECKS, len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED: %s" % f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
