# -*- coding: utf-8 -*-
"""
真实模型「幻影通道风险」离线扫描器（零设备、零 CI）。

用法:
  python3 scan-model-phantom-risk.py <model-ops.tsv> [--json]
输入 TSV 每行至少三列:  op_name \t kind \t out_channels \t weight_width
  kind 取值: conv / conv2d / depthwise / pointwise / dw ...
  省略 weight_width 时默认 1（1x1）。

原理（源码依据 gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c）:
  v7:193  kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)
          cores_used = MIN2(output_channels, nn_core_count)   (默认 nn_core_count=8)
  v7:225/231/305/389  out_channel = core*kis + kernel + sb*d*cores_used
  ⇒ 通道总数被向上取整到 cores_used 的倍数，末批必多出
     m = cores_used*DRoU(C, cores_used) - C = (cores_used - C%cores_used) % cores_used
     个「幻影通道」，它们会被真实执行 ⇒ 三处越界（权重读 / bias 读 / 输出偏移写）。
  ⇒ m>0 <=> C % cores_used != 0

判据:
  必要条件 (m>0)        => 该算子在该核数下「有幻影通道」，属风险点
  充分条件 (m==0)       => 该机制不可能触发（安全）
  经验律 (rem==1 & m>0) => OCH 的 10/10 经验律（本工具同时列出，供交叉核对）

输出:
  风险算子清单 + 统计 + 「补齐到 8 的倍数需要补多少通道」的手术建议。
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

NCC = 8


def align(x, a):
    return (x + a - 1) & ~(a - 1)


def dru(a, b):
    return (a + b - 1) // b


def superblock_profile(C, ncc=NCC, w=1, tile_y=12, abd=64, im=1):
    """复刻 calc_superblocks (v7:19-39)，标定 ncc=8 ibd=12 abd=64 im=1 geometry 50x50"""
    if C <= 0:
        return None
    cores_used = min(C, ncc)
    kpc0 = dru(C, cores_used)                       # v7:193
    foo = (abd * im) // tile_y                      # v7:26
    if w == 1:
        foo = min(foo, abd // 3)                    # v7:28-29
    foo = min(foo, kpc0)                            # v7:31
    foo = min(foo, 127)                             # v7:32
    kpc = dru(C, ncc * foo)                         # v7:34
    nk = dru(C, kpc * ncc)                          # v7:35
    sb = dru(dru(C, ncc), nk)                       # v7:36
    d = dru(kpc0, sb)                               # v7:220
    rem = kpc0 - d * (sb - 1)                       # v7:222
    m = cores_used * kpc0 - C                       # 幻影通道数
    return dict(cores_used=cores_used, kpc0=kpc0, sb=sb, d=d, rem=rem, m=m)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", " ").split(" ") if p.strip()]
            if len(parts) < 3:
                continue
            name = parts[0]
            kind = parts[1]
            try:
                C = int(parts[2])
            except ValueError:
                continue
            w = 1
            for p in parts[3:]:
                try:
                    w = int(p)
                    break
                except ValueError:
                    continue
            rows.append((ln, name, kind, C, w))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    rows = load_rows(path)

    risky, safe, law_hit = [], [], []
    for ln, name, kind, C, w in rows:
        pr = superblock_profile(C, w=w)
        if pr is None:
            continue
        rec = dict(line=ln, name=name, kind=kind, out_channels=C, weight_width=w, **pr)
        if pr["m"] > 0:
            rec["pad_to"] = C + ((NCC - C % NCC) % NCC)
            risky.append(rec)
            if pr["rem"] == 1:
                law_hit.append(rec)
        else:
            safe.append(rec)

    if as_json:
        print(json.dumps(dict(risk=risky, safe_count=len(safe),
                              law_hit=[r["name"] for r in law_hit]),
                         ensure_ascii=False, indent=2))
        return 0

    print("=" * 96)
    print("真实模型幻影通道风险扫描: %s" % path)
    print("=" * 96)
    print("总 conv 算子 %d 个 | 有幻影通道(m>0) %d 个 | 零幻影(m==0) %d 个"
          % (len(risky) + len(safe), len(risky), len(safe)))
    print()
    print("%-28s%-12s%-8s%-7s%-6s%-6s%-6s%-8s%s"
          % ("op", "kind", "out_c", "w", "kpc0", "sb", "rem", "m(幻影)", "经验律rem==1"))
    for r in risky:
        print("%-28s%-12s%-8d%-7d%-6d%-6d%-6d%-8d%s"
              % (r["name"][:28], r["kind"][:12], r["out_channels"], r["weight_width"],
                 r["kpc0"], r["sb"], r["rem"], r["m"],
                 "★命中(高风险)" if r["rem"] == 1 else ""))
    print()
    print("★ 经验律(rem==1 且 m>0) 命中 %d 个: %s"
          % (len(law_hit), [r["name"] for r in law_hit]))
    print("★ 模型手术建议（仅对 1x1/逐通道独立的算子有效）: out_c 补齐到 8 的倍数")
    for r in risky:
        if r["weight_width"] == 1:
            print("    %-28s %d -> %d (补 %d 个零权重通道, host 侧裁掉)"
                  % (r["name"][:28], r["out_channels"], r["pad_to"],
                     r["pad_to"] - r["out_channels"]))
    print()
    print("注: depthwise/多核共享权重的算子不能简单补零通道(会改变分组语义),")
    print("    需另走驱动侧钳位(P2)或拆算子。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
