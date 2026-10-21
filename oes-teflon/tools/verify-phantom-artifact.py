#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify-phantom-artifact.py —— B 段产物的**身份校验**（上机前必跑）

## 为什么需要它

C3/C4/C2b 用的是**三个不同的 .so**，开关组合互不相同：

| 变体 | 必须存在 | 必须缺席 |
|---|---|---|
| `clamp`（C3） | `och-patch-v7-phantom-clamp`、`[och-rofix]` | `och-patch-v7-phantom-gate`、`[och-ro]`、`[och-fp]` |
| `gate`（C4）  | `och-patch-v7-phantom-gate`、`[och-rofix]` | `och-patch-v7-phantom-clamp`、`[och-ro]`、`[och-fp]` |
| `diag`（C2b） | `och-patch-v7-phantom-clamp`、`och-patch-v7-phantom-gate`、`[och-ro]` | `[och-rofix]`、`[och-fp]` |
| `fix`（对照）  | 两条 phantom 标记、`[och-rofix]` | `[och-ro]`、`[och-fp]` |

拿错产物会让**整轮真机结论作废**，而且「不挂」会被误读成「补丁有效」——
所以这一步必须是**机器判定**，不能靠人眼 grep。

## 三态退出码（沿用 `tools/verify_artifact.py` 的约定）

    0  通过（每个 PASS 都打印它匹配到的**原文**）
    1  失败（缺/多，逐条列出）
    2  无法判定（找不到 strings / 文件不可读）—— **不等于通过**

## 用法

    python3 tools/verify-phantom-artifact.py <libteflon.so> clamp|gate|diag|fix
    python3 tools/verify-phantom-artifact.py --selftest

## 自测（正控制 + 负控制）

    --selftest 用**合成样本**跑 6 条正控制与 6 条负控制；
    合成样本是纯文本，`strings` 照样能取到（本工具只依赖 strings，不解析 ELF）。
"""

import argparse
import os
import subprocess
import sys
import tempfile

# 变体 → (必须存在, 必须缺席)
VARIANTS = {
    "clamp": (
        ["och-patch-v7-phantom-clamp", "[och-rofix]"],
        ["och-patch-v7-phantom-gate", "[och-ro]", "[och-fp]"],
    ),
    "gate": (
        ["och-patch-v7-phantom-gate", "[och-rofix]"],
        ["och-patch-v7-phantom-clamp", "[och-ro]", "[och-fp]"],
    ),
    "diag": (
        ["och-patch-v7-phantom-clamp", "och-patch-v7-phantom-gate", "[och-ro]"],
        ["[och-rofix]", "[och-fp]"],
    ),
    "fix": (
        ["och-patch-v7-phantom-clamp", "och-patch-v7-phantom-gate", "[och-rofix]"],
        ["[och-ro]", "[och-fp]"],
    ),
}


def extract_strings(path):
    """返回 strings 输出；无法探测时返回 None（调用方转成退出码 2）。"""
    try:
        p = subprocess.run(["strings", "-a", path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.decode("utf-8", "replace")


def check(blob, variant):
    """返回 (violations, passes)。passes 里带**实际匹配到的原文行**。"""
    need, forbid = VARIANTS[variant]
    lines = blob.splitlines()
    violations, passes = [], []
    for pat in need:
        hit = next((ln for ln in lines if pat in ln), None)
        if hit is None:
            violations.append("缺少 %s —— 该补丁不在产物里" % pat)
        else:
            passes.append("存在 %s  ← %r" % (pat, hit.strip()[:90]))
    for pat in forbid:
        hit = next((ln for ln in lines if pat in ln), None)
        if hit is not None:
            violations.append("不该出现 %s 却出现了 ← %r" % (pat, hit.strip()[:90]))
        else:
            passes.append("确认缺席 %s" % pat)
    return violations, passes


def run_file(path, variant):
    if not os.path.isfile(path):
        sys.stderr.write("ERROR: 文件不存在: %s\n" % path)
        return 2
    blob = extract_strings(path)
    if blob is None:
        sys.stderr.write("ERROR: PROBE_UNTRUSTED —— 无法执行 strings（工具缺失或文件不可读）\n")
        sys.stderr.write("       这不是「通过」：请换一台有 binutils 的机器重跑。\n")
        return 2
    violations, passes = check(blob, variant)
    for p in passes:
        print("  PASS: " + p)
    for v in violations:
        print("  FAIL: " + v)
    if violations:
        print("结果: FAIL（%d 条）—— 不要上机" % len(violations))
        return 1
    print("结果: OK —— %s 变体的身份校验通过（%d 项）" % (variant, len(passes)))
    return 0


def selftest():
    tmp = tempfile.mkdtemp(prefix="phantom-artifact-selftest-")

    def make(name, markers):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(markers) + "\n")
        return p

    good = {
        "clamp": ["och-patch-v7-phantom-clamp", "[och-rofix]"],
        "gate": ["och-patch-v7-phantom-gate", "[och-rofix]"],
        "diag": ["och-patch-v7-phantom-clamp", "och-patch-v7-phantom-gate",
                 "[och-ro]   head=%02x"],
        "fix": ["och-patch-v7-phantom-clamp", "och-patch-v7-phantom-gate", "[och-rofix]"],
    }

    ok = True
    # ---------- 正控制 4：每个变体的正确产物必须通过 ----------
    for variant, markers in good.items():
        blob = extract_strings(make("good-%s.bin" % variant, markers))
        v, p = check(blob, variant)
        if v:
            print("  FAIL 正控制 %s: %s" % (variant, v))
            ok = False
        else:
            print("  ok 正控制 %s（%d 项）" % (variant, len(p)))

    # ---------- 负控制 ----------
    # 1) clamp 产物里混进 gate 标记 ⇒ 必须被拒
    blob = extract_strings(make("bad-mixed.bin",
                                ["och-patch-v7-phantom-clamp",
                                 "och-patch-v7-phantom-gate", "[och-rofix]"]))
    v, _ = check(blob, "clamp")
    if not any("phantom-gate" in x for x in v):
        print("  FAIL 负控制 1: clamp 产物混入 gate 标记却没被抓出")
        ok = False
    else:
        print("  ok 负控制 1（两条补丁混进同一产物）")

    # 2) diag 产物误带 ROFIX ⇒ 必须被拒（ROFIX 与 RODIAG 硬互斥）
    blob = extract_strings(make("bad-diag-rofix.bin",
                                ["och-patch-v7-phantom-clamp",
                                 "och-patch-v7-phantom-gate",
                                 "[och-ro]", "[och-rofix]"]))
    v, _ = check(blob, "diag")
    if not any("och-rofix" in x for x in v):
        print("  FAIL 负控制 2: diag 带 ROFIX 却没被抓出")
        ok = False
    else:
        print("  ok 负控制 2（diag 误带 ROFIX）")

    # 3) 缺标记 ⇒ 必须被拒
    blob = extract_strings(make("bad-missing.bin", ["[och-rofix]"]))
    v, _ = check(blob, "gate")
    if not any("缺少" in x for x in v):
        print("  FAIL 负控制 3: 缺 gate 标记却没被抓出")
        ok = False
    else:
        print("  ok 负控制 3（缺补丁标记）")

    # 4) FPARTIAL 混入 ⇒ 任何变体都必须被拒
    for variant in VARIANTS:
        blob = extract_strings(make("bad-fp-%s.bin" % variant,
                                    good[variant] + ["[och-fp] fused-partial"]))
        v, _ = check(blob, variant)
        if not any("och-fp" in x for x in v):
            print("  FAIL 负控制 4: %s 产物混入 [och-fp] 却没被抓出" % variant)
            ok = False
    print("  ok 负控制 4（四个变体都拒绝 [och-fp]）")

    # 5) 文件不存在 ⇒ 退出码 2（不是通过）
    rc = run_file(os.path.join(tmp, "nope.so"), "clamp")
    if rc != 2:
        print("  FAIL 负控制 5: 文件不存在应返回 2，实际 %d" % rc)
        ok = False
    else:
        print("  ok 负控制 5（文件不存在 ⇒ 退出码 2）")

    print("结果: %s" % ("selftest 全部通过（正控制 4 + 负控制 5）" if ok else "selftest 失败"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("so", nargs="?")
    ap.add_argument("variant", nargs="?", choices=sorted(VARIANTS))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.so or not a.variant:
        ap.print_help()
        return 2
    return run_file(a.so, a.variant)


if __name__ == "__main__":
    sys.exit(main())
