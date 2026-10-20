#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-readout-diag-compose.py —— RODIAG 与 ROFIX 叠加形态的自测

## 门守什么

2026-10-20 事故（verify/reports/38）：

  `phantomdiag` 产物是 **ROFIX=0 + RODIAG=1** 建的。它把
  `read_outputs` 的**范围对齐修复去掉了** ⇒ map 用 `(0, pipe_buffer_size(res))`、
  读取用 `tensor->size`，越界读重新出现。该产物被当成「低危诊断仪器」送上真机，
  **作业启动后几秒内整机硬挂**（kern.log 之后再无任何内核输出，需断电重启）。

  取证打印本身不改行为；危险的是**丢掉 ROFIX**。

本门守三件事：

  1. 「ROFIX=0 的取证产物」不能再被当成正常产物悄悄产出（build 脚本硬拒绝）；
  2. 取证**能**叠加在 ROFIX 之上（两者不再互斥），且叠加后行为不变
     （不新增 `etna_ml_get_tensor` 调用、不碰 map 参数）；
  3. build 脚本里的「先 ROFIX、后取证」顺序不能被后续改动打乱。

## 测试矩阵

  正控制 3：
    1) 真实源码 + RODIAG（独立模式）⇒ 成功，且**明确警告**「本产物不含 ROFIX」；
    2) 真实源码 + ROFIX ⇒ RODIAG（叠加模式）⇒ 成功；`[och-ro]` ×3、`[och-rofix]` ×1，
       且 `etna_ml_get_tensor(` 调用数**不增加**；
    3) 叠加版函数体括号配平（可解析性的弱检查）。

  负控制 5：
    1) 重复应用 RODIAG ⇒ 拒绝；
    2) RODIAG 之后打 ROFIX ⇒ 拒绝（保护顺序）；
    3) 真实源码删掉 `assert(src_map);` ⇒ 独立模式拒绝（STRUCT 校验）；
    4) 叠加版改掉 `res = tensor->resource;` ⇒ 叠加模式拒绝（STRUCT_ROFIX 校验）；
    5) 无关文件 ⇒ 拒绝。

  回归 2（build 脚本结构，防「改了顺序但门还是绿的」）：
    R1) 必须含硬门文案「TEFLON_RODIAG=1 必须同时设 TEFLON_ROFIX=1」；
    R2) 3j-2（取证叠加）段落必须**晚于** 3j（ROFIX 应用）。

用法:
    python3 selftest-readout-diag-compose.py
退出码: 0 通过 / 1 失败 / 2 样本不可信（缺真实源码 ⇒ 本门拒绝给出结论）
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DIAG = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-diag.py")
ROFIX = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")
BUILD = os.path.join(ROOT, "build-recipe", "build-teflon.sh")

FIXTURE = os.path.join(ROOT, "build-recipe", "fixtures", "etnaviv_ml.ref.c")
FALLBACK = os.path.join(ROOT, "third_party", "mesa-src-26.2.5",
                        "src", "gallium", "drivers", "etnaviv", "etnaviv_ml.c")

FAILS = []


def fail(msg):
    FAILS.append(msg)
    sys.stderr.write("  [FAIL] " + msg + "\n")


def ok(msg):
    print("  [ok] " + msg)


def run(script, path):
    return subprocess.run([sys.executable, script, path],
                          capture_output=True, text=True)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def fn_body(src, fn="etna_ml_subgraph_read_outputs(struct pipe_context"):
    at = src.find(fn)
    if at < 0:
        return None
    i = src.index("{", at)
    d, j = 0, i
    while True:
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return src[at:j + 1]
        j += 1


def main() -> int:
    for p in (DIAG, ROFIX, BUILD):
        if not os.path.isfile(p):
            sys.stderr.write("ERROR: 缺文件 " + p + "\n")
            return 2

    sample = FIXTURE if os.path.isfile(FIXTURE) else FALLBACK
    if not os.path.isfile(sample):
        sys.stderr.write("ERROR: 找不到真实源码样本（%s 或 %s）\n" % (FIXTURE, FALLBACK))
        sys.stderr.write("       ⚠️ 本门**拒绝**在合成样本上给结论（真实锚点是本门的前提）。\n")
        return 2

    base = read(sample)
    print("== 样本: %s (%d B) ==" % (os.path.relpath(sample, ROOT), len(base)))

    tmp = tempfile.mkdtemp(prefix="selftest-rodiag-")
    try:
        def put(name, text):
            p = os.path.join(tmp, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            return p

        # ── 正控制 1：独立模式 ────────────────────────────────────────────
        p1 = put("standalone.c", base)
        r = run(DIAG, p1)
        out1 = read(p1)
        if r.returncode != 0:
            fail("正控制 1 失效：独立模式应成功，实际 rc=%d\n%s" % (r.returncode, r.stderr))
        elif out1.count("[och-ro]") != 3:
            fail("正控制 1 失效：独立模式 [och-ro] 应为 3 处，实际 %d" % out1.count("[och-ro]"))
        elif "不含 ROFIX" not in r.stdout:
            fail("正控制 1 失效：独立模式必须**明确警告**本产物不含 ROFIX（否则又会被当低危仪器）")
        else:
            ok("正控制 1：独立模式可用，且如实警告「不含 ROFIX」")

        # ── 正控制 2：ROFIX + 叠加 ────────────────────────────────────────
        p2 = put("compose.c", base)
        r = run(ROFIX, p2)
        if r.returncode != 0:
            fail("正控制 2 失效：ROFIX 应先成功，实际 rc=%d\n%s" % (r.returncode, r.stderr))
        else:
            before = read(p2)
            n_get_before = before.count("etna_ml_get_tensor(")
            r2 = run(DIAG, p2)
            after = read(p2)
            n_get_after = after.count("etna_ml_get_tensor(")
            if r2.returncode != 0:
                fail("正控制 2 失效：叠加模式应成功，实际 rc=%d\n%s" % (r2.returncode, r2.stderr))
            elif "叠加在 ROFIX 之上" not in r2.stdout:
                fail("正控制 2 失效：未进入叠加模式（说明 ROFIX 标记没被识别）")
            elif after.count("[och-ro]") != 3:
                fail("正控制 2 失效：叠加版 [och-ro] 应为 3 处，实际 %d" % after.count("[och-ro]"))
            elif after.count("[och-rofix]") != 1:
                fail("正控制 2 失效：叠加版应保留 ROFIX 的 [och-rofix] 1 处，实际 %d"
                     % after.count("[och-rofix]"))
            elif n_get_after != n_get_before:
                fail("正控制 2 失效：叠加新增了 etna_ml_get_tensor 调用（%d → %d）"
                     % (n_get_before, n_get_after))
            else:
                ok("正控制 2：ROFIX + 取证叠加成功；%d 个 [och-ro]、ROFIX 标记保留、未新增取张量调用"
                   % after.count("[och-ro]"))

            # ── 正控制 3：括号配平 ────────────────────────────────────────
            body = fn_body(after)
            if body is None:
                fail("正控制 3 失效：叠加版里找不到 read_outputs 函数体")
            elif body.count("{") != body.count("}") or body.count("(") != body.count(")"):
                fail("正控制 3 失效：叠加版括号不配平（{=%d }=%d ( =%d )=%d）"
                     % (body.count("{"), body.count("}"), body.count("("), body.count(")")))
            else:
                ok("正控制 3：叠加版函数体括号配平")

        # ── 负控制 1：重复应用 ────────────────────────────────────────────
        p3 = put("dup.c", read(p2))
        r = run(DIAG, p3)
        if r.returncode == 0:
            fail("负控制 1 失效：重复应用应拒绝（否则会双重注入）")
        else:
            ok("负控制 1：重复应用被拒绝")

        # ── 负控制 2：顺序反了（RODIAG 之后打 ROFIX）─────────────────────
        p4 = put("order.c", base)
        run(DIAG, p4)
        r = run(ROFIX, p4)
        if r.returncode == 0:
            fail("负控制 2 失效：RODIAG 之后仍能打 ROFIX（顺序门失效 ⇒ 会产出未对齐的 map）")
        else:
            ok("负控制 2：RODIAG 之后拒绝再打 ROFIX")

        # ── 负控制 3：真实样本被破坏（删 assert）────────────────────────
        broken = base.replace("         assert(src_map);\n", "", 1)
        if broken == base:
            fail("负控制 3 失效：样本里找不到 `assert(src_map);`（锚点已变，请重跑 gen-rodiag-patch.py）")
        else:
            p5 = put("noassert.c", broken)
            r = run(DIAG, p5)
            if r.returncode == 0:
                fail("负控制 3 失效：删掉 assert 后仍能应用（无取证可锚点也放行）")
            else:
                ok("负控制 3：删掉 `assert(src_map);` 被拒绝")

        # ── 负控制 4：叠加版结构被破坏 ───────────────────────────────────
        compose_txt = read(p2)
        broken2 = compose_txt.replace(
            "      struct pipe_resource *res = tensor->resource;\n",
            "      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);\n", 1)
        if broken2 == compose_txt:
            fail("负控制 4 失效：叠加版里找不到 `res = tensor->resource;`")
        else:
            p6 = put("compose_broken.c", broken2)
            # 先去掉已有取证，才能重新触发叠加分支
            p6b = put("compose_broken2.c", broken2.replace(
                "      /* och patch: read_outputs 越界取证（叠加在 ROFIX 之上；只读，不改行为）\n", "", 1))
            r = run(DIAG, p6b)
            if r.returncode == 0:
                fail("负控制 4 失效：叠加版结构被破坏后仍能应用")
            else:
                ok("负控制 4：叠加版结构被破坏后被拒绝")

        # ── 负控制 5：无关文件 ───────────────────────────────────────────
        p7 = put("unrelated.c", "int main(void) { return 0; }\n")
        r = run(DIAG, p7)
        if r.returncode == 0:
            fail("负控制 5 失效：无关文件应拒绝")
        else:
            ok("负控制 5：无关文件被拒绝")

        # ── 回归 R1/R2：build 脚本结构 ───────────────────────────────────
        build = read(BUILD)
        if "TEFLON_RODIAG=1 必须同时设 TEFLON_ROFIX=1" not in build:
            fail("回归 R1 失效：build-teflon.sh 缺少「RODIAG 必须配 ROFIX」硬门文案")
        else:
            ok("回归 R1：build-teflon.sh 有「RODIAG 必须配 ROFIX」硬门")

        at_j2 = build.find("[3j-2]")
        at_j = build.find("[3j/6] 打【修复】read_outputs 范围对齐")
        if at_j < 0 or at_j2 < 0:
            fail("回归 R2 失效：build-teflon.sh 里找不到 3j 或 3j-2 段（顺序无法核对）")
        elif at_j2 < at_j:
            fail("回归 R2 失效：3j-2（取证叠加）出现在 3j（ROFIX）**之前** ⇒ 锚点会冲突")
        else:
            ok("回归 R2：取证叠加段位于 ROFIX 之后")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILS:
        sys.stderr.write("\n结果：失败 %d 项\n" % len(FAILS))
        for m in FAILS:
            sys.stderr.write("  - " + m.splitlines()[0] + "\n")
        return 1

    print("\n  [ok] selftest-readout-diag-compose 全部通过（正控制 3 + 负控制 5 + 回归 2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
