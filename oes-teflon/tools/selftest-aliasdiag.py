#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIASDIAG 补丁（alias 批量元数据迁移取证）本地正/反例自测（CI Preflight 用）。

覆盖：1 正例 + 10 反例 + 幂等 + **与 ROFIX 共存（双向顺序）** + 生成器一致性。

本轮特有判据：
  * 三口径 —— 三处锚点的「新旧计数」口径**不同**，自测必须逐条钉死：
      ref  —— 段**中间**插入 ⇒ OLD_REF 消失（0）
      rec  —— 只追加型     ⇒ OLD_REC **保留 1**（它嵌在 NEW_REC 里）
      mark —— 只追加型     ⇒ ANCHOR_MARK **保留 1**（嵌在 NEW_MARK 里）
    一律查「消失」会把 rec/mark 的正例全部误拒（och 自己踩了两次）。
  * 共存 —— ALIASDIAG 改 reference_tensor_with_offset / recreate_tensor，
    ROFIX 改 etna_ml_subgraph_read_outputs：不同函数 ⇒ 必须能同时打进去，
    **两种顺序都要过**（och 只说「本地验证过共存」，没给顺序，这里补齐）。
  * 共性陷阱 —— recreate_tensor **只改 size 不改 offset**；STRUCT_REC 里
    不得出现 offset 共性（按共性校验必然失败）。这里钉死这条不许被改回去。
  * 无 TRAP 项 —— 本补丁没有「剥注释后反查」型自检（它的门全是「必须存在」型），
    所以不存在 RODIAG/ROFIX 那种「不剥注释会误拒正例」的陷阱。

⚠️ 本自测只证明「补丁自身行为 + 共存 + 与生成器的载荷一致性」；
   「真实 Mesa 源码里的锚点是否仍逐字一致」由构建后
   `tools/check-aliasdiag-anchors.py` 负责。

用法： python3 oes-teflon/tools/selftest-aliasdiag.py
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-alias-diag.py")
ROFIX_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")
GEN_PATH = os.path.join(HERE, "gen-aliasdiag-patch.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = load(PATCH_PATH, "aliasdiag_patch")
ROFIX = load(ROFIX_PATH, "rofix_patch_for_coexist")

FAILS = []
WARNS = []

# ── 合成源码：锚点部分**逐字取自补丁常量**（绝不手写缩进）────────────────
STUB_IO = "#include <stdio.h>\n"

ALIAS_SRC = (
    "static void\n"
    + PATCH.TARGET_FN_REF
    + " *subgraph, unsigned src_tensor, unsigned dst_tensor,\n"
    "                             unsigned offset, unsigned size)\n"
    "{\n"
    "   struct etna_ml_tensor *src = etna_ml_get_tensor(subgraph, src_tensor);\n"
    "   struct etna_ml_tensor *dst = etna_ml_get_tensor(subgraph, dst_tensor);\n"
    "   struct pipe_resource *old_res = dst->resource;\n"
    "   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);\n"
    "   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);\n"
    "   ML_DBG(\"stub\");\n"
    "   pipe_resource_reference(&dst->resource, src->resource);\n"
    "   dst->offset = offset;\n"
    + PATCH.OLD_REF +
    "      for (int i = 0; i < num_tensors; i++) {\n"
    "         if (etna_ml_get_resource(subgraph, i) == old_res) {\n"
    "            pipe_resource_reference(&tensors[i]->resource, src->resource);\n"
    "            tensors[i]->size = size;\n"
    "            tensors[i]->offset = offset;\n"
    "         }\n"
    "      }\n"
    "   }\n"
    "}\n"
    "\n"
    "static void\n"
    + PATCH.TARGET_FN_REC
    + " *subgraph, unsigned idx, unsigned size)\n"
    "{\n"
    "   struct pipe_resource *old_res = etna_ml_get_resource(subgraph, idx);\n"
    "   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);\n"
    "   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);\n"
    + PATCH.OLD_REC +
    "\n"
    "   etna_ml_destroy_tensor(subgraph, idx);\n"
    "   etna_ml_create_tensor(subgraph, idx, size);\n"
    "   new_res = etna_ml_get_resource(subgraph, idx);\n"
    "\n"
    "   if (old_res) {\n"
    "      for (int i = 0; i < num_tensors; i++) {\n"
    "         if (etna_ml_get_resource(subgraph, i) == old_res) {\n"
    "            pipe_resource_reference(&tensors[i]->resource, new_res);\n"
    "            tensors[i]->size = size;\n"
    "         }\n"
    "      }\n"
    "   }\n"
    "}\n"
)

RO_SRC = (
    "void\n"
    + ROFIX.TARGET_FN + " *context, struct pipe_ml_subgraph *psubgraph,\n"
    "                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],\n"
    "                              bool is_signed[])\n"
    "{\n"
    "   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)psubgraph;\n"
    "   (void)context; (void)subgraph;\n"
    "\n"
    + ROFIX.OLD +
    "}\n"
)

MOCK = STUB_IO + ALIAS_SRC
MOCK_BOTH = STUB_IO + RO_SRC + ALIAS_SRC


def mock_selfcheck(label, text, patcher):
    """mock 自检：括号配平 + 结构 token 计数 + 锚点恰 1（否则结论失真）。"""
    bad = []
    if text.count("{") != text.count("}"):
        bad.append("括号不配平 {=%d }=%d" % (text.count("{"), text.count("}")))
    for tok, want in patcher.STRUCT_REF:
        body = patcher._body(text, patcher.TARGET_FN_REF)
        got = body.count(tok) if body else -1
        if got != want:
            bad.append("REF 内 `%s` 实得 %d（期望 %d）" % (tok[:40], got, want))
    for tok, want in patcher.STRUCT_REC:
        body = patcher._body(text, patcher.TARGET_FN_REC)
        got = body.count(tok) if body else -1
        if got != want:
            bad.append("REC 内 `%s` 实得 %d（期望 %d）" % (tok[:40], got, want))
    for name, old in (("OLD_REF", patcher.OLD_REF), ("OLD_REC", patcher.OLD_REC),
                      ("ANCHOR_MARK", patcher.ANCHOR_MARK)):
        got = text.count(old)
        if got != 1:
            bad.append("%s 在 mock 里 %d 处（期望 1）" % (name, got))
    print("  mock 自检(%s): %s" % (label, "PASS" if not bad else "FAIL " + "; ".join(bad)))
    if bad:
        FAILS.append("mock 自检 " + label)
        sys.exit(1)


mock_selfcheck("MOCK", MOCK, PATCH)
mock_selfcheck("MOCK_BOTH", MOCK_BOTH, PATCH)
# MOCK_BOTH 里 read_outputs 侧也必须满足 ROFIX 的结构门（否则共存测试无意义）
# ⚠️ ROFIX 补丁器里那个函数签名是 `_fn_body(src)`（无函数名参数），本轮是 `_body(src, fn)`
for tok, want in ROFIX.STRUCT:
    got = ROFIX._fn_body(MOCK_BOTH).count(tok)
    if got != want:
        sys.stderr.write("ERROR: MOCK_BOTH 里 ROFIX token `%s` 实得 %d（期望 %d）\n"
                         % (tok[:50], got, want))
        sys.exit(1)
print("  命名  ROFIX 取函数体是 _fn_body(src)（本轮补丁器是 _body(src, fn)）PASS")
print("  mock 自检: MOCK_BOTH 满足 ROFIX 结构门 PASS")


def run_patch(text, patcher_path=None, new_overrides=None):
    """跑一次补丁器；new_overrides = {常量名: 值}；返回 (rc, out, err, after)。"""
    mod = load(patcher_path or PATCH_PATH, "aliasdiag_patch_run")
    for k, v in (new_overrides or {}).items():
        setattr(mod, k, v)
    d = tempfile.mkdtemp(prefix="aliasdiag_selftest_")
    p = os.path.join(d, "etnaviv_ml.c")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = [os.path.basename(patcher_path or PATCH_PATH), p]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main()
    except Exception as exc:            # noqa: BLE001 —— 故意兜住，fail-closed 要记名
        rc = -1
        err.write("EXCEPTION: %r\n" % (exc,))
    finally:
        sys.argv = saved
    with open(p, encoding="utf-8") as f:
        after = f.read().replace("\r\n", "\n")
    return rc, out.getvalue(), err.getvalue(), after


def report(label, ok, detail=""):
    print("  %-54s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def warn(label, detail):
    print("  %-54s WARN" % label)
    for line in detail.split("\n"):
        print("        " + line)
    WARNS.append(label)


def case(label, text, expect_err=None, expect_rc=1, patcher_path=None,
         new_overrides=None, checks=None):
    rc, out, err, after = run_patch(text, patcher_path=patcher_path,
                                    new_overrides=new_overrides)
    detail = []
    if rc != expect_rc:
        detail.append("rc=%s（期望 %s）" % (rc, expect_rc))
    if expect_err is not None and expect_err not in err:
        last = (err.strip().splitlines() or [""])[-1]
        detail.append("stderr 未含 %r；末行=%r" % (expect_err, last))
    if expect_rc == 0 and checks is not None:
        for desc, cond in checks(after, out, err):
            if not cond:
                detail.append(desc)
    report(label, not detail, "; ".join(detail))
    return after, out, err


print()
print("=== ALIASDIAG 补丁 正/反例自测（1 正例 + 10 反例 + 幂等 + 共存 + 生成器）===")

# ── 正例 + 三口径 ──────────────────────────────────────────────────────
after, out, err = case(
    "GOOD  合成源码", MOCK, expect_rc=0,
    checks=lambda after, out, err: [
        ("含 MARKER", PATCH.MARKER in after),
        ("含 [och-alias]", "[och-alias]" in after),
        ("含 migrate idx=", "migrate idx=" in after),
        ("含编译期标记 %r" % PATCH.COMPILE_MARK, PATCH.COMPILE_MARK in after),
        ("NEW_REF 恰好 1 份", after.count(PATCH.NEW_REF) == 1),
        ("NEW_REC 恰好 1 份", after.count(PATCH.NEW_REC) == 1),
        ("NEW_MARK 恰好 1 份", after.count(PATCH.NEW_MARK) == 1),
        ("【口径 ref】OLD_REF 已消失（0）", after.count(PATCH.OLD_REF) == 0),
        ("【口径 rec】OLD_REC 仍恰好 1（只追加型）", after.count(PATCH.OLD_REC) == 1),
        ("【口径 mark】ANCHOR_MARK 仍恰好 1（只追加型）",
         after.count(PATCH.ANCHOR_MARK) == 1),
        ("补丁后括号仍配平", after.count("{") == after.count("}")),
        ("dst->size = size; 未被注入块挤掉", "dst->size = size;" in after),
    ])
report("GOOD  补丁器 stdout 含两行 [och-alias]",
       out.count("  [och-alias]") == 2, "实得 %d 行" % out.count("  [och-alias]"))

# ── 共性陷阱：STRUCT_REC 不得含 offset 共性 ────────────────────────────
off_in_rec = [t for t, _ in PATCH.STRUCT_REC if "offset" in t]
report("陷阱  STRUCT_REC 不含 offset 共性（recreate 只改 size）",
       not off_in_rec, "出现了: %s" % off_in_rec)

# ── 幂等 ──────────────────────────────────────────────────────────────
d = tempfile.mkdtemp(prefix="aliasdiag_idem_")
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
mod = load(PATCH_PATH, "aliasdiag_patch_idem")
saved = sys.argv
sys.argv = ["patch", p]
o, e = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
    rc1 = mod.main()
with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
    rc2 = mod.main()
sys.argv = saved
report("幂等  同一文件跑两次（第二次必须拒绝）", rc1 == 0 and rc2 == 1,
       "rc1=%s rc2=%s" % (rc1, rc2))
report("幂等  第二次 stderr 含「已包含本补丁」", "已包含本补丁" in e.getvalue())

# ── 反例 n1..n10 ───────────────────────────────────────────────────────
case("n1  reference 的 if (old_res) { 被改写",
     MOCK.replace("   if (old_res) {\n      for (int i = 0;",
                  "   if (old_res2) {\n      for (int i = 0;", 1),
     "出现 0 处")

case("n2  reference 的 tensors[i]->offset = offset; 被删",
     MOCK.replace("            tensors[i]->offset = offset;\n", "", 1),
     "出现 0 处")

case("n3  pipe_resource_reference(&dst->resource, ...) 被改写",
     MOCK.replace("pipe_resource_reference(&dst->resource, src->resource);",
                  "pipe_resource_reference(&dst->resource, old_res);", 1),
     "出现 0 处")

case("n4  recreate 的 tensors[i]->size = size; 被删",
     MOCK.replace("            tensors[i]->size = size;\n", "", 1),
     "出现 0 处")

case("n5  recreate 的 struct pipe_resource *new_res; 被改写",
     MOCK.replace("   struct pipe_resource *new_res;",
                  "   struct pipe_resource *newres;", 1),
     "出现 0 处")

case("n6  函数整体改名（TARGET_FN_REF）",
     MOCK.replace("reference_tensor_with_offset(struct etna_ml_subgraph",
                  "reference_tensor_with_offset_XX(struct etna_ml_subgraph", 1),
     "找不到函数")

# ⚠️ 不能直接把 OLD_REF 复制一份来测「匹配到 2 处」：OLD_REF 以未配对的 `{`
#    结尾，复制即破坏括号平衡，于是**先**被 _body 的 IndexError 拦下（方向仍
#    fail-closed，但测不到计数门）。所以复制**整个函数**（改函数名 ⇒ 括号仍配平），
#    让 STRUCT 门在第一个函数体上通过，再由文件级计数门报「匹配到 2 处」。
DUP_FN = ALIAS_SRC.replace("reference_tensor_with_offset(struct",
                           "reference_tensor_with_offset_dup(struct", 1)
case("n7  ref 锚点存在两个同形函数（计数门）", MOCK + DUP_FN,
     "锚点匹配到 2 处")

# n7b：直接复制 OLD_REF 文本会破坏括号 —— 实测记录（不是「干净拒绝」）
rc7b, _, err7b, _ = run_patch(MOCK.replace(PATCH.OLD_REF,
                                           PATCH.OLD_REF + PATCH.OLD_REF, 1))
report("n7b 复制 OLD_REF 文本 ⇒ 拒绝（rc!=0）", rc7b != 0, "rc=%s" % rc7b)
if rc7b == -1:
    report("n7b 拒绝方式是 IndexError（括号被复制破坏），已记名",
           "EXCEPTION" in err7b)

case("n8  #include <stdio.h> 出现两次（标记锚点撞车）",
     MOCK.replace(STUB_IO, STUB_IO + STUB_IO, 1),
     "标记锚点 锚点匹配到 2 处")

case("n9  rec 锚点被复制（同形 2 处）",
     MOCK.replace(PATCH.OLD_REC, PATCH.OLD_REC + PATCH.OLD_REC, 1),
     "2 处")

rc10, _, err10, _ = run_patch(MOCK.replace("   }\n", "   {\n", 1))
report("n10 括号不平衡 ⇒ 拒绝（rc!=0，允许是异常）", rc10 != 0, "rc=%s" % rc10)
if rc10 == -1:
    report("n10 失败方式是异常（IndexError 等），已如实记名", "EXCEPTION" in err10)
    warn("n10 备注：_body 对括号不平衡会 IndexError",
         "方向 fail-closed（不会静默放行），但错误信息退化成 traceback；\n"
         "根治 = 给 _body 加 `while j < len(src)` 边界（1 行）。\n"
         "这是本项目**第三轮**同一处 nit（RODIAG / ROFIX / ALIASDIAG），建议一次性修三个文件。")

# ── 共存：ALIASDIAG × ROFIX，双向顺序 ─────────────────────────────────
print()
print("=== 共存（不同函数 ⇒ 可同时开；两种顺序都要过）===")


def run_both(order):
    """按 order 依次跑两个补丁器，返回 (最终文本, [(步, rc, err)])。"""
    text = MOCK_BOTH
    steps = []
    for name in order:
        path = PATCH_PATH if name == "alias" else ROFIX_PATH
        rc, _out, err, text = run_patch(text, patcher_path=path)
        steps.append((name, rc, err.strip()[-160:]))
        if rc != 0:
            break
    return text, steps


for order, label in ((( "alias", "rofix"), "ALIASDIAG 先、ROFIX 后"),
                     (("rofix", "alias"), "ROFIX 先、ALIASDIAG 后")):
    text, steps = run_both(order)
    ok = all(rc == 0 for _, rc, _ in steps)
    detail = "; ".join("%s rc=%s%s" % (n, rc, (" err=" + e) if e else "")
                       for n, rc, e in steps)
    if not ok:
        report("共存  " + label, False, detail)
        continue
    bad = []
    if PATCH.MARKER not in text or ROFIX.MARKER not in text:
        bad.append("两个 marker 不都在")
    if "[och-alias]" not in text or "[och-rofix]" not in text:
        bad.append("[och-alias]/[och-rofix] 不都在")
    if text.count("{") != text.count("}"):
        bad.append("括号不配平 {=%d }=%d" % (text.count("{"), text.count("}")))
    if text.count(PATCH.NEW_REF) != 1:
        bad.append("NEW_REF %d 份" % text.count(PATCH.NEW_REF))
    if text.count(ROFIX.NEW) != 1:
        bad.append("ROFIX.NEW %d 份" % text.count(ROFIX.NEW))
    report("共存  " + label, not bad, "; ".join(bad) + " | " + detail)

text_ad_ro, _ = run_both(("alias", "rofix"))
text_ro_ad, _ = run_both(("rofix", "alias"))
report("共存  两种顺序产出的结果完全一致", text_ad_ro == text_ro_ad,
       "顺序影响结果 —— 说明两补丁在同一文件上并非完全独立")

# ── 生成器 ↔ 补丁 一致性 ──────────────────────────────────────────────
print()
print("=== 生成器 ↔ 补丁 一致性（GEN）===")
d2 = tempfile.mkdtemp(prefix="aliasdiag_gen_")
mock_path = os.path.join(d2, "etnaviv_ml.c")
with open(mock_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
regen_path = os.path.join(d2, "regen", "patch-etnaviv-alias-diag.py")
os.makedirs(os.path.dirname(regen_path), exist_ok=True)

gen_mod = load(GEN_PATH, "aliasdiag_gen_run")
gen_mod.SRC = mock_path
gen_mod.OUT = regen_path
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gen_mod.main()
except SystemExit as exc:
    report("GEN  生成器能在同形 mock 上跑通", False,
           "生成器 die()，exit=%s" % exc.code)
else:
    report("GEN  生成器能在同形 mock 上跑通", True, "")
    if not os.path.exists(regen_path):
        report("GEN  生成器产出补丁文件", False, "文件不存在")
    else:
        REGEN = load(regen_path, "aliasdiag_regen")
        keys = ("MARKER", "COMPILE_MARK", "TARGET_FN_REF", "TARGET_FN_REC",
                "STRUCT_REF", "STRUCT_REC", "OLD_REF", "NEW_REF", "OLD_REC",
                "NEW_REC", "ANCHOR_MARK", "NEW_MARK")
        diff = [k for k in keys if getattr(REGEN, k) != getattr(PATCH, k)]
        report("GEN  12 个语义载荷与仓库补丁逐字节一致", not diff,
               "不一致: %s" % diff)
        rcg, _, errg, _ = run_patch(MOCK, patcher_path=regen_path)
        report("GEN  生成器产物在同一 mock 上通过正例", rcg == 0,
               "rc=%s err=%r" % (rcg, errg.strip()[:160]))

print()
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
if WARNS:
    print("RESULT: ALL PASS（含 %d 条 WARN：%s）" % (len(WARNS), "; ".join(WARNS)))
else:
    print("RESULT: ALL PASS")
