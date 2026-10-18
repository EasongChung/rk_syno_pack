#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CREATEDIAG 补丁（create/destroy_tensor 分配取证）本地正/反例自测（CI Preflight 用）。

覆盖：1 正例 + 9 反例 + 幂等 + **三诊断共存（ROFIX + ALIASDIAG + CREATEDIAG）**
      + 生成器一致性 + **打印纪律门（print-only）**。

## 打印纪律门（本轮新增的纪律，也是本轮唯一的红项）

「诊断补丁只加打印」是一条硬纪律：把注释与 `ML_DBG` 语句剥掉后，
注入前后的**可执行代码必须逐字相同**。理由不是洁癖：

* 诊断轮的产物要用来解释**交付候选**的行为。一旦诊断补丁自己改了行为，
  设备上跑出来的现象就不能直接归因到交付候选。
* 本轮实测就撞上了：`NEW_CREATE` 把 `if (res != NULL)` 改成
  `if (res != NULL && tensor->size == size)` 并新增一条**静默重分配**路径。
  后果是：真源码里那条 `assert(size == tensor->size)`（在关 assert 的构建里
  是「静默早退」，在我们的构建里是**当场 abort**）被绕过了 ——
  恰好把本轮要取证的那条假说所依赖的可观测信号给消掉了。

⚠️ 这道门**只适用于声明为纯打印的补丁**。ALIASDIAG 不适用：
它为打印「迁移明细」必须新增一个只读 for 循环（那一处我逐行核过是只读的）。

⚠️ 本自测只证明「补丁自身行为 + 共存 + 与生成器的载荷一致性」；
   真实锚点由构建后的 tools/check-creatediag-anchors.py 负责。

用法： python3 oes-teflon/tools/selftest-creatediag.py
"""

import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P_CREATE = os.path.join(ROOT, "build-recipe", "patch-etnaviv-create-diag.py")
P_ALIAS = os.path.join(ROOT, "build-recipe", "patch-etnaviv-alias-diag.py")
P_ROFIX = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")
G_PATH = os.path.join(HERE, "gen-creatediag-patch.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load(P_CREATE, "creatediag_patch")
A = load(P_ALIAS, "aliasdiag_patch_for_coexist")
R = load(P_ROFIX, "rofix_patch_for_coexist")

FAILS = []
WARNS = []

STUB_IO = "#include <stdio.h>\n"

# ⚠️ v2 起 OLD_CREATE **本身就是完整函数**（从签名到收尾的 `}`）——
#    v1 只截到 `return;`，需要外面再套一层壳。两种形态都会踩到「mock 括号不配平」，
#    所以这里先判形态再拼，别写死。
CREATE_IS_WHOLE_FN = P.OLD_CREATE.rstrip().endswith("}") and P.FN_CREATE in P.OLD_CREATE
if CREATE_IS_WHOLE_FN:
    # 原生函数已经自带签名/花括号/结尾换行，直接用；后面**不要**再补 `}`
    CREATE_SRC = P.OLD_CREATE
    CREATE_TAIL = ""
else:
    CREATE_SRC = (
        "void\n"
        + P.FN_CREATE + " *subgraph, unsigned idx, unsigned size)\n"
        "{\n"
        "   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);\n"
        "\n"
        "   assert(idx < util_dynarray_num_elements(&subgraph->tensors, "
        "struct etna_ml_tensor *));\n"
        "   assert(size > 0);\n"
        "\n"
        + P.OLD_CREATE
        + "   }\n"
    )
    CREATE_TAIL = ""
if CREATE_IS_WHOLE_FN:
    CREATE_TAIL = (
        "   }\n"
        "\n"
        "   res = etna_ml_create_resource(&subgraph->screen->base, size);\n"
        "   tensor->resource = res;\n"
        "   tensor->size = size;\n"
        "\n"
        '   ML_DBG("created resource %p for tensor %d with size %d\\n", res, idx, size);\n'
        "}\n")
    CREATE_SRC = P.OLD_CREATE
DESTROY_SRC = (
    "static void\n"
    + P.FN_DESTROY + " *subgraph, unsigned idx)\n"
    "{\n"
    "   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);\n"
    "\n"
    + P.OLD_DESTROY +
    "   tensor->offset = 0;\n"
    "   tensor->size = 0;\n"
    "}\n"
)
ALIAS_SRC = (
    "static void\n"
    + A.TARGET_FN_REF
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
    + A.OLD_REF +
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
    + A.TARGET_FN_REC + " *subgraph, unsigned idx, unsigned size)\n"
    "{\n"
    "   struct pipe_resource *old_res = etna_ml_get_resource(subgraph, idx);\n"
    "   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);\n"
    "   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);\n"
    + A.OLD_REC +
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
    + R.TARGET_FN + " *context, struct pipe_ml_subgraph *psubgraph,\n"
    "                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],\n"
    "                              bool is_signed[])\n"
    "{\n"
    "   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)psubgraph;\n"
    "   (void)context; (void)subgraph;\n"
    "\n"
    + R.OLD +
    "}\n"
)

MOCK = STUB_IO + CREATE_SRC + DESTROY_SRC
MOCK_ALL = STUB_IO + CREATE_SRC + DESTROY_SRC + ALIAS_SRC + RO_SRC


def strip_code(text):
    """去注释 + 去 ML_DBG 语句 + 去空行，只留可执行代码（打印纪律门用）。"""
    t = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r"ML_DBG\s*\(.*?\)\s*;", "", t, flags=re.S)
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def selfcheck(label, text, patcher, extra=()):
    bad = []
    if text.count("{") != text.count("}"):
        bad.append("括号不配平")
    for fn, struct in ((patcher.FN_CREATE, patcher.STRUCT_CREATE),
                       (patcher.FN_DESTROY, patcher.STRUCT_DESTROY)):
        body = patcher._body(text, fn)
        if body is None:
            bad.append("找不到 %s" % fn)
            continue
        for tok, want in struct:
            got = body.count(tok)
            if got != want:
                bad.append("%s 内 %r 实得 %d（期望 %d）" % (fn[-12:], tok[:40], got, want))
    for nm, old in (("OLD_CREATE", patcher.OLD_CREATE), ("OLD_DESTROY", patcher.OLD_DESTROY),
                    ("ANCHOR_MARK", patcher.ANCHOR_MARK)):
        got = text.count(old)
        if got != 1:
            bad.append("%s 在 mock 里 %d 处（期望 1）" % (nm, got))
    for mod_, fn, struct in extra:
        # ROFIX 只有单函数，用 _fn_body(src)；其余用 _body(src, fn)
        body = (mod_._fn_body(text) if hasattr(mod_, "_fn_body")
                else mod_._body(text, fn))
        if body is None:
            bad.append("共存体找不到 %s" % fn[:40])
            continue
        for tok, want in struct:
            got = body.count(tok)
            if got != want:
                bad.append("共存体 %r 实得 %d（期望 %d）" % (tok[:40], got, want))
    print("  mock 自检(%s): %s" % (label, "PASS" if not bad else "FAIL " + "; ".join(bad)))
    if bad:
        FAILS.append("mock 自检 " + label)
        sys.exit(1)


selfcheck("MOCK", MOCK, P)
selfcheck("MOCK_ALL", MOCK_ALL, P,
          extra=((A, A.TARGET_FN_REF, A.STRUCT_REF), (R, R.TARGET_FN, R.STRUCT)))


def run_patch(text, patcher_path=None, overrides=None):
    mod = load(patcher_path or P_CREATE, "creatediag_run")
    for k, v in (overrides or {}).items():
        setattr(mod, k, v)
    d = tempfile.mkdtemp(prefix="creatediag_selftest_")
    p = os.path.join(d, "etnaviv_ml.c")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = [os.path.basename(patcher_path or P_CREATE), p]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main()
    except Exception as exc:            # noqa: BLE001 —— fail-closed 要记名
        rc = -1
        err.write("EXCEPTION: %r\n" % (exc,))
    finally:
        sys.argv = saved
    with open(p, encoding="utf-8") as f:
        after = f.read().replace("\r\n", "\n")
    return rc, out.getvalue(), err.getvalue(), after


def report(label, ok, detail=""):
    print("  %-56s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def warn(label, detail):
    print("  %-56s WARN" % label)
    for line in detail.split("\n"):
        print("        " + line)
    WARNS.append(label)


def case(label, text, expect_err=None, expect_rc=1, overrides=None, checks=None):
    rc, out, err, after = run_patch(text, overrides=overrides)
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
    return after


print()
print("=== CREATEDIAG 补丁 正/反例自测 ===")

after = case(
    "GOOD  合成源码", MOCK, expect_rc=0,
    checks=lambda after, out, err: [
        ("含 MARKER", P.MARKER in after),
        ("含 [och-create]", "[och-create]" in after),
        ("含 EARLY-RETURN 分支", "EARLY-RETURN" in after),
        ("含编译期标记 %r" % P.COMPILE_MARK, P.COMPILE_MARK in after),
        ("NEW_CREATE 恰好 1 份", after.count(P.NEW_CREATE) == 1),
        ("NEW_DESTROY 恰好 1 份", after.count(P.NEW_DESTROY) == 1),
        ("NEW_MARK 恰好 1 份", after.count(P.NEW_MARK) == 1),
        ("【口径 create】OLD_CREATE 已消失（中间插入型）", after.count(P.OLD_CREATE) == 0),
        ("【口径 destroy】OLD_DESTROY 仍恰好 1（只追加型）", after.count(P.OLD_DESTROY) == 1),
        ("【口径 mark】ANCHOR_MARK 仍恰好 1（只追加型）",
         after.count(P.ANCHOR_MARK) == 1),
        ("补丁后括号仍配平", after.count("{") == after.count("}")),
    ])

# ── ★ 打印纪律门（本轮新增的纪律）────────────────────────────────────
same_create = strip_code(P.OLD_CREATE) == strip_code(P.NEW_CREATE)
same_destroy = strip_code(P.OLD_DESTROY) == strip_code(P.NEW_DESTROY)
if same_create and same_destroy:
    report("纪律  注入是纯打印（可执行代码逐字未变）", True)
else:
    report("纪律  注入是纯打印（可执行代码逐字未变）", False,
           "create=%s destroy=%s" % (same_create, same_destroy))
    warn("纪律门红项：诊断补丁改了可执行代码",
         "create_tensor 注入把 `if (res != NULL)` 改成\n"
         "  `if (res != NULL && tensor->size == size)` 并新增一条**静默重分配**路径，\n"
         "同时把真源码里那条 `assert(size == tensor->size)` 变成不可达死代码。\n"
         "后果（已逐路径追过）：\n"
         "  · res == NULL（正常路径）：行为与上游一致，只是多打了两行 ⇒ 无害；\n"
         "  · res != NULL 且 tensor->size != size（**本轮假说的核心**）：\n"
         "    上游 = assert 当场 abort（我们的构建 assert 是开的，ROFIX 的真机 abort 就是证据）；\n"
         "    本补丁 = 静默重新分配 + 覆盖指针（旧 resource 泄漏）⇒ **可观测信号被消掉**。\n"
         "最小修法（不动控制流，只在 if 内加一行打印）：\n"
         "  if (res != NULL) {\n"
         "     ML_DBG(\"[och-create]   EARLY-RETURN idx=%u req_size=%u tensor_size=%u res=%p actual=%u %s\\n\",\n"
         "            idx, size, tensor->size, (void *)res, pipe_buffer_size(res),\n"
         "            (tensor->size == size) ? \"<size 一致>\" : \"*** size 不一致！***\");\n"
         "     assert(size == tensor->size);\n"
         "     return;\n"
         "  }\n"
         "（`created` 那行打印建议改锚到 `   tensor->size = size;` 之后，\n"
         "  这样连重发射都不需要，OLD/NEW 的可执行代码完全一致。）")

# ── 幂等 ────────────────────────────────────────────────────────────
d = tempfile.mkdtemp(prefix="creatediag_idem_")
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
mod = load(P_CREATE, "creatediag_idem")
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

# ── 反例 n1..n9 ─────────────────────────────────────────────────────
case("n1  create 的 if (res != NULL) { 被改写",
     MOCK.replace("   if (res != NULL) {\n      assert(size == tensor->size);",
                  "   if (res != NULL && tensor->size == size) {\n"
                  "      assert(size == tensor->size);", 1),
     "出现 0 处")
case("n2  create 的 assert 被删",
     MOCK.replace("      assert(size == tensor->size);\n", "", 1),
     "出现 0 处")
case("n3  create 的取 resource 行被改（漏 struct 前缀的坑）",
     MOCK.replace("   struct pipe_resource *res = tensor->resource;",
                  "   pipe_resource *res = tensor->resource;", 1),
     "出现 0 处")
case("n4  create_resource 调用行被改",
     MOCK.replace("res = etna_ml_create_resource(&subgraph->screen->base, size);",
                  "res = etna_ml_create_resource(&subgraph->screen->base, size + 0);", 1),
     "出现 0 处")
case("n5  tensor->size = size; 被删（create 体内那处）",
     MOCK.replace("   tensor->size = size;\n\n   ML_DBG(\"created resource",
                  "\n   ML_DBG(\"created resource", 1),
     "出现 0 处")
case("n6  destroy 函数消失",
     MOCK.replace(P.FN_DESTROY.split("(")[0], P.FN_DESTROY.split("(")[0] + "_XX", 1),
     "找不到函数")
# v2 起 OLD_CREATE 是**整函数**锚点（含签名），所以「同形副本」必须是**逐字节相同**的
# 副本才撞车；只改签名的副本反而撞不上（那是更强的性质，不是漏洞）。
# —— 第一版我把副本签名改名，反例因此作废、门判绿是对的。已踩一次。
DUP = CREATE_SRC if CREATE_IS_WHOLE_FN else CREATE_SRC.replace(
    P.FN_CREATE, P.FN_CREATE + "_dup", 1)
case("n7  存在两个逐字节相同的 create 函数（文件级计数门）", MOCK + DUP,
     "锚点匹配到 2 处")
case("n7b create 函数被改名（整函数锚点应找不到）",
     MOCK.replace(P.FN_CREATE.split("(")[0], P.FN_CREATE.split("(")[0] + "_XX", 1),
     "找不到函数")
case("n8  #include <stdio.h> 出现两次（标记锚点撞车）",
     MOCK.replace(STUB_IO, STUB_IO + STUB_IO, 1), "锚点匹配到 2 处")
rc9, _, err9, _ = run_patch(MOCK.replace("   }\n", "   {\n", 1))
report("n9 括号不平衡 ⇒ 拒绝（rc!=0，允许异常）", rc9 != 0, "rc=%s" % rc9)
if rc9 == -1:
    report("n9 拒绝方式是异常（IndexError 等），已记名", "EXCEPTION" in err9)
    warn("n9 备注：_body 括号不平衡会 IndexError",
         "方向 fail-closed；根治 = 给 _body 加 `while j < len(src)`（1 行）。\n"
         "这是同一处 nit 的**第四次**出现（RODIAG/ROFIX/ALIASDIAG/CREATEDIAG）。")

# ── 三诊断共存 ──────────────────────────────────────────────────────
print()
print("=== 三诊断共存（ROFIX + ALIASDIAG + CREATEDIAG）===")


def run_seq(text, seq):
    for name in seq:
        path = {"create": P_CREATE, "alias": P_ALIAS, "rofix": P_ROFIX}[name]
        rc, _o, err, text = run_patch(text, patcher_path=path)
        if rc != 0:
            return None, (name, rc, err.strip()[-140:])
    return text, None


def run_seq_override(text, seq, new_mark=None):
    """按 seq 打补丁，并可覆盖 CREATEDIAG 的 NEW_MARK（用于构造「旧形态」负控制）。"""
    for name in seq:
        path = {"create": P_CREATE, "alias": P_ALIAS, "rofix": P_ROFIX}[name]
        ov = {"NEW_MARK": new_mark} if (name == "create" and new_mark) else None
        rc, _o, err, text = run_patch(text, patcher_path=path, overrides=ov)
        if rc != 0:
            return None, (name, rc, err.strip()[-140:])
    return text, None


def coexist_problems(text):
    """共存门：除了 marker 在场，还必须断言**每个补丁器自己的 NEW_* / ANCHOR_MARK
    仍是连续文本且各恰好 1 处** —— 那正是各自构建后锚点门依赖的不变量。

    ⚠️ 第一版共存门只查「marker 字符串在不在 / 某处 NEW 是不是 1 份」，
       于是 run 39145972254 的真实失败（ALIASDIAG 的 NEW_MARK 被 CREATEDIAG
       从中间劈开 ⇒ **它的**构建后锚点门红）在我这里**显示为绿**。
       教训：共存门必须断言每个补丁器**自己的**锚点不变量，而不是「打进去了没有」。
    """
    bad = []
    for nm, blocks, anchor in (("CREATEDIAG",
                                (P.NEW_CREATE, P.NEW_DESTROY, P.NEW_MARK), P.ANCHOR_MARK),
                               ("ALIASDIAG", (A.NEW_REF, A.NEW_REC, A.NEW_MARK),
                                A.ANCHOR_MARK),
                               ("ROFIX", (R.NEW,), None)):
        for i, blk in enumerate(blocks):
            n = text.count(blk)
            if n != 1:
                bad.append("%s 的第 %d 个 NEW_* 命中 %d（须恰好 1：块被别的补丁劈开了）"
                           % (nm, i + 1, n))
        if anchor is not None and text.count(anchor) != 1:
            bad.append("%s 的 ANCHOR_MARK 命中 %d（须 1）" % (nm, text.count(anchor)))
    if text.count("{") != text.count("}"):
        bad.append("括号不配平")
    return bad


for seq, label in ((("create", "alias", "rofix"), "CREATEDIAG→ALIASDIAG→ROFIX"),
                   (("rofix", "alias", "create"), "ROFIX→ALIASDIAG→CREATEDIAG"),
                   (("rofix", "create", "alias"), "ROFIX→CREATEDIAG→ALIASDIAG")):
    text, failinfo = run_seq(MOCK_ALL, seq)
    if failinfo:
        report("共存  " + label, False, "%s rc=%s %s" % failinfo)
        continue
    bad = []
    for m in (P.MARKER, A.MARKER, R.MARKER):
        if m not in text:
            bad.append("缺 marker %r" % m[:24])
    for c in (P.COMPILE_MARK, A.COMPILE_MARK):
        if c not in text:
            bad.append("缺编译期标记 %r" % c)
    bad += coexist_problems(text)
    report("共存  " + label, not bad, "; ".join(bad))

# ── 共存门的负控制：把 NEW_MARK 换回「插在锚点之后」的旧形态 ──────────
# 这一步证明共存门**真能**抓到 run 39145972254 那类劈开；否则上面的「全绿」又是假绿。
print()
if P.NEW_MARK.endswith(P.ANCHOR_MARK):
    INJ = P.NEW_MARK[:-len(P.ANCHOR_MARK)]
    OLD_MARK = P.ANCHOR_MARK + INJ                    # 旧形态：锚点在前、块在后
    after_old, fail_old = run_seq_override(MOCK_ALL, ("rofix", "alias", "create"),
                                           new_mark=OLD_MARK)
    n_alias = after_old.count(A.NEW_MARK) if after_old else -1
    report("负控  旧形态共存必须被判红（否则共存门是假绿）",
           bool(after_old) and n_alias != 1,
           "打补丁 %s；ALIASDIAG.NEW_MARK 命中 %d（应 !=1）" % (fail_old or "成功", n_alias))
    if after_old and n_alias != 1:
        print("        ⇒ 旧形态下 ALIASDIAG 的构建后锚点门必红 —— 这就是 run 39145972254 的成因")
else:
    report("负控  旧形态共存必须被判红", False, "NEW_MARK 不以 ANCHOR_MARK 结尾，无法构造旧形态")

# ── 生成器一致性 ────────────────────────────────────────────────────
print()
print("=== 生成器 ↔ 补丁 一致性（GEN）===")
d2 = tempfile.mkdtemp(prefix="creatediag_gen_")
mock_path = os.path.join(d2, "etnaviv_ml.c")
with open(mock_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
regen = os.path.join(d2, "regen", "patch-etnaviv-create-diag.py")
os.makedirs(os.path.dirname(regen), exist_ok=True)
gm = load(G_PATH, "creatediag_gen_run")
gm.SRC = mock_path
gm.OUT = regen
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gm.main()
except SystemExit as exc:
    report("GEN  生成器能在同形 mock 上跑通", False, "die() exit=%s" % exc.code)
else:
    report("GEN  生成器能在同形 mock 上跑通", True, "")
    if not os.path.exists(regen):
        report("GEN  生成器产出补丁文件", False, "文件不存在")
    else:
        RG = load(regen, "creatediag_regen")
        keys = ("MARKER", "COMPILE_MARK", "FN_CREATE", "FN_DESTROY", "STRUCT_CREATE",
                "STRUCT_DESTROY", "OLD_CREATE", "NEW_CREATE", "OLD_DESTROY",
                "NEW_DESTROY", "ANCHOR_MARK", "NEW_MARK")
        diff = [k for k in keys if getattr(RG, k) != getattr(P, k)]
        report("GEN  12 个语义载荷与仓库补丁逐字节一致", not diff, "不一致: %s" % diff)
        rcg, _, errg, _ = run_patch(MOCK, patcher_path=regen)
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
