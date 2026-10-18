#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROFIX 补丁（read_outputs 映射范围对齐 invoke）本地正/反例自测（CI Preflight 用）。

覆盖：1 正例 + 13 反例 + 幂等 + 括号不平衡(fail-closed) + 生成器一致性。

本轮特有的判据：
  * TRAP  —— 补丁注入的说明注释里**引用了** `pipe_buffer_size(res)`（H2 的证据）。
              所以「注入的**可执行代码**里不得含这个名字」这道自检
              **必须先剥注释**再查；不剥 ⇒ 正例会被自己误拒
              （同一个坑 och 踩过两次，已写进补丁注释）。
              注：`etna_ml_get_size` **没有**出现在注释里（注释引用的是
              `etna_ml_get_tensor`），所以那道自检是纯防回归门 ——
              它「会不会响」由反例 n11 证明，而不是由 TRAP 项证明。
  * n10   —— 把 map 长度真改回 `pipe_buffer_size(res)`（这就是 H2 本体）⇒ 必须拦。
  * n11   —— 把循环上界真改回 `etna_ml_get_size` ⇒ 必须拦。
  * n12   —— 源里注入 RODIAG 标记（两补丁互斥）⇒ 必须拦。
  * n13   —— 括号不平衡 ⇒ `_fn_body` IndexError，方向 fail-closed，但要**记名**
             （不允许把「崩掉」当「干净拒绝」悄悄放过）。

⚠️ 本自测只证明「补丁自身行为 + 与生成器的载荷一致性」，
   不证明真实 Mesa 源码里有这些锚点 —— 后者由构建后运行
   `tools/check-rofix-anchors.py`（对着真源码）负责。

用法： python3 oes-teflon/tools/selftest-rofix.py
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
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")
GEN_PATH = os.path.join(HERE, "gen-rofix-patch.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = load(PATCH_PATH, "rofix_patch")

FAILS = []
WARNS = []

# ── 合成源码：函数体 = 补丁自己的 OLD **逐字**（绝不手写锚点）─────────────
#    头尾只有声明与收尾，且**不得**再含任何 STRUCT token（下面有 mock 自检）
MOCK_HEAD = (
    "/* 合成 Mesa 片段：仅用于校验补丁与生成器的逻辑，不代表真实源码。 */\n"
    "\n"
    "void\n"
    + PATCH.TARGET_FN + " *context, struct pipe_ml_subgraph *psubgraph,\n"
    "                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],\n"
    "                              bool is_signed[])\n"
    "{\n"
    "   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)psubgraph;\n"
    "   (void)context; (void)subgraph;\n"
    "\n"
)
# ⚠️ MOCK_HEAD 只到「函数体开头」，**不含** OLD 也不含收尾 ——
#    否则反例里想拼「函数体内出现两个循环块」时，头部自带的收尾会先把
#    函数截断，括号配平后计数门结论就失真了（自己把自己判红的坑）。
MOCK = MOCK_HEAD + PATCH.OLD + "}\n"

# mock 自检 1：括号必须配平，否则后面所有结论都无意义
if MOCK.count("{") != MOCK.count("}"):
    sys.stderr.write("ERROR: mock 括号不配平（{=%d }=%d）—— 自测自身失效\n"
                     % (MOCK.count("{"), MOCK.count("}")))
    sys.exit(1)
# mock 自检 2：每个 STRUCT token 在 mock 里必须恰好出现 want 次（否则计数门结论失真）
for _tok, _want in PATCH.STRUCT:
    if MOCK.count(_tok) != _want:
        sys.stderr.write("ERROR: mock 里 `%s` 出现 %d 次（期望 %d）—— 自测自身失效\n"
                         % (_tok, MOCK.count(_tok), _want))
        sys.exit(1)


def run_patch(text, patcher_path=None, new_override=None):
    """把 text 写进临时 .c 跑一次补丁器；返回 (rc, out, err, after)。

    rc == -1 表示补丁器**抛异常**（fail-closed，但要在报告里记名）。
    after 已把 CRLF 归一成 LF（Windows 上文本写盘会引入 CRLF，不归一必假红）。
    """
    path_mod = patcher_path or PATCH_PATH
    mod = load(path_mod, "rofix_patch_run")
    if new_override is not None:
        mod.NEW = new_override
    d = tempfile.mkdtemp(prefix="rofix_selftest_")
    p = os.path.join(d, "etnaviv_ml.c")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = ["patch-etnaviv-readout-offset-fix.py", p]
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
    print("  %-52s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def warn(label, detail):
    print("  %-52s WARN" % label)
    for line in detail.split("\n"):
        print("        " + line)
    WARNS.append(label)


def case(label, text, expect_err=None, new_override=None, checks=None,
         expect_rc=1):
    rc, out, err, after = run_patch(text, new_override=new_override)
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


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)


print("=== ROFIX 补丁 正/反例自测（1 正例 + 13 反例 + 幂等 + 生成器）===")

# ── 正例 ──────────────────────────────────────────────────────────────
after, out, err = case(
    "GOOD  合成源码", MOCK, expect_rc=0,
    checks=lambda after, out, err: [
        ("补丁后含标记串（MARKER）", PATCH.MARKER in after),
        ("补丁后含输出前缀 [och-rofix]", "[och-rofix]" in after),
        ("补丁后含 util_dynarray_num_elements", "util_dynarray_num_elements" in after),
        ("补丁后含 tensor->offset 与 tensor->size",
         "tensor->offset" in after and "tensor->size" in after),
        ("NEW 恰好 1 份", after.count(PATCH.NEW) == 1),
        ("OLD 已消失（0 份）", after.count(PATCH.OLD) == 0),
    ])
report("GOOD  补丁器 stdout 含两行 [och-rofix]",
       out.count("  [och-rofix]") == 2, "实得 %d 行" % out.count("  [och-rofix]"))

# ── TRAP：注释里就引用着这两个名字 —— 门必须剥注释后再查 ─────────────
_stripped = strip_comments(PATCH.NEW)
report("TRAP  注入注释里引用了 pipe_buffer_size(res)（这道自检必须先剥注释）",
       'pipe_buffer_size(res)' in PATCH.NEW)
report("TRAP  剥掉注释后 pipe_buffer_size(res) 消失（门不会被自己误触）",
       'pipe_buffer_size(res)' not in _stripped)
report("TRAP  etna_ml_get_size 不在 NEW 的注释里（该自检纯防回归；"
       "由 n11 证明它会响）",
       'etna_ml_get_size' not in PATCH.NEW)

# ── 幂等 ──────────────────────────────────────────────────────────────
d = tempfile.mkdtemp(prefix="rofix_idem_")
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
mod = load(PATCH_PATH, "rofix_patch_idem")
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

# ── 反例 n1..n9（改源码：门必须全部拦住）──────────────────────────────
case("n1  循环头改写（outputs_count -> n_out）",
     MOCK.replace("for (int i = 0; i < outputs_count; i++) {",
                  "for (int i = 0; i < n_out; i++) {"),
     "出现 0 处")

case("n2  assert(src_map) 被删",
     MOCK.replace("assert(src_map);\n", "", 1),
     "出现 0 处")

case("n3  取 resource 语句被改（output_idxs[i] -> [0]）",
     MOCK.replace("etna_ml_get_resource(subgraph, output_idxs[i])",
                  "etna_ml_get_resource(subgraph, output_idxs[0])"),
     "出现 0 处")

case("n4  map 长度行字面被改（pipe_buffer_size(res) + 0）",
     MOCK.replace("0, pipe_buffer_size(res),",
                  "0, pipe_buffer_size(res) + 0,"),
     "出现 0 处")

case("n5  赋值行被改（- 128 -> - 127）",
     MOCK.replace("src_map[k] - 128;", "src_map[k] - 127;"),
     "出现 0 处")

case("n6  else 分支 pipe_buffer_read 被改（outputs[i] -> NULL）",
     MOCK.replace("pipe_buffer_read(context, res, 0, "
                  "etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);",
                  "pipe_buffer_read(context, res, 0, "
                  "etna_ml_get_size(subgraph, output_idxs[i]), NULL);"),
     "出现 0 处")

case("n7  函数整体消失",
     MOCK.replace("etna_ml_subgraph_read_outputs(struct pipe_context",
                  "etna_ml_subgraph_read_outputs_XX(struct pipe_context"),
     "找不到函数")

case("n8  循环块被复制（同形出现 2 处）",
     MOCK_HEAD + PATCH.OLD + PATCH.OLD + "}\n",
     "出现 2 处")

case("n9  锚点同形微变（} else { 变双空格）—— 字节级锚点门独立成立",
     MOCK.replace("      } else {", "      }  else {"),
     "匹配到 0 处")

# ── n10 / n11：真·注入回归（H2 本体与上界回退），门须独立成立 ────────
EVIL_MAP = re.sub(r"res,\n\s+tensor->offset,\n\s+tensor->size,\n",
                  "res,\n                                                     "
                  "0, pipe_buffer_size(res),\n",
                  PATCH.NEW, count=1)
if EVIL_MAP == PATCH.NEW:
    report("n10 注入真·map 长度回退 pipe_buffer_size(res)", False,
           "替换没生效 —— 自测自身失效（NEW 结构与预期不符）")
else:
    case("n10 注入真·map 长度回退 pipe_buffer_size(res)（H2 本体）",
         MOCK, "注入代码里仍有 pipe_buffer_size(res)", new_override=EVIL_MAP)

EVIL_SIZE = PATCH.NEW.replace(
    "for (unsigned k = 0; k < tensor->size; k++) {",
    "for (unsigned k = 0; k < "
    "etna_ml_get_size(subgraph, output_idxs[i]); k++) {", 1)
if EVIL_SIZE == PATCH.NEW:
    report("n11 注入真·循环上界回退 etna_ml_get_size", False,
           "替换没生效 —— 自测自身失效（NEW 结构与预期不符）")
else:
    case("n11 注入真·循环上界回退 etna_ml_get_size",
         MOCK, "注入代码里仍有 etna_ml_get_size", new_override=EVIL_SIZE)

# ── n12：源里已有 RODIAG 标记 ⇒ 互斥，必须拦 ─────────────────────────
case("n12 源里注入 RODIAG 标记（互斥）",
     "/* " + PATCH.DIAG_MARKER + " */\n" + MOCK,
     "已含 RODIAG 取证注入")

# ── n13：括号不平衡 —— fail-closed 但必须记名（不是"干净拒绝"）───────
rc13, _, err13, _ = run_patch(MOCK.replace("   }\n", "   {\n", 1))
report("n13 括号不平衡 ⇒ 拒绝（rc!=0，允许是异常）", rc13 != 0, "rc=%s" % rc13)
if rc13 == -1:
    report("n13 失败方式是异常（IndexError 等），已如实记名",
           "EXCEPTION" in err13)
    warn("n13 备注：_fn_body 对括号不平衡会 IndexError",
         "方向 fail-closed（不会静默放行），但错误信息退化成 traceback；\n"
         "根治办法是给 _fn_body 加 `while j < len(src)` 边界（1 行）。\n"
         "RODIAG 轮的同一备注仍待 och 一并处理（两轮共 2 处）。")

# ── 生成器 ↔ 补丁 一致性 ──────────────────────────────────────────────
print()
print("=== 生成器 ↔ 补丁 一致性（GEN）===")
d2 = tempfile.mkdtemp(prefix="rofix_gen_")
mock_path = os.path.join(d2, "etnaviv_ml.c")
with open(mock_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
regen_path = os.path.join(d2, "regen", "patch-etnaviv-readout-offset-fix.py")
os.makedirs(os.path.dirname(regen_path), exist_ok=True)

gen_mod = load(GEN_PATH, "rofix_gen_run")
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
        REGEN = load(regen_path, "rofix_regen")
        keys = ("MARKER", "DIAG_MARKER", "STRUCT", "OLD", "NEW")
        diff = [k for k in keys if getattr(REGEN, k) != getattr(PATCH, k)]
        report("GEN  5 个语义载荷与仓库补丁逐字节一致", not diff,
               "不一致: %s" % diff)
        rc3, out3, err3, _ = run_patch(MOCK, patcher_path=regen_path)
        # 本轮生成器模板自带 comment-strip ⇒ 不存在 RODIAG 那轮的已知 WARN，
        # 正例被拒一律按硬失败处理。
        report("GEN  生成器产物在同一 mock 上通过正例", rc3 == 0,
               "rc=%s err=%r" % (rc3, err3.strip()[:160]))

print()
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
if WARNS:
    print("RESULT: ALL PASS（含 %d 条 WARN：%s）" % (len(WARNS), "; ".join(WARNS)))
else:
    print("RESULT: ALL PASS")
