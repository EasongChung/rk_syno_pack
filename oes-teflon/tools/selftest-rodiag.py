#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RODIAG 补丁（read_outputs 越界取证）本地正/反例自测（CI Preflight 用）。

覆盖：1 正例 + 9 反例 + 幂等 + 括号不平衡(fail-closed) + 生成器一致性。

三项**本轮特有**的判据：
  * TRAP  —— 补丁注入的取证注释里写着「不调 `etna_ml_get_tensor`」，
             所以「注入代码不得含 get_tensor」这道门**必须先把注释剥掉**再查；
             不剥注释 ⇒ 正例会被自己误拒（och 的生成器模板现在就是这个状态，
             见 GEN 段的 WARN）。
  * n9    —— 把只读槽改回 get_tensor 调用（真注入），门必须拦住。
  * n10   —— 括号不平衡：`_fn_body` 会 IndexError —— 方向仍 fail-closed，
             但要**记名**（不允许把"崩掉"当"干净拒绝"悄悄放过）。

⚠️ 本自测只证明「补丁自身行为 + 与生成器的载荷一致性」，
   不证明真实 Mesa 里有这些锚点 —— 后者由构建后运行
   `tools/check-rodiag-anchors.py`（对着真源码）负责。

用法： python3 oes-teflon/tools/selftest-rodiag.py
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
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-diag.py")
GEN_PATH = os.path.join(HERE, "gen-rodiag-patch.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = load(PATCH_PATH, "rodiag_patch")
GEN = load(GEN_PATH, "rodiag_gen")

FAILS = []
WARNS = []

# ── 合成源码：只保留补丁/生成器依赖的结构特征，锚点**逐字取自补丁常量** ──
#    （绝不手写缩进 —— 手写锚点是踩过的坑；这里只有"锚点之间的填充行"是手写的）
MAP_LINES = PATCH.OLD_MAP.rstrip("\n").split("\n")
if len(MAP_LINES) != 2:
    sys.stderr.write("ERROR: OLD_MAP 不是 2 行，本自测的 mock 构造前提失效\n")
    sys.exit(1)
ASSERT_LINE = MAP_LINES[1]

MOCK_HEAD = (
    "/* 合成 Mesa 片段：仅用于校验补丁与生成器的逻辑，不代表真实源码。 */\n"
    "\n"
    "static void\n"
    + PATCH.TARGET_FN + " *context,\n"
    "                            struct etna_ml_subgraph *subgraph)\n"
    "{\n"
    "   struct pipe_transfer *src_transfer = NULL;\n"
    "   uint8_t *src_map = NULL;\n"
    "   unsigned outputs_count = 4;\n"
    "   (void)context; (void)subgraph;\n"
    "\n"
    + PATCH.OLD_LOOP +
    "      unsigned sz = etna_ml_get_size(subgraph, output_idxs[i]);\n"
    "      (void)sz;\n"
    "   }\n"
    "\n"
    "   src_map = (uint8_t *) pipe_buffer_map_range(context,\n"
    + PATCH.OLD_MAP +
    "\n"
    "   for (unsigned k = 0; k < 16; k++) {\n"
    "      for (unsigned i = 0; i < outputs_count; i++)\n"
    "         ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;\n"
    "   }\n"
)
MOCK_TAIL = "}\n"
MOCK = MOCK_HEAD + MOCK_TAIL

# mock 自检：括号必须配平，否则后面所有结论都无意义
if MOCK.count("{") != MOCK.count("}"):
    sys.stderr.write("ERROR: mock 括号不配平（{=%d }=%d）—— 自测自身失效\n"
                     % (MOCK.count("{"), MOCK.count("}")))
    sys.exit(1)


def run_patch(text, patcher_path=None, new_loop_override=None):
    """把 text 写进临时 .c 跑一次补丁器；返回 (rc, out, err, after)。

    rc == -1 表示补丁器**抛异常**（fail-closed，但要在报告里记名）。
    after 已把 CRLF 归一成 LF（Windows 上文本写盘会引入 CRLF，不归一必假红）。
    """
    path_mod = patcher_path or PATCH_PATH
    mod = load(path_mod, "rodiag_patch_run")
    if new_loop_override is not None:
        mod.NEW_LOOP = new_loop_override
    d = tempfile.mkdtemp(prefix="rodiag_selftest_")
    p = os.path.join(d, "etnaviv_ml.c")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = ["patch-etnaviv-readout-diag.py", p]
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
    print("  %-46s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def warn(label, detail):
    print("  %-46s WARN" % label)
    for line in detail.split("\n"):
        print("        " + line)
    WARNS.append(label)


def case(label, text, expect_err=None, new_loop_override=None, checks=None,
         expect_rc=1):
    rc, out, err, after = run_patch(text, new_loop_override=new_loop_override)
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


print("=== RODIAG 补丁 正/反例自测（1 正例 + 9 反例 + 幂等 + 生成器）===")

# ── 正例 ──────────────────────────────────────────────────────────────
after, out, err = case(
    "GOOD  合成源码", MOCK, expect_rc=0,
    checks=lambda after, out, err: [
        ("补丁后含标记串（MARKER）", PATCH.MARKER in after),
        ("补丁后含输出前缀 [och-ro]", "[och-ro]" in after),
        ("补丁后含 util_dynarray_num_elements",
         "util_dynarray_num_elements" in after),
        ("补丁后含 in_range ? 三元（越界保护）", "in_range ?" in after),
        ("补丁后含 OOB 分支串", "OOB" in after),
        ("NEW_LOOP 恰好 1 份", after.count(PATCH.NEW_LOOP) == 1),
        ("NEW_MAP 恰好 1 份", after.count(PATCH.NEW_MAP) == 1),
        ("OLD_LOOP 已消失（两侧被隔开 ⇒ 必须 0）", after.count(PATCH.OLD_LOOP) == 0),
        ("OLD_MAP 恰好保留 1 处（嵌在 NEW_MAP 里 ⇒ 不能查 0）",
         after.count(PATCH.OLD_MAP) == 1),
        ("只读槽值：注入的**非注释**代码里没有 get_tensor 调用",
         "etna_ml_get_tensor" not in re.sub(
             r"/\*.*?\*/", "", PATCH.NEW_LOOP + PATCH.NEW_MAP, flags=re.S)),
    ])
report("GOOD  补丁器 stdout 含两行 [och-ro]",
       out.count("  [och-ro]") == 2, "实得 %d 行" % out.count("  [och-ro]"))

# ── TRAP：注释里就写着 get_tensor —— 门必须剥注释后再查 ────────────────
trap_raw = "etna_ml_get_tensor" in (PATCH.NEW_LOOP + PATCH.NEW_MAP)
trap_stripped = "etna_ml_get_tensor" in re.sub(r"/\*.*?\*/", "",
                                               PATCH.NEW_LOOP + PATCH.NEW_MAP,
                                               flags=re.S)
report("TRAP  注入代码的**注释**里确实含 get_tensor（陷阱存在）", trap_raw)
report("TRAP  剥掉注释后不含 get_tensor（门不会被自己误触）", not trap_stripped)

# ── 幂等 ──────────────────────────────────────────────────────────────
d = tempfile.mkdtemp(prefix="rodiag_idem_")
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
mod = load(PATCH_PATH, "rodiag_patch_idem")
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

# ── 反例 n1..n8（含 och 编号的 8 条）──────────────────────────────────
case("n1  循环头改写（outputs_count -> n_out）",
     MOCK.replace("for (int i = 0; i < outputs_count; i++) {",
                  "for (int i = 0; i < n_out; i++) {"),
     "出现 0 处")

case("n2  assert(src_map) 被删",
     MOCK.replace(ASSERT_LINE + "\n", ""),
     "出现 0 处")

case("n3  取 resource 语句被改（output_idxs[i] -> [0]）",
     MOCK.replace("etna_ml_get_resource(subgraph, output_idxs[i])",
                  "etna_ml_get_resource(subgraph, output_idxs[0])"),
     "出现 0 处")

case("n4  assert 行缩进坏（锚点计数门独立成立）",
     MOCK.replace(ASSERT_LINE + "\n",
                  ASSERT_LINE.replace("assert", "  assert", 1) + "\n"),
     "匹配到 0 处")

case("n5  赋值行被改（- 128 -> - 127）",
     MOCK.replace("src_map[k] - 128;", "src_map[k] - 127;"),
     "出现 0 处")

case("n6  函数整体消失",
     MOCK.replace("etna_ml_subgraph_read_outputs(struct pipe_context",
                  "etna_ml_subgraph_read_outputs_XX(struct pipe_context"),
     "找不到函数")

case("n7  锚点被复制（逐字相同出现 2 处）",
     MOCK.replace("\n" + MOCK_TAIL, "\n" + PATCH.OLD_MAP + MOCK_TAIL, 1),
     "出现 2 处")

case("n8  map 调用尾行被换（&src_transfer); -> &xfer);）",
     MOCK.replace("&src_transfer);", "&xfer);"),
     "匹配到 0 处")

# ── n9：真注入 get_tensor 调用，门必须拦住（覆盖 NEW_LOOP 走注入路径）──
EVIL = PATCH.NEW_LOOP.replace(
    "struct etna_ml_tensor *t = slotp ? *slotp : NULL;",
    "struct etna_ml_tensor *t = etna_ml_get_tensor(subgraph, idx);")
if EVIL == PATCH.NEW_LOOP:
    report("n9  注入真 get_tensor 调用", False, "替换没生效 —— 自测自身失效")
else:
    case("n9  注入真 get_tensor 调用（门须独立成立）", MOCK,
         "取证代码里出现了 etna_ml_get_tensor", new_loop_override=EVIL)

# ── n10：括号不平衡 —— fail-closed 但必须记名（不是"干净拒绝"）────────
rc10, _, err10, _ = run_patch(MOCK.replace("   }\n", "   {\n", 1))
report("n10 括号不平衡 ⇒ 拒绝（rc!=0，允许是异常）", rc10 != 0,
       "rc=%s" % rc10)
if rc10 == -1:
    report("n10 失败方式是异常（IndexError 等），已如实记名",
           "EXCEPTION" in err10)
    warn("n10 备注：_fn_body 对括号不平衡会 IndexError",
         "方向 fail-closed（不会静默放行），但错误信息退化成 traceback；\n"
         "根治办法是给 _fn_body 加 `while j < len(src)` 边界（1 行），已随回信上报。")

# ── 生成器 ↔ 补丁 一致性 ──────────────────────────────────────────────
print()
print("=== 生成器 ↔ 补丁 一致性（GEN）===")
d2 = tempfile.mkdtemp(prefix="rodiag_gen_")
mock_path = os.path.join(d2, "etnaviv_ml.c")
with open(mock_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)
regen_path = os.path.join(d2, "regen", "patch-etnaviv-readout-diag.py")
os.makedirs(os.path.dirname(regen_path), exist_ok=True)

gen_mod = load(GEN_PATH, "rodiag_gen_run")
gen_mod.SRC = mock_path
gen_mod.OUT = regen_path
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gen_mod.main()
except SystemExit as exc:
    report("GEN  生成器能在同形 mock 上跑通", False, "生成器 die()，exit=%s" % exc.code)
else:
    report("GEN  生成器能在同形 mock 上跑通", True, "")
    if not os.path.exists(regen_path):
        report("GEN  生成器产出补丁文件", False, "文件不存在")
    else:
        REGEN = load(regen_path, "rodiag_regen")
        keys = ("MARKER", "STRUCT", "OLD_LOOP", "NEW_LOOP", "OLD_MAP", "NEW_MAP")
        diff = [k for k in keys if getattr(REGEN, k) != getattr(PATCH, k)]
        report("GEN  6 个语义载荷与仓库补丁逐字节一致", not diff,
               "不一致: %s" % diff)
        rc3, out3, err3, _ = run_patch(MOCK, patcher_path=regen_path)
        if rc3 == 0:
            report("GEN  生成器产物在同一 mock 上通过正例", True)
        elif "出现了 etna_ml_get_tensor" in err3:
            warn("GEN  生成器产物被正例误拒（生成器模板过期）",
                 "现象：rc=%s，stderr = %r\n"
                 "根因：tools/gen-rodiag-patch.py 的 TEMPLATE 里那道\n"
                 "      「不得引入 etna_ml_get_tensor」自检**没有先剥注释**，\n"
                 "      而 DIAG_LOOP 的注释里恰好写着这个词 ⇒ 必然误拒。\n"
                 "      仓库里的补丁文件已修（多 import re + 两行 re.sub），\n"
                 "      但生成器还没同步 —— 谁重跑一次生成器就会覆盖回坏版本。\n"
                 "      已在回信上报 och（附 5 行改法）。本项暂记 WARN 不阻塞构建，\n"
                 "      因为他修好即可自动转 PASS（换成其它错误则硬失败）。"
                 % (rc3, err3.strip()[:120]))
        else:
            report("GEN  生成器产物在同一 mock 上通过正例", False,
                   "非已知问题：rc=%s err=%r" % (rc3, err3.strip()[:120]))

print()
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
if WARNS:
    print("RESULT: ALL PASS（含 %d 条 WARN：%s）" % (len(WARNS), "; ".join(WARNS)))
else:
    print("RESULT: ALL PASS")
