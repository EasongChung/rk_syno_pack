#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""per-channel 量化门补丁（weight/bias）的正/反例自测（CI Preflight 用）。

覆盖：
  正例 1
  反例 8：锚点被改 / 函数被改名 / 四连调用缺一项 / input 被误改 / output 被误改 /
          重复应用 / 括号不平衡 / 上游把注释删了
  ★ **关键负控制**：反例「input/output 被误改成宽松判定」必须被拦住 ——
    这是本补丁唯一会**静默算错**的改法（input/output 只读标量 scale）。
  与 FSGUARD 共存（两者都改 etna_ml_operation_supported 附近，必须并存）

⚠️ 这是**功能补丁**（改行为），不是诊断补丁 ⇒ **不适用 print-only 纪律**。
   它的正确性靠：① 锚点逐字唯一 ② STRUCT 计数 ③ 替换后不变量断言
                ④ 反例必须全拦 ⑤ 与其它补丁共存。
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-perchannel-quant.py")
FSGUARD = os.path.join(ROOT, "build-recipe", "patch-etnaviv-filter-shape-guard.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load(PATCH, "perchannel_patch")
G = load(FSGUARD, "fsguard_for_coexist")

FAILS = []
WARNS = []

# ── 合成源码 ────────────────────────────────────────────────────────
# ★ 必须**逐字嵌入 FSGUARD 的 OLD**：它打的也是 etna_ml_operation_supported，
#   锚点结尾正是 `// Dilation and per-axis quantization not yet implemented`
#   —— 就在本补丁要改的四连调用**上面**。第一版我用自己手写的上下文，
#   FSGUARD 找不到锚点 ⇒ 共存用例假红（是 mock 的错，不是补丁的错）。
MOCK = """\
#include <stdint.h>

struct pipe_tensor {
   void    *data;
   unsigned index;
   unsigned dims[4];
   float    scale;
   float   *scales;
   int      zero_point;
   int     *zero_points;
   bool     is_signed;
   uint8_t  type_size;
};

""" + P.OLD_GATE + """
bool
""" + P.FN_USE_INPUT + """
                              const struct pipe_ml_operation *operation)
{
   bool supported = false;
   switch (operation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
         struct pipe_tensor *input_tensor = operation->input_tensors[0];
         struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
         struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
""" + G.OLD + """
   if (""" + P.CALL_OLD + """
       operation->conv.dilation_width_factor == 1) {
      supported = true;
   }
      }
      break;
   default:
      supported = false;
      break;
   }
   return supported;
}
"""


def selfcheck(label, text):
    bad = []
    if text.count("{") != text.count("}"):
        bad.append("括号不配平")
    for fn, struct in ((P.FN_GATE, P.STRUCT_GATE), (P.FN_USE_INPUT, P.STRUCT_USE)):
        b = P._body(text, fn)
        if b is None:
            bad.append("找不到 %s" % fn)
            continue
        for tok, want in struct:
            got = b.count(tok)
            if got != want:
                bad.append("%s 内 %r 实得 %d（期望 %d）" % (fn[-18:], tok[:40], got, want))
    print("  mock 自检(%s): %s" % (label, "PASS" if not bad else "FAIL " + "; ".join(bad)))
    if bad:
        FAILS.append("mock 自检 " + label)
        sys.exit(1)


selfcheck("MOCK", MOCK)


def run_patch(text, patcher_path=None, overrides=None):
    mod = load(patcher_path or PATCH, "perchannel_run")
    for k, v in (overrides or {}).items():
        setattr(mod, k, v)
    d = tempfile.mkdtemp(prefix="perchannel_selftest_")
    p = os.path.join(d, "etnaviv_ml.c")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = ["patch", p]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main()
    except Exception as exc:            # noqa: BLE001
        rc = -1
        err.write("EXCEPTION: %r\n" % (exc,))
    finally:
        sys.argv = saved
    with open(p, encoding="utf-8") as f:
        return rc, err.getvalue(), f.read().replace("\r\n", "\n")


def report(label, ok, detail=""):
    print("  %-56s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def case(label, text, expect_err=None, expect_rc=1, checks=None):
    rc, err, after = run_patch(text)
    detail = []
    if rc != expect_rc:
        detail.append("rc=%s（期望 %s）" % (rc, expect_rc))
    if expect_err is not None and expect_err not in err:
        detail.append("stderr 未含 %r；末行=%r"
                      % (expect_err, (err.strip().splitlines() or [""])[-1]))
    if expect_rc == 0 and checks is not None:
        for item in checks(after, err):
            if isinstance(item, tuple):
                desc, ok = item
            else:
                desc, ok = item, False
            if not ok:
                detail.append(desc)
    report(label, not detail, "; ".join(detail))
    return after


print()
print("=== per-channel 量化门补丁 正/反例自测 ===")

case("GOOD  合成源码", MOCK, expect_rc=0, checks=lambda after, err: [
    ("含标记", P.MARKER in after),
    ("新判定已加入", "tensor_quantization_supported_per_axis" in after),
    ("weight 走宽松判定",
     after.count("tensor_quantization_supported_per_axis(weight_tensor)") == 1),
    ("bias 走宽松判定",
     after.count("tensor_quantization_supported_per_axis(bias_tensor)") == 1),
    ("★ input 仍严格",
     after.count("tensor_quantization_supported(input_tensor)") == 1),
    ("★ output 仍严格",
     after.count("tensor_quantization_supported(output_tensor)") == 1),
    ("旧 gate 纯追加保留 1 处", after.count(P.OLD_GATE) == 1),
    ("括号仍配平", after.count("{") == after.count("}")),
])

# ★ 负控制：把 input/output 也改成宽松 ⇒ 补丁必须拒绝（否则会静默算错）
def mutate_which(old, new):
    """构造一个「上游把 input/output 也用宽松判定」的源码，让锚点不匹配。"""
    return MOCK.replace(old, new, 1)


case("★negA input 被换成宽松判定（必须 FAIL）",
     mutate_which("tensor_quantization_supported(input_tensor)",
                  "tensor_quantization_supported_per_axis(input_tensor)"),
     "出现 0 处")
case("★negB output 被换成宽松判定（必须 FAIL）",
     mutate_which("tensor_quantization_supported(output_tensor)",
                  "tensor_quantization_supported_per_axis(output_tensor)"),
     "出现 0 处")

case("negC gate 的 return 被上游改写", MOCK.replace(
    "return tensor->scales == NULL && tensor->zero_points == NULL;",
    "return tensor->scales == NULL;", 1), "出现 0 处")

case("negD 上游删掉了 per-axis 注释", MOCK.replace(
    "    * Per-axis quantization not supported, for details see:\n", "", 1),
    "出现 0 处")

case("negE 四连调用缺 bias 一项", MOCK.replace(
    "             tensor_quantization_supported(bias_tensor) &&\n", "", 1),
    "出现 0 处")

case("negF 整个 gate 函数被改名", MOCK.replace(
    P.FN_GATE, P.FN_GATE + "_XX", 1), "匹配到 0 处")

case("negG 四连调用被复制一份（同形 2 处）", MOCK + MOCK.replace(
    P.FN_USE_INPUT, P.FN_USE_INPUT + "_dup", 1), "匹配到 2 处")

# 幂等
rc, _e1, after1 = run_patch(MOCK)
rc2, err2, _a2 = run_patch(after1)
report("幂等  同一文件跑两次（第二次必须拒绝）", rc == 0 and rc2 == 1,
       "rc1=%s rc2=%s" % (rc, rc2))
report("幂等  第二次 stderr 含「已包含本补丁」", "已包含本补丁" in err2)

# 括号不平衡
rc3, err3, _ = run_patch(MOCK.replace("   return tensor->scales == NULL && "
                                      "tensor->zero_points == NULL;\n", "   {\n", 1))
report("negH 括号不平衡 ⇒ 拒绝（rc!=0）", rc3 != 0, "rc=%s" % rc3)
if rc3 == -1:
    warn_msg = "_body 括号不平衡会 IndexError（fail-closed 方向对，错误信息退化）"
    print("  %-56s WARN" % warn_msg)
    WARNS.append(warn_msg)

# ── 与 FSGUARD 共存（两者都动 operation_supported 附近）──────────────
print()
print("=== 与 FSGUARD 共存 ===")
d = tempfile.mkdtemp(prefix="perchannel_coexist_")
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(MOCK)


def apply_seq(path, seq):
    for name in seq:
        mod = load({"pc": PATCH, "fg": FSGUARD}[name], "seq_" + name)
        saved = sys.argv
        sys.argv = ["patch", path]
        o, e = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                rc = mod.main()
        finally:
            sys.argv = saved
        if rc != 0:
            return rc, name, e.getvalue().strip()[-160:]
    return 0, None, ""


for seq, label in ((("fg", "pc"), "FSGUARD→perchannel"),
                   (("pc", "fg"), "perchannel→FSGUARD")):
    d2 = tempfile.mkdtemp(prefix="pc_co_")
    p2 = os.path.join(d2, "etnaviv_ml.c")
    shutil.copyfile(os.path.join(d, "etnaviv_ml.c"), p2)
    rc, failed, err = apply_seq(p2, seq)
    if rc != 0:
        report("共存  " + label, False, "%s rc=%s %s" % (failed, rc, err))
        continue
    txt = open(p2, encoding="utf-8").read()
    bad = []
    for tok in (P.MARKER, G.MARKER, "tensor_quantization_supported_per_axis"):
        if tok not in txt:
            bad.append("缺 %r" % tok[:40])
    if txt.count(P.OLD_GATE) != 1:
        bad.append("gate 纯追加不成立")
    if txt.count("{") != txt.count("}"):
        bad.append("括号不配平")
    report("共存  " + label, not bad, "; ".join(bad))

print()
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("RESULT: ALL PASS%s" % ("（含 %d 条 WARN）" % len(WARNS) if WARNS else ""))