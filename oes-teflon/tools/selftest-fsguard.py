#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FSGUARD 补丁 + 锚点生成器 的本地正/反例自测（CI Preflight 用）。

覆盖 och 编号的 1 正例 + 7 反例 + 幂等，另加两项：
  * n8  —— 把守卫出口从 `return false;` 换成 `break;`，验「守卫块内必须有
            return false」这道门**独立**成立（不靠文件里别处的 return false 蒙过）。
  * GEN —— 用 OLD 构造同形 mock，跑生成器，与仓库里的补丁文件**逐字节**比对，
            捕捉「补丁文件被手工改过 / 生成器改了但没重新生成」。

⚠️ 本自测**只证明补丁与生成器的内部一致性**，不证明真实 Mesa 里存在该锚点 ——
   后者只能由真实源码那次运行（生成器 / 补丁器自身）证明。

用法： python3 oes-teflon/tools/selftest-fsguard.py
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_PATH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-filter-shape-guard.py")
GEN_PATH = os.path.join(HERE, "gen-fsguard-patch.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCH = load(PATCH_PATH, "fsguard_patch")
GEN = load(GEN_PATH, "fsguard_gen")

ANCHOR = PATCH.OLD.rstrip("\n").split("\n")
if len(ANCHOR) != 3:
    sys.stderr.write("ERROR: 补丁 OLD 不是 3 行，本自测的 mock 构造前提失效\n")
    sys.exit(1)

# 合成源码：只保留补丁/生成器依赖的结构特征；函数尾部形态与真实源码一致
# （`default: return false;` + 尾部 `return supported;`）。
MOCK_TPL = """/* 合成 Mesa 片段：仅用于校验补丁与生成器的逻辑，不代表真实源码。 */

bool
etna_ml_operation_supported(struct pipe_ml_device *device,
                            const struct etna_ml_operation *operation)
{
   const struct pipe_tensor *input_tensor = operation->input_tensors[0];
   bool supported = false;

   (void)device;

   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
%s
      if (tensor_quantization_supported(input_tensor) &&
          operation->conv.dilation_width_factor == 1 &&
          operation->conv.weight_tensor->dims[0] != 0)
         supported = true;
      break;

   default:
      return false;
   }

   return supported;
}
"""

MOCK = MOCK_TPL % "\n".join(ANCHOR)

FAILS = []
GEN_WARN = []


def toks(s):
    """Python token 流：忽略纯格式差异，但保留缩进(INDENT/DEDENT)与字符串/注释内容。

    比「折叠空白」准确：能看穿 `x: y` vs `x:y`，又能抓到缩进漂移与字符串改动。
    """
    out = []
    drop = (tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER)
    for t in tokenize.generate_tokens(io.StringIO(s).readline):
        if t.type in drop:
            continue
        out.append((t.type, t.string))
    return out


def run_patch(text, new_override=None):
    """把 text 写进临时 .c，跑一次补丁器，返回 (rc, stdout, stderr, 打完后的内容)。"""
    mod = load(PATCH_PATH, "fsguard_patch_run")
    if new_override is not None:
        mod.NEW = new_override
    d = tempfile.mkdtemp()
    path = os.path.join(d, "etnaviv_ml.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = ["patch-etnaviv-filter-shape-guard.py", path]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main()
    finally:
        sys.argv = saved
    with open(path, encoding="utf-8") as f:
        after = f.read()
    return rc, out.getvalue(), err.getvalue(), after


def report(label, ok, detail=""):
    print("  %-44s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        FAILS.append(label)


def case(label, text, expect_rc, expect_err=None, new_override=None, checks=None):
    """checks 必须是 (after, out, err) -> [(描述, 布尔)] 的函数，惰性求值。"""
    rc, out, err, after = run_patch(text, new_override)
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


print("=== FSGUARD 补丁 正/反例自测（1 正例 + 7 反例 + 幂等 + 1 加固 + 生成器一致性）===")

# ── 正例 ──────────────────────────────────────────────────────────────
after, out, err = case(
    "GOOD  合成源码（函数尾部 return supported）", MOCK, 0,
    checks=lambda after, out, err: [
        ("补丁后含 ML_DBG 真字面量 `filter in_c (%u) != input channels`",
         "filter in_c (%u) != input channels" in after),
        ("补丁后含标记串 och patch: filter 深度与输入通道一致性守卫",
         "och patch: filter 深度与输入通道一致性守卫" in after),
        ("补丁后含 depthwise 豁免 `!operation->conv.depthwise`",
         "!operation->conv.depthwise" in after),
        ("新代码恰好 1 份", after.count(PATCH.NEW) == 1),
        ("原锚点已消失", after.count(PATCH.OLD) == 0),
        ("守卫块内是 return false（不是 break）",
         "return false;" in after[after.index(PATCH.MARKER):after.index("}", after.index(PATCH.MARKER))]),
        ("fail-closed：return false 出现在 supported = true 之前",
         after.index("return false;") < after.index("supported = true;")),
        ("守卫同时用 weight_tensor->dims[3] 与 input_tensor->dims[3]",
         "operation->conv.weight_tensor->dims[3] != input_tensor->dims[3]" in after),
    ])
report("GOOD  补丁器 stdout 含 [och-fsguard] 两行",
       out.count("[och-fsguard]") == 2,
       "实得 %d 行" % out.count("[och-fsguard]"))

# ── 幂等 ──────────────────────────────────────────────────────────────
d = tempfile.mkdtemp()
p = os.path.join(d, "etnaviv_ml.c")
with open(p, "w", encoding="utf-8") as f:
    f.write(MOCK)
mod = load(PATCH_PATH, "fsguard_patch_idem")
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

# ── 反例 n1..n7 ───────────────────────────────────────────────────────
case("n1  注释行被删（锚点 3 行变 2 行）",
     MOCK.replace("         // Dilation and per-axis quantization not yet implemented\n", ""),
     1, "匹配到 0 处")

case("n2  注释行缩进坏一格（9 -> 8 空格）",
     MOCK.replace("\n         // Dilation and per-axis quantization not yet implemented\n",
                  "\n        // Dilation and per-axis quantization not yet implemented\n"),
     1, "匹配到 0 处")

case("n3  conv case 消失",
     MOCK.replace("   case PIPE_ML_OPERATION_TYPE_CONVOLUTION:",
                  "   case PIPE_ML_OPERATION_TYPE_SOMETHING_ELSE:"),
     1, "case PIPE_ML_OPERATION_TYPE_CONVOLUTION:")

case("n4  dilation 检查被删",
     MOCK.replace("operation->conv.dilation_width_factor == 1",
                  "operation->conv.dilation_width_factor == 2"),
     1, "operation->conv.dilation_width_factor == 1")

case("n5  quant 检查被删",
     MOCK.replace("tensor_quantization_supported(input_tensor)",
                  "tensor_quantization_supported(output_tensor)"),
     1, "tensor_quantization_supported(input_tensor)")

case("n6  函数整体消失",
     MOCK.replace("etna_ml_operation_supported(struct pipe_ml_device",
                  "etna_ml_operation_supported_XX(struct pipe_ml_device"),
     1, "找不到函数")

case("n7  fail-open 形态（supported = true 在守卫之前）",
     MOCK.replace("   case PIPE_ML_OPERATION_TYPE_CONVOLUTION:\n",
                  "   case PIPE_ML_OPERATION_TYPE_CONVOLUTION:\n      supported = true;\n"),
     1, "出现在守卫之前")

# ── 加固项 n8（我加的）：守卫出口换成 break，门必须独立拦住 ─────────────
BREAK_NEW = PATCH.NEW.replace("            return false;\n", "            break;\n")
if BREAK_NEW == PATCH.NEW:
    report("n8  守卫出口换成 break", False, "替换没生效 —— 自测自身失效")
else:
    case("n8  守卫出口换成 break（门须独立成立）", MOCK, 1,
         "守卫块内没有 return false", new_override=BREAK_NEW)

# ── 生成器 ↔ 补丁文件 逐字节一致性 ────────────────────────────────────
print()
print("=== 生成器 ↔ 补丁文件 一致性（GEN）===")
try:
    gen_old, gen_comment = GEN.extract_anchor(MOCK)
    gen_new = GEN.build_new(gen_old, gen_comment)
    gen_body = (GEN.TEMPLATE
                .replace("__OLD__", repr(gen_old))
                .replace("__NEW__", repr(gen_new))
                .replace("__STRUCT__", repr(GEN.STRUCT)))
except SystemExit as exc:
    gen_body = None
    report("GEN  生成器能在同形 mock 上跑通", False, "生成器 die(), exit=%s" % exc.code)
else:
    report("GEN  生成器能在同形 mock 上跑通", True)
    report("GEN  生成器产出的 OLD 与补丁一致（构造保证）", gen_old == PATCH.OLD)
    report("GEN  生成器产出的 NEW 与补丁逐字节一致", gen_new == PATCH.NEW,
           "首个差异 @%s" % (
               next((i for i, (a, b) in enumerate(zip(gen_new, PATCH.NEW)) if a != b),
                    min(len(gen_new), len(PATCH.NEW)))))
    with open(PATCH_PATH, encoding="utf-8") as f:
        actual = f.read()
    if gen_body == actual:
        report("GEN  整份补丁文件与生成器输出逐字节一致", True)
    else:
        i = next((i for i, (a, b) in enumerate(zip(gen_body, actual)) if a != b),
                 min(len(gen_body), len(actual)))
        warn_byte = ("首个字节差异 @%d 生成=%r 文件=%r"
                     % (i, gen_body[i:i + 48], actual[i:i + 48]))
        try:
            tg, ta = toks(gen_body), toks(actual)
            same_tokens = tg == ta
        except tokenize.TokenError as exc:
            same_tokens = False
            warn_byte += "（tokenize 失败: %s）" % exc
        if same_tokens:
            # 仅格式差异：不阻塞构建，但明确打印（说明两者未逐字节同步）
            print("  %-44s WARN   <- 仅格式差异，token 流等价；%s" % ("GEN  逐字节一致", warn_byte))
            GEN_WARN.append(warn_byte)
        else:
            report("GEN  整份补丁文件与生成器输出（token 流等价）", False, warn_byte)

print()
if GEN_WARN:
    print("WARN: 生成器与补丁文件存在纯空白差异（不阻塞，但需 och 统一）:")
    for w in GEN_WARN:
        print("      " + w)
if FAILS:
    print("RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("RESULT: ALL PASS" + ("（含 %d 条 GEN 空白告警）" % len(GEN_WARN) if GEN_WARN else ""))
