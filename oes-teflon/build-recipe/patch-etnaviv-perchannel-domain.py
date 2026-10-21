#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修 **per-channel（per-axis）权重折叠后的域错误**：signed 分支多减了 0x80。

## 现象（报告 46，真机实测）

权重 scale **不均匀**的 int8 模型（C=248，1×1，单位阵）：

    NPU 输出：全 255（int8 的 127）—— 饱和垃圾
    CPU 基准：min=-128 max=33 mean=-72.15 —— 正常

日志里的硬证据：

    [och-nn]  conv_scale=0.002026 shift=24 ... coef_zero_point = 0
    [och-pcl] RW-OUT ... new_size=61504 head=2a 80 80 80 80 80 80 80 ...

`head` 的第一个字节 0x2a=42；离线复算（`verify/perchannel-prediction.py`）：

    requantized[0] = round(scale[0]·255/max_scale) = round(0.8·255/1.2) = 170
    170 - 128 = 42 = 0x2a            ✓ 命中
    其余字节 0x80=128，而**非对角权重本该是 0**

## 根因（源码定位）

`etnaviv_ml_nn.c` 的 `requantize_weights()`，signed 分支：

    int8_t requantized = round(dequantized / operation->weight_scale);
    map_out[oc][w][h][ic] = requantized + operation->weight_zero_point - 0x80;

两处错：

1. **`int8_t requantized`**：折叠后的值是 **uint8 域**的（`calc_quant_params` 把
   [min,max] 映射到 [0,255]），可以到 255；存进 `int8_t` 就被截断成 [-128,127]
   （170 → −86）。
2. **再减 `0x80`**：把 uint8 域的值写成了 int8 域的值。

而描述符的 `coef_zero_point`（`nn.c:1168`：`map->coef_zero_point = operation->weight_zero_point`）
在 per-channel 路径下是 `calc_quant_params()` 给的 **zp_new**（本例 min=0 ⇒ **0**），
**不是 128** ⇒ 硬件按 `字节 − 0` 解释 ⇒

    非对角权重：0（应）→ 128（实）  ⇒ 每个非对角项都带 +128
    对角权重  ：170（应）→ 42（实）  ⇒ 偏小 128

⇒ 248 个输入通道各贡献一个 +128 ⇒ 巨额正累加 ⇒ **输出饱和到 255**。

（per-tensor 路径没有这个问题：那里 `weight_zero_point = etna_tensor_zero_point()`
⇒ signed 张量返回 `zp+128` ⇒ coef_zp=128，与 int8 域自洽。）

## 修法

signed 分支与 unsigned 分支**统一到 uint8 域**：

    int32_t requantized = (int32_t)round(dequantized / operation->weight_scale);
    int32_t och_qu = requantized + operation->weight_zero_point;   /* uint8 域 */
    map_out[oc][w][h][ic] = (uint8_t)CLAMP(och_qu, 0, 255);

硬件算 `字节 − coef_zp(=zp_new)` ⇒ `requantized` ⇒ 真实权重
`= scale_new · requantized` ⇒ 与折叠前的 per-channel 权重一致（精度损失只剩
「压成单一 scale」这一项，那是 upstream 明确声明的设计）。

**为什么不用「保留 int8 + 把 coef_zp 改成 zp_new+128」**：`operation->weight_zero_point`
同时被 `requantize_bias()` 和描述符使用，改它会连带改偏置的域 ⇒ 风险更大。

## 门控

运行期 `TEFLON_PCDOMAIN=1` 才生效（不设 = 与上游逐字节一致），便于同产物 A/B。
默认关闭：这是**改行为**的修复，取证性质，交付前需另行决策。

## 自证

    python3 build-recipe/patch-etnaviv-perchannel-domain.py --selftest

正控制：真实源码副本上标记出现、门控在、int32 中间量在、CLAMP 在、锚点唯一。
负控制：重复应用被拒；锚点缺失必须报错；原文件其余行逐字不变。
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

MARKER = "och patch: per-channel weight domain (signed branch)"

ANCHOR = """               if (operation->weight_signed) {
                  int8_t val = map_in[oc][src_ic][w][h];
                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
                  int8_t requantized = round(dequantized / operation->weight_scale);
                  map_out[oc][ic][w][h] = requantized + operation->weight_zero_point - 0x80;
               } else {
"""

NEW = """               if (operation->weight_signed) {
                  int8_t val = map_in[oc][src_ic][w][h];
                  double dequantized = weight_tensor->scales[oc] * (val - weight_tensor->zero_points[oc]);
                  /* och patch: per-channel weight domain (signed branch)
                   *
                   * 上游把折叠后的权重写成了 **int8 域**的值：
                   *   int8_t requantized = round(...);        <- 截断到 [-128,127]
                   *   map_out[...] = requantized + zp - 0x80;  <- 又减了一次 0x80
                   * 而折叠后的量本来就在 **uint8 域**（calc_quant_params 把 [min,max]
                   * 映射到 [0,255]），且 per-channel 路径的 coef_zero_point 是
                   * zp_new（本例 0，**不是 128**）=> 硬件按 `字节 - 0` 解释。
                   *
                   * 后果（真机实测，报告 46）：非对角权重 0 -> 128、对角 170 -> 42
                   * => 248 项各带 +128 => 巨额正累加 => 输出饱和到 255。
                   *
                   * 这里统一到 uint8 域：中间量用 int32（不截断），并夹到 [0,255]。
                   * 硬件算 `字节 - coef_zp(=zp_new)` => requantized => 真实权重
                   * = scale_new * requantized => 与折叠前的 per-channel 权重一致。
                   *
                   * 运行期门控：TEFLON_PCDOMAIN=1 才生效。
                   *
                   * ★ 锚点是 **perchannel-layout 补丁之后**的形态
                   *   （map_in[oc][src_ic][w][h] / map_out[oc][ic][w][h]）——
                   *   那个补丁整函数重写了 requantize_weights，所以本补丁**必须排在它后面**
                   *   （build-teflon.sh 的 3m2 段）。自测也按这个顺序跑。
                   */
                  int32_t och_requant = (int32_t)round(dequantized / operation->weight_scale);
                  int32_t och_qu = och_requant + operation->weight_zero_point;
                  /* 门控语义（2026-10-21 交付版）：**默认开**，TEFLON_PCDOMAIN=0 才关 */
                  const char *och_pd = getenv("TEFLON_PCDOMAIN");
                  if (och_pd == NULL || och_pd[0] != '0') {
                     map_out[oc][ic][w][h] = (uint8_t)CLAMP(och_qu, 0, 255);
                  } else {
                     map_out[oc][ic][w][h] = (uint8_t)((int8_t)och_requant +
                                                          operation->weight_zero_point - 0x80);
                  }
                  #if defined(__GNUC__) || defined(__clang__)
                  /* 编译期标记：只有 __attribute__((used)) 的对象才留在 .rodata，
                   * 于是 `strings libteflon.so | grep [och-pcdomain]` 能真正判出
                   * 「这个 .so 里有没有本补丁」。 */
                  __attribute__((used)) static const char och_pcdomain_marker[] =
                     "[och-pcdomain]";
                  (void)och_pcdomain_marker;
                  #endif
               } else {
"""


def apply(path):
    src = io.open(path, encoding="utf-8").read()
    if MARKER in src:
        sys.stderr.write('ERROR: %s 已包含本补丁（标记 "%s"），拒绝重复应用。\n'
                         % (path, MARKER))
        return 1
    n = src.count(ANCHOR)
    if n != 1:
        sys.stderr.write("ERROR: 锚点在 %s 中出现 %d 次（要求恰好 1 次）。\n"
                         "       源码可能已升级/被其它补丁改动，拒绝盲目套用。\n"
                         % (path, n))
        return 1
    io.open(path, "w", encoding="utf-8").write(src.replace(ANCHOR, NEW, 1))
    print("  [och-pcdomain] 已注入 per-channel 权重域修复: %s" % path)
    print("  [och-pcdomain] 运行期门控：TEFLON_PCDOMAIN=1 才生效")
    return 0


def _dangling_requantized(text):
    """signed 分支里是否残留裸 `requantized` 引用（删了声明却留着引用）。

    ★ 必须用**括号配对**定位 signed 块：里面还嵌着本补丁自己的
    `if (getenv(...)) { ... } else { ... }` ⇒ 非贪婪正则会停在内层的 `} else {`
    上，把真正出问题的 else 分支漏在检查范围之外（第一版就是这么漏判的）。
    """
    m_fn = re.search(r'requantize_weights\(struct etna_ml_subgraph', text)
    if not m_fn:
        return False
    i = text.index("if (operation->weight_signed) {", m_fn.start())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                seg = text[i:j]
                break
    else:
        return True                      # 括号不配平 ⇒ 判失败（fail-closed）
    seg = re.sub(r'/\*.*?\*/', '', seg, flags=re.S)   # 剥注释（注释里会提到 requantized）
    seg = seg.replace("och_requant", "")               # 去掉我们自己的变量名
    return "requantized" in seg


def _compile_inserted(text):
    """把**本补丁插入的那段代码**抽出来，用宿主 gcc 真编译一次。

    动机：run 39566597876 的失败是**编译期**才发现的（删了声明却留着引用），
    而那时的自测只验「补丁能落地」。这里补上「落地的代码能编译」这一层。

    抽出的片段只用到 dequantized / operation->weight_scale /
    operation->weight_zero_point / map_out / getenv / CLAMP ⇒ 可用最小 stub 编译。
    返回 (ok, 编译器输出)。
    """
    m = re.search(r'int32_t och_requant.*?#endif\n', text, re.S)
    if not m:
        return False, "抽不出插入的代码片段"
    frag = m.group(0)
    harness = """
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#define CLAMP(X, MIN, MAX) ((X) > (MIN) ? ((X) > (MAX) ? (MAX) : (X)) : (MIN))
struct op_stub { float weight_scale; int weight_zero_point; };
static void probe(struct op_stub *operation, uint8_t (*map_out)[4][1][1], double dequantized) {
   for (unsigned oc = 0; oc < 1; oc++)
   for (unsigned ic = 0; ic < 1; ic++)
   for (unsigned w = 0; w < 1; w++)
   for (unsigned h = 0; h < 1; h++) {
%s
   }
}
static uint8_t out[1][4][1][1];
int main(void) {
   struct op_stub o = { 0.002026f, 0 };
   probe(&o, out, 0.5);
   return out[0][0][0][0];
}
""" % frag
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        cc = os.environ.get("CC", "cc")
        r = subprocess.run([cc, "-std=c99", "-Wall", "-Wextra", "-Werror",
                            "-fsyntax-only", path],
                           capture_output=True, text=True)
        return (r.returncode == 0), (r.stdout + r.stderr)
    except FileNotFoundError:
        return True, "（无 C 编译器，跳过）"
    finally:
        os.unlink(path)


def _gate_check(text, env, prefix):
    """门控语义的行为级检查（本补丁只有开/关两态）。"""
    try:
        from gate_semantics_check import check_gate_simple
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gate_semantics_check import check_gate_simple
    return check_gate_simple(text, env, prefix)


def selftest(mesa_src):
    ok = True
    real_nn = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml_nn.c")
    real_v7 = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c")
    if not os.path.isfile(real_nn):
        print("  [SKIP] 找不到真实源码 %s" % real_nn)
        return True

    tmpdir = tempfile.mkdtemp(prefix="och-pcdomain-")
    try:
        dst = os.path.join(tmpdir, "etnaviv_ml_nn.c")
        shutil.copyfile(real_nn, dst)
        raw = io.open(real_nn, encoding="utf-8").read()

        # ---- 前置：先打 perchannel-layout 补丁（build 里的真实顺序）----
        layout = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "patch-etnaviv-perchannel-layout.py")
        if os.path.isfile(layout) and os.path.isfile(real_v7):
            v7 = os.path.join(tmpdir, "etnaviv_ml_nn_v7.c")
            shutil.copyfile(real_v7, v7)
            r = subprocess.run([sys.executable, layout, dst, "--v7", v7, "--diag", "1"],
                               capture_output=True, text=True)
            print("  [%s] 前置：perchannel-layout 补丁应用成功"
                  % ("ok" if r.returncode == 0 else "BAD"))
            if r.returncode != 0:
                print("        %s" % (r.stderr or "")[:200])
                return False
        else:
            print("  [WARN] 找不到 perchannel-layout 补丁或 v7 源文件 —— "
                  "只测本补丁的锚点唯一性（不测组合）")

        # ★ 基线快照：**在**打本补丁**之前**取（否则后面的 diff 没有可比对象）
        laid = io.open(dst, encoding="utf-8").read()

        if apply(dst) != 0:
            print("  [BAD] 正控制：应用到真实源码失败")
            return False
        out = io.open(dst, encoding="utf-8").read()

        # ★ 悬空引用守卫（run 39566597876 的失败类别）：
        #   删了 requantized 的声明却留下引用它的那一行 ⇒ 编译期才炸。
        #   这里在**打完的文本**里查：函数体内除 och_requant 外不得再有裸 requantized。
        dangling = _dangling_requantized(out)
        _compile_ok, _compile_msg = _compile_inserted(out)
        if not _compile_ok:
            print('        编译输出: %s' % _compile_msg.strip()[:300])

        _gate_ok, _gate_msg = _gate_check(out, "TEFLON_PCDOMAIN", "och_pd")
        if not _gate_ok:
            print("        %s" % _gate_msg)
        for cond, desc in [
            (MARKER in out, "标记已写入"),
            ('getenv("TEFLON_PCDOMAIN")' in out, "运行期门控存在"),
            (out.count('getenv("TEFLON_PCDOMAIN")') == 1, "门控只出现一次"),
            (_gate_ok, "门控语义行为级正确（unset=开 / =0=关 / =1=开）"),

            ("int32_t och_requant" in out, "中间量用 int32（不截断）"),
            ("CLAMP(och_qu, 0, 255)" in out, "夹到 uint8 域 [0,255]"),
            ("- 0x80" in out, "未设门控时保留上游原样（int8 域）"),
            ("[och-pcdomain]" in out, "C 代码自带产物可查的标记"),
            (not dangling, "打完的 requantize_weights 里没有裸 requantized 引用（悬空引用守卫）"),
            (_compile_ok, "插入的代码片段能真的编译过（宿主 gcc -fsyntax-only）"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # ★ 守卫自身的负控：把打完的文本改回「有悬空引用」的形态，
        #   守卫必须判 True —— 否则这道守卫自己可能是假绿。
        buggy = out.replace(
            "map_out[oc][ic][w][h] = (uint8_t)((int8_t)och_requant +",
            "map_out[oc][ic][w][h] = requantized + (int8_t)och_requant +")
        cond = (buggy != out) and _dangling_requantized(buggy)
        print("  [%s] 负控制 0：守卫自己会失败（造 bug 版必须被判中）"
              % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 1：重复应用被拒
        cond = (apply(dst) == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 2：锚点缺失必须报错（在「布局补丁之后」的文本上构造）
        dst2 = os.path.join(tmpdir, "nn_nomod.c")
        broken = laid.replace(ANCHOR, "", 1)
        if broken == laid:
            print("  [BAD] 负控制 2：无法构造锚点缺失样本")
            ok = False
        else:
            io.open(dst2, "w", encoding="utf-8").write(broken)
            cond = (apply(dst2) == 1)
            print("  [%s] 负控制 2：锚点缺失时报错（非静默跳过）"
                  % ("ok" if cond else "BAD"))
            ok = ok and cond

        # 负控制 3：改动**必须只发生在锚点那一处**
        import difflib
        a = laid.splitlines()
        b = out.splitlines()
        sm = difflib.SequenceMatcher(None, a, b)
        hunks = [op for op in sm.get_opcodes() if op[0] != "equal"]
        anchor_start = None
        for i, l in enumerate(a):
            if l.strip() == "int8_t requantized = round(dequantized / operation->weight_scale);":
                anchor_start = i
                break
        cond = (len(hunks) == 1)
        print("  [%s] 负控制 3：只有 1 处改动块（实际 %d）" % ("ok" if cond else "BAD", len(hunks)))
        ok = ok and cond
        if hunks:
            tag, i1, i2, j1, j2 = hunks[0]
            print("        改动位置: 布局补丁后文本行 %d..%d → 新文本行 %d..%d" % (i1, i2, j1, j2))
            cond = (anchor_start is not None and abs(i1 - anchor_start) <= 2)
            print("  [%s] 负控制 3b：改动块紧邻锚点（锚点行 %s）"
                  % ("ok" if cond else "BAD", anchor_start))
            ok = ok and cond
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    mesa = os.environ.get("OCH_MESA_SRC",
                          os.path.join(here, "..", "third_party", "mesa-src-26.2.5"))
    if len(argv) == 2 and argv[1] == "--selftest":
        ok = selftest(mesa)
        print("自测%s" % ("通过" if ok else "失败"))
        return 0 if ok else 1
    if len(argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for p in argv[1:]:
        rc |= apply(p)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
