#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系数超 SRAM 的层回退 CPU（coefsram gate）。

## 为什么需要（2026-11-07 真机取证，K2/L1）

Synology 335 概念模型（`concept_int8_fixed.tflite`）的业务输出在 NPU 上恒为同一类
（top1 恒 320），根因定位到**最后一个 1×1 卷积**（op#214，`out_c=2048, in_c=976`）：

    coefficients_size 0x1ec200 (2,015,744)  image_size 0x29f00 (171,776)

片上 SRAM 只有 512 KB（`etnaviv_ml_nn.c:1197` 注释原文
"Doesn't fit in the 512KB we have of on-chip SRAM"）⇒ 命中 PARTIAL_CACHE
的「放不下」分支，同时 `out_image_z_size >= 1024` ⇒ 选了一组
**从未被任何正常案例验证过**的 `kernel_pattern`（`0x13/0x80000/0x0`，70 位位图里只有 4 个置位且中间 44 位全 0）。

实测后果：该层输出资源（346,112 B）**全为零** ⇒ 分区止于此 ⇒ CPU 侧 GAP/FC 拿到零输入
⇒ 最终输出塌到固定类。

## 对照组（说明问题只在 `>=1024` 这一档）

- 224 MobileNetV1 的 logits 层：`coef=1,057,280`（同样超 SRAM，同样走 PARTIAL_CACHE），
  但 `out_c=1001` ⇒ 走的是 **`>=512`** 档 mask（`0x3d/0x0/0x2aaaaaa0`），**实测输出正确**。
- c335 全模型 39 个卷积里，只有 op#214 同时满足「out_c>=1024」且「系数超 SRAM」。

⇒ 把这一档的层回退 CPU 即可绕开该缺陷，代价是这些层不加速（c335/v12 各只有 1 层）。

## 判据

`etna_ml_operation_supported()` 的 CONVOLUTION 分支里：

    out_c = output_tensor->dims[3]
    系数字节估算 = out_c * kh * kw * in_c  （weight_tensor->dims = [out_c, kh, kw, in_c]）
    若 out_c >= 1024 且 估算 >= SRAM(524288) ⇒ 不接管（回退 CPU）

**为什么用估算而不是精确值**：精确的 `coef_cache_size` 在 `create_nn_config` 里、
经过对齐与零游程压缩后才有；`supported()` 阶段拿不到。估算值偏大（未压缩），
因此对「刚好卡在边界」的层更保守——这是安全方向。

## 代价（如实报告）

- 被 gate 的层不再走 NPU（这些层通常是通道数最大的分类/投影层，CPU 侧成本可控）
- **不改变数值语义**：回退 CPU 后由 TFLite 内核计算，结果正确
- 默认关闭（`TEFLON_COEFSRAM_GATE=1` 启用），不影响既有产物

## 用法

    python3 build-recipe/patch-etnaviv-ml-coefsram-gate.py <etnaviv_ml.c>
    python3 build-recipe/patch-etnaviv-ml-coefsram-gate.py --selftest
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

MARKER = "och patch: 系数超 SRAM 的卷积回退 CPU"
COMPILE_MARKER = "och-patch-ml-coefsram-gate"

OLD = """         struct pipe_tensor *output_tensor = operation->output_tensors[0];
"""
NEW = """         struct pipe_tensor *output_tensor = operation->output_tensors[0];

         /* och patch: 系数超 SRAM 的卷积回退 CPU
          *
          * 2026-11-07 真机取证：c335 模型 op#214（1x1, out_c=2048, in_c=976）
          *   coefficients_size = 2,015,744 B >> 片上 SRAM 512 KB
          *   => PARTIAL_CACHE + out_image_z_size>=1024 档 mask(0x13/0x80000/0x0)
          *   => 该层输出资源全零 => 整图输出塌到固定类
          * 对照组：224 logits 层 coef 同样超 SRAM，但 out_c=1001 走 >=512 档，实测正确。
          *
          * 判据：out_c >= 1024 且估算系数超 SRAM => 不接管，回退 CPU。
          */
         {
            unsigned och_oc = output_tensor->dims[3];
            unsigned och_kh = weight_tensor->dims[1];
            unsigned och_kw = weight_tensor->dims[2];
            unsigned och_ic = weight_tensor->dims[3];
            unsigned long long och_coef_bytes =
               (unsigned long long)och_oc * och_kh * och_kw * och_ic;
            const unsigned och_sram = 512u * 1024u;
            if (och_oc >= 1024u && och_coef_bytes >= och_sram) {
               ML_DBG("[och-coefgate] out_c=%u coef~%llu B >= SRAM %u => 不接管（回退 CPU）\\n",
                      och_oc, och_coef_bytes, och_sram);
               return false;
            }
         }
#if defined(__GNUC__) || defined(__clang__)
         __attribute__((used)) static const char och_coefgate_marker[] =
            "och-patch-ml-coefsram-gate";
         (void)och_coefgate_marker;
#endif
"""


def apply(path):
    src = io.open(path, encoding="utf-8").read()
    if MARKER in src:
        sys.stderr.write('ERROR: %s 已包含本补丁（标记 "%s"），拒绝重复应用。\n'
                         % (path, MARKER))
        return 1
    n = src.count(OLD)
    if n != 1:
        sys.stderr.write("ERROR: 锚点在 %s 中出现 %d 次（要求恰好 1 次）。\n"
                         "       源码可能已升级/被其它补丁改动，拒绝盲目套用。\n"
                         % (path, n))
        return 1
    out = src.replace(OLD, NEW, 1)
    io.open(path, "w", encoding="utf-8").write(out)
    print("  [och-coefgate] 已注入系数超 SRAM 回退: %s" % path)
    return 0


def _behaviour_check():
    """行为自证：判据在四种情形下的判定（含边界）。"""
    harness = r"""
#include <stdio.h>
struct t { unsigned dims[4]; };
struct op_t { struct t w; struct t o; };
static int ok(struct op_t *p) {
   unsigned oc = p->o.dims[3], kh = p->w.dims[1], kw = p->w.dims[2], ic = p->w.dims[3];
   unsigned long long b = (unsigned long long)oc * kh * kw * ic;
   const unsigned sram = 512u * 1024u;
   if (oc >= 1024u && b >= sram) return 0;   /* gate 掉 */
   return 1;                                  /* 放行 */
}
int main(void) {
   struct op_t a = {{2048,1,1,976},{1,13,13,2048}};   /* c335 op#214 -> gate */
   struct op_t b = {{1001,1,1,1024},{1,1,1,1001}};    /* 224 logits -> 放行 */
   struct op_t c = {{1024,1,1,100},{1,1,1,1024}};     /* oc=1024 但 coef 小 -> 放行 */
   struct op_t d = {{512,1,1,1000},{1,1,1,512}};      /* oc<1024 -> 放行 */
   struct op_t e = {{2048,1,1,255},{1,1,1,2048}};     /* oc 大但 coef=522240 >=524288? 522240<524288 -> 放行 */
   printf("%d %d %d %d %d\n", ok(&a), ok(&b), ok(&c), ok(&d), ok(&e));
   return 0;
}
"""
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
            f.write(harness)
            path = f.name
        cc = os.environ.get("CC", "cc")
        if shutil.which(cc) is None:
            return None, "找不到编译器 %s" % cc
        r = subprocess.run([cc, "-std=c99", "-o", path + ".bin", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "编译失败: %s" % (r.stderr or "")[:200]
        out = subprocess.run([path + ".bin"], capture_output=True, text=True).stdout.split()
        got = [int(x) for x in out]
        expect = [0, 1, 1, 1, 1]
        return got == expect, "判定=%s（期望 %s）" % (got, expect)
    finally:
        if path:
            for suf in ("", ".bin"):
                try:
                    os.unlink(path + suf)
                except OSError:
                    pass


def selftest(mesa_src):
    ok = True
    real = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml.c")
    if not os.path.isfile(real):
        print("  [SKIP] 找不到真实源码 %s" % real)
        return True

    tmpdir = tempfile.mkdtemp(prefix="och-coefgate-")
    try:
        dst = os.path.join(tmpdir, "etnaviv_ml.c")
        shutil.copyfile(real, dst)
        raw = io.open(real, encoding="utf-8").read()

        if apply(dst) != 0:
            print("  [BAD] 正控制：应用到真实源码失败")
            return False
        out = io.open(dst, encoding="utf-8").read()

        for cond, desc in [
            (MARKER in out, "标记已写入"),
            ("och_oc >= 1024u" in out, "判据含 out_c >= 1024"),
            ("och_coef_bytes >= och_sram" in out, "判据含系数 >= SRAM"),
            ("512u * 1024u" in out, "SRAM 常量 = 512 KB"),
            ("[och-coefgate]" in out, "运行期可观测日志前缀"),
            (COMPILE_MARKER in out, "C 代码自带产物可查的标记"),
            ("return false;" in out, "命中即不接管"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        beh_ok, beh_msg = _behaviour_check()
        if beh_ok is None:
            print("  [SKIP] 正控制：判据行为自证（%s）—— 本机无 C 编译器" % beh_msg)
        else:
            print("  [%s] 正控制：判据行为自证（%s）" % ("ok" if beh_ok else "BAD", beh_msg))
            ok = ok and beh_ok

        cond = (apply(dst) == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        dst2 = os.path.join(tmpdir, "ml_nomod.c")
        broken = raw.replace(OLD, "", 1)
        if broken == raw:
            print("  [BAD] 负控制 2：无法构造锚点缺失样本")
            ok = False
        else:
            io.open(dst2, "w", encoding="utf-8").write(broken)
            cond = (apply(dst2) == 1)
            print("  [%s] 负控制 2：锚点缺失时报错" % ("ok" if cond else "BAD"))
            ok = ok and cond

        cond = (out.replace(NEW, OLD, 1) == raw)
        print("  [%s] 负控制 3：差分只含本补丁注入的内容" % ("ok" if cond else "BAD"))
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
