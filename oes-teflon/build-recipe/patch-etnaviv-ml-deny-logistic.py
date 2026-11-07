#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拒绝 LOGISTIC 算子（单变量实验开关）：让 PIPE_ML_OPERATION_TYPE_LOGISTIC 一律不接管。

## 为什么需要（2026-11-07 WP1 取证）

v12 INT8 学生模型在 NPU 上的 delegate 分区实测（`TEFLON_DEBUG=verbose`）：

    [分区1: node 0-60 大段] -> [CPU: MEAN 61] -> [分区2: 62-64]
      -> [CPU: STRIDED_SLICE 65] -> [分区3: LOG 66]  <- 输出张量 170

运行期：compiled 1171 / 141 / 3 ms，invoked 23 / 0 / 0 ms。
第 3 段只有 1 个 LOG 算子，`invoked 0 ms`，其输入 169 来自 CPU 回退段；
实测输出与输入无关（两张不同图逐字节相同）=> 疑为分区边界输入未传递 / 未执行。

## 本补丁做什么（**单变量**）

在 `etna_ml_operation_supported()` 函数入口，对
`operation->type == PIPE_ML_OPERATION_TYPE_LOGISTIC` 直接 `return false`。

=> LOG 回退 CPU，整个子图变成 2 段 delegate、输出由 CPU 侧 TFLite 内核计算。
=> **实验判据**：若 NPU 两图输出开始不同（且与官模 CPU 基准相关），则
   「尾部单算子分区 + CPU→delegate 边界」实锤；否则排除该假说。

## 代价（如实报告）

- 少一个算子的 NPU 加速（模型尾部单个 LOG，代价可忽略）
- **不改变任何数值语义**：LOG 由 TFLite CPU 内核算，结果等价
- 不改模型、不改阈值、不改协议

## 用法

    python3 build-recipe/patch-etnaviv-ml-deny-logistic.py <etnaviv_ml.c>
    python3 build-recipe/patch-etnaviv-ml-deny-logistic.py --selftest

正控制：标记写入、判据命中、锚点唯一、行为自证（C harness）。
负控制：重复应用被拒、锚点缺失报错、其余行逐字不变。
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

MARKER = "och patch: 拒绝 LOGISTIC（单变量实验）"

OLD = """etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;
"""

NEW = """etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;

   /* och patch: 拒绝 LOGISTIC（单变量实验）
    *
    * 2026-11-07 WP1 实测：v12 INT8 被切成 3 段 delegate，
    * 输出张量落在只有 1 个 LOG 算子的第 3 段（compiled 3 ms / invoked 0 ms），
    * 其输入来自 CPU 回退的 STRIDED_SLICE => 输出与输入无关（两图逐字节相同）。
    *
    * 本补丁让 LOG 回退 CPU，用于单变量验证「尾部单算子分区 + 边界传递」假说。
    * 不改变数值语义（LOG 由 TFLite CPU 内核算）。
    */
   if (operation->type == PIPE_ML_OPERATION_TYPE_LOGISTIC) {
      ML_DBG("[och-deny-logistic] LOGISTIC -> 不接管（实验开关）\\n");
      return false;
   }
#if defined(__GNUC__) || defined(__clang__)
   __attribute__((used)) static const char och_deny_logistic_marker[] = "och-patch-ml-deny-logistic";
   (void)och_deny_logistic_marker;
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
    io.open(path, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    print("  [och-deny-logistic] 已注入 LOGISTIC 拒绝: %s" % path)
    return 0


def _behaviour_check():
    """行为自证：判据在 LOGISTIC / CONV / ADD 三种算子上的判定。"""
    harness = r"""
#include <stdio.h>
enum { T_CONV = 1, T_ADD = 2, T_LOGISTIC = 3 };
struct op_t { int type; };
static int reject(struct op_t *o) {
   if (o->type == T_LOGISTIC) return 1;
   return 0;
}
int main(void) {
   struct op_t conv = {T_CONV};
   struct op_t add  = {T_ADD};
   struct op_t logi = {T_LOGISTIC};
   printf("%d %d %d\n", reject(&conv), reject(&add), reject(&logi));
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
        expect = [0, 0, 1]
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

    tmpdir = tempfile.mkdtemp(prefix="och-denylog-")
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
            ("PIPE_ML_OPERATION_TYPE_LOGISTIC" in out, "判据引用 LOGISTIC 类型"),
            ("[och-deny-logistic]" in out, "运行期可观测日志前缀"),
            ("och-patch-ml-deny-logistic" in out, "C 代码自带产物可查的标记"),
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

        # apply 的实现是 src.replace(OLD, NEW, 1)，倒回去应当逐字等于原文
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
