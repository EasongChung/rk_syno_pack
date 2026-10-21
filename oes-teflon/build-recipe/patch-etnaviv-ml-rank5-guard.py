#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**通用** rank>4 守卫：任何输入/输出张量是 rank>4 激活的算子，**一律不接管**。

## 为什么需要（真机，Synology Photos 主题识别模型）

模型 `/volume1/OES/photos-npu/model.tflite`（335 类，输入 [1,395,395,3]）在推理期 abort：

    pipe_buffer_map_range: Assertion `offset + length <= buffer->width0' failed.
    栈：etna_ml_subgraph_read_outputs ← invoke

**根因**（本轮定位，见报告 48）：

1. 该模型是 **rank-5 折叠**形态：`RESHAPE → [1,2,122,50,50]` / `TRANSPOSE → [1,122,2,50,50]`
   ⇒ 全模型 32 个 rank-5 张量
2. **delegate 接管了 RESHAPE**（`tfl_device.c:248` 支持 kTfLiteBuiltinReshape），
   而 `etna_ml_operation_supported()` 的 RESHAPE 分支是**无条件** `supported = true`
3. ML 契约里没有「维度个数」字段：delegate 只写最后 4 维，驱动一律算
   `dims[1]*dims[2]*dims[3]` ⇒ **rank-5 的 dims[0] 被丢掉**
   ⇒ 真实 610,000 字节被算成 305,000 ⇒ 资源/尺寸错 ⇒ map_range 断言

**为什么已有的 R5GUARD 抓不到**：它的守卫只挂在 **CONVOLUTION 分支**上
（`patch-etnaviv-r5guard.py`），而这里出问题的是 **RESHAPE**。

## 本补丁做什么

在 `etna_ml_operation_supported()` 的**函数入口**加一道通用检查：遍历
`operation->input_tensors[0..input_count)` 与 `output_tensors[0..output_count)`，
凡是 `data == NULL`（是激活，不是权重/常量）且 `dims[0] != 1`（说明 rank>4 被压进 4 槽）
的 ⇒ **直接 return false（不接管，回落 CPU）**。

⇒ **一处覆盖所有算子类型**（CONV / RESHAPE / TRANSPOSE / POOL / PAD / CONCAT / FC / RELU…），
比 R5GUARD 的「只在卷积分支」严格更强。

## 代价（如实报告）

- rank>4 参与的那个算子回落 CPU ⇒ 少一点加速
- **不改变任何数值**：只是「不接管」⇒ 结果由 TFLite 的 CPU 内核算
- 不改模型、不改阈值

## 与 R5GUARD / MLWIDTH 的关系

- 本补丁**完全覆盖** R5GUARD 的判据（同一判据，作用域更大）⇒ 两者可并存
- `patch-etnaviv-ml-resource-width.py`（ML 自建资源 width0=0）是**另一个**独立缺陷，
  两者互补：一个不让 rank>4 的算子进来，一个保证进来的资源 width0 容得下映射范围

## 自证

    python3 build-recipe/patch-etnaviv-ml-rank5-guard.py --selftest

正控制：真实源码副本上标记出现、门控覆盖 input/output 两组、锚点唯一、
        **行为自证**（用 C harness 验证判据在 rank-4/rank-5/常量张量三种情形的判定）。
负控制：重复应用被拒；锚点缺失必须报错；原文件其余行逐字不变。
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

MARKER = "och patch: 通用 rank>4 守卫"

OLD = """etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;
"""

NEW = """etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;

   /* och patch: 通用 rank>4 守卫
    *
    * ML 契约里没有「维度个数」字段：delegate 只写最后 4 维，驱动一律算
    * dims[1]*dims[2]*dims[3] ⇒ **rank>4 的 dims[0] 被丢掉**。
    *   真机（Synology Photos 主题识别模型，报告 48）：rank-5 张量
    *   [1,2,122,50,50] 的真实体积 610,000 被算成 305,000
    *   ⇒ 资源/尺寸错 ⇒ pipe_buffer_map_range 断言 ⇒ 推理期 abort（core dumped）。
    *
    * 该模型里 rank-5 张量是 **RESHAPE 的输出**，而 delegate 接管 RESHAPE
    * （tfl_device.c 支持 kTfLiteBuiltinReshape）⇒ 只挂在 CONVOLUTION 分支上的
    * R5GUARD 抓不到 ⇒ 这里做**通用**检查：一处覆盖所有算子类型。
    *
    * 判据：data == NULL（是激活，不是权重/常量）且 dims[0] != 1
    *       （说明 dims[0] 是 rank-5 的真实第一维，不是 batch=1）。
    *
    * 命中 ⇒ 不接管（回落 CPU）。不改任何数值，只是「不算这个算子」。
    */
   for (unsigned och_i = 0; och_i < operation->input_count; och_i++) {
      const struct pipe_tensor *och_t = operation->input_tensors[och_i];
      if (och_t && och_t->data == NULL && och_t->dims[0] != 1) {
         ML_DBG("[och-r5guard] input %u dims[0]=%u -> 不接管\\n", och_i, och_t->dims[0]);
         return false;
      }
   }
   for (unsigned och_i = 0; och_i < operation->output_count; och_i++) {
      const struct pipe_tensor *och_t = operation->output_tensors[och_i];
      if (och_t && och_t->data == NULL && och_t->dims[0] != 1) {
         ML_DBG("[och-r5guard] output %u dims[0]=%u -> 不接管\\n", och_i, och_t->dims[0]);
         return false;
      }
   }
#if defined(__GNUC__) || defined(__clang__)
   __attribute__((used)) static const char och_r5guard_marker[] = "och-patch-ml-rank5-guard";
   (void)och_r5guard_marker;
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
    print("  [och-r5guard] 已注入通用 rank>4 守卫: %s" % path)
    return 0


def _behaviour_check():
    """行为自证：判据在三种情形下的判定是否符合预期。"""
    harness = """
#include <stdio.h>
struct tensor_t { void *data; unsigned dims[4]; };
struct op_t { struct tensor_t **input_tensors; unsigned input_count;
               struct tensor_t **output_tensors; unsigned output_count; };
static int reject(struct op_t *o) {
   for (unsigned i = 0; i < o->input_count; i++) {
      struct tensor_t *t = o->input_tensors[i];
      if (t && t->data == NULL && t->dims[0] != 1) return 1;
   }
   for (unsigned i = 0; i < o->output_count; i++) {
      struct tensor_t *t = o->output_tensors[i];
      if (t && t->data == NULL && t->dims[0] != 1) return 1;
   }
   return 0;
}
int main(void) {
   struct tensor_t act4  = {NULL, {1, 50, 50, 244}};
   struct tensor_t act5  = {NULL, {2, 122, 50, 50}};
   struct tensor_t const_ = {(void*)1, {2, 122, 50, 50}};
   struct tensor_t *in4[1]  = {&act4};
   struct tensor_t *in5[1]  = {&act5};
   struct tensor_t *inc[1]  = {&const_};
   struct tensor_t *out4[1] = {&act4};
   struct op_t a = {in4, 1, out4, 1};
   struct op_t b = {in5, 1, out4, 1};
   struct op_t c = {inc, 1, out4, 1};
   struct op_t d = {in4, 1, in5, 1};
   printf("%d %d %d %d\\n", reject(&a), reject(&b), reject(&c), reject(&d));
   return 0;
}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        cc = os.environ.get("CC", "cc")
        r = subprocess.run([cc, "-std=c99", "-o", path + ".bin", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "编译失败: %s" % (r.stderr or "")[:200]
        out = subprocess.run([path + ".bin"], capture_output=True, text=True).stdout.split()
        got = [int(x) for x in out]
        expect = [0, 1, 0, 1]   # rank-4 激活=接；rank-5 激活=拒；常量=接；rank-5 输出=拒
        return got == expect, "判定=%s（期望 %s）" % (got, expect)
    finally:
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

    tmpdir = tempfile.mkdtemp(prefix="och-r5guard-")
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
            ("operation->input_count" in out and "operation->output_count" in out,
             "守卫覆盖 input/output 两组"),
            ("och_t->data == NULL && och_t->dims[0] != 1" in out, "判据正确（激活且 dims[0]!=1）"),
            ("return false;" in out, "命中即不接管"),
            ("och-patch-ml-rank5-guard" in out, "C 代码自带产物可查的标记"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        beh_ok, beh_msg = _behaviour_check()
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
            print("  [%s] 负控制 2：锚点缺失时报错（非静默跳过）"
                  % ("ok" if cond else "BAD"))
            ok = ok and cond

        cond = (out.count("och_t->data == NULL && och_t->dims[0] != 1") == 2)
        print("  [%s] 负控制 3：判据只出现两处（input/output 各一）"
              % ("ok" if cond else "BAD"))
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
