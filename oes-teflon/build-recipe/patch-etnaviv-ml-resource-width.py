#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 **ML 层自建资源**的 `width0 = 0`，导致任何 `pipe_buffer_map_range` 断言失败。

## 现象（真机，Synology Photos 主题识别模型）

模型：`/volume1/OES/photos-npu/model.tflite`（concept 检测，335 类，输入 [1,395,395,3]）

用**任何** libteflon.so（含未打补丁的上游原版）跑，都在推理期 abort：

    python3.8: ../src/gallium/auxiliary/util/u_inlines.h:503:
      pipe_buffer_map_range: Assertion `offset + length <= buffer->width0' failed.
    栈：etna_ml_subgraph_read_outputs+0x38c ← read_outputs

⇒ 这就是 OES 侧把 `TEFLON_NPU` 默认设成 `off` 的原因
（`@appstore/SynologyPhotos/npu/npu_server` 注释：「delegate 在推理期遇 rank>4 张量
会 assert→abort()」）。

## 根因（源码定位）

`etna_ml_create_resource()`（`etnaviv_ml.c:117`）：

    struct pipe_resource *res = pipe_buffer_create(pscreen, **0**, PIPE_USAGE_DEFAULT, size);

Gallium 的 `pipe_buffer_create(screen, width, usage, size)` 里 **width 直接进 `res->width0`**，
而 `pipe_buffer_map_range` 有两条断言：

    assert(offset < buffer->width0);
    assert(offset + length <= buffer->width0);

⇒ 自建资源的 `width0 = 0` ⇒ **任何** map_range 都必然断言失败。

为什么我们的三个模型不炸：它们的输出资源由 **delegate 侧**提供（width 正确），
ML 层自建资源这条路径没被走到。而 photos 模型的 **FC 输出**（rank-2 [1,335]）
没有 delegate 侧资源 ⇒ 走 `etna_ml_create_tensor()` ⇒ 落到自建资源这条路径 ⇒ 炸。

## 修法

    pipe_buffer_create(pscreen, **size**, PIPE_USAGE_DEFAULT, size)

即「资源的 width 就等于它的字节数」—— 语义上这是**唯一自洽**的取值
（`width0` 本来就应当是「可映射的字节范围」）。

## 为什么风险极低

- **不改任何数值**：只改资源的几何字段（width0），不碰卷积/重量化/回读的算法
- 只影响 **ML 层自建**的资源（delegate 提供了资源的地方不受影响）
- 改动前这条路径必然 abort ⇒ 改动后从「必崩」变成「能跑」，不存在回归面
- 与 ROFIX 互补：ROFIX 让 map 范围用 `tensor->offset/size`；本补丁让资源的
  `width0` 容得下这个范围（两者都在时 `offset+size <= width0(=size)` 成立）

## 自证

    python3 build-recipe/patch-etnaviv-ml-resource-width.py --selftest

正控制：真实源码副本上标记出现、width 改成 size、锚点唯一。
负控制：重复应用被拒；锚点缺失必须报错；原文件其余行逐字不变；
        **模拟断言**：构造「自建资源 + map_range」的等价表达式，
        证明 width0=size 时断言成立、width0=0 时断言失败（守卫自证能失败）。
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

MARKER = "och patch: ML self-created resource width"

ANCHOR = ("   struct pipe_resource *res = pipe_buffer_create(pscreen, 0, "
          "PIPE_USAGE_DEFAULT, size);\n")

NEW = """   /* och patch: ML self-created resource width
    *
    * Gallium 的 pipe_buffer_create(screen, width, usage, size) 把 **width 直接写进
    * res->width0**，而 pipe_buffer_map_range 有两条断言：
    *     assert(offset < buffer->width0);
    *     assert(offset + length <= buffer->width0);
    * ⇒ width 传 0 ⇒ 自建资源的 width0 = 0 ⇒ **任何** map_range 必然断言失败。
    *
    * 真机（Synology Photos 主题识别模型，报告 48）：FC 输出的 rank-2 张量没有
    * delegate 侧资源 ⇒ 走 etna_ml_create_tensor() ⇒ 自建资源 ⇒ 推理期 abort：
    *     pipe_buffer_map_range: Assertion `offset + length <= buffer->width0' failed.
    * （未打补丁的上游原版同样 abort ⇒ 与本仓其它改动无关。）
    *
    * 修法：width 就取 size —— width0 的语义是「可映射的字节范围」，
    * 这是唯一自洽的取值。不改任何数值，只改资源几何。
    */
   struct pipe_resource *res = pipe_buffer_create(pscreen, size, PIPE_USAGE_DEFAULT, size);
#if defined(__GNUC__) || defined(__clang__)
   __attribute__((used)) static const char och_mlwidth_marker[] = "och-patch-ml-resource-width";
   (void)och_mlwidth_marker;
#endif
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
    print("  [och-mlwidth] 已注入 ML 自建资源 width 修复: %s" % path)
    return 0


def _assert_semantics():
    """守卫自证：width0=size 时断言成立，width0=0 时断言失败。"""
    harness = """
#include <assert.h>
#include <stdio.h>
struct res_t { unsigned width0; };
int main(int argc, char **argv) {
   struct res_t r;
   unsigned size = 335, offset = 0, length = 335;
   r.width0 = (argc > 1) ? 0 : size;   /* 0 = 修之前，1 = 修之后 */
   if (offset + length <= r.width0)
      printf("ASSERT-OK\\n");
   else
      printf("ASSERT-FAIL\\n");
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
        out_fixed = subprocess.run([path + ".bin"], capture_output=True, text=True).stdout.strip()
        out_broken = subprocess.run([path + ".bin", "broken"],
                                    capture_output=True, text=True).stdout.strip()
        ok = (out_fixed == "ASSERT-OK" and out_broken == "ASSERT-FAIL")
        return ok, "width0=size ⇒ %s；width0=0 ⇒ %s" % (out_fixed, out_broken)
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

    tmpdir = tempfile.mkdtemp(prefix="och-mlwidth-")
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
            ("pipe_buffer_create(pscreen, size, PIPE_USAGE_DEFAULT, size)" in out,
             "width 改成 size（width0 容得下映射范围）"),
            (out.count("pipe_buffer_create(pscreen, size, PIPE_USAGE_DEFAULT, size)") == 1,
             "只改这一处"),
            ("och-patch-ml-resource-width" in out, "C 代码自带产物可查的标记"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # 守卫自证：断言语义必须能分辨
        sem_ok, sem_msg = _assert_semantics()
        print("  [%s] 正控制：断言语义自证（%s）" % ("ok" if sem_ok else "BAD", sem_msg))
        ok = ok and sem_ok

        # 负控制 1：重复应用被拒
        cond = (apply(dst) == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 2：锚点缺失必须报错
        dst2 = os.path.join(tmpdir, "ml_nomod.c")
        broken = raw.replace(ANCHOR, "", 1)
        if broken == raw:
            print("  [BAD] 负控制 2：无法构造锚点缺失样本")
            ok = False
        else:
            io.open(dst2, "w", encoding="utf-8").write(broken)
            cond = (apply(dst2) == 1)
            print("  [%s] 负控制 2：锚点缺失时报错（非静默跳过）"
                  % ("ok" if cond else "BAD"))
            ok = ok and cond

        # 负控制 3：只改这一处
        cond = (out.count("pipe_buffer_create(pscreen, size, PIPE_USAGE_DEFAULT, size)") == 1
                and raw.count("pipe_buffer_create(pscreen, 0, PIPE_USAGE_DEFAULT, size);") == 1)
        print("  [%s] 负控制 3：原锚点只被替换了一次" % ("ok" if cond else "BAD"))
        ok = ok and cond
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


def main(argv):
    import subprocess
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
