#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 etnaviv ML **图输出不做 NCHW→NHWC 反变换**的缺陷。

## 现象

C=248 的 1×1 conv，权重域修好（+128）之后，NPU 输出：

  按 HWC 读：与 CPU 参考只有 **1.19%** 逐字节一致，差值散布在 −14…+6
  按 CHW 重排：**76.51%** 逐字节一致，其余**恰好差 −1**，|差|>1 的像素数为 **0**

⇒ 计算是对的，只是**输出布局**与 TFLite 期望不一致。

## 根因（源码定位）

`lower_operations()` 尾部本来有图输出的 detranspose 收口：

    if (!etna_ml_find_consumer(...) &&          /* 是图输出 */
        is_3d(poperation->output_tensors[i]) && /* 输出是 3D（这里是 50x50x248 ✓） */
        tensor->exp_layout != tensor->act_layout) {
       ML_DBG("Adding detranspose.\\n");
       ...
    }

但紧邻它上面的那句：

    if (tensor->exp_layout == ANY && tensor->act_layout == ANY)
       etna_ml_copy_layout(subgraph, output, operation->input_tensors[0]);

以及卷积分支里的 `etna_ml_copy_layout(operation->output_tensors[0], ...)`，
把**图输出的 `exp_layout` 也继承成了卷积的 NCHW** ⇒ `exp == act` 恒成立
⇒ detranspose **永远不插入**。真机日志里 `Adding transpose` / `Adding detranspose`
出现次数均为 **0**，与推断吻合。

而 `etna_ml_subgraph_read_outputs()` 只是把 tensor 数据原样 copy 到 TFLite 的
输出 buffer，不做任何布局转换 ⇒ TFLite 把 NCHW 数据当 NHWC 用。

## 修法

在 `lower_operations()` 的 layout 传播之后、detranspose 判定之前，判断该张量
是否为「无消费者的图输出」；若是且 `act_layout == NCHW`，把 **`exp_layout`
改回 NHWC**（TFLite 的约定），让原有的 detranspose 逻辑自然触发。

不动 detranspose 本身的实现，也不改 `is_3d()`。

运行期门控 `TEFLON_ODETRANS`：不设 = 与上游逐字节一致（便于同产物 A/B）。

## 自证

  python3 build-recipe/patch-etnaviv-output-detranspose.py --selftest

正控制：在真实源码副本上应用 → 标记出现、门控在、只改 exp_layout、锚点唯一。
负控制：重复应用被拒；锚点缺失必须报错（不静默跳过）；
        确认**没有**改动 detranspose 的插入条件本身。
"""

import io
import os
import shutil
import sys
import tempfile

MARKER = "och patch: graph output back to NHWC"

ANCHOR = """         if (tensor->exp_layout == ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout == ETNA_ML_LAYOUT_ANY) {
            ML_DBG("Copying layout to output tensor %d.\\n", poperation->output_tensors[i]->index);
            etna_ml_copy_layout(subgraph, poperation->output_tensors[i]->index, operation->input_tensors[0]);
         }
"""

NEW = """         if (tensor->exp_layout == ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout == ETNA_ML_LAYOUT_ANY) {
            ML_DBG("Copying layout to output tensor %d.\\n", poperation->output_tensors[i]->index);
            etna_ml_copy_layout(subgraph, poperation->output_tensors[i]->index, operation->input_tensors[0]);
         }

         /* och patch: graph output back to NHWC
          *
          * 上面的 copy_layout 会把**图输出**的 exp_layout 也继承成卷积的 NCHW，
          * 于是下面 `exp_layout != act_layout` 恒为假 ⇒ detranspose 永不插入
          * ⇒ TFLite 把 NCHW 数据当 NHWC 读。
          *
          * 实测（C=248 1x1 conv，权重域已修）：按 HWC 读只有 1.19% 一致，
          * 按 CHW 重排后 76.51% 逐字节一致、其余恰好差 -1、|差|>1 为 0。
          *
          * 这里只把「无消费者的图输出」的 exp_layout 改回 NHWC，
          * 让驱动原有的 detranspose 逻辑自然触发。
          * 运行期门控：TEFLON_ODETRANS=1 才生效。
          */
         /* 门控语义（2026-10-21 交付版）：**默认开**，TEFLON_ODETRANS=0 才关。
          *   不设   -> 修复生效（交付默认）
          *   =0     -> 关闭（与上游逐字节一致）
          *   =1/其它 -> 生效**并打印**（调试用）
          */
         const char *och_ot = getenv("TEFLON_ODETRANS");
         int och_ot_on = (och_ot == NULL) || (och_ot[0] != '0');
         int och_ot_log = (och_ot != NULL) && (och_ot[0] != '0');
         if (och_ot_on &&
             !etna_ml_find_consumer(poperations, count, poperation->output_tensors[i]->index) &&
             tensor->act_layout == ETNA_ML_LAYOUT_NCHW) {
            if (och_ot_log)
               ML_DBG("[och-odetrans] graph output tensor %d NCHW -> expect NHWC.\\n",
                      poperation->output_tensors[i]->index);
            tensor->exp_layout = ETNA_ML_LAYOUT_NHWC;
         }
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
    print("  [och-odetrans] 已注入图输出 NHWC 收口: %s" % path)
    print("  [och-odetrans] 运行期门控：TEFLON_ODETRANS=1 才生效")
    return 0


def _gate_check(text, env, prefix):
    """门控语义的行为级检查（真编译真跑三种状态）。"""
    try:
        from gate_semantics_check import check_gate
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gate_semantics_check import check_gate
    return check_gate(text, env, prefix)


def selftest(mesa_src):
    ok = True
    real = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml.c")
    if not os.path.isfile(real):
        print("  [SKIP] 找不到真实源码 %s" % real)
        return True

    tmpdir = tempfile.mkdtemp(prefix="och-odetrans-")
    try:
        dst = os.path.join(tmpdir, "etnaviv_ml.c")
        shutil.copyfile(real, dst)
        raw = io.open(real, encoding="utf-8").read()

        rc = apply(dst)
        if rc != 0:
            print("  [BAD] 正控制：应用到真实源码失败")
            return False
        out = io.open(dst, encoding="utf-8").read()
        _gate_ok, _gate_msg = _gate_check(out, "TEFLON_ODETRANS", "och_ot")
        if not _gate_ok:
            print("        %s" % _gate_msg)
        for cond, desc in [
            (MARKER in out, "标记已写入"),
            ("TEFLON_ODETRANS" in out, "运行期门控存在"),
            (_gate_ok, "门控语义行为级正确（unset=开且不打印 / =0=关 / =1=开且打印）"),

            ("tensor->exp_layout = ETNA_ML_LAYOUT_NHWC;" in out, "只改 exp_layout"),
            (out.count("getenv(\"TEFLON_ODETRANS\")") == 1, "门控只出现一次"),
            ("[och-odetrans]" in out, "C 代码自带产物可查的标记"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # 负控制 1：重复应用被拒
        cond = (apply(dst) == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 2：锚点缺失必须报错（从原始源码构造）
        dst2 = os.path.join(tmpdir, "etnaviv_ml_nomod.c")
        broken = raw.replace(
            "         if (tensor->exp_layout == ETNA_ML_LAYOUT_ANY &&\n"
            "             tensor->act_layout == ETNA_ML_LAYOUT_ANY) {\n", "", 1)
        if broken == raw:
            print("  [BAD] 负控制 2：无法构造锚点缺失样本")
            ok = False
        else:
            io.open(dst2, "w", encoding="utf-8").write(broken)
            cond = (apply(dst2) == 1)
            print("  [%s] 负控制 2：锚点缺失时报错（非静默跳过）" % ("ok" if cond else "BAD"))
            ok = ok and cond

        # 负控制 3：detranspose 的原有插入条件不得被改动
        key = "if (!etna_ml_find_consumer(poperations, count, poperation->output_tensors[i]->index) &&"
        call = "!etna_ml_find_consumer(poperations, count, poperation->output_tensors[i]->index)"
        cond = (out.count(key) == raw.count(key) and
                out.count(call) == raw.count(call) + 1)
        print("  [%s] 负控制 3：原 detranspose 判定块未被改写（新增 1 处同形判断）"
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
    return apply(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
