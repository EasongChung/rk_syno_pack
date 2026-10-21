#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 NN 描述符的 `rounding_mode` 做成**运行期门控探针**，用于实测它的取值含义。

## 背景（报告 42）

权重域（+128）与图输出收口（NCHW→NHWC）都修好之后，`both` 组与 CPU 基准比：

    完全一致 474339/620000 = 76.51%
    差值分布 {0: 474339, -1: 145661}      ← 只有 0 和 −1
    npu<cpu = 145661   npu>cpu = 0         ← 单侧，方向锁死

这是**舍入方式**差异，不是算术错误：

    硬件：(acc * post_mul) >> shift         向 −∞ 截断
    TFLite：RoundingDivideByPOT            四舍五入

## 为什么只能实测

    etnaviv_ml_nn.c:71    FIELD(rounding_mode, 2)     ← 2 bit，4 个可能值
    etnaviv_ml_nn.c:1042  map->rounding_mode = 0x1;   ← 上游固定写 1

仓库与 `/root/clawtemp/mesa_src` 里**没有任何取值含义的注释或枚举**，
`etnaviv_ml_tp.c` 的 `flat_rounding_mode` / `integer_rounding_mode` 同样只赋 0x1。
⇒ 唯一可靠的办法是在真机上把 0/1/2/3 各跑一遍，看哪个与 TFLite 一致。

## 为什么不用「bias 补偿」

加 δ 个 acc 单位到 bias 等价于把判决门限从 0 挪到 δ × 0.430524：

    δ=1 → 0.4305（残差 −0.0695）    δ=2 → 0.8610（残差 +0.3610）

0.5 / 0.430524 = 1.1613，**没有整数 δ 能精确等于 0.5** ⇒ bias 只能近似改善，
不是修复。故先做本探针。

## 门控

    TEFLON_ROUNDMODE=0|1|2|3   覆盖描述符字段（不设 = 与上游逐字节一致）
    非法值（<0 或 >3）不覆盖，只打一行告警

前 4 次生效各打一行 `[och-roundmode] rounding_mode -> N`（避免 28 个卷积刷屏），
既便于产物级 `strings` 校验，也便于日志侧确认真的走到。

## 自证

  python3 build-recipe/patch-etnaviv-roundmode.py --selftest

正控制：真实源码副本上标记出现、门控在、范围钳制在、默认值 0x1 未被改动。
负控制：重复应用被拒；锚点缺失必须报错；原文件所有其它行逐字不变。
"""

import io
import os
import shutil
import sys
import tempfile

MARKER = "och patch: rounding-mode probe"

ANCHOR = "   map->rounding_mode = 0x1;\n"

NEW = """   map->rounding_mode = 0x1;

   /* och patch: rounding-mode probe
    *
    * Mesa 固定写 0x1，实测行为是「向 -inf 截断」，而 TFLite 参考实现是
    * 四舍五入（RoundingDivideByPOT）。C=248 1x1 conv 上表现为：76.51%
    * 逐字节一致、其余恰好差 -1、npu>cpu 恒为 0（报告 42）。
    *
    * rounding_mode 是 2 bit 字段（本文件 FIELD(rounding_mode, 2)），上游没有
    * 任何取值含义的注释或枚举 ⇒ 只能实测。这里做成运行期门控探针：
    *
    *   TEFLON_ROUNDMODE=0..3  覆盖默认值（不设 = 与上游逐字节一致）
    *
    * 非法值不覆盖，只告警；前 4 次生效各打一行，避免 28 个卷积刷屏。
    */
   {
      const char *och_rm = getenv("TEFLON_ROUNDMODE");

      if (och_rm != NULL) {
         const int och_v = atoi(och_rm);

         if (och_v < 0 || och_v > 3) {
            fprintf(stderr, "[och-roundmode] invalid value %d (want 0..3), keep 1\\n", och_v);
         } else {
            static int och_rm_reports = 0;

            map->rounding_mode = (unsigned)och_v;
            if (och_rm_reports < 4) {
               fprintf(stderr, "[och-roundmode] rounding_mode -> %d\\n", och_v);
               och_rm_reports++;
            }
         }
      }
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
        sys.stderr.write("ERROR: 锚点 `map->rounding_mode = 0x1;` 在 %s 中出现 %d 次"
                         "（要求恰好 1 次）。\n"
                         "       源码可能已升级/被其它补丁改动，拒绝盲目套用。\n"
                         % (path, n))
        return 1
    io.open(path, "w", encoding="utf-8").write(src.replace(ANCHOR, NEW, 1))
    print("  [och-roundmode] 已注入舍入模式探针: %s" % path)
    print("  [och-roundmode] 运行期门控：TEFLON_ROUNDMODE=0..3（不设 = 上游行为）")
    return 0


def selftest(mesa_src):
    ok = True
    real = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml_nn.c")
    if not os.path.isfile(real):
        print("  [SKIP] 找不到真实源码 %s" % real)
        return True

    tmpdir = tempfile.mkdtemp(prefix="och-roundmode-")
    try:
        dst = os.path.join(tmpdir, "etnaviv_ml_nn.c")
        shutil.copyfile(real, dst)
        raw = io.open(real, encoding="utf-8").read()

        if apply(dst) != 0:
            print("  [BAD] 正控制：应用到真实源码失败")
            return False
        out = io.open(dst, encoding="utf-8").read()

        for cond, desc in [
            (MARKER in out, "标记已写入"),
            (out.count('getenv("TEFLON_ROUNDMODE")') == 1, "运行期门控唯一"),
            ("[och-roundmode]" in out, "C 代码自带产物可查的标记"),
            ("atoi(och_rm)" in out, "取值解析存在"),
            ("och_v < 0 || och_v > 3" in out, "取值范围钳制存在"),
            # 负控制性质的正控：默认行为必须原样保留
            (out.count("   map->rounding_mode = 0x1;\n") == 1,
             "默认值 0x1 原样保留（不设环境变量 = 上游行为）"),
            (out.count("map->rounding_mode") == 2,
             "只在原赋值 + 门控覆盖处各写一次，无第三处"),
        ]:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # 负控制 1：重复应用被拒
        cond = (apply(dst) == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 2：锚点缺失必须报错
        dst2 = os.path.join(tmpdir, "nn_nomod.c")
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

        # 负控制 3：原文件的其余每一行都必须还在（只插入，不改写）
        raw_lines = [l for l in raw.splitlines() if l.strip()]
        out_set = io.open(dst, encoding="utf-8").read()
        missing = [l for l in raw_lines if l not in out_set]
        cond = (missing == [])
        print("  [%s] 负控制 3：原有非空行全部保留（缺 %d 行）"
              % ("ok" if cond else "BAD", len(missing)))
        ok = ok and cond
        for l in missing[:3]:
            print("        MISSING: %s" % l[:100])
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
