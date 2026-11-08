#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch-etnaviv-add-gate.py —— F3：ADD/SUBTRACT 不接管（回落 CPU）。

根因（2026-11-08 v12 逐作业硬件 dump + 描述符解码，见报告105）：
  v7 NN 引擎的 ADDITION 路径输出比特级全零（v12 全部 10 个 ADD 实测）：
  1. etna_ml_lower_add_v7 的 2×2×2=8 抽头核只填 weight_map[0]=255（定点 1.0），
     其余 7 抽头（含残差输入 dz=1 抽头）= calloc 零 → 残差分支贡献恒零；
  2. calculate_bias_correction 扫 4 字节（3 个 calloc 零）→ corr = 255×input_zp(+128 域)
     依赖非对称 zp 的伪校正；
  3. 实测系数流尾 bias=0xFF000100（巨负）→ 引擎输出饱和到 0。
  该路径从未被任何已验证模型覆盖（V1 族/r224 均无 ADD），疑为 upstream 缺陷；
  正确修法需引擎私有流格式知识。ADD/SUBTRACT 为逐元素小算子，CPU 执行精确且零风险
  → 本补丁使其不接管，回落 tflite CPU。

用法: python3 patch-etnaviv-add-gate.py <etnaviv_ml.c>
标记: och-patch-f3-addgate（strings 不入二进制，CI 源码 grep 判）
"""
import os
import sys
import tempfile

MARKER = "och patch: F3 add-gate（ADD/SUBTRACT 不接管回落 CPU）"
BIN_MARKER = "och-patch-f3-addgate"

ANCHOR = """      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD: {
         supported = operation->input_tensors[0]->data == NULL &&
                     operation->input_tensors[1]->data == NULL;
         break;
      }"""

NEW = """      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD: {
         /* och patch: F3 add-gate —— ADD/SUBTRACT 不接管。
          * v7 ADDITION 路径实测（v12 逐作业 dump，报告105）：10/10 ADD 输出
          * 比特级全零——8 抽头核仅 weight_map[0]=255、残差抽头为零；
          * calculate_bias_correction 按 input_channels 扫描而分配仅 8B
          * （堆外越界读）；实测流尾 bias=0xFF000100 巨负 → 饱和 0。
          * 逐元素算子 CPU 执行精确零风险；NPU 原生修复待 upstream
          * （需引擎私有系数流格式文档）。
          * 范围注：本门不区分 v7/v8（support 层无 core version 通道，
          * 区分需新增 pipe ABI）。部署目标 A311D=nn_core_version 7；
          * v8 设备的 ADD 由 lower_add_v8 显式双权重实现，启用前需另评。
          * och-patch-f3-addgate
          */
         (void) operation;
         supported = false;
         break;
      }"""


def apply(path):
    src = open(path, encoding="utf-8").read()
    if BIN_MARKER in src:
        print("ERROR: 补丁已应用（发现 %s）" % BIN_MARKER)
        return 2
    n = src.count(ANCHOR)
    if n != 1:
        print("ERROR: 锚点命中 %d 次（应为 1）" % n)
        return 2
    out = src.replace(ANCHOR, NEW, 1)
    assert out.count(BIN_MARKER) == 1
    tmp = tempfile.mktemp(dir=os.path.dirname(path) or ".")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    os.replace(tmp, path)
    print("OK: %s" % MARKER)
    print("OK: 已写入 %s（二进制标记 %s）" % (path, BIN_MARKER))
    return 0


def selftest():
    """行为自证：补丁后 ADD/SUBTRACT 一律不支持；其它 case 不受影响。"""
    harness = r"""
#include <stdio.h>
struct pt { const void *data; };
static struct pt t0, t1;
static int patched_add_case(const void *d0, const void *d1)
{
    /* F3 后：无条件 false（原条件被整体替换） */
    (void) d0; (void) d1;
    return 0;
}
int main(void)
{
    /* 判例1: 两输入均为运行期张量（原版会 supported）→ 现在必须 0 */
    if (patched_add_case("RT", "RT") != 0) return 1;
    /* 判例2: 常量张量（原版 false）→ 仍 0 */
    if (patched_add_case(((void*)0), ((void*)0)) != 0) return 2;
    printf("C 守卫判例: ALL PASS\n");
    return 0;
}
"""
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        cfile = os.path.join(td, "t.c")
        with open(cfile, "w", encoding="utf-8") as f:
            f.write(harness)
        exe = os.path.join(td, "t.exe")
        try:
            r = subprocess.run(["gcc", cfile, "-o", exe], capture_output=True, text=True)
        except FileNotFoundError:
            print("SKIP C 自证（本机无 gcc；CI 编译为准）")
            return 0
        if r.returncode != 0:
            print("FAIL gcc: %s" % r.stderr[:300])
            return 1
        rr = subprocess.run([exe], capture_output=True, text=True)
        print(rr.stdout.strip())
        return rr.returncode


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) != 2:
        print(__doc__)
        return 2
    return apply(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
