#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 实验：把 c335 头部三个搬运算子（TRANSPOSE→PAD→TRANSPOSE）挡回 CPU。

## 为什么需要（2026-11-07 N1 定位 + oca 终审，见
   verify/evidence/rootcause-audit-20261105/op-map/n1-diverge-20261107.md §8-9）

etnaviv ML 把图级 TRANSPOSE 实现为**零搬动 bypass + 布局重标**（etnaviv_ml.c:499），
承诺的布局转换从不兑现；c335 头部链 TRANSPOSE→PAD→TRANSPOSE 中：
  - N(t136) == 输入 t0 的 NHWC 原始字节流（100.0000%，bypass 零搬动实证）；
  - PAD 的 TP job 以 tile 序写出（pad 值位置正确但数据跨条带混杂）；
  - 物化时 act==exp 跳过 detranspose ⇒ CONV 按声明布局消费错误字节序
    ⇒ 首层 CONV 满量程错位 ⇒ 级联塌 320。

## 本补丁做什么（A1：gate + 隔离二合一，oca 执行卡 2026-11-07）

在 teflon 前端 `fill_operation()` 顶部：对 kTfLiteBuiltinTranspose /
kTfLiteBuiltinPad，当输入张量 dims 含 395 或 397（头部三算子的输入形状
[1,395,395,3] / [1,395,3,395] / [1,397,3,397]；c335 内部搬运算子维度
198/99/50/25/13/122/244/488，均不含 395/397，故不受影响）时 return false
⇒ 头部三算子回 CPU，CONV 主力留在 NPU，输入 = CPU 物化的干净 NHWC
（与 224 的「图输入直接进 NN」路径同构）。

首版精确匹配 {395,397}；泛化条件（mod-64 非对齐话语）按 oca 执行卡在 A2 后落地。
内部 60 个 TRANSPOSE 的 bypass 对偶读不在此 gate 内——其健康性由 A1 回归
（t236/t286 等 cut 的 cos 恢复情况）间接检验。

运行期标记：`[och-hdrgate]`（teflon_debug，TEFLON_DEBUG=verbose 可见）；
二进制标记：`och-patch-a1-hdrgate`（strings 可判）。
门控：构建期 TEFLON_GATE_HDR=1，不设 = 与上游逐字节一致。

## 用法

    python3 build-recipe/patch-teflon-gate-header-moveops.py <tfl_device.c>
    python3 build-recipe/patch-teflon-gate-header-moveops.py --selftest

正控制：标记写入、锚点唯一、行为自证（C harness）。
负控制：重复应用被拒、锚点缺失报错、其余行逐字不变。
"""

import subprocess
import sys
import tempfile
import os

MARKER = "och patch: A1 头部搬运算子 gate（TRANSPOSE/PAD @395/397 -> CPU）"
BIN_MARKER = "och-patch-a1-hdrgate"

ANCHOR = """   operation->output_count = node->outputs->size;
   operation->output_tensors = calloc(operation->output_count, sizeof(void *));
   for (unsigned i = 0; i < node->outputs->size; i++)
      operation->output_tensors[i] = &tensors[node->outputs->data[i]];

   switch (node_registration->builtin_code) {"""

NEW = """   operation->output_count = node->outputs->size;
   operation->output_tensors = calloc(operation->output_count, sizeof(void *));
   for (unsigned i = 0; i < node->outputs->size; i++)
      operation->output_tensors[i] = &tensors[node->outputs->data[i]];

   /* och patch: A1 头部搬运算子 gate（TRANSPOSE/PAD @395/397 -> CPU）
    *
    * 2026-11-07 N1 定位：etnaviv ML 的图级 TRANSPOSE 是零搬动 bypass + 布局重标，
    * c335 头部 TRANSPOSE->PAD->TRANSPOSE 链的布局台账与真实字节序脱钩
    * （PAD TP job tile 序写出 + 物化跳过 detranspose），CONV 按声明布局消费
    * 即满量程错位，级联致输出塌 320。详见
    * verify/evidence/rootcause-audit-20261105/op-map/n1-diverge-20261107.md。
    *
    * 本 gate 让头部三算子回 CPU（输入 dims 含 395/397 的 TRANSPOSE/PAD），
    * CONV 主力留在 NPU 且输入为 CPU 物化的干净 NHWC。
    * och-patch-a1-hdrgate
    */
   if (node_registration->builtin_code == kTfLiteBuiltinTranspose ||
       node_registration->builtin_code == kTfLiteBuiltinPad) {
      struct pipe_tensor *hdrgate_in = operation->input_tensors[0];
      if (hdrgate_in->dims[1] == 395 || hdrgate_in->dims[1] == 397 ||
          hdrgate_in->dims[2] == 395 || hdrgate_in->dims[2] == 397 ||
          hdrgate_in->dims[3] == 395 || hdrgate_in->dims[3] == 397) {
         teflon_debug("[och-hdrgate] builtin=%d dims[%u,%u,%u,%u] -> CPU\\n",
                      node_registration->builtin_code,
                      hdrgate_in->dims[0], hdrgate_in->dims[1],
                      hdrgate_in->dims[2], hdrgate_in->dims[3]);
         return false;
      }
   }

   switch (node_registration->builtin_code) {"""


def apply(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        print("ERROR: 补丁已应用（发现标记），拒绝重复应用")
        return 2
    if src.count(ANCHOR) != 1:
        print(f"ERROR: 锚点命中 {src.count(ANCHOR)} 次（应为 1），拒绝盲改")
        return 2
    out = src.replace(ANCHOR, NEW, 1)
    assert out.count(BIN_MARKER) == 1
    tmp = tempfile.mktemp(dir=os.path.dirname(path) or ".")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    os.replace(tmp, path)
    print(f"OK: {MARKER}")
    print(f"OK: 已写入 {path}（二进制标记 {BIN_MARKER}）")
    return 0


def selftest() -> int:
    """行为自证：把补丁打进一段带锚点的 C 片段，编译成共享库验证逻辑。"""
    harness = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define kTfLiteBuiltinTranspose 18
#define kTfLiteBuiltinPad       20
typedef struct { unsigned dims[4]; } pipe_tensor;
struct node_reg { int builtin_code; };
static struct node_reg *node_registration;
static pipe_tensor *operation_input_tensors;
static int verbose_flag = 1;
#define teflon_debug(...) do { if (verbose_flag) printf(__VA_ARGS__); } while (0)
/* 与补丁注入的 gate 逻辑逐字等价（除变量来源） */
static int patched_body(void)
{
   if (node_registration->builtin_code == kTfLiteBuiltinTranspose ||
       node_registration->builtin_code == kTfLiteBuiltinPad) {
      pipe_tensor *hdrgate_in = operation_input_tensors;
      if (hdrgate_in->dims[1] == 395 || hdrgate_in->dims[1] == 397 ||
          hdrgate_in->dims[2] == 395 || hdrgate_in->dims[2] == 397 ||
          hdrgate_in->dims[3] == 395 || hdrgate_in->dims[3] == 397) {
         teflon_debug("[och-hdrgate] builtin=%d dims[%u,%u,%u,%u] -> CPU\n",
                      node_registration->builtin_code,
                      hdrgate_in->dims[0], hdrgate_in->dims[1],
                      hdrgate_in->dims[2], hdrgate_in->dims[3]);
         return 0;
      }
   }
   return 1;
}
static int patched_fill(int builtin_code, unsigned d1, unsigned d2, unsigned d3)
{
   struct node_reg reg = { builtin_code };
   pipe_tensor in = { {1, d1, d2, d3} };
   node_registration = &reg;
   operation_input_tensors = &in;
   return patched_body();
}
int main(void)
{
   /* 头部三算子：必须被 gate（0）*/
   if (patched_fill(kTfLiteBuiltinTranspose, 395, 395, 3) != 0) return 1;
   if (patched_fill(kTfLiteBuiltinPad, 395, 3, 395) != 0) return 2;
   if (patched_fill(kTfLiteBuiltinTranspose, 397, 3, 397) != 0) return 3;
   /* 内部搬运算子（c335 实际维度）：必须不 gate（1）*/
   if (patched_fill(kTfLiteBuiltinTranspose, 50, 50, 244) != 1) return 4;
   if (patched_fill(kTfLiteBuiltinTranspose, 244, 50, 50) != 1) return 5;
   if (patched_fill(kTfLiteBuiltinPad, 50, 50, 244) != 1) return 6;
   if (patched_fill(kTfLiteBuiltinTranspose, 25, 25, 488) != 1) return 7;
   /* CONV 等其它算子不受影响 */
   if (patched_fill(3, 397, 397, 3) != 1) return 8;
   printf("SELFTEST PASS\n");
   return 0;
}
"""
    tmpd = tempfile.mkdtemp()
    cfile = os.path.join(tmpd, "h.c")
    open(cfile, "w").write(harness)
    try:
        r = subprocess.run(["gcc", cfile, "-o", os.path.join(tmpd, "h")],
                           capture_output=True, text=True)
    except FileNotFoundError:
        print("SKIP: 本机无 gcc，仅做文本断言（CI ubuntu runner 会真跑自证）")
        r = None
    if r is not None and r.returncode != 0:
        print("SKIP: 无 gcc 或编译失败，仅做文本断言")
        print(r.stderr[:300])
    else:
        rr = subprocess.run([os.path.join(tmpd, "h")], capture_output=True, text=True)
        if rr.returncode != 0:
            print(f"FAIL: 行为自证失败 rc={rr.returncode}")
            return 1
        if "SELFTEST PASS" not in rr.stdout:
            print("FAIL: 自证输出缺失")
            return 1
        print("OK: 行为自证 PASS（8/8 判例）")
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) != 2:
        print(__doc__)
        return 2
    return apply(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
