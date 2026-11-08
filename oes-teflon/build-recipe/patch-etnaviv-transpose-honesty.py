#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch-etnaviv-transpose-honesty.py —— F2 转置诚实性补丁（etnaviv_ml.c，两站点）。

根因（2026-11-08 slot 代数定谳，详见 verify/evidence/rootcause-audit-20261105/op-map/）：
  etnaviv_ml.c 对显式 TRANSPOSE 无条件 add_bypass（零搬动 + act=exp=NCHW 硬编码）。
  bypass 仅在「输入字节序恰为输出声明序」时精确：
    - T[0,3,1,2] + 输入 act=NCHW（conv 输出后）：精确 ✓（muse 块间 op#12 型，17 处）
    - T[0,3,1,2] + 输入 NHWC：不精确 → 必须真 TP_TRANSPOSE
    - T[0,2,3,1]：**永不精确**（consumer 读得 T_in[c,h,w]，逻辑要求 T_in[w,c,h]，
      H==W 时表现为空间 h↔w 翻转级腐败）——现状无条件 bypass 直接投毒（29 处）

修复：
  站点1（etna_ml_operation_supported）：[0,2,3,1] 移出支持矩阵 → 落 CPU
  （CPU row-major 读写与子图边界天然一致 = 构造性诚实；TP_DETRANSPOSE 配置
   只对规范 dims 成立，不复用到 [1,C,H,W] 形状）。
  站点2（lower_operations TRANSPOSE case）：[0,3,1,2] bypass 加守卫——
  输入 act==NCHW 才 bypass（现网行为不变，零回归）；否则真 TP_TRANSPOSE
  （etna_ml_lower_transpose，r224 输入隐式转置路径，设备已证）。

推演结论：两站点后每个 conv 输入要么 NPU-NCHW（bypass 态）要么 CPU 真 NHWC
（隐式真转置接管），:344 边界谎言对 conv 全部消失，无需改边界标签逻辑。

自证：
  1) numpy slot 代数：可执行证明有效性表（bypass==真转置仅在表内成立）
  2) gcc：把站点2守卫逻辑编译执行，判例全过

用法: python3 build-recipe/patch-etnaviv-transpose-honesty.py <etnaviv_ml.c>
标记: och-patch-f2-transpose-honesty（strings 可判）
"""
import os
import subprocess
import sys
import tempfile

MARKER = "och patch: F2 转置诚实性（bypass 有效性守卫 + [0,2,3,1] 退役）"
BIN_MARKER = "och-patch-f2-transpose-honesty"

ANCHOR_SUP = """      case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 3 &&
             operation->transpose.perm[2] == 1 &&
             operation->transpose.perm[3] == 2)
            supported = true;  /* Channels-last to channels-first */
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 2 &&
             operation->transpose.perm[2] == 3 &&
             operation->transpose.perm[3] == 1)
            supported = true;  /* Channels-first to channels-last */
         break;
      }"""

NEW_SUP = """      case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 3 &&
             operation->transpose.perm[2] == 1 &&
             operation->transpose.perm[3] == 2)
            supported = true;  /* Channels-last to channels-first */
         /* och patch: F2 转置诚实性 —— [0,2,3,1]（channels-first to last）退役。
          * 该方向的零搬动 bypass 永不精确（slot 代数：consumer 读得 T_in[c,h,w]，
          * 逻辑要求 T_in[w,c,h]），过去 29 处直接投毒。落 CPU 执行 =
          * row-major 读写与子图边界天然一致，构造性诚实。
          * och-patch-f2-transpose-honesty
          */
         break;
      }"""

ANCHOR_LOW = """         case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
            add_bypass(poperation, input_tensors[0], operation, etna_operations);
            struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, operation->output_tensors[0]);
            tensor->act_layout = ETNA_ML_LAYOUT_NCHW;
            tensor->exp_layout = ETNA_ML_LAYOUT_NCHW;

            break;
         }"""

NEW_LOW = """         case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
            /* och patch: F2 转置诚实性 —— bypass 有效性守卫。
             * 零搬动 bypass 仅当输入字节已是 C-outer（act==NCHW，conv 输出态）时
             * 与真转置逐字节等价；否则走真 TP_TRANSPOSE（r224 输入隐式转置路径，
             * 设备已证）。输出标签 act=exp=NCHW：字节为 C-outer，
             * 边界 (act==exp) 不发 detranspose，交 tflite/CPU 的即 row-major。
             * och-patch-f2-transpose-honesty
             */
            struct etna_ml_tensor *tfix_in =
               etna_ml_get_tensor(subgraph, input_tensors[0]);
            if (tfix_in->act_layout == ETNA_ML_LAYOUT_NCHW) {
               add_bypass(poperation, input_tensors[0], operation, etna_operations);
            } else {
               ML_DBG("Adding real transpose (bypass invalid).\\n");
               etna_ml_lower_transpose(subgraph, poperation->input_tensors[0],
                                       operation);
               operation->input_tensors[0] = input_tensors[0];
               operation->output_tensors[0] = poperation->output_tensors[0]->index;
               list_addtail(&operation->link, etna_operations);
            }
            struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, operation->output_tensors[0]);
            tensor->act_layout = ETNA_ML_LAYOUT_NCHW;
            tensor->exp_layout = ETNA_ML_LAYOUT_NCHW;

            break;
         }"""


def apply(path):
    src = open(path, encoding="utf-8").read()
    if BIN_MARKER in src:
        print("ERROR: 补丁已应用（发现 %s），拒绝重复" % BIN_MARKER)
        return 2
    for name, anchor in (("站点1", ANCHOR_SUP), ("站点2", ANCHOR_LOW)):
        n = src.count(anchor)
        if n != 1:
            print("ERROR: %s 锚点命中 %d 次（应为 1），拒绝盲改" % (name, n))
            return 2
    out = src.replace(ANCHOR_SUP, NEW_SUP, 1).replace(ANCHOR_LOW, NEW_LOW, 1)
    assert out.count(BIN_MARKER) == 2
    tmp = tempfile.mktemp(dir=os.path.dirname(path) or ".")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    os.replace(tmp, path)
    print("OK: %s" % MARKER)
    print("OK: 已写入 %s（二进制标记 %s）" % (path, BIN_MARKER))
    return 0


def selftest_numpy():
    """有效性表可执行证明。

    比较对象（slot 代数的数值化）：
      bypass_read  = 零搬动后，消费者按 NCHW-role 读输出 dims（= buf.reshape(输出C先序)）
      required_log = tflite 真转置的逻辑结果（x_log.transpose(perm)）
    bypass 合法 ⟺ 两者逐元素相等。
    """
    import numpy as np
    fails = []
    H = W = C = 4  # 方形特征图（c335 全部 50/25/13 平方）
    rng = np.random.RandomState(7)
    x_log = rng.randint(-128, 128, size=(1, H, W, C)).astype(np.int64)
    bytes_nhwc = x_log.ravel().copy()                        # row-major [H,W,C]
    bytes_couter = np.transpose(x_log, (0, 3, 1, 2)).ravel().copy()  # C-outer

    def consumer_read(buf, out_dims_cfirst):
        # NCHW-role 读：线性 = c*H*W + h*W + w == reshape(C,H,W)
        return buf.reshape(out_dims_cfirst)

    # 判例1: T[0,3,1,2]，输入 C-outer（conv 输出态）→ bypass == 真转置（合法）
    got = consumer_read(bytes_couter, (C, H, W))
    req = x_log.transpose(0, 3, 1, 2)          # out[c,h,w] = x_log[h,w,c]
    if not np.array_equal(got, req[0]):
        fails.append("判例1 应等价")

    # 判例2: T[0,3,1,2]，输入 NHWC → bypass != 真转置（须真搬运）
    got = consumer_read(bytes_nhwc, (C, H, W))
    req = x_log.transpose(0, 3, 1, 2)
    if np.array_equal(got, req[0]):
        fails.append("判例2 应不等价")

    # 判例3: T[0,2,3,1]，输入 C-outer（[1,C,H,W] dims）→ bypass != 真转置（投毒实证）
    got = consumer_read(bytes_couter, (H, W, C))
    req = x_log.transpose(0, 2, 3, 1)          # out[h,w,c] = x_log[c,h,w]
    if np.array_equal(got, req[0]):
        fails.append("判例3 应不等价")

    # 判例4: 判例3 的交付物与真结果的精确关系（文档化腐败形态）：
    # bypass 交付 = 真结果的 (1,2,0) 轴循环置换（数值唯一匹配，非简单翻转）
    got = consumer_read(bytes_couter, (H, W, C))
    req = x_log.transpose(0, 2, 3, 1)
    if not np.array_equal(got, np.transpose(req[0], (1, 2, 0))):
        fails.append("判例4 轴置换关系不成立")

    for f in fails:
        print("FAIL " + f)
    print("numpy 有效性表: %s" % ("FAILED" if fails else "ALL PASS"))
    return 1 if fails else 0


def selftest_c():
    """站点2守卫逻辑 C 自证（判例驱动）。"""
    harness = r"""
#include <stdio.h>
#include <stdint.h>
#define ETNA_ML_LAYOUT_ANY   0
#define ETNA_ML_LAYOUT_NHWC  1
#define ETNA_ML_LAYOUT_NCHW  2
static int verbose_flag = 1;
#define ML_DBG(...) do { if (verbose_flag) printf(__VA_ARGS__); } while (0)
struct ml_tensor { int act_layout; int exp_layout; };
static struct ml_tensor tensors[8];
static struct ml_tensor *etna_ml_get_tensor(int idx) { return &tensors[idx]; }
static int bypass_called, real_called;
static void add_bypass(void) { bypass_called++; }
static void etna_ml_lower_transpose(void) { real_called++; }
#define list_addtail(op, list) do { } while (0)
struct list_head;
struct etna_operation { int pad; };
static struct etna_operation the_op;
static int input_tensors0;
static int pop_out_index;
static struct etna_operation *operation = &the_op;

/* 与补丁注入的站点2守卫逐字等价（除类型细节） */
static int patched_transpose_case(void)
{
   struct ml_tensor *tfix_in = etna_ml_get_tensor(input_tensors0);
   if (tfix_in->act_layout == ETNA_ML_LAYOUT_NCHW) {
      add_bypass();
   } else {
      ML_DBG("Adding real transpose (bypass invalid).\n");
      etna_ml_lower_transpose();
   }
   struct ml_tensor *tensor = etna_ml_get_tensor(pop_out_index);
   tensor->act_layout = ETNA_ML_LAYOUT_NCHW;
   tensor->exp_layout = ETNA_ML_LAYOUT_NCHW;
   return 0;
}

int main(void)
{
   /* 判例1: act=NCHW → bypass */
   tensors[1].act_layout = ETNA_ML_LAYOUT_NCHW;
   input_tensors0 = 1; pop_out_index = 2; bypass_called = real_called = 0;
   patched_transpose_case();
   if (bypass_called != 1 || real_called != 0) return 1;
   if (tensors[2].act_layout != ETNA_ML_LAYOUT_NCHW) return 2;
   /* 判例2: act=NHWC → 真 TP_TRANSPOSE */
   tensors[3].act_layout = ETNA_ML_LAYOUT_NHWC;
   input_tensors0 = 3; pop_out_index = 4; bypass_called = real_called = 0;
   patched_transpose_case();
   if (bypass_called != 0 || real_called != 1) return 3;
   /* 判例3: act=ANY（未知）→ 不 bypass，真搬运兜底 */
   tensors[5].act_layout = ETNA_ML_LAYOUT_ANY;
   input_tensors0 = 5; pop_out_index = 6; bypass_called = real_called = 0;
   patched_transpose_case();
   if (bypass_called != 0 || real_called != 1) return 4;
   printf("C 守卫判例: ALL PASS\n");
   return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        cfile = os.path.join(td, "t.c")
        with open(cfile, "w", encoding="utf-8") as f:
            f.write(harness)
        exe = os.path.join(td, "t.exe")
        try:
            r = subprocess.run(["gcc", cfile, "-o", exe],
                               capture_output=True, text=True)
        except FileNotFoundError:
            print("SKIP C 自证（本机无 gcc；CI 编译+AMD 盒复跑为准）")
            return 0
        if r.returncode != 0:
            print("FAIL gcc: %s" % r.stderr[:400])
            return 1
        rr = subprocess.run([exe], capture_output=True, text=True)
        print(rr.stdout.strip())
        if rr.returncode != 0:
            print("FAIL 判例 exit=%d" % rr.returncode)
            return 1
    return 0


def selftest():
    rc = selftest_numpy()
    rc |= selftest_c()
    print("== 自证 %s ==" % ("FAILED" if rc else "ALL PASS"))
    return rc


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) != 2:
        print(__doc__)
        return 2
    return apply(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
