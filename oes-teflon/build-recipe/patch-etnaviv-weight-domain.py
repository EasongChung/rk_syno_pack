#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 etnaviv ML 的 **per-tensor** 卷积权重补上 int8 → uint8 域转换（+128）。

## 为什么

`etnaviv_ml_nn.c` 的 `etna_tensor_zero_point()` 注释已经把约定写死了：

    Since the hardware only supports unsigned 8-bit integers, signed tensors
    are shifted from the -128..127 range to 0..255 by adding 128 when
    uploading and subtracting 128 when downloading the tensor.
    Tensor zero point and weight coefficients have to be adapted to account
    for this.

实际实现只做了两处：

1. **输入**：`etna_ml_subgraph_invoke()` 里 `dst_map[k] = src[k] + 128`
2. **zero point**：`etna_tensor_zero_point()` 返回 `zero_point + 128`

而**权重系数**的 +128 只在 **per-channel** 路径里由 `requantize_weights()`
完成（`map_out = requantized + weight_zero_point - 0x80`）。
**per-tensor 路径**（`weight_tensor->scales == NULL`）走的是裸 `memcpy`，
于是 int8 的 `0x01 / 0x00` 被原样塞进系数 BO，而描述符里
`coef_zero_point = 128` ⇒ 硬件按 `字节 - 128` 解释 ⇒ 每个权重变成 −128/−127。

实测（C=248，1×1 conv，权重=单位阵）：

    [och-coef] nonzero=1607 head=40 1f 00 00 ...   ← 系数 BO 有数据
    [och-coef] wscale=0.430524 wzp=128 wt_nonzero=248 wt_head=01 00 00 00
    描述符解码: coef_zero_point = 128

⇒ 累加巨额负数 ⇒ ReLU 全钳零；而且把权重换成全 1、把偏置换成 +20000，
NPU 输出**逐字节不变**（三次 hash 都是 0xbb9e90a3c13f4a03），完全吻合。

## 怎么修

在 `etna_ml_lower_convolution()` 的权重复制之后，**仅当**满足全部条件时逐字节 +128：

  * 环境变量 `TEFLON_WDOMAIN` 已设置（运行期门控，便于同产物 A/B）
  * `weight_tensor->is_signed`（int8 张量）
  * `weight_tensor->scales == NULL`（per-tensor；per-channel 已被
    `requantize_weights()` 正确转换，**不能**重复加）

## 为什么门控而不是直接改

同一次上机里 A/B 对比（不设 = 旧行为全 0；设 = 应有实数输出），
避免「改完变好了但无法归因」。默认不设 = 与上游完全一致。

## 自证

  python3 build-recipe/patch-etnaviv-weight-domain.py --selftest

正控制：在真实源码副本上应用 → 标记出现、锚点唯一、`+128` 只在 conv 路径。
负控制：重复应用被拒；锚点被人为改动后必须报错（不静默跳过）。
"""

import io
import os
import re
import shutil
import sys
import tempfile

MARKER = "och patch: weight int8->uint8 domain"

ANCHOR = """   operation->weight_tensor = malloc(weight_size);
   memcpy(operation->weight_tensor, poperation->conv.weight_tensor->data, weight_size);
   operation->weight_tensor_size = weight_size;
"""

NEW = """   operation->weight_tensor = malloc(weight_size);
   memcpy(operation->weight_tensor, poperation->conv.weight_tensor->data, weight_size);
   operation->weight_tensor_size = weight_size;

   /* och patch: weight int8->uint8 domain
    *
    * 硬件只认无符号 8 位（见 etna_tensor_zero_point 的注释），输入与 zero point
    * 都已经 +128，唯独 **per-tensor 的权重系数** 漏了这一步：
    *   - per-channel 路径由 requantize_weights() 输出 uint8 域（+wzp-0x80），正确；
    *   - per-tensor 路径走裸 memcpy，int8 的 0/1 原样进系数 BO，
    *     而描述符 coef_zero_point=128 ⇒ 硬件算成 -128/-127 ⇒ ReLU 后全零。
    *
    * 运行期门控：不设 TEFLON_WDOMAIN 时行为与上游逐字节一致。
    */
   if (getenv("TEFLON_WDOMAIN") != NULL &&
       poperation->conv.weight_tensor->is_signed &&
       poperation->conv.weight_tensor->scales == NULL) {
      uint8_t *och_wp = (uint8_t *)operation->weight_tensor;
      for (unsigned och_i = 0; och_i < weight_size; och_i++)
         och_wp[och_i] = (uint8_t)(och_wp[och_i] + 128);
      fprintf(stderr, "[och-wdom] weights shifted +128: size=%u (signed=1 per_tensor=1)\\n",
              weight_size);
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
        sys.stderr.write(
            "ERROR: 锚点在 %s 中出现 %d 次（要求恰好 1 次）。\n"
            "       源码可能已升级/被其它补丁改动，拒绝盲目套用。\n" % (path, n))
        return 1

    io.open(path, "w", encoding="utf-8").write(src.replace(ANCHOR, NEW, 1))
    print("  [och-wdom] 已注入权重域转换: %s" % path)
    print("  [och-wdom] 运行期门控：TEFLON_WDOMAIN=1 才生效")
    return 0


def selftest(mesa_src):
    ok = True
    real = os.path.join(mesa_src, "src/gallium/drivers/etnaviv/etnaviv_ml_nn.c")
    if not os.path.isfile(real):
        print("  [SKIP] 找不到真实源码 %s" % real)
        return True

    tmpdir = tempfile.mkdtemp(prefix="och-wdom-")
    try:
        dst = os.path.join(tmpdir, "etnaviv_ml_nn.c")
        shutil.copyfile(real, dst)

        # 正控制：应用成功、标记出现、+128 落在 conv 路径、per-channel 条件在
        rc = apply(dst)
        if rc != 0:
            print("  [BAD] 正控制：应用到真实源码失败")
            return False
        out = io.open(dst, encoding="utf-8").read()
        checks = [
            (MARKER in out, "标记已写入"),
            ("TEFLON_WDOMAIN" in out, "运行期门控存在"),
            ("scales == NULL" in out, "per-tensor 判据存在"),
            ("is_signed" in out, "signed 判据存在"),
            (out.count("och_wp[och_i] = (uint8_t)(och_wp[och_i] + 128)") == 1,
             "+128 只加一次"),
        ]
        for cond, desc in checks:
            print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
            ok = ok and cond

        # 负控制 1：重复应用必须被拒
        rc2 = apply(dst)
        cond = (rc2 == 1)
        print("  [%s] 负控制 1：重复应用被拒" % ("ok" if cond else "BAD"))
        ok = ok and cond

        # 负控制 2：锚点被改动（模拟上游升级）必须报错，绝不静默跳过。
        # 注意：必须从**原始源码**构造，否则样本自带 MARKER，测到的是「重复应用」。
        dst2 = os.path.join(tmpdir, "etnaviv_ml_nn_modified.c")
        raw = io.open(real, encoding="utf-8").read()
        broken = raw.replace(
            "   operation->weight_tensor = malloc(weight_size);\n"
            "   memcpy(operation->weight_tensor, poperation->conv.weight_tensor->data, weight_size);\n",
            "   operation->weight_tensor = calloc(1, weight_size);\n", 1)
        if broken == raw:
            print("  [BAD] 负控制 2：无法构造锚点缺失样本")
            ok = False
        else:
            io.open(dst2, "w", encoding="utf-8").write(broken)
            rc3 = apply(dst2)
            cond = (rc3 == 1)
            print("  [%s] 负控制 2：锚点缺失时报错（非静默跳过）" % ("ok" if cond else "BAD"))
            ok = ok and cond

        # 负控制 3：per-channel 路径（requantize_weights）不得被本补丁改写
        cond = out.count("map_out[oc][w][h][ic] = requantized + operation->weight_zero_point - 0x80;") == \
               io.open(real, encoding="utf-8").read().count(
                   "map_out[oc][w][h][ic] = requantized + operation->weight_zero_point - 0x80;")
        print("  [%s] 负控制 3：per-channel 重量化代码未被改动" % ("ok" if cond else "BAD"))
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
