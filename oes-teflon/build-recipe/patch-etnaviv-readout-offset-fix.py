#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：read_outputs 的映射范围与读取范围对齐（与 invoke 同构）。

## 根因（真机运行时数据，`0078f5b2` + `TEFLON_RODIAG=1`）

    etna_ml_subgraph_read_outputs+0x220 → etnaviv_ml.c:1095
      = `((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;`

    [och-ro] idx=151 n_tensors=152 IN  bsize=5000  off_expected=610000

map 长度用 `pipe_buffer_size(res)`（5000），循环上界用 `tensor->size`（610000）
⇒ **越界读 605000 字节** → SIGSEGV。

三条假说判定：**H1 排除**（idx 全部 IN）、**H2 确认**、**H3 排除**（src_map 合法）。

## 本补丁

1. 边界检查：`idx >= n_tensors` 跳过（`etna_ml_get_tensor` 越界会**写入**野地址）。
2. map 参数改成 `(tensor->offset, tensor->size)`，与 `invoke`（`etnaviv_ml.c:908`）同构。

**不加 min() 截断** —— resource 真不够大时 map 返回 NULL → assert 明确失败。
把「越界」换成「静默错误数值」比崩更糟。

## 互斥

与 `TEFLON_RODIAG=1` 互斥（RODIAG 也改这个循环体，锚点冲突 ⇒ 门会报错）。
由 `build-teflon.sh` 显式检查。

## 用法

    python3 patch-etnaviv-readout-offset-fix.py <etnaviv_ml.c 路径>
"""

import argparse
import sys

MARKER = 'och patch: read_outputs 范围对齐 invoke'
DIAG_MARKER = 'och patch: read_outputs 越界取证'
TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
STRUCT = (('for (int i = 0; i < outputs_count; i++) {', 1), ('struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);', 1), ('0, pipe_buffer_size(res),', 1), ('assert(src_map);', 1), ('((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;', 1), ('pipe_buffer_read(context, res, 0, etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);', 1))

# ↓↓↓ 逐字锚点（由 tools/gen-rofix-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = '   for (int i = 0; i < outputs_count; i++) {\n      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);\n      if (is_signed[i]) {\n         struct pipe_transfer *src_transfer;\n         uint8_t *src_map;\n         src_map = (uint8_t *) pipe_buffer_map_range(context,\n                                                     res,\n                                                     0, pipe_buffer_size(res),\n                                                     PIPE_MAP_READ,\n                                                     &src_transfer);\n         assert(src_map);\n         for (unsigned k = 0; k < etna_ml_get_size(subgraph, output_idxs[i]); k++) {\n            ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;\n         }\n         pipe_buffer_unmap(context, src_transfer);\n      } else {\n         pipe_buffer_read(context, res, 0, etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);\n      }\n   }\n'
NEW = '   for (int i = 0; i < outputs_count; i++) {\n      /* och patch: read_outputs 范围对齐 invoke\n       *\n       * 真机（0078f5b2 + RODIAG）取证：\n       *   idx=151 n_tensors=152 IN  bsize=5000  off_expected=610000\n       *   → map 的长度用 pipe_buffer_size(res)=5000，循环上界用 tensor->size=610000\n       *     ⇒ 越界读 605000 字节 → SIGSEGV（addr2line: etnaviv_ml.c:1095）\n       *   另两组：139 正常；149 是 bsize=2×tsize（map 多余，不致命）。\n       *   H1（idx 越界）已排除：全部 IN。H3（野指针）已排除：src_map 是合法 mmap。\n       *\n       * 修法：把 map 参数改成与 invoke（etnaviv_ml.c:908）**完全同构** ——\n       *       用 tensor->offset / tensor->size，让「映射范围」与「读取范围」一致。\n       *\n       * 为什么不加 min() 截断：前两处缺陷的教训是「改单点会把越界换成\n       * 静默错误数值」，那比崩更糟。resource 真不够大时，\n       * pipe_buffer_map_range 返回 NULL → assert 明确失败（可观测）。\n       *\n       * 为什么先做边界检查：etna_ml_get_tensor 在越界时会 calloc 并**写入**\n       * 该槽（等于往野地址写），必须先挡住。 */\n      unsigned n_ten = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);\n      if (output_idxs[i] >= n_ten) {\n         ML_DBG("[och-rofix] idx %u >= n_tensors %u，跳过（拒绝读野指针）\\n",\n                output_idxs[i], n_ten);\n         continue;\n      }\n      struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, output_idxs[i]);\n      struct pipe_resource *res = tensor->resource;\n      if (is_signed[i]) {\n         struct pipe_transfer *src_transfer;\n         uint8_t *src_map;\n         src_map = (uint8_t *) pipe_buffer_map_range(context,\n                                                     res,\n                                                     tensor->offset,\n                                                     tensor->size,\n                                                     PIPE_MAP_READ,\n                                                     &src_transfer);\n         assert(src_map);\n         for (unsigned k = 0; k < tensor->size; k++) {\n            ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;\n         }\n         pipe_buffer_unmap(context, src_transfer);\n      } else {\n         pipe_buffer_read(context, res, tensor->offset, tensor->size, outputs[i]);\n      }\n   }\n'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def _fn_body(src):
    at = src.find(TARGET_FN)
    if at < 0:
        return None
    i = src.index("{", at)
    d = 0
    j = i
    while True:
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return src[at:j + 1]
        j += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁，拒绝重复应用。\n")
        return 1
    if DIAG_MARKER in src:
        sys.stderr.write(
            "ERROR: 源文件已含 RODIAG 取证注入（`" + DIAG_MARKER + "`）。\n"
            "       本补丁替换整个循环体，两者互斥 —— 交付构建请只开 TEFLON_ROFIX=1。\n")
        return 1

    body = _fn_body(src)
    if body is None:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    for token, want in STRUCT:
        got = body.count(token)
        if got != want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 "
                + str(got) + " 处，期望 " + str(want) + " 处。\n"
                "       修复依赖这些点，请人工核对后重新生成补丁"
                "（tools/gen-rofix-patch.py），不要跳过校验。\n")
            return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            "ERROR: 锚点在文件中匹配到 " + str(n) + " 处，期望恰好 1 处 —— "
            "不要跳过校验（会静默产出未修复的 .so，崩溃会照旧）。\n")
        return 1
    if NEW == OLD:
        sys.stderr.write("ERROR: 新代码与原文相同（替换未生效）\n")
        return 1

    src = src.replace(OLD, NEW, 1)

    # 反向自检
    if src.count(OLD) != 0:
        sys.stderr.write("ERROR: 锚点仍在源文件中，替换未生效\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 新代码出现次数为 " + str(src.count(NEW)) + "（期望 1）\n")
        return 1
    for token in (MARKER, "[och-rofix]",
                  "util_dynarray_num_elements",
                  "tensor->offset", "tensor->size"):
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: " + token + "\n")
            return 1
    # 关键：map 参数必须用 tensor 的 offset/size，且 pipe_buffer_size 不得再出现在
    # 本补丁注入的**可执行代码**里（那正是 H2 的成因）。
    # ⚠️ 必须**先剥掉注释**再查：本补丁的说明注释里恰好引用了 pipe_buffer_size(res)
    #    与 etna_ml_get_size 作为证据，不剥会把正例全部误拒（同一个坑踩第二次）。
    import re as _re
    injected = _re.sub(r"/\*.*?\*/", "", NEW, flags=_re.S)
    injected = _re.sub(r"//[^\n]*", "", injected)
    if "pipe_buffer_size(res)" in injected:
        sys.stderr.write("ERROR: 注入代码里仍有 pipe_buffer_size(res) —— "
                         "H2 未修复\n")
        return 1
    if "etna_ml_get_size" in injected:
        sys.stderr.write("ERROR: 注入代码里仍有 etna_ml_get_size —— "
                         "循环上界应改用 tensor->size\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-rofix] 已加 read_outputs 范围对齐 + idx 边界检查")
    print("  [och-rofix] 预期：不崩；若 resource 真不够大则 assert 明确失败（可观测）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
