#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：alias（批量元数据迁移）取证。

## 目的

ROFIX（`6acfef1f`）把野指针 SIGSEGV 换成可观测的 assert：

    u_inlines.h:503: pipe_buffer_map_range:
    Assertion `offset + length <= buffer->width0' failed.

⇒ `tensor` 的元数据（offset/size）与 resource 实际大小不一致。

RODIAG（`0078f5b2`）实测：

| idx | n_tensors | bsize | off_expected |
|---|---|---|---|
| 139 | 141 | 940896 | 940896 |
| 149 | 152 | 610000 | 305000 |
| 151 | 152 | **5000** | **610000** |

三个 `off_expected` 全是 5000 的整数倍，`toff` 全 0 —— 静态分析无法完全解释。

## 可疑代码

`reference_tensor_with_offset`（`:198-224`）与 `recreate_tensor`（`:226-246`）：

    if (old_res) {
       for (int i = 0; i < num_tensors; i++)
          if (etna_ml_get_resource(subgraph, i) == old_res) {
             tensors[i]->size = size;      // 一律同一 size
             tensors[i]->offset = offset;  // 一律同一 offset
          }
    }

调用点 `:540/:552/:562/:569` = CONCAT / SPLIT /
**NN 且 input_count>1（Add/Subtract）** / BYPASS。

## 打印

    [och-alias] ref  src=%u dst=%u offset=%u size=%u old_res=%p new_res=%p
    [och-alias]   migrate idx=%d res=%p old(size=%u,off=%u) -> new(size=%u,off=%u)
    [och-alias] recreate idx=%u new_size=%u old_res=%p old(size=%u,off=%u)

**迁移明细在改之前捕获**旧值 —— 否则改完就看不出原本各是多少。

## 与其它开关的关系

改的是 `reference_tensor_with_offset` / `recreate_tensor`，
与 ROFIX / RODIAG 改的 `etna_ml_subgraph_read_outputs` **不同函数**，
⇒ **可与 `TEFLON_ROFIX=1` 同时开**。

⚠️ 上机必须设 `ETNA_MESA_DEBUG=ml_msgs`（`ML_DBG` 是运行期判断）。
⚠️ 该产物含取证打印，**不可交付**。
"""

import argparse
import sys

MARKER = 'och patch: alias 迁移取证'
COMPILE_MARK = 'och-patch-alias-diag'
TARGET_FN_REF = 'reference_tensor_with_offset(struct etna_ml_subgraph'
TARGET_FN_REC = 'recreate_tensor(struct etna_ml_subgraph'
STRUCT_REF = (('if (old_res) {', 1), ('tensors[i]->size = size;', 1), ('tensors[i]->offset = offset;', 1), ('pipe_resource_reference(&dst->resource, src->resource);', 1))
STRUCT_REC = (('if (old_res) {', 1), ('tensors[i]->size = size;', 1), ('struct pipe_resource *new_res;', 1))

# ↓↓↓ 逐字锚点（由 tools/gen-aliasdiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_REF = '   dst->size = size;\n\n   if (old_res) {\n'
NEW_REF = '   dst->size = size;\n   /* och patch: alias 迁移取证（reference_tensor_with_offset）\n    *\n    * ROFIX 上机（6acfef1f）：map 层 assert\n    *   `offset + length <= buffer->width0\' failed\n    * ⇒ tensor 元数据与 resource 实际大小不一致。\n    * RODIAG 实测三组 tsize 全是 5000 的整数倍、bsize 恰为 5000、toff 全 0。\n    *\n    * 这里在**改之前**把每个将被迁移的 tensor 的旧 size/offset 抓出来，\n    * 否则改完就看不出「原本各是多少」。 */\n   {\n      ML_DBG("[och-alias] ref  src=%u dst=%u offset=%u size=%u old_res=%p new_res=%p\\n",\n             src_tensor, dst_tensor, offset, size,\n             (void *)old_res, (void *)src->resource);\n      if (old_res) {\n         for (int i = 0; i < (int)num_tensors; i++) {\n            struct etna_ml_tensor *t = etna_ml_get_tensor(subgraph, i);\n            if (t->resource == old_res)\n               ML_DBG("[och-alias]   migrate idx=%d res=%p old(size=%u,off=%u) -> new(size=%u,off=%u)\\n",\n                      i, (void *)t->resource, t->size, t->offset, size, offset);\n         }\n      }\n   }\n\n   if (old_res) {\n'
OLD_REC = '   struct pipe_resource *new_res;\n'
NEW_REC = '   /* och patch: alias 迁移取证（recreate_tensor） */\n   ML_DBG("[och-alias] recreate idx=%u new_size=%u old_res=%p old(size=%u,off=%u)\\n",\n          idx, size, (void *)old_res,\n          old_res ? etna_ml_get_tensor(subgraph, idx)->size : 0u,\n          old_res ? etna_ml_get_tensor(subgraph, idx)->offset : 0u);\n   struct pipe_resource *new_res;\n'
ANCHOR_MARK = '#include <stdio.h>\n'
NEW_MARK = '#include <stdio.h>\n\n/* och patch: alias 迁移取证 —— 编译期标记，供 strings 硬门判定 */\n__attribute__((used)) static const char och_aliasdiag_marker[] = "och-patch-alias-diag";\n'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def _body(src, fn):
    at = src.find(fn)
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

    for fn, struct in ((TARGET_FN_REF, STRUCT_REF), (TARGET_FN_REC, STRUCT_REC)):
        b = _body(src, fn)
        if b is None:
            sys.stderr.write("ERROR: 找不到函数 " + fn + "（Mesa 结构可能已变）\n")
            return 1
        for token, want in struct:
            got = b.count(token)
            if got != want:
                sys.stderr.write(
                    "ERROR: 在 " + fn + " 内 `" + token + "` 出现 " + str(got)
                    + " 处，期望 " + str(want) + " 处。\n"
                    "       取证依赖这些点，请重新生成补丁"
                    "（tools/gen-aliasdiag-patch.py），不要跳过校验。\n")
                return 1

    for tag, old, new in (("reference_tensor_with_offset", OLD_REF, NEW_REF),
                          ("recreate_tensor", OLD_REC, NEW_REC),
                          ("标记锚点", ANCHOR_MARK, NEW_MARK)):
        n = src.count(old)
        if n != 1:
            sys.stderr.write(
                "ERROR: " + tag + " 锚点匹配到 " + str(n) + " 处，期望恰好 1 处 —— "
                "不要跳过校验（会静默产出无取证的 .so）。\n")
            return 1
        if old == new:
            sys.stderr.write("ERROR: " + tag + " 替换未生效\n")
            return 1

    src = src.replace(OLD_REF, NEW_REF, 1)
    src = src.replace(OLD_REC, NEW_REC, 1)
    src = src.replace(ANCHOR_MARK, NEW_MARK, 1)

    # 反向自检。⚠️ 三处口径**不同**，不能一律查「OLD 必须消失」：
    #   ref  —— 在段**中间**插入 ⇒ OLD_REF 被打断，应当消失（0 处）
    #   rec  —— 在锚点**前面**插入（只追加型）⇒ NEW_REC ⊃ OLD_REC，OLD 应恰好 1 处
    #   mark —— 同上，OLD 应恰好 1 处
    # 一律查「消失」会把 rec/mark 的正例全部误拒（这个坑本轮第二次踩）。
    if src.count(OLD_REF) != 0:
        sys.stderr.write("ERROR: ref 锚点应在插入后消失，实际仍有 "
                         + str(src.count(OLD_REF)) + " 处\n")
        return 1
    for tag, old in (("rec", OLD_REC), ("mark", ANCHOR_MARK)):
        if src.count(old) != 1:
            sys.stderr.write("ERROR: " + tag + " 锚点应恰好保留 1 处"
                             "（它嵌在新代码里），实际 " + str(src.count(old)) + " 处\n")
            return 1
    for tag, new in (("ref", NEW_REF), ("rec", NEW_REC), ("mark", NEW_MARK)):
        if src.count(new) != 1:
            sys.stderr.write("ERROR: " + tag + " 新代码出现次数为 "
                             + str(src.count(new)) + "（期望 1）\n")
            return 1
    for token in (MARKER, "[och-alias]", "migrate idx=", COMPILE_MARK):
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: " + token + "\n")
            return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-alias] 已加 alias 迁移取证：打印 ref/recreate 参数 + 迁移前后对照")
    print("  [och-alias] 需同时设 ETNA_MESA_DEBUG=ml_msgs；该产物不可交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
