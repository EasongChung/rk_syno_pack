#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：create_tensor / destroy_tensor 的分配取证。

## 要回答的问题

`tensor->size` = 610000 正确（与模型一致），但 `pipe_buffer_size(res)` = **5000**。
**错的是 buffer 分配大小，不是元数据。**

已排除（真机数据）：`recreate_tensor` 0 次调用；批量迁移 13 次值都没变。

嫌疑：`etnaviv_ml.c:70-89` 的早退 `if (res != NULL && tensor->size == size) return;`

## 打印

    [och-create] enter     idx=%u req_size=%u have_res=%p have_size=%u have_actual=%u
    [och-create]   EARLY-RETURN idx=%u size=%u res=%p actual=%u
    [och-create]   destroy   idx=%u res=%p actual=%u
    [och-create]   created   idx=%u size=%u res=%p actual=%u

**`actual` = `pipe_buffer_size(res)`** 是关键列，直接区分三种可能：

| 观测 | 指向 |
|---|---|
| `actual == req_size` 但与 tensor 对不上 | 问题在指针/索引 |
| `actual < req_size` | `etna_ml_create_resource` 有对齐/最小值/复用逻辑 |
| 早退频繁且 `have_size != req_size` | `tensor->size` 被别处改写，早退条件被绕过 |

## 与 ALIASDIAG 的关系

注入**函数不同** ⇒ **可与 `TEFLON_ALIASDIAG=1` 同时开**。
标记锚点复用 `#include <stdio.h>`（ALIASDIAG 在其后插标记，该行仍唯一）。

⚠️ 上机须设 `ETNA_MESA_DEBUG=ml_msgs`；该产物**不可交付**。
"""

import argparse
import sys

MARKER = 'och patch: create_tensor 取证'
COMPILE_MARK = 'och-patch-creatediag'
FN_CREATE = 'etna_ml_create_tensor(struct etna_ml_subgraph'
FN_DESTROY = 'etna_ml_destroy_tensor(struct etna_ml_subgraph'
STRUCT_CREATE = (('struct pipe_resource *res = tensor->resource;', 1), ('if (res != NULL) {', 1), ('assert(size == tensor->size);', 1), ('res = etna_ml_create_resource(&subgraph->screen->base, size);', 1), ('tensor->resource = res;', 1), ('tensor->size = size;', 1))
STRUCT_DESTROY = (('pipe_resource_reference(&tensor->resource, NULL);', 1),)

# ↓↓↓ 逐字锚点（由 tools/gen-creatediag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_CREATE = 'etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)\n{\n   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);\n\n   assert(idx < util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *));\n   assert(size > 0);\n\n   struct pipe_resource *res = tensor->resource;\n\n   if (res != NULL) {\n      assert(size == tensor->size);\n      return;\n   }\n\n   res = etna_ml_create_resource(&subgraph->screen->base, size);\n   tensor->resource = res;\n   tensor->size = size;\n\n   ML_DBG("created resource %p for tensor %d with size %d\\n", res, idx, size);\n}\n'
NEW_CREATE = 'etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)\n{\n   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);\n\n   assert(idx < util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *));\n   assert(size > 0);\n\n   struct pipe_resource *res = tensor->resource;\n   /* och patch: create_tensor 取证 —— 只加打印，不改控制流 */\n   ML_DBG("[och-create] enter     idx=%u req_size=%u have_res=%p have_size=%u have_actual=%u\\n",\n          idx, size, (void *)res, res ? tensor->size : 0u,\n          res ? pipe_buffer_size(res) : 0u);\n\n   if (res != NULL) {\n      ML_DBG("[och-create]   EARLY-RETURN idx=%u req_size=%u tensor_size=%u res=%p actual=%u %s\\n",\n             idx, size, tensor->size, (void *)res, pipe_buffer_size(res),\n             (tensor->size == size) ? "<size 一致>" : "*** size 不一致！***");\n      assert(size == tensor->size);\n      return;\n   }\n\n   res = etna_ml_create_resource(&subgraph->screen->base, size);\n   tensor->resource = res;\n   tensor->size = size;\n   /* och patch: create_tensor 取证 —— 记录实际分配大小 */\n   ML_DBG("[och-create]   created   idx=%u req_size=%u res=%p actual=%u\\n",\n          idx, size, (void *)res, res ? pipe_buffer_size(res) : 0u);\n\n   ML_DBG("created resource %p for tensor %d with size %d\\n", res, idx, size);\n}\n'
OLD_DESTROY = '   pipe_resource_reference(&tensor->resource, NULL);\n'
NEW_DESTROY = '   /* och patch: create_tensor 取证 —— destroy 侧，只加打印 */\n   ML_DBG("[och-create]   destroy   idx=%u res=%p actual=%u\\n",\n          idx, (void *)tensor->resource,\n          tensor->resource ? pipe_buffer_size(tensor->resource) : 0u);\n   pipe_resource_reference(&tensor->resource, NULL);\n'
ANCHOR_MARK = '#include <stdio.h>\n'
NEW_MARK = '#include <stdio.h>\n\n/* och patch: create_tensor 分配取证 —— 编译期标记 */\n__attribute__((used)) static const char och_creatediag_marker[] = "och-patch-creatediag";\n'
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

    for fn, struct in ((FN_CREATE, STRUCT_CREATE), (FN_DESTROY, STRUCT_DESTROY)):
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
                    "（tools/gen-creatediag-patch.py），不要跳过校验。\n")
                return 1

    for tag, old, new in (("create_tensor", OLD_CREATE, NEW_CREATE),
                          ("destroy_tensor", OLD_DESTROY, NEW_DESTROY),
                          ("标记锚点", ANCHOR_MARK, NEW_MARK)):
        n = src.count(old)
        if n != 1:
            sys.stderr.write("ERROR: " + tag + " 锚点匹配到 " + str(n)
                             + " 处，期望恰好 1 处 —— 不要跳过校验"
                               "（会静默产出无取证的 .so）。\n")
            return 1
        if old == new:
            sys.stderr.write("ERROR: " + tag + " 替换未生效\n")
            return 1

    src = src.replace(OLD_CREATE, NEW_CREATE, 1)
    src = src.replace(OLD_DESTROY, NEW_DESTROY, 1)
    src = src.replace(ANCHOR_MARK, NEW_MARK, 1)

    # 反向自检：create 是**中间插入**（OLD 被打断 → 应消失）；
    # destroy / mark 是**只追加型**（OLD 嵌在 NEW 里 → 应恰好 1 处）。
    # 一律查「消失」会把后两者的正例误拒（本轮第三次踩这个坑的变体）。
    if src.count(OLD_CREATE) != 0:
        sys.stderr.write("ERROR: create_tensor 锚点应在插入后消失\n")
        return 1
    for tag, old in (("destroy_tensor", OLD_DESTROY), ("标记锚点", ANCHOR_MARK)):
        if src.count(old) != 1:
            sys.stderr.write("ERROR: " + tag + " 锚点应恰好保留 1 处"
                             "（嵌在新代码里），实际 " + str(src.count(old)) + " 处\n")
            return 1
    for tag, new in (("create_tensor", NEW_CREATE), ("destroy_tensor", NEW_DESTROY),
                     ("标记", NEW_MARK)):
        if src.count(new) != 1:
            sys.stderr.write("ERROR: " + tag + " 新代码出现次数为 "
                             + str(src.count(new)) + "（期望 1）\n")
            return 1
    for token in (MARKER, "[och-create]", "EARLY-RETURN", COMPILE_MARK,
                  "pipe_buffer_size(res)"):
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: " + token + "\n")
            return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-create] 已加 create/destroy_tensor 分配取证（含 pipe_buffer_size 实际大小）")
    print("  [och-create] 需同时设 ETNA_MESA_DEBUG=ml_msgs；该产物不可交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
