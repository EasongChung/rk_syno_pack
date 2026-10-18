#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-create-diag.py。

## 目的：查清「tensor->size 正确、resource 只有 5000 字节」的来源

已确定的事实（真机数据 + 本地模型核对）：

* `[och-alias]`（`64933494`）：`ref src=150 dst=151 offset=0 size=610000`
  ⇒ **`tensor->size` = 610000 是对的**，与模型一致
* RODIAG（`0078f5b2`）：`idx=151` 的 `pipe_buffer_size(res)` = **5000**
  ⇒ **错的是 buffer 分配大小，不是元数据**

已排除（数据否证，不是推理）：

* `recreate_tensor` —— `[och-alias]` 里 **0 次调用**
* 批量元数据迁移 —— 13 次 `migrate` 全部 `old(X,0) -> new(X,0)`，**值没变**

嫌疑（待取证，**不是结论**）—— `etnaviv_ml.c:70-89`：

```c
void
etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);
   pipe_resource *res = tensor->resource;

   if (res != NULL && tensor->size == size)
      return;                       /* ★ 早退：已存在且大小相同就不重建 */

   res = etna_ml_create_resource(&subgraph->screen->base, size);
   tensor->resource = res;
   tensor->size = size;
   ML_DBG("created resource %p for tensor %d with size %d\\n", res, idx, size);
}
```

## 本补丁打印什么

```
[och-create] enter     idx=%u req_size=%u have_res=%p have_size=%u
[och-create]   EARLY-RETURN idx=%u size=%u res=%p actual=%u      ← 早退路径
[och-create]   destroy   idx=%u res=%p actual=%u                 ← destroy 前
[och-create]   created   idx=%u size=%u res=%p actual=%u         ← 建好后
```

**`actual` = `pipe_buffer_size(res)`** —— 这是最关键的一列：
它直接回答「实际分配了多大」，从而区分三种可能：

| 观测 | 指向 |
|---|---|
| `actual == req_size` 但 `res` 与 tensor 对不上 | 问题在别处（指针/索引） |
| `actual < req_size` | `etna_ml_create_resource` 有对齐/最小值/复用逻辑 |
| 早退频繁且 `have_size != req_size` | 早退条件被绕过（`tensor->size` 已被别处改写） |

## 与 ALIASDIAG 的关系

同改 `etnaviv_ml.c`，但注入**函数不同**
（`etna_ml_create_tensor` / `etna_ml_destroy_tensor` vs
`reference_tensor_with_offset` / `recreate_tensor`）⇒ **可同时开**。

标记锚点复用 `#include <stdio.h>`：ALIASDIAG 在它之后插入自己的标记块，
该行**仍然唯一**，所以 CREATEDIAG 用它作锚点不受影响（已本地实测）。

由 `build-teflon.sh` 的 `TEFLON_CREATEDIAG=1` 启用，默认关。
⚠️ 该产物含取证打印，**不可交付**；上机须设 `ETNA_MESA_DEBUG=ml_msgs`。
"""

import os
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = ("/root/workspace/oes-teflon/build-recipe/"
       "patch-etnaviv-create-diag.py")

MARKER = "och patch: create_tensor 分配取证"
COMPILE_MARK = "och-patch-creatediag"

FN_CREATE = "etna_ml_create_tensor(struct etna_ml_subgraph"
FN_DESTROY = "etna_ml_destroy_tensor(struct etna_ml_subgraph"

# ⚠️ 真实源码的早退条件是 `if (res != NULL)`（**只看有没有 resource**），
#    内部 `assert(size == tensor->size)` —— 不是我最初以为的
#    `res != NULL && tensor->size == size`。差别很关键：
#    只要有 resource 就早退，release 构建关掉 assert 后会**静默早退**。
STRUCT_CREATE = (
    ("struct pipe_resource *res = tensor->resource;", 1),
    ("if (res != NULL) {", 1),
    ("assert(size == tensor->size);", 1),
    ("res = etna_ml_create_resource(&subgraph->screen->base, size);", 1),
    ("tensor->resource = res;", 1),
    ("tensor->size = size;", 1),
)
STRUCT_DESTROY = (
    ("pipe_resource_reference(&tensor->resource, NULL);", 1),
)

INJ_CREATE = r'''   /* och patch: create_tensor 分配取证
    *
    * 已确定：tensor->size = 610000 正确（与模型一致），但 resource 实际只有 5000 字节。
    * 已排除：recreate_tensor（0 次调用）、批量迁移（13 次值都没变）。
    *
    * 这里在**早退判断之前**打印已有状态，并在每条出口打印
    * `pipe_buffer_size(res)`（实际分配大小）—— 那是回答本问题的关键列。 */
   ML_DBG("[och-create] enter     idx=%u req_size=%u have_res=%p have_size=%u have_actual=%u\n",
          idx, size, (void *)res, res ? tensor->size : 0u,
          res ? pipe_buffer_size(res) : 0u);
   if (res != NULL && tensor->size == size) {
      ML_DBG("[och-create]   EARLY-RETURN idx=%u size=%u res=%p actual=%u\n",
             idx, size, (void *)res, pipe_buffer_size(res));
      return;
   }
   res = etna_ml_create_resource(&subgraph->screen->base, size);
   tensor->resource = res;
   tensor->size = size;
   ML_DBG("[och-create]   created   idx=%u size=%u res=%p actual=%u\n",
          idx, size, (void *)res, res ? pipe_buffer_size(res) : 0u);
'''

INJ_DESTROY = r'''   /* och patch: create_tensor 分配取证 —— destroy 侧 */
   ML_DBG("[och-create]   destroy   idx=%u res=%p actual=%u\n",
          idx, (void *)tensor->resource,
          tensor->resource ? pipe_buffer_size(tensor->resource) : 0u);
'''

INJ_MARK = r'''
/* och patch: create_tensor 分配取证 —— 编译期标记 */
__attribute__((used)) static const char och_creatediag_marker[] = "och-patch-creatediag";
'''

TEMPLATE = r'''#!/usr/bin/env python3
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

MARKER = __MARKER__
COMPILE_MARK = __COMPILE_MARK__
FN_CREATE = __FN_CREATE__
FN_DESTROY = __FN_DESTROY__
STRUCT_CREATE = __STRUCT_CREATE__
STRUCT_DESTROY = __STRUCT_DESTROY__

# ↓↓↓ 逐字锚点（由 tools/gen-creatediag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_CREATE = __OLD_CREATE__
NEW_CREATE = __NEW_CREATE__
OLD_DESTROY = __OLD_DESTROY__
NEW_DESTROY = __NEW_DESTROY__
ANCHOR_MARK = __ANCHOR_MARK__
NEW_MARK = __NEW_MARK__
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
'''


def _body_gen(src, fn):
    at = src.find(fn)
    if at < 0:
        die("找不到 " + fn)
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


def _extract(src):
    lines = src.split("\n")

    def rng(sig):
        idxs = [i for i, l in enumerate(lines) if sig in l]
        if not idxs:
            die("找不到 " + sig)
        i = idxs[0]
        depth = 0
        started = False
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                started = True
            if started and depth == 0:
                return i, j
        die(sig + " 找不到配平花括号")

    # create_tensor：从 `pipe_resource *res = …` 到 `      return;`
    a, b = rng(FN_CREATE)
    i0 = next((i for i in range(a, b + 1)
               if lines[i].strip() == "struct pipe_resource *res = tensor->resource;"), -1)
    if i0 < 0:
        die("create_tensor 内找不到 `struct pipe_resource *res = tensor->resource;`")
    i1 = next((i for i in range(i0, b + 1)
               if lines[i].strip() == "return;"), -1)
    if i1 < 0:
        die("create_tensor 内找不到早退的 `return;`")
    old_create = "\n".join(lines[i0:i1 + 1]) + "\n"
    new_create = (lines[i0] + "\n" + INJ_CREATE + "\n".join(lines[i0 + 1:i1 + 1]) + "\n")

    # destroy_tensor：在 `pipe_resource_reference(&tensor->resource, NULL);` 之前
    a, b = rng(FN_DESTROY)
    i2 = next((i for i in range(a, b + 1)
               if lines[i].strip() == "pipe_resource_reference(&tensor->resource, NULL);"), -1)
    if i2 < 0:
        die("destroy_tensor 内找不到 `pipe_resource_reference(&tensor->resource, NULL);`")
    old_destroy = lines[i2] + "\n"
    new_destroy = INJ_DESTROY + lines[i2] + "\n"

    # 标记锚点：#include <stdio.h>
    i3 = next((i for i, l in enumerate(lines) if l.strip() == "#include <stdio.h>"), -1)
    if i3 < 0:
        die("找不到 `#include <stdio.h>`")
    anchor_mark = lines[i3] + "\n"
    new_mark = lines[i3] + "\n" + INJ_MARK

    return old_create, new_create, old_destroy, new_destroy, anchor_mark, new_mark


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def main():
    src = open(SRC, encoding="utf-8").read()

    for fn, struct in ((FN_CREATE, STRUCT_CREATE), (FN_DESTROY, STRUCT_DESTROY)):
        body = _body_gen(src, fn)
        if body is None:
            die("找不到 " + fn)
        for token, want in struct:
            got = body.count(token)
            if got != want:
                die("STRUCT 期望值与真实源码不符: `" + fn + "` 内 `" + token
                    + "` 实际 " + str(got) + " 处，期望 " + str(want) + " 处")

    oc, nc, od, nd, am, nm = _extract(src)
    for x, y in ((oc, nc), (od, nd), (am, nm)):
        if x == y:
            die("替换未生效")
    if MARKER not in nc or MARKER not in nd:
        die("注入块缺标记串")
    if COMPILE_MARK not in nm:
        die("标记块缺编译期字符串")

    out = (TEMPLATE
           .replace("__MARKER__", repr(MARKER))
           .replace("__COMPILE_MARK__", repr(COMPILE_MARK))
           .replace("__FN_CREATE__", repr(FN_CREATE))
           .replace("__FN_DESTROY__", repr(FN_DESTROY))
           .replace("__STRUCT_CREATE__", repr(STRUCT_CREATE))
           .replace("__STRUCT_DESTROY__", repr(STRUCT_DESTROY))
           .replace("__OLD_CREATE__", repr(oc))
           .replace("__NEW_CREATE__", repr(nc))
           .replace("__OLD_DESTROY__", repr(od))
           .replace("__NEW_DESTROY__", repr(nd))
           .replace("__ANCHOR_MARK__", repr(am))
           .replace("__NEW_MARK__", repr(nm)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    os.chmod(OUT, 0o755)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()