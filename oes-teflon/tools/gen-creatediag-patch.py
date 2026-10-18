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
import re
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = ("/root/workspace/oes-teflon/build-recipe/"
       "patch-etnaviv-create-diag.py")

MARKER = "och patch: create_tensor 取证"
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

# ★★★ 纪律门（本项目最严重的一次教训，oca 在真源码上实读抓到）★★★ #
# 本诊断补丁**只能加打印，绝不改控制流**。
#
# 初版把 `if (res != NULL && tensor->size == size) { ... }` 连同 create 块**整块重发射**，
# 而那个条件正是我在正文里刚更正掉的**错误形态**（真实源码是 `if (res != NULL)`）。
# 后果：`res != NULL && size != size` 这条路径，上游会 `assert` 当场 abort，
#       本补丁却变成**静默重分配 + 覆盖指针**，原生块还变成不可达死代码。
# ⇒ 我要查的正是「size 不一致会怎样」，而补丁把那个现象改成了「悄悄修好」。
#   这就是前两轮反复拒绝的「把可观测失败换成静默行为」，我自己犯了。
#
# 现在改成**三处独立切片 + 纯替换**：条件 / assert / return / create 块逐字保留，
# 注入的只有 ML_DBG。
#
# 另有一条事实修正（oca 提供，重要）：**assert 在我们的构建里是开的**
# —— ROFIX 真机那次 abort 就是 `u_inlines.h:503` 的 assert 消息。
# ⇒ 「release/NDEBUG 下静默早退」在我们这里**已被排除**：
#   size 不一致不可能静默，要么没走过，要么走过就崩，而崩点可 addr2line。
# ⇒ 本轮真正能立功的是 `created … size=5000` 那类行，以及有没有 destroy 后重建。

DIAG_ENTER = (
    "   struct pipe_resource *res = tensor->resource;\n"
    "   /* och patch: create_tensor 取证 —— 只加打印，不改控制流 */\n"
    "   ML_DBG(\"[och-create] enter     idx=%u req_size=%u have_res=%p have_size=%u have_actual=%u\\n\",\n"
    "          idx, size, (void *)res, res ? tensor->size : 0u,\n"
    "          res ? pipe_buffer_size(res) : 0u);\n")

DIAG_EARLY = (
    "   if (res != NULL) {\n"
    "      ML_DBG(\"[och-create]   EARLY-RETURN idx=%u req_size=%u tensor_size=%u res=%p actual=%u %s\\n\",\n"
    "             idx, size, tensor->size, (void *)res, pipe_buffer_size(res),\n"
    "             (tensor->size == size) ? \"<size 一致>\" : \"*** size 不一致！***\");\n"
    "      assert(size == tensor->size);\n"
    "      return;\n")

DIAG_CREATED = (
    "   tensor->size = size;\n"
    "   /* och patch: create_tensor 取证 —— 记录实际分配大小 */\n"
    "   ML_DBG(\"[och-create]   created   idx=%u req_size=%u res=%p actual=%u\\n\",\n"
    "          idx, size, (void *)res, res ? pipe_buffer_size(res) : 0u);\n")


INJ_DESTROY = r'''   /* och patch: create_tensor 取证 —— destroy 侧，只加打印 */
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


def _strip_diag(t):
    """去注释、去**整条** ML_DBG 语句（含跨行参数）。"""
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    out, skip = [], False
    for l in t.split("\n"):
        if not skip and "ML_DBG(" in l:
            skip = not l.rstrip().endswith(");")
            continue
        if skip:
            skip = not l.rstrip().endswith(");")
            continue
        out.append(l.rstrip())
    return "\n".join(x for x in out if x.strip())


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

    # create_tensor：**在完整函数体上做三次独立替换**。
    # ⚠️ 踩坑：曾用「三段切片各自替换后**拼接**」—— 那会把切片之间的
    #    原生代码（`res = etna_ml_create_resource(...)`、`tensor->resource = res;`）
    #    **整段丢掉**。拼接是错的，只能在完整文本上逐个 replace。
    a, b = rng(FN_CREATE)
    full = "".join(l + "\n" for l in lines[a:b + 1])

    def sub(old, new, what):
        n = full.count(old)
        if n != 1:
            die("create_tensor 内 `" + old.strip().split("\n")[0][:52]
                + "` 匹配到 " + str(n) + " 处（期望 1）：" + what)
        return full.replace(old, new, 1), new

    for anchor, repl, what in (
        ("   struct pipe_resource *res = tensor->resource;\n", DIAG_ENTER, "enter 锚点"),
        ("   if (res != NULL) {\n      assert(size == tensor->size);\n      return;\n",
         DIAG_EARLY, "早退锚点"),
        ("   tensor->size = size;\n", DIAG_CREATED, "created 锚点"),
    ):
        full, _tail = sub(anchor, repl, what)

    old_create = "".join(l + "\n" for l in lines[a:b + 1])
    new_create = full
    # 原生上下文逐字保留的三条硬性检查
    for keep in ("      assert(size == tensor->size);\n",
                 "      return;\n",
                 "   res = etna_ml_create_resource(&subgraph->screen->base, size);\n",
                 "   tensor->resource = res;\n",
                 "   if (res != NULL) {\n"):
        if keep not in new_create:
            die("NEW_CREATE 丢了原生语句 `" + keep.strip() + "` —— 控制流被改了")

    # destroy_tensor：在 `pipe_resource_reference(&tensor->resource, NULL);` 之前
    a, b = rng(FN_DESTROY)
    i2 = next((i for i in range(a, b + 1)
               if lines[i].strip() == "pipe_resource_reference(&tensor->resource, NULL);"), -1)
    if i2 < 0:
        die("destroy_tensor 内找不到 `pipe_resource_reference(&tensor->resource, NULL);`")
    old_destroy = lines[i2] + "\n"
    new_destroy = INJ_DESTROY + lines[i2] + "\n"
    # destroy 是**纯追加**（打印插在原生语句之前），原生语句逐字保留在 NEW 里

    # 标记锚点：#include <stdio.h>
    i3 = next((i for i, l in enumerate(lines) if l.strip() == "#include <stdio.h>"), -1)
    if i3 < 0:
        die("找不到 `#include <stdio.h>`")
    anchor_mark = lines[i3] + "\n"
    # ★ 标记块必须插在锚点行**之前**（INJ_MARK + 锚点），不能插在之后。
    #
    # 原因（run 39145972254 真实失败 + 本地两向复现）：ALIASDIAG 与 CREATEDIAG
    # 共用同一行锚点 `#include <stdio.h>`，而且两边都是「锚点 + 自己的块」形态。
    #   先应用的那个块会被后一个**从中间劈开**（劈在 include 行与自己的注释之间），
    #   于是**先应用**者的 `NEW_MARK` 不再是连续文本 ⇒ 它自己的构建后保真逆变换
    #   前提不成立 ⇒ 那个锚点门必红（实测：ALIAS 先则 ALIAS 门红；CREATED 先则
    #   CREATED 门红）。**与顺序无关，两个里必红一个。**
    #
    # 插在之前则顺序无关地都成立：无论谁先谁后，最终文本都是
    #   [created 块] #include <stdio.h> \n\n [alias 块]
    # 两边的 `NEW_MARK` 都保持连续，`ANCHOR_MARK` 也都恰好 1 处。
    #
    # 前向检查不受影响：`ANCHOR_MARK` 在任一时刻只出现 1 次（另一家的块里那 1 次
    # 就是它自己 NEW_MARK 的组成部分），所以 `n != 1` 的拒绝逻辑照样成立。
    new_mark = INJ_MARK + lines[i3] + "\n"

    globals()["_EQ_PAIRS"] = (("create", old_create, full),
                              ("destroy", old_destroy, old_destroy.replace(
                                  INJ_DESTROY + old_destroy, "")))
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
    # ★ 纪律门（本项目最严重教训）：**不用禁 token**，那会误报
    #   —— 替换块必须逐字保留 `if (res != NULL) {` 等原生行，那正是替换的一部分。
    #   正确的守门是下面的「去注释去 ML_DBG 后与原生逐字等价」：
    #   它直接验证「可执行代码一字未改」，且不会有假阳性。

    # 反向：原生语句必须逐字保留在 NEW 里
    for keep in ("      assert(size == tensor->size);\n", "      return;\n",
                 "   res = etna_ml_create_resource(&subgraph->screen->base, size);\n",
                 "   tensor->resource = res;\n"):
        if keep not in nc:
            die("NEW_CREATE 丢了原生语句 `" + keep.strip() + "` —— 控制流被改了")

    # ★ 最强的一道门：把「补丁后的函数」与「原生函数」都去注释、去**整条 ML_DBG 语句**
    #   （含跨行参数），然后**逐字比对**。相等 ⇒ 控制流一字未改。
    #   这是 oca 在真源码上实读抓到的那类问题的**自动化守门**。
    def _strip_diag(t):
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
        t = re.sub(r"//[^\n]*", "", t)
        out, skip = [], False
        for l in t.split("\n"):
            if not skip and "ML_DBG(" in l:
                skip = not l.rstrip().endswith(");")
                continue
            if skip:
                skip = not l.rstrip().endswith(");")
                continue
            out.append(l.rstrip())
        return "\n".join(x for x in out if x.strip())

    def _fn(text, sig):
        i = text.index(sig)
        i = text.rfind("\n", 0, i)
        seg = text[i:]
        j = seg.index("{")
        d = 0
        k = j
        while True:
            if seg[k] == "{":
                d += 1
            elif seg[k] == "}":
                d -= 1
                if d == 0:
                    return seg[:k + 1]
            k += 1

    for tag, before_txt, after_txt in _EQ_PAIRS:
        if _strip_diag(before_txt) != _strip_diag(after_txt):
            die("★ 纪律门未通过：" + tag + " 打补丁后与原生的可执行代码**不等价**"
                "（去注释、去整条 ML_DBG 语句后逐字比对失败）"
                "—— 诊断补丁只能加打印，不得改控制流")
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