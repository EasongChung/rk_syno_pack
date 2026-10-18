#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-alias-diag.py。

## 目的

ROFIX（`6acfef1f`）把野指针 SIGSEGV 换成了可观测的 assert：

    u_inlines.h:503: pipe_buffer_map_range:
    Assertion `offset + length <= buffer->width0' failed.

⇒ **`tensor` 的元数据（offset/size）与 resource 实际大小不一致。**

RODIAG（`0078f5b2`）已测出三组对照：

| idx | n_tensors | bsize | off_expected | toff |
|---|---|---|---|---|
| 139 | 141 | 940896 | 940896 | 0 |
| 149 | 152 | 610000 | 305000 | 0 |
| 151 | 152 | **5000** | **610000** | 0 |

三个 `off_expected` 全是 **5000 的整数倍**（305000=61×5000、610000=122×5000），
而 `bsize` 恰为 5000，`toff` 全为 0。**静态分析无法完全解释这组关系**，需要运行时数据。

## 可疑代码（本地静态定位）

`etnaviv_ml.c` 两个函数做**批量元数据迁移**，模式相同：

    reference_tensor_with_offset  (:198-224)
    recreate_tensor               (:226-246)

    if (old_res) {
       for (int i = 0; i < num_tensors; i++)
          if (etna_ml_get_resource(subgraph, i) == old_res) {   // 所有共享 old_res 的 tensor
             pipe_resource_reference(&tensors[i]->resource, src->resource);
             tensors[i]->size = size;      // ← 一律改成同一个 size
             tensors[i]->offset = offset;  // ← 一律改成同一个 offset
          }
    }

调用点 `:540 / :552 / :562 / :569` = `CONCAT` / `SPLIT` /
**`NN` 且 `input_count > 1`（Add/Subtract）** / `BYPASS`。

以 Add/Subtract 为例：

    recreate_tensor(subgraph, in0, size0 + size1);              /* in0 buffer 变成 size0+size1 */
    reference_tensor_with_offset(subgraph, in0, in1, size0, size1);  /* in1 → (offset=size0, size=size1) */

**可疑点**：批量分支把「所有共享 `old_res` 的 tensor」都改成同一 `(offset, size)`，
但它们本该各自占据 `old_res` 的**不同段**。

## 本补丁打印什么

    [och-alias] ref  src=%u dst=%u offset=%u size=%u old_res=%p new_res=%p
    [och-alias]   migrate idx=%u res=%p old(size=%u,off=%u) -> new(size=%u,off=%u)
    [och-alias] recreate idx=%u new_size=%u old_res=%p
    [och-alias] marker compile-check

**关键**：迁移明细在**改之前**捕获旧 size/offset，否则看不到改了什么。

## 与其它开关的关系（重要）

本补丁改的是 `reference_tensor_with_offset` 与 `recreate_tensor`，
**与 ROFIX / RODIAG 改的函数完全不同**（它们改 `etna_ml_subgraph_read_outputs`），
因此 **可以与 `TEFLON_ROFIX=1` 同时开**，一次构建同时拿到：
「read_outputs 侧的范围」+「alias 迁移过程」。

由 `build-teflon.sh` 的 `TEFLON_ALIASDIAG=1` 启用，默认关。
"""

import os
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = ("/root/workspace/oes-teflon/build-recipe/"
       "patch-etnaviv-alias-diag.py")

MARKER = "och patch: alias 迁移取证"
COMPILE_MARK = "och-patch-alias-diag"

TARGET_FN_REF = "reference_tensor_with_offset(struct etna_ml_subgraph"
TARGET_FN_REC = "recreate_tensor(struct etna_ml_subgraph"

# 每个函数各自的特征（token, 期望次数）—— 分开是因为两个函数**各 1 处**，
# 合成一个「共 2 处」的表会让人按单函数校验时永远失败（本轮踩过）。
STRUCT_REF = (
    ("if (old_res) {", 1),
    ("tensors[i]->size = size;", 1),
    ("tensors[i]->offset = offset;", 1),
    ("pipe_resource_reference(&dst->resource, src->resource);", 1),
)
# ⚠️ recreate_tensor **只改 size、不改 offset**（与 reference_tensor_with_offset 不同）——
#    这个不对称本身是条线索，别把它写进 STRUCT 里当共性（本轮踩过：按共性校验必然失败）。
STRUCT_REC = (
    ("if (old_res) {", 1),
    ("tensors[i]->size = size;", 1),
    ("struct pipe_resource *new_res;", 1),
)

# ── 注入块 1：reference_tensor_with_offset，在批量迁移之前 ──
INJ_REF = r'''   /* och patch: alias 迁移取证（reference_tensor_with_offset）
    *
    * ROFIX 上机（6acfef1f）：map 层 assert
    *   `offset + length <= buffer->width0' failed
    * ⇒ tensor 元数据与 resource 实际大小不一致。
    * RODIAG 实测三组 tsize 全是 5000 的整数倍、bsize 恰为 5000、toff 全 0。
    *
    * 这里在**改之前**把每个将被迁移的 tensor 的旧 size/offset 抓出来，
    * 否则改完就看不出「原本各是多少」。 */
   {
      ML_DBG("[och-alias] ref  src=%u dst=%u offset=%u size=%u old_res=%p new_res=%p\n",
             src_tensor, dst_tensor, offset, size,
             (void *)old_res, (void *)src->resource);
      if (old_res) {
         for (int i = 0; i < (int)num_tensors; i++) {
            struct etna_ml_tensor *t = etna_ml_get_tensor(subgraph, i);
            if (t->resource == old_res)
               ML_DBG("[och-alias]   migrate idx=%d res=%p old(size=%u,off=%u) -> new(size=%u,off=%u)\n",
                      i, (void *)t->resource, t->size, t->offset, size, offset);
         }
      }
   }
'''

# ── 注入块 2：recreate_tensor，在 destroy 之前 ──
INJ_REC = r'''   /* och patch: alias 迁移取证（recreate_tensor） */
   ML_DBG("[och-alias] recreate idx=%u new_size=%u old_res=%p old(size=%u,off=%u)\n",
          idx, size, (void *)old_res,
          old_res ? etna_ml_get_tensor(subgraph, idx)->size : 0u,
          old_res ? etna_ml_get_tensor(subgraph, idx)->offset : 0u);
'''

# ── 编译期标记（放进 read_outputs 之外，用文件作用域；放函数里也行）──
INJ_MARK = r'''
/* och patch: alias 迁移取证 —— 编译期标记，供 strings 硬门判定 */
__attribute__((used)) static const char och_aliasdiag_marker[] = "och-patch-alias-diag";
'''

TEMPLATE = r'''#!/usr/bin/env python3
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

MARKER = __MARKER__
COMPILE_MARK = __COMPILE_MARK__
TARGET_FN_REF = __TARGET_FN_REF__
TARGET_FN_REC = __TARGET_FN_REC__
STRUCT_REF = __STRUCT_REF__
STRUCT_REC = __STRUCT_REC__

# ↓↓↓ 逐字锚点（由 tools/gen-aliasdiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_REF = __OLD_REF__
NEW_REF = __NEW_REF__
OLD_REC = __OLD_REC__
NEW_REC = __NEW_REC__
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
'''


def _body(src, fn):
    """取目标函数体（花括号配平）。"""
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


def _extract(src):
    """逐字取锚点（按行切片，不手写缩进）。"""
    lines = src.split("\n")

    def fn_range(fn_sig):
        idxs = [i for i, l in enumerate(lines) if fn_sig in l]
        if not idxs:
            die("找不到 " + fn_sig)
        i = idxs[0]
        depth = 0
        started = False
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                started = True
            if started and depth == 0:
                return i, j
        die(fn_sig + " 找不到配平的右花括号")

    # 注入点 1：reference_tensor_with_offset 里第一处 `if (old_res) {`
    # ⚠️ 锚点必须**带函数内唯一上下文**：`if (old_res) {` 在两个函数里逐字相同
    #    （都是 3 空格缩进），只取它会让文件级匹配到 2 处 —— 本轮踩过。
    #    故取 `dst->size = size;` 起、到 `if (old_res) {` 止的整段。
    a, b = fn_range(TARGET_FN_REF)
    hits = [i for i in range(a, b + 1) if lines[i].strip() == "if (old_res) {"]
    if len(hits) != 1:
        die("reference_tensor_with_offset 内 `if (old_res) {` 出现 "
            + str(len(hits)) + " 处（期望 1）")
    i1 = hits[0]
    ctx = [k for k in range(a, i1) if lines[k].strip() == "dst->size = size;"]
    if len(ctx) != 1:
        die("reference_tensor_with_offset 内 `dst->size = size;` 出现 "
            + str(len(ctx)) + " 处（期望 1）")
    old_ref = "\n".join(lines[ctx[0]:i1 + 1]) + "\n"
    new_ref = lines[ctx[0]] + "\n" + INJ_REF + "\n".join(lines[ctx[0] + 1:i1 + 1]) + "\n"
    if "dst->size = size;" not in new_ref:
        die("NEW_REF 丢了 `dst->size = size;` 这一行")

    # 注入点 2：recreate_tensor 里 `struct pipe_resource *new_res;`
    a, b = fn_range(TARGET_FN_REC)
    hits = [i for i in range(a, b + 1)
            if lines[i].strip() == "struct pipe_resource *new_res;"]
    if len(hits) != 1:
        die("recreate_tensor 内 `struct pipe_resource *new_res;` 出现 "
            + str(len(hits)) + " 处（期望 1）")
    i2 = hits[0]
    old_rec = lines[i2] + "\n"
    new_rec = INJ_REC + lines[i2] + "\n"

    # 注入点 3：文件作用域标记，锚在第一个 `#include <stdio.h>` 之后
    hits = [i for i, l in enumerate(lines) if l.strip() == "#include <stdio.h>"]
    if len(hits) != 1:
        die("#include <stdio.h> 出现 " + str(len(hits)) + " 处（期望 1）")
    anchor_mark = lines[hits[0]] + "\n"
    new_mark = lines[hits[0]] + "\n" + INJ_MARK

    return (old_ref, new_ref, old_rec, new_rec, anchor_mark, new_mark)


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def main():
    src = open(SRC, encoding="utf-8").read()

    for fn, struct in ((TARGET_FN_REF, STRUCT_REF), (TARGET_FN_REC, STRUCT_REC)):
        body = _body(src, fn)
        if body is None:
            die("找不到 " + fn)
        for token, want in struct:
            got = body.count(token)
            if got != want:
                die("STRUCT 期望值与真实源码不符: `" + fn + "` 内 `" + token
                    + "` 实际 " + str(got) + " 处，期望 " + str(want) + " 处")
    for token, want in ((TARGET_FN_REF, 1), (TARGET_FN_REC, 1)):
        got = src.count(token)
        if got != want:
            die("函数签名 `" + token + "` 出现 " + str(got) + " 处，期望 "
                + str(want) + " 处")

    old_ref, new_ref, old_rec, new_rec, anchor_mark, new_mark = _extract(src)
    for nm, o, n in (("REF", old_ref, new_ref), ("REC", old_rec, new_rec),
                    ("MARK", anchor_mark, new_mark)):
        if o == n:
            die(nm + " 替换未生效")
    if MARKER not in new_ref or MARKER not in new_rec:
        die("注入块缺标记串")
    if COMPILE_MARK not in new_mark:
        die("标记块缺编译期字符串")

    out = (TEMPLATE
           .replace("__MARKER__", repr(MARKER))
           .replace("__COMPILE_MARK__", repr(COMPILE_MARK))
           .replace("__TARGET_FN_REF__", repr(TARGET_FN_REF))
           .replace("__TARGET_FN_REC__", repr(TARGET_FN_REC))
           .replace("__STRUCT_REF__", repr(STRUCT_REF))
           .replace("__STRUCT_REC__", repr(STRUCT_REC))
           .replace("__OLD_REF__", repr(old_ref))
           .replace("__NEW_REF__", repr(new_ref))
           .replace("__OLD_REC__", repr(old_rec))
           .replace("__NEW_REC__", repr(new_rec))
           .replace("__ANCHOR_MARK__", repr(anchor_mark))
           .replace("__NEW_MARK__", repr(new_mark)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    os.chmod(OUT, 0o755)

    print("已生成:", OUT)
    for nm, o in (("OLD_REF", old_ref), ("NEW_REF", new_ref),
                  ("OLD_REC", old_rec), ("NEW_REC", new_rec),
                  ("ANCHOR_MARK", anchor_mark)):
        print(f"--- {nm} ---")
        for l in o.split("\n"):
            if l:
                print(l[:112])


if __name__ == "__main__":
    main()