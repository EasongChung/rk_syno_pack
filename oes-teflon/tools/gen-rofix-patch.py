#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-readout-offset-fix.py。

## 根因（真机运行时数据，RODIAG 取证，`0078f5b2`）

崩溃点（新产物重取，旧偏移已作废）：

    etna_ml_subgraph_read_outputs+0x220  →  etnaviv_ml.c:1095
      = `((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;`

`[och-ro]` 逐行输出：

    idx=139 n_tensors=141 IN  bsize=940896 off_expected=940896   ← 一致
    idx=149 n_tensors=152 IN  bsize=610000 off_expected=305000   ← bsize = 2×tsize
    idx=151 n_tensors=152 IN  bsize=5000   off_expected=610000   ← ★ 致命

**三条假说的判定**：

| # | 假说 | 判定 | 依据 |
|---|---|---|---|
| H1 | `output_idxs[i]` 越界索引 dynarray | ❌ 排除 | 三组全部 `IN`（139<141 / 149<152 / 151<152） |
| **H2** | `pipe_buffer_size(res) != tensor->size` | ✅ **确认** | `idx=151`：resource 实际 5000 B，循环要读 610000 B ⇒ **越界 605000 B** |
| H3 | resource 跨 context 失效 | ❌ 排除 | `src_map=0x7f83da9000` 等均为合法 mmap 地址，非野指针 |

**算术**：map 的长度用 `pipe_buffer_size(res)`，循环上界用 `tensor->size`，
两者不等时即越界。致命组 610000 − 5000 = 605000 字节越界读 → SIGSEGV。

**附带观察（未深究）**：三个 `tsize` 全是 5000 的整数倍
（305000 = 61×5000、610000 = 122×5000），而 `bsize` 恰为 5000，
且 `tensor->offset` 全为 0。指向 `etna_ml_tensor_offset`
（`etnaviv_ml.c:212-223`）的批量重定位逻辑 —— 它把所有
`resource == old_res` 的 tensor 一律改成同一个 `size`/`offset`。
**这是下一步的问题，不在本补丁范围内。**

## 本补丁做什么（只两件事）

1. **边界检查**：`idx >= util_dynarray_num_elements(...)` 则跳过。
   必须有 —— `etna_ml_get_tensor` 在越界时会 `calloc` 并**写入**该槽，
   等于往野地址写，会掩盖现场。
2. **map 参数与读取范围对齐**，且**与 `invoke`（`:908`）完全同构**：
   用 `tensor->offset` / `tensor->size`，不再用 `0` / `pipe_buffer_size(res)`。

## 为什么不给循环加 clamp

前两处缺陷的教训：**改单点会把「越界」换成「静默错误数值」** ——
那比崩更糟。所以这里**不**加 `min()` 截断：
若 resource 真不够大，`pipe_buffer_map_range` 会返回 NULL →
`assert(src_map)` 明确失败（SIGABRT，可观测），而不是野指针或错误数值。

## 互斥

**与 `TEFLON_RODIAG=1` 互斥** —— RODIAG 也改这个循环体，
两者锚点冲突。门会因此报错（明确失败，不是静默错误），
`build-teflon.sh` 另有显式互斥检查。
"""

import os
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = ("/root/workspace/oes-teflon/build-recipe/"
       "patch-etnaviv-readout-offset-fix.py")

TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
MARKER = "och patch: read_outputs 范围对齐 invoke"

# 目标函数体内必须精确匹配的结构特征（token, 期望次数）
STRUCT = (
    ("for (int i = 0; i < outputs_count; i++) {", 1),
    ("struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);", 1),
    ("0, pipe_buffer_size(res),", 1),
    ("assert(src_map);", 1),
    ("((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;", 1),
    ("pipe_buffer_read(context, res, 0, etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);", 1),
)

DIAG_MARKER = "och patch: read_outputs 越界取证"   # RODIAG 的标记，互斥用

NEW_BODY = r'''   for (int i = 0; i < outputs_count; i++) {
      /* och patch: read_outputs 范围对齐 invoke
       *
       * 真机（0078f5b2 + RODIAG）取证：
       *   idx=151 n_tensors=152 IN  bsize=5000  off_expected=610000
       *   → map 的长度用 pipe_buffer_size(res)=5000，循环上界用 tensor->size=610000
       *     ⇒ 越界读 605000 字节 → SIGSEGV（addr2line: etnaviv_ml.c:1095）
       *   另两组：139 正常；149 是 bsize=2×tsize（map 多余，不致命）。
       *   H1（idx 越界）已排除：全部 IN。H3（野指针）已排除：src_map 是合法 mmap。
       *
       * 修法：把 map 参数改成与 invoke（etnaviv_ml.c:908）**完全同构** ——
       *       用 tensor->offset / tensor->size，让「映射范围」与「读取范围」一致。
       *
       * 为什么不加 min() 截断：前两处缺陷的教训是「改单点会把越界换成
       * 静默错误数值」，那比崩更糟。resource 真不够大时，
       * pipe_buffer_map_range 返回 NULL → assert 明确失败（可观测）。
       *
       * 为什么先做边界检查：etna_ml_get_tensor 在越界时会 calloc 并**写入**
       * 该槽（等于往野地址写），必须先挡住。 */
      unsigned n_ten = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
      if (output_idxs[i] >= n_ten) {
         ML_DBG("[och-rofix] idx %u >= n_tensors %u，跳过（拒绝读野指针）\n",
                output_idxs[i], n_ten);
         continue;
      }
      struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, output_idxs[i]);
      struct pipe_resource *res = tensor->resource;
      if (is_signed[i]) {
         struct pipe_transfer *src_transfer;
         uint8_t *src_map;
         src_map = (uint8_t *) pipe_buffer_map_range(context,
                                                     res,
                                                     tensor->offset,
                                                     tensor->size,
                                                     PIPE_MAP_READ,
                                                     &src_transfer);
         assert(src_map);
         for (unsigned k = 0; k < tensor->size; k++) {
            ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;
         }
         pipe_buffer_unmap(context, src_transfer);
      } else {
         pipe_buffer_read(context, res, tensor->offset, tensor->size, outputs[i]);
      }
   }
'''

TEMPLATE = r'''#!/usr/bin/env python3
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

MARKER = __MARKER__
DIAG_MARKER = __DIAG_MARKER__
TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
STRUCT = __STRUCT__

# ↓↓↓ 逐字锚点（由 tools/gen-rofix-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = __OLD__
NEW = __NEW__
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
'''


def _fn_body(src):
    """取目标函数的完整函数体（花括号配平，不靠 find("\\n}\\n") —— 那会命中内层块）。"""
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


def _extract(src):
    """逐字取出要替换的循环体（从 for 头到其配平的右花括号）。

    **不手写任何缩进** —— 空格数从真实源码按行切片取。
    """
    lines = src.split("\n")

    starts = [i for i, l in enumerate(lines)
              if l.strip() == "for (int i = 0; i < outputs_count; i++) {"]
    if len(starts) != 1:
        die("read_outputs 的循环头出现 " + str(len(starts)) + " 处（期望 1）")
    i0 = starts[0]

    # 定位配平的右花括号（函数体内唯一的 for）
    depth = 0
    end = -1
    for i in range(i0, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            end = i
            break
    if end < 0:
        die("找不到与循环头配平的右花括号")
    # 循环体必须包含 map 与拷贝语句，否则说明结构变了
    blk = "\n".join(lines[i0:end + 1])
    for must in ("pipe_buffer_size(res)", "src_map[k] - 128",
                 "pipe_buffer_read(context, res"):
        if must not in blk:
            die("循环体里找不到 `" + must + "`，Mesa 结构可能已变")

    old = "\n".join(lines[i0:end + 1]) + "\n"
    return old, NEW_BODY


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def main():
    src = open(SRC, encoding="utf-8").read()

    at = src.find(TARGET_FN)
    if at < 0:
        die("找不到 " + TARGET_FN)
    body = _fn_body(src)
    if body is None:
        die("_fn_body 失败")
    for token, want in STRUCT:
        got = body.count(token)
        if got != want:
            die("STRUCT 期望值与真实源码不符: `" + token + "` 实际 "
                + str(got) + " 处，期望 " + str(want) + " 处")

    old, new = _extract(src)
    if old == new:
        die("替换未生效")
    if MARKER not in new:
        die("NEW 里缺标记串")

    out = (TEMPLATE
           .replace("__MARKER__", repr(MARKER))
           .replace("__DIAG_MARKER__", repr(DIAG_MARKER))
           .replace("__STRUCT__", repr(STRUCT))
           .replace("__OLD__", repr(old))
           .replace("__NEW__", repr(new)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    os.chmod(OUT, 0o755)

    print("已生成:", OUT)
    print("--- OLD（前 3 行 / 后 2 行）---")
    ol = old.split("\n")
    for l in ol[:3] + ["   …"] + ol[-3:]:
        print(l[:110])
    print("--- NEW（前 3 行 / 后 3 行）---")
    nl = new.split("\n")
    for l in nl[:3] + ["   …"] + nl[-3:]:
        print(l[:110])


if __name__ == "__main__":
    main()