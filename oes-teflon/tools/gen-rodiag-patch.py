#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-readout-diag.py。

## 为什么需要它

`675a1091` / `c8e8e658` 在真机 Invoke 阶段 SIGSEGV：

    fsguard.so(etna_ml_subgraph_read_outputs+0x188)
    fsguard.so(+0x8564c)  ==  partition_invoke  (frontends/teflon/tfl_device.c:702)

addr2line（CI + 本地复核一致）：

    0x2307e8  etna_ml_subgraph_read_outputs  etnaviv_ml.c:1076
              （FSGUARD 注入 +24 行 ⇒ 干净源码 1052 行 = `outputs[i][k] = src_map[k] - 128;`）

反汇编故障指令：

    2307d0: blr  x6                ← pipe_buffer_map_range()
    2307d4: mov  x24, x0           ← x24 = src_map
    2307d8: cbz  x0, 2309c0        ← NULL 分支走 __assert_fail（assert 已编译进去）
    2307e8: ldrb w0, [x24, w26]    ← ★ SIGSEGV

⇒ **src_map 非 NULL 但是野指针**。

## 三条假说，静态层面都能自洽，必须用运行时数据区分

| # | 假说 | 本地静态核算结果 |
|---|---|---|
| H1 | `output_idxs[i]` 越界索引 `subgraph->tensors` dynarray | **已算过：partition 1 输出 tensor=139、dynarray 大小=140，不越界**（除非 partition 划分与节点表不符） |
| H2 | `pipe_buffer_size(res) != tensor->size` | `read_outputs` 绕过 `tensor->offset` 直接 map 整个 resource，而循环上界用 `tensor->size`；`etna_ml_tensor_offset`(`:212-223`) 会按「共享 old_res」批量重定位 |
| H3 | resource 跨 context 失效 | `subgraph` 是 device 级（`tfl_device.c:603`），`context` 每次 invoke 新建（`:666`），未验证 |

本补丁打印区分这三者所需的**全部**量。

## 本补丁的自我保护

诊断代码**不得改变行为**：

- 只读 `util_dynarray_element` 的槽值，**不调用 `etna_ml_get_tensor`**
  （它在槽为 NULL 时会 `calloc` 并**写入**该槽 —— 若 idx 越界就是往野地址写，
  会掩盖 H1 的现场）；
- idx 越界时（`idx >= n_tensors`）**不打 t->…**，只打 `OOB`；
- 断言存在 `util_dynarray_num_elements` 与 `ML_DBG` 可用。

## 用法

    python3 patch-etnaviv-readout-diag.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_RODIAG=1` 启用，默认关。
**诊断轮必须同时开运行期开关** `ETNA_MESA_DEBUG=ml_msgs`，否则 `ML_DBG` 不打印
（已实证：格式串进 `.rodata`，但 `DBG_ENABLED` 是运行期判断）。
"""

import os
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = ("/root/workspace/oes-teflon/build-recipe/"
       "patch-etnaviv-readout-diag.py")

TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
MARKER = "och patch: read_outputs 越界取证"

# 目标函数体内必须精确匹配的结构特征（token, 期望次数）
STRUCT = (
    ("for (int i = 0; i < outputs_count; i++) {", 1),
    ("struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);", 1),
    ("src_map = (uint8_t *) pipe_buffer_map_range(context,", 1),
    ("assert(src_map);", 1),
    ("((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;", 1),
)

MAP_TAIL = ("&src_transfer);\n", "         assert(src_map);\n")

DIAG_LOOP = r'''      /* och patch: read_outputs 越界取证 */
      {
         unsigned n_ten = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
         unsigned idx = output_idxs[i];
         int in_range = (idx < n_ten);
         /* 只读槽值，不调 etna_ml_get_tensor：它会在槽为 NULL 时 calloc 并写入，
          * idx 越界时等于往野地址写，会掩盖现场。 */
         struct etna_ml_tensor **slotp = in_range
            ? util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, idx)
            : NULL;
         struct etna_ml_tensor *t = slotp ? *slotp : NULL;
         ML_DBG("[och-ro] i=%d idx=%u n_tensors=%u %s slot=%p res=%p tsize=%u toff=%u\n",
                i, idx, n_ten, in_range ? "IN " : "OOB", (void *)slotp,
                (void *)(t ? t->resource : NULL), t ? t->size : 0u, t ? t->offset : 0u);
      }
'''

DIAG_MAP = r'''         /* och patch: read_outputs map 结果取证 */
         ML_DBG("[och-ro]   map res=%p bsize=%u off_expected=%u -> src_map=%p\n",
                (void *)res, res ? pipe_buffer_size(res) : 0u,
                etna_ml_get_size(subgraph, output_idxs[i]), (void *)src_map);
'''

TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：在 etna_ml_subgraph_read_outputs 里打印区分三条假说所需的全部量。

## 背景（真机 + addr2line + 反汇编 三重互证）

    675a1091 / c8e8e658 在 Invoke 阶段 SIGSEGV：
        etna_ml_subgraph_read_outputs+0x188   ← etnaviv_ml.c:1076（干净源码 1052）
                                               ← outputs[i][k] = src_map[k] - 128;
        +0x8564c == partition_invoke（tfl_device.c:702 的 ml_subgraph_read_output 调用点）

    故障指令 `ldrb w0, [x24, w26]`（src_map[k]），且 NULL 分支走 __assert_fail
    ⇒ src_map 非 NULL 但是野指针。

## 要区分的三条假说

| # | 假说 | 需要打印的量 |
|---|---|---|
| H1 | `output_idxs[i]` 越界索引 dynarray | idx、n_tensors、槽指针 |
| H2 | `pipe_buffer_size(res) != tensor->size` | tsize、bsize |
| H3 | resource 跨 context 失效 | res、src_map |

输出前缀 `[och-ro]`。

## 本补丁不改变行为

- 只读 `util_dynarray_element` 的槽值，**不调 `etna_ml_get_tensor`**
  （它在槽为 NULL 时会 calloc 并**写入**，idx 越界时等于往野地址写，会掩盖 H1 现场）；
- idx 越界时只打 `OOB`，不解引用 t。

## 用法

    python3 patch-etnaviv-readout-diag.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_RODIAG=1` 启用，默认关。
⚠️ 诊断轮必须同时开运行期开关 `ETNA_MESA_DEBUG=ml_msgs`，
   否则 `ML_DBG` 不打印（格式串进 .rodata，但 `DBG_ENABLED` 是运行期判断）。
"""

import argparse
import sys

MARKER = __MARKER__
TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
STRUCT = __STRUCT__

# ↓↓↓ 逐字锚点（由 tools/gen-rodiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_LOOP = __OLD_LOOP__
NEW_LOOP = __NEW_LOOP__
OLD_MAP = __OLD_MAP__
NEW_MAP = __NEW_MAP__
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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
                break
        j += 1
    return src[at:j + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁，拒绝重复应用。\n")
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
                "       取证依赖这些点，请人工核对后重新生成补丁。\n")
            return 1

    for tag, old, new in (("循环锚点", OLD_LOOP, NEW_LOOP),
                          ("map 锚点", OLD_MAP, NEW_MAP)):
        n = src.count(old)
        if n != 1:
            sys.stderr.write(
                "ERROR: " + tag + " 在文件中匹配到 " + str(n)
                + " 处，期望恰好 1 处 —— 不要跳过校验"
                  "（会静默产出无取证的 .so，白跑一轮真机验证）。\n"
                  "       请重跑 tools/gen-rodiag-patch.py 重新生成。\n")
            return 1

    if NEW_LOOP == OLD_LOOP or NEW_MAP == OLD_MAP:
        sys.stderr.write("ERROR: 生成的新代码与原文相同（替换未生效）\n")
        return 1

    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    src = src.replace(OLD_MAP, NEW_MAP, 1)

    # 反向自检。
    # ⚠️ 不能一律查「OLD 必须消失」：OLD_MAP 是 NEW_MAP 的**相邻前缀**
    #    （只追加型补丁），NEW ⊃ OLD 时正例会被全部误拒 —— 这个坑踩过。
    #    正确口径：OLD_LOOP 两侧被 DIAG_LOOP 隔开 ⇒ 必须消失；
    #              OLD_MAP 原样嵌在 NEW_MAP 里 ⇒ 必须**恰好保留 1 处**。
    if src.count(OLD_LOOP) != 0:
        sys.stderr.write("ERROR: 循环锚点仍在源文件中，替换未生效\n")
        return 1
    if src.count(OLD_MAP) != 1:
        sys.stderr.write("ERROR: map 锚点应恰好保留 1 处（它嵌在 NEW_MAP 里），"
                         "实际 " + str(src.count(OLD_MAP)) + " 处\n")
        return 1
    if src.count(NEW_LOOP) != 1 or src.count(NEW_MAP) != 1:
        sys.stderr.write("ERROR: 新代码出现次数异常\n")
        return 1
    for token in (MARKER, "[och-ro]", "util_dynarray_num_elements",
                  "in_range ?", "OOB"):
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: " + token + "\n")
            return 1
    # 不得引入 etna_ml_get_tensor 的新调用（会写越界槽）
    injected = NEW_LOOP + NEW_MAP
    if "etna_ml_get_tensor" in injected:
        sys.stderr.write("ERROR: 取证代码里出现了 etna_ml_get_tensor —— "
                         "它在槽为 NULL 时会写入，越界时掩盖现场\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-ro] 已加 read_outputs 取证：打印 idx / n_tensors / 槽 / res / tsize / bsize")
    print("  [och-ro] 预期输出前缀 [och-ro]；需同时设 ETNA_MESA_DEBUG=ml_msgs 才会打印")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _extract(src):
    """逐字提取两个锚点并拼出新代码。

    关键：**不手写任何缩进** —— 空格数从真实源码按行切片取。
    （此前用 awk 打印把前导空格吃掉，导致缩进误判两次。）
    """
    lines = src.split("\n")

    # 锚点 1：循环头 + 其后一行（取 resource）
    hits = [i for i, l in enumerate(lines)
            if l.strip() == "for (int i = 0; i < outputs_count; i++) {"]
    if len(hits) != 1:
        die("read_outputs 的循环头出现 " + str(len(hits)) + " 处（期望 1）")
    i_loop = hits[0]
    if lines[i_loop + 1].strip() != \
            "struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);":
        die("循环头下一行不是预期的取 resource 语句：\n" + lines[i_loop + 1])
    old_loop = lines[i_loop] + "\n" + lines[i_loop + 1] + "\n"
    new_loop = lines[i_loop] + "\n" + DIAG_LOOP + lines[i_loop + 1] + "\n"

    # 锚点 2：&src_transfer); 行 + assert(src_map); 行
    cands = [i for i, l in enumerate(lines) if l.strip() == "&src_transfer);"]
    if len(cands) != 1:
        die("`&src_transfer);` 出现 " + str(len(cands)) + " 处（期望 1）")
    i_map = cands[0]
    if lines[i_map + 1].strip() != "assert(src_map);":
        die("&src_transfer); 下一行不是 assert(src_map);：\n" + lines[i_map + 1])
    old_map = lines[i_map] + "\n" + lines[i_map + 1] + "\n"
    new_map = old_map + DIAG_MAP

    return old_loop, new_loop, old_map, new_map


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def main():
    src = open(SRC, encoding="utf-8").read()

    # 生成前先确认 STRUCT 的期望值与真实源码一致
    at = src.find(TARGET_FN)
    if at < 0:
        die("找不到 " + TARGET_FN)
    i = src.index("{", at)
    d = 0
    j = i
    while True:
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                break
        j += 1
    fn_body = src[at:j + 1]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got != want:
            die("STRUCT 期望值与真实源码不符: `" + token + "` 实际 "
                + str(got) + " 处，期望 " + str(want) + " 处")

    old_loop, new_loop, old_map, new_map = _extract(src)

    for nm, o, n in (("OLD_LOOP", old_loop, new_loop), ("OLD_MAP", old_map, new_map)):
        if o == n:
            die(nm + " 替换未生效")
    if MARKER not in new_loop:
        die("NEW_LOOP 缺标记串")

    body = (TEMPLATE
            .replace("__MARKER__", repr(MARKER))
            .replace("__STRUCT__", repr(STRUCT))
            .replace("__OLD_LOOP__", repr(old_loop))
            .replace("__NEW_LOOP__", repr(new_loop))
            .replace("__OLD_MAP__", repr(old_map))
            .replace("__NEW_MAP__", repr(new_map)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(OUT, 0o755)

    print("已生成:", OUT)
    print("--- OLD_LOOP ---"); print(old_loop)
    print("--- NEW_LOOP ---"); print(new_loop)
    print("--- OLD_MAP ---"); print(old_map)
    print("--- NEW_MAP ---"); print(new_map)


if __name__ == "__main__":
    main()