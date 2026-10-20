#!/usr/bin/env python3
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

## 写端填充探针（`[och-wd]`，运行期 `TEFLON_WFILL=1`）

只打印读端无法区分「NPU 没写」与「写了但 CPU 读到旧值」。探针在**提交前**
把输出 BO 按 `tensor->offset/size` 整段填 `0xAA`，再交由正常的
`context->flush()` 提交作业；提交后由 `[och-ro] head/sample` 看被改写范围：

| 观察 | 结论 |
|---|---|
| 提交后仍全 `0xaa` | NPU **一个字节都没写进这个 BO** ⇒ 写地址侧（B2） |
| 部分/全部被改写 | NPU 确实写了 ⇒ CPU 侧读到的不一致才是问题（B3） |

⚠️ 探针会**破坏**输出内容 ⇒ 只在诊断产物上、且只在上机时按需打开。
不设 `TEFLON_WFILL` 时该段代码整体跳过，行为与上游完全一致
（这也是它不另开构建开关、直接随 RODIAG 产物的原因）。

## 用法

    python3 patch-etnaviv-readout-diag.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_RODIAG=1` 启用，默认关。
⚠️ 诊断轮必须同时开运行期开关 `ETNA_MESA_DEBUG=ml_msgs`，
   否则 `ML_DBG` 不打印（格式串进 .rodata，但 `DBG_ENABLED` 是运行期判断）。
"""

import argparse
import re
import sys

MARKER = 'och patch: read_outputs 越界取证'
TARGET_FN = "etna_ml_subgraph_read_outputs(struct pipe_context"
STRUCT = (('for (int i = 0; i < outputs_count; i++) {', 1), ('struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);', 1), ('src_map = (uint8_t *) pipe_buffer_map_range(context,', 1), ('assert(src_map);', 1), ('((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;', 1))

# ↓↓↓ 逐字锚点（由 tools/gen-rodiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD_LOOP = '   for (int i = 0; i < outputs_count; i++) {\n      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);\n'
NEW_LOOP = '   for (int i = 0; i < outputs_count; i++) {\n      /* och patch: read_outputs 越界取证 */\n      {\n         unsigned n_ten = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);\n         unsigned idx = output_idxs[i];\n         int in_range = (idx < n_ten);\n         /* 只读槽值，不调 etna_ml_get_tensor：它会在槽为 NULL 时 calloc 并写入，\n          * idx 越界时等于往野地址写，会掩盖现场。 */\n         struct etna_ml_tensor **slotp = in_range\n            ? util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, idx)\n            : NULL;\n         struct etna_ml_tensor *t = slotp ? *slotp : NULL;\n         ML_DBG("[och-ro] i=%d idx=%u n_tensors=%u %s slot=%p res=%p tsize=%u toff=%u\\n",\n                i, idx, n_ten, in_range ? "IN " : "OOB", (void *)slotp,\n                (void *)(t ? t->resource : NULL), t ? t->size : 0u, t ? t->offset : 0u);\n      }\n      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);\n'
OLD_MAP = '                                                     &src_transfer);\n         assert(src_map);\n'
NEW_MAP = '                                                     &src_transfer);\n         assert(src_map);\n         /* och patch: read_outputs map 结果取证 */\n         ML_DBG("[och-ro]   map res=%p bsize=%u off_expected=%u -> src_map=%p\\n",\n                (void *)res, res ? pipe_buffer_size(res) : 0u,\n                etna_ml_get_size(subgraph, output_idxs[i]), (void *)src_map);\n         /* och patch: 头 16 字节 —— 判别「没写回(全 0x7f/0xff) / 别名到输入(输入图案) /\n          * 真的算出了 int8 值」。守卫 pipe_buffer_size(res) >= 16，避免本诊断自己越界读。 */\n         if (src_map && pipe_buffer_size(res) >= 16u) {\n            ML_DBG("[och-ro]   head=%02x %02x %02x %02x %02x %02x %02x %02x "\n                   "%02x %02x %02x %02x %02x %02x %02x %02x\\n",\n                   src_map[0], src_map[1], src_map[2], src_map[3],\n                   src_map[4], src_map[5], src_map[6], src_map[7],\n                   src_map[8], src_map[9], src_map[10], src_map[11],\n                   src_map[12], src_map[13], src_map[14], src_map[15]);\n         }\n         /* och patch: 空间采样 —— 配合 TEFLON_WFILL 判断「写了多少」（独立模式按 bsize 取样）*/\n         if (src_map && pipe_buffer_size(res) >= 64u) {\n            unsigned och_sz = pipe_buffer_size(res);\n            ML_DBG("[och-ro]   sample sz=%u @0=%02x @1/8=%02x @1/4=%02x"\n                   " @1/2=%02x @3/4=%02x @7/8=%02x @last=%02x\\n",\n                   och_sz, src_map[0], src_map[och_sz / 8u],\n                   src_map[och_sz / 4u], src_map[och_sz / 2u],\n                   src_map[och_sz * 3u / 4u], src_map[och_sz * 7u / 8u],\n                   src_map[och_sz - 1u]);\n         }\n'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# ── 叠加模式：补丁打在【已应用 ROFIX】的源码上 ─────────────────────────────
#
# 为什么必须有这个模式（2026-10-20 真机事故）：
#   `phantomdiag` 产物是 ROFIX=0 建的。它把 read_outputs 的范围对齐**去掉了**，
#   ⇒ map 用 (0, pipe_buffer_size)、读用 tsize，越界读重新出现。
#   该产物被当成「低危诊断仪器」送上真机，作业启动后几秒内**整机硬挂**
#   （kern.log 之后再无任何内核输出，需断电重启）。见 verify/reports/38。
#
#   取证打印本身不改行为；危险的是**丢掉 ROFIX**。所以正确形态是
#   「ROFIX 在前 + 取证叠加在后」，两者不再互斥。
#
# ROFIX 之后循环体的形态（关键变量已存在，取证直接复用，不再自己查槽）：
#
#    for (int i = 0; i < outputs_count; i++) {
#       /* och patch: read_outputs 范围对齐 invoke ... */
#       unsigned n_ten = util_dynarray_num_elements(...);
#       if (output_idxs[i] >= n_ten) { ...; continue; }
#       struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, output_idxs[i]);
#       struct pipe_resource *res = tensor->resource;      ← 取证块插在这两行之后
#
ROFIX_MARKER = 'och patch: read_outputs 范围对齐 invoke'

# ROFIX 之后的函数体必须含这些点（与独立模式不同：res 已改由 tensor 提供）
STRUCT_ROFIX = (('for (int i = 0; i < outputs_count; i++) {', 1),
                ('struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, output_idxs[i]);', 1),
                ('struct pipe_resource *res = tensor->resource;', 1),
                ('src_map = (uint8_t *) pipe_buffer_map_range(context,', 1),
                ('assert(src_map);', 1),
                ('((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;', 1))

_TAIL_ROFIX = '      struct pipe_resource *res = tensor->resource;\n'
ON_ROFIX_ANCHOR = ('      struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, output_idxs[i]);\n'
                   + _TAIL_ROFIX)

NEW_LOOP_ROFIX = (ON_ROFIX_ANCHOR +
                  '      /* och patch: read_outputs 越界取证（叠加在 ROFIX 之上；只读，不改行为）\n'
                  '       *\n'
                  '       * ★ 必须叠加在 ROFIX 之上：ROFIX 负责把 map 范围与读取范围对齐，\n'
                  '       *   丢掉它就是把越界读重新引入（2026-10-20 整机硬挂，见报告 38）。\n'
                  '       * 复用 ROFIX 已算好的 n_ten / tensor / res，不新增 etna_ml_get_tensor。 */\n'
                  '      {\n'
                  '         ML_DBG("[och-ro] i=%d idx=%u n_tensors=%u IN  res=%p tsize=%u toff=%u\\n",\n'
                  '                i, output_idxs[i], n_ten, (void *)res,\n'
                  '                tensor->size, tensor->offset);\n'
                  '      }\n')

NEW_MAP_ROFIX = ('                                                     &src_transfer);\n'
                 '         assert(src_map);\n'
                 '         /* och patch: read_outputs map 结果取证（ROFIX 之后：映射长度就是 tensor->size）*/\n'
                 '         ML_DBG("[och-ro]   map res=%p bsize=%u off_expected=%u -> src_map=%p\\n",\n'
                 '                (void *)res, res ? pipe_buffer_size(res) : 0u,\n'
                 '                tensor->size, (void *)src_map);\n'
                 '         if (src_map && tensor->size >= 16u) {\n'
                 '            ML_DBG("[och-ro]   head=%02x %02x %02x %02x %02x %02x %02x %02x "\n'
                 '                   "%02x %02x %02x %02x %02x %02x %02x %02x\\n",\n'
                 '                   src_map[0], src_map[1], src_map[2], src_map[3],\n'
                 '                   src_map[4], src_map[5], src_map[6], src_map[7],\n'
                 '                   src_map[8], src_map[9], src_map[10], src_map[11],\n'
                 '                   src_map[12], src_map[13], src_map[14], src_map[15]);\n'
                 '         }\n'
                 '         /* och patch: 空间采样 —— 配合 TEFLON_WFILL 判断「写了多少」*/\n'
                 '         if (src_map && tensor->size >= 64u) {\n'
                 '            unsigned och_sz = tensor->size;\n'
                 '            ML_DBG("[och-ro]   sample sz=%u @0=%02x @1/8=%02x @1/4=%02x"\n'
                 '                   " @1/2=%02x @3/4=%02x @7/8=%02x @last=%02x\\n",\n'
                 '                   och_sz, src_map[0], src_map[och_sz / 8u],\n'
                 '                   src_map[och_sz / 4u], src_map[och_sz / 2u],\n'
                 '                   src_map[och_sz * 3u / 4u], src_map[och_sz * 7u / 8u],\n'
                 '                   src_map[och_sz - 1u]);\n'
                 '         }\n')


# ── 写端填充探针（B2 / B3 判决）──────────────────────────────────────────────
#
# 读端打印只能证明「读到的全 0」，无法区分：
#   B2 写地址算错 ⇒ NPU 把结果写到了别处（甚至没写）
#   B3 缓存陈旧   ⇒ NPU 写了，CPU 读的是旧映射
# 在**提交前**把输出 BO 填成 0xAA，提交后看有没有被改写，就能一刀切开：
#   仍全 0xaa ⇒ 没写（B2）    被改写 ⇒ 写了（B3 才可能成立）
#
# ★ 运行期门控（getenv TEFLON_WFILL），不新增构建开关：
#   不设环境变量时这段代码整段跳过，产物行为与上游一致。
# ★ 不调 etna_ml_get_tensor：它会在槽为 NULL 时 calloc 并写入，
#   直接读槽值 + 先判越界，与 ROFIX/取证块同一纪律。
WFILL_MARKER = 'och patch: 写端填充探针'

WFILL_INC_OLD = '#include <stdio.h>\n'
WFILL_INC_NEW = ('#include <stdio.h>\n'
                 '#include <stdlib.h>   /* och patch: TEFLON_WFILL 需要 getenv */\n')

WFILL_ANCHOR = '   if (operation_count > 0) {\n'

WFILL_NEW = (
    '   /* och patch: 写端填充探针（运行期 TEFLON_WFILL=1；不设则行为与上游一致）\n'
    '    *\n'
    '    * 目的：把「NPU 到底有没有写进输出 BO」变成可观测事实。\n'
    '    *   提交前整段填 0xaa → 提交后看读端 head/sample 哪些位置被改写。\n'
    '    *   全 0xaa ⇒ 一个字节都没写 ⇒ 写地址侧（B2）\n'
    '    *   被改写  ⇒ NPU 真的写了   ⇒ 读端问题（B3）\n'
    '    * ⚠️ 本探针破坏输出内容，只许在诊断产物上按需打开。 */\n'
    '   {\n'
    '      const char *och_wfill = getenv("TEFLON_WFILL");\n'
    '      if (och_wfill && och_wfill[0] == \'1\') {\n'
    '         unsigned och_n_ten = util_dynarray_num_elements(&subgraph->tensors,\n'
    '                                                         struct etna_ml_tensor *);\n'
    '         for (unsigned och_w = 0; och_w < outputs_count; och_w++) {\n'
    '            /* 先判越界再取槽：越界时 util_dynarray_element 会算出野地址 */\n'
    '            if (output_idxs[och_w] >= och_n_ten) {\n'
    '               ML_DBG("[och-wd] idx %u >= n_tensors %u，跳过（拒绝 map 野槽）\\n",\n'
    '                      output_idxs[och_w], och_n_ten);\n'
    '               continue;\n'
    '            }\n'
    '            struct etna_ml_tensor **och_slotp = util_dynarray_element(\n'
    '               &subgraph->tensors, struct etna_ml_tensor *, output_idxs[och_w]);\n'
    '            struct etna_ml_tensor *och_t = och_slotp ? *och_slotp : NULL;\n'
    '            if (!och_t || !och_t->resource || och_t->size == 0u)\n'
    '               continue;\n'
    '            struct pipe_transfer *och_tr = NULL;\n'
    '            uint8_t *och_map = (uint8_t *) pipe_buffer_map_range(context, och_t->resource,\n'
    '                                                                 och_t->offset, och_t->size,\n'
    '                                                                 PIPE_MAP_WRITE, &och_tr);\n'
    '            if (!och_map) {\n'
    '               ML_DBG("[och-wd] idx=%u res=%p off=%u size=%u MAP_FAILED\\n",\n'
    '                      output_idxs[och_w], (void *)och_t->resource,\n'
    '                      och_t->offset, och_t->size);\n'
    '               continue;\n'
    '            }\n'
    '            memset(och_map, 0xAA, och_t->size);\n'
    '            ML_DBG("[och-wd] idx=%u res=%p va=%x off=%u size=%u bsize=%u fill=0xaa\\n",\n'
    '                   output_idxs[och_w], (void *)och_t->resource,\n'
    '                   (unsigned)etna_bo_gpu_va(etna_buffer_resource(och_t->resource)->bo),\n'
    '                   och_t->offset, och_t->size, pipe_buffer_size(och_t->resource));\n'
    '            pipe_buffer_unmap(context, och_tr);\n'
    '         }\n'
    '      }\n'
    '   }\n'
    + WFILL_ANCHOR)



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
    if WFILL_MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含写端填充探针，拒绝重复应用。\n")
        return 1

    on_rofix = ROFIX_MARKER in src

    body = _fn_body(src)
    if body is None:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1

    for token, want in (STRUCT_ROFIX if on_rofix else STRUCT):
        got = body.count(token)
        if got != want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 "
                + str(got) + " 处，期望 " + str(want) + " 处。\n"
                "       取证依赖这些点，请人工核对后重新生成补丁。\n")
            return 1

    if on_rofix:
        pairs = (("叠加锚点（ROFIX 之后）", ON_ROFIX_ANCHOR, NEW_LOOP_ROFIX),
                 ("map 锚点", OLD_MAP, NEW_MAP_ROFIX))
        mode = "叠加在 ROFIX 之上"
    else:
        pairs = (("循环锚点", OLD_LOOP, NEW_LOOP),
                 ("map 锚点", OLD_MAP, NEW_MAP))
        mode = "独立（源码未含 ROFIX）"

    # 写端填充探针：两种模式都要（它是「提交前」的动作，与读端形态无关）
    pairs = pairs + (("写端填充探针锚点", WFILL_ANCHOR, WFILL_NEW),)

    n_get_before = src.count("etna_ml_get_tensor(")

    for tag, old, new in pairs:
        n = src.count(old)
        if n != 1:
            sys.stderr.write(
                "ERROR: " + tag + " 在文件中匹配到 " + str(n)
                + " 处，期望恰好 1 处 —— 不要跳过校验"
                  "（会静默产出无取证的 .so，白跑一轮真机验证）。\n"
                  "       请重跑 tools/gen-rodiag-patch.py 重新生成。\n")
            return 1
        if new == old:
            sys.stderr.write("ERROR: 生成的新代码与原文相同（替换未生效）\n")
            return 1

    for tag, old, new in pairs:
        src = src.replace(old, new, 1)

    # 头文件：填充探针用 getenv，必须显式引入 stdlib.h（不赌传递包含）
    if src.count(WFILL_INC_OLD) != 1:
        sys.stderr.write("ERROR: `#include <stdio.h>` 应恰好 1 处，实际 "
                         + str(src.count(WFILL_INC_OLD)) + " 处（无法安全插入 stdlib.h）\n")
        return 1
    src = src.replace(WFILL_INC_OLD, WFILL_INC_NEW, 1)

    # 反向自检。
    # ⚠️ 不能一律查「OLD 必须消失」：map 锚点是 NEW 的**相邻前缀**
    #    （只追加型补丁），NEW ⊃ OLD 时正例会被全部误拒 —— 这个坑踩过。
    #    正确口径：map 锚点原样嵌在 NEW 里 ⇒ 必须**恰好保留 1 处**。
    if src.count(OLD_MAP) != 1:
        sys.stderr.write("ERROR: map 锚点应恰好保留 1 处（它嵌在 NEW 里），"
                         "实际 " + str(src.count(OLD_MAP)) + " 处\n")
        return 1
    for tag, old, new in pairs:
        if src.count(new) != 1:
            sys.stderr.write("ERROR: " + tag + " 新代码出现次数异常\n")
            return 1

    if on_rofix:
        # 叠加模式：锚点两行必须原样保留 1 处；且**不得新增**任何
        # etna_ml_get_tensor 调用（ROFIX 已经查过一次，取证只复用）。
        if src.count(ON_ROFIX_ANCHOR) != 1:
            sys.stderr.write("ERROR: 叠加锚点应恰好保留 1 处，实际 "
                             + str(src.count(ON_ROFIX_ANCHOR)) + " 处\n")
            return 1
        if src.count("etna_ml_get_tensor(") != n_get_before:
            sys.stderr.write("ERROR: 叠加模式新增了 etna_ml_get_tensor 调用 —— "
                             "取证只许复用 ROFIX 已取到的 tensor\n")
            return 1
        must = (MARKER, "[och-ro]", "head=", "tensor->size", "n_ten")
    else:
        if src.count(OLD_LOOP) != 0:
            sys.stderr.write("ERROR: 循环锚点仍在源文件中，替换未生效\n")
            return 1
        must = (MARKER, "[och-ro]", "util_dynarray_num_elements",
                "in_range ?", "OOB")

    # 写端填充探针的自检（与模式无关）
    must = must + ("[och-wd]", WFILL_MARKER, 'getenv("TEFLON_WFILL")',
                   "memset(och_map, 0xAA", "output_idxs[och_w] >= och_n_ten")

    for token in must:
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: " + token + "\n")
            return 1

    # 顺序门：填充必须发生在**提交之前**。若插到 flush 之后，探测到的永远是
    # NPU 已经写过（或被跳过）的场面，B2/B3 就再也分不开 —— 而且不会报错。
    body_after = _fn_body(src)
    if body_after is None:
        sys.stderr.write("ERROR: 打完后找不到 read_outputs 函数体（无法核对顺序）\n")
        return 1
    at_fill = body_after.find(WFILL_MARKER)
    at_flush = body_after.find("context->flush(context, NULL, 0);")
    if at_fill < 0 or at_flush < 0 or at_fill > at_flush:
        sys.stderr.write("ERROR: 写端填充探针不在 context->flush() 之前 "
                         "(fill=%d flush=%d) ⇒ 探针无效\n" % (at_fill, at_flush))
        return 1
    # 不得引入 etna_ml_get_tensor 的新调用（它会在槽为 NULL 时 calloc 并写入，
    # idx 越界时等于往野地址写，会掩盖 H1 现场）。
    # ⚠️ 必须**先剥掉注释**再查：本文件的取证注释里恰好写了「不调 etna_ml_get_tensor」，
    #    不剥注释会把正例全部误拒（这个坑踩过一次）。
    injected = NEW_LOOP + NEW_MAP
    stripped = re.sub(r"/\*.*?\*/", "", injected, flags=re.S)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    if "etna_ml_get_tensor" in stripped:
        sys.stderr.write("ERROR: 取证代码里出现了 etna_ml_get_tensor —— "
                         "它在槽为 NULL 时会写入，越界时掩盖现场\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-ro] 已加 read_outputs 取证（" + mode + "）")
    print("  [och-ro]   打印 idx / n_tensors / 槽 / res / tsize / toff / bsize / head / sample")
    print("  [och-wd] 已加写端填充探针（运行期 TEFLON_WFILL=1 才生效，默认不改行为）")
    print("  [och-wd]   提交前输出 BO 填 0xaa，提交后看 head/sample 是否被改写 ⇒ 分 B2/B3")
    print("  [och-ro] 预期输出前缀 [och-ro]；需同时设 ETNA_MESA_DEBUG=ml_msgs 才会打印")
    if not on_rofix:
        print("  [och-ro] ⚠️⚠️ 本产物【不含 ROFIX】：read_outputs 的越界读重新存在。")
        print("  [och-ro]     2026-10-20 该形态上机后整机硬挂（报告 38）。")
        print("  [och-ro]     build-teflon.sh 已强制要求 TEFLON_ROFIX=1 才能开 RODIAG。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
