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
NEW_MAP = '                                                     &src_transfer);\n         assert(src_map);\n         /* och patch: read_outputs map 结果取证 */\n         ML_DBG("[och-ro]   map res=%p bsize=%u off_expected=%u -> src_map=%p\\n",\n                (void *)res, res ? pipe_buffer_size(res) : 0u,\n                etna_ml_get_size(subgraph, output_idxs[i]), (void *)src_map);\n         /* och patch: 头 16 字节 —— 判别「没写回(全 0x7f/0xff) / 别名到输入(输入图案) /\n          * 真的算出了 int8 值」。守卫 pipe_buffer_size(res) >= 16，避免本诊断自己越界读。 */\n         if (src_map && pipe_buffer_size(res) >= 16u) {\n            ML_DBG("[och-ro]   head=%02x %02x %02x %02x %02x %02x %02x %02x "\n                   "%02x %02x %02x %02x %02x %02x %02x %02x\\n",\n                   src_map[0], src_map[1], src_map[2], src_map[3],\n                   src_map[4], src_map[5], src_map[6], src_map[7],\n                   src_map[8], src_map[9], src_map[10], src_map[11],\n                   src_map[12], src_map[13], src_map[14], src_map[15]);\n         }\n'
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

    print("  [och-ro] 已加 read_outputs 取证：打印 idx / n_tensors / 槽 / res / tsize / bsize")
    print("  [och-ro] 预期输出前缀 [och-ro]；需同时设 ETNA_MESA_DEBUG=ml_msgs 才会打印")
    return 0


if __name__ == "__main__":
    sys.exit(main())
