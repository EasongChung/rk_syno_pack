#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成 build-recipe/patch-etnaviv-filter-shape-guard.py。

## 根因（真机 + 本地 addr2line + 离线模型审计 四重互证）

1. `b9936131`（clamp 版）与 `f9bf1d51`（重映射版）在真机上**崩在完全相同的地址**
   `+0x22fda0`，本地 `addr2line`（对着 unstripped 产物）解出：

       subsample   etnaviv_ml_nn.c:424
       out[x][y] = in[in_x][in_y][in_z];

2. 调用链：`etna_ml_lower_convolution` → `strided_to_normal`（仅 `stride > 1` 时）
   → `reshape` → `subsample`。

3. 离线模型审计（`verify/audit_stride2_ops.py` + 逐算子 filter 形状核对）定位到崩溃分区里的
   具体算子：

       op#58   CONV_2D  input=[1,52,52,244]  filter=[244,3,3,1]  stride=2x2
       op#153  CONV_2D  input=[1,25,25,488]  filter=[488,3,3,1]  stride=2x2

   **filter 的 in_c（最后一维）= 1，而输入张量通道数 = 244 / 488。**

4. 算术完全对得上：
   - `etna_ml_lower_convolution:571` 取 `operation->input_channels = input_tensors[0]->dims[3]` = **244**；
   - `strided_to_normal` 里 `wdims_in = {output_channels, w, h, input_channels=244}`；
   - `reshape` 里 `map_in = input + out_channel * dims_in[1]*dims_in[2]*dims_in[3]`
     = `input + out_channel * 3*3*244` = `input + out_channel * 2196`；
   - 而权重张量实际只有 `244*3*3*1 = 2196` 字节；
   - ⇒ **`out_channel >= 1` 时 `map_in` 已越过张量末尾** → `subsample` 读越界 → SIGSEGV。

## 为什么改法是「不接管」而不是「改系数代码」

驱动在 `etna_ml_lower_convolution:582-585` 用 filter 的**实际 dims** 算权重大小，
但 `:571` 的 `input_channels` 取自**输入张量**。当两者不一致时，
**从 coeff 构造（`write_core_*` 用 `weights_map[input_channels][w][h]`）到
`strided_to_normal` 的整条路径都建立在「filter 深度 == 输入通道」这个不成立的前提上**。

因此正确的修法不是去改某处系数排布（那样只会把越界换成静默错误数值），
而是**让驱动识别这种算子并拒绝接管**，回退到 CPU —— 这是 Mesa 上游风格
（`etna_ml_operation_supported` 本来就是干这个的，它已经拒绝了 dilation 与 per-axis 量化）。

**判据**：非 depthwise 卷积必须满足 `weight_tensor->dims[3] == input_tensors[0]->dims[3]`。
depthwise 的 filter 布局是 `[1, h, w, out_c]`（`dims[3]` 是输出通道、`dims[0]` 恒为 1），
不适用该判据，必须豁免。

## 用法

    python3 patch-etnaviv-filter-shape-guard.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_FSGUARD=1` 启用，默认关。
"""

import os
import sys

SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
       "src/gallium/drivers/etnaviv/etnaviv_ml.c")
OUT = "/root/workspace/oes-teflon/build-recipe/patch-etnaviv-filter-shape-guard.py"

TARGET_FN = "etna_ml_operation_supported(struct pipe_ml_device"
MARKER = "och patch: filter 深度与输入通道一致性守卫"

# 目标函数体内必须精确匹配的结构特征（token, 期望次数）
STRUCT = (
    ("case PIPE_ML_OPERATION_TYPE_CONVOLUTION:", 1),
    ("operation->conv.weight_tensor", 1),
    ("tensor_quantization_supported(input_tensor)", 1),
    ("operation->conv.dilation_width_factor == 1", 1),
)


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def extract_anchor(src: str):
    """在目标函数里定位 conv case 的 if 条件行，向上取两行组成唯一锚点。"""
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        die("源码里找不到函数 " + TARGET_FN)
    nxt = src.find("\n}\n", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]

    hit = -1
    for i, l in enumerate(fn_body.split("\n")):
        if "// Dilation and per-axis quantization not yet implemented" in l:
            hit = i
            break
    if hit < 0:
        die("函数内找不到注释行「Dilation and per-axis quantization」")

    lines = fn_body.split("\n")
    ctx = lines[hit - 2: hit + 1]
    old = "\n".join(ctx) + "\n"
    n = src.count(old)
    if n != 1:
        die("锚点在整个文件出现 " + str(n) + " 处（期望 1）")
    return old, lines[hit]


def build_new(old: str, comment_line: str) -> str:
    lines = old.rstrip("\n").split("\n")
    head = "\n".join(lines[:-1])
    indent = comment_line[: len(comment_line) - len(comment_line.lstrip())]
    body = (
        indent + "/* och patch: filter 深度与输入通道一致性守卫 */\n"
        + indent + "/* 崩溃算子：op#58 CONV_2D input=[1,52,52,244] filter=[244,3,3,1] stride=2\n"
        + indent + " *           op#153 CONV_2D input=[1,25,25,488] filter=[488,3,3,1] stride=2\n"
        + indent + " * filter 的 in_c（最后一维）是 1，而输入张量通道数是 244/488。\n"
        + indent + " *\n"
        + indent + " * 后果（真机 SIGSEGV，addr2line → subsample etnaviv_ml_nn.c:424）：\n"
        + indent + " *   input_channels 取自输入张量 = 244 → strided_to_normal 的 wdims_in[3]=244\n"
        + indent + " *   → reshape 里 map_in = input + out_channel*3*3*244 = input + out_channel*2196\n"
        + indent + " *   而权重张量只有 244*3*3*1 = 2196 字节 → out_channel>=1 必然越界读。\n"
        + indent + " *\n"
        + indent + " * 这里**不接管**而不是改系数排布：从 coeff 构造（weights_map[input_channels]）\n"
        + indent + " * 到 strided_to_normal 整条路径都建立在「filter 深度 == 输入通道」这个\n"
        + indent + " * 不成立的前提上，改单点只会把越界换成静默错误数值。\n"
        + indent + " * depthwise 的 filter 布局是 [1,h,w,out_c]，dims[3] 是输出通道，豁免。 */\n"
        + indent + "if (!operation->conv.depthwise &&\n"
        + indent + "    operation->conv.weight_tensor->dims[3] != input_tensor->dims[3]) {\n"
        + indent + "   ML_DBG(\"filter in_c (%u) != input channels (%u)，回退 CPU\\n\",\n"
        + indent + "          operation->conv.weight_tensor->dims[3], input_tensor->dims[3]);\n"
        + indent + "   /* 用 return false 而不是 break：\n"
        + indent + "    * break 依赖「此刻 supported 仍是 false」这个隐含前提，\n"
        + indent + "    * 将来若有人把 supported = true 提到守卫之前，break 就会变成 fail-open。\n"
        + indent + "    * return false 是显式的、与上下文无关的失败出口。 */\n"
        + indent + "   return false;\n"
        + indent + "}\n"
    )
    return head + "\n" + body


TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复补丁：不接管「filter 深度 ≠ 输入通道数」的卷积，回退 CPU。

## 根因（真机 + 本地 addr2line + 离线模型审计 四重互证）

1. `b9936131`（clamp 版）与 `f9bf1d51`（重映射版）在真机上**崩在完全相同的地址**
   `+0x22fda0`；本地 `addr2line`（对着 unstripped 产物）解出
   `subsample  etnaviv_ml_nn.c:424`（`out[x][y] = in[in_x][in_y][in_z];`）。
2. 调用链：`etna_ml_lower_convolution` → `strided_to_normal`（stride>1）
   → `reshape` → `subsample`。
3. 离线模型审计定位到崩溃分区里的算子：

       op#58   CONV_2D input=[1,52,52,244] filter=[244,3,3,1] stride=2x2
       op#153  CONV_2D input=[1,25,25,488] filter=[488,3,3,1] stride=2x2

   **filter 的 in_c（最后一维）= 1，输入张量通道数 = 244 / 488。**
4. 算术对得上：`input_channels` 取自输入张量 = 244 → `wdims_in[3]=244`
   → `map_in = input + out_channel*3*3*244 = input + out_channel*2196`，
   而权重张量只有 `244*3*3*1 = 2196` 字节 → `out_channel>=1` 必然越界 → SIGSEGV。

## 本补丁做什么

在 `etna_ml_operation_supported()` 的 CONVOLUTION 分支里加一致性守卫：
非 depthwise 卷积若 `weight_tensor->dims[3] != input_tensor->dims[3]` 则**拒绝接管**。

**为什么不改系数排布**：从 coeff 构造（`weights_map[input_channels][w][h]`）
到 `strided_to_normal` 的整条路径都建立在「filter 深度 == 输入通道」这个不成立的前提上，
改单点只会把越界换成**静默错误数值**。回退 CPU 是唯一不产生错误数值的做法。

## 验收

「回退 CPU」这件事本身可以验证：接管数会下降（那 2 个算子不再被接管），
但**不崩**、且第 ④ 级的逐字节比对应当通过（因为那 2 个算子走 CPU，结果正确）。

## 用法

    python3 patch-etnaviv-filter-shape-guard.py <etnaviv_ml.c 路径>

由 `build-teflon.sh` 的 `TEFLON_FSGUARD=1` 启用，默认关。
"""

import argparse
import sys

MARKER = "och patch: filter 深度与输入通道一致性守卫"
TARGET_FN = "etna_ml_operation_supported(struct pipe_ml_device"

# (特征串, 期望出现次数) —— 期望值来自真实源码统计
STRUCT = __STRUCT__

# ↓↓↓ 逐字锚点（由 tools/gen-fsguard-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = __OLD__
NEW = __NEW__
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁，拒绝重复应用。\n")
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处锚点，期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对并重新生成补丁"
            "（python3 tools/gen-fsguard-patch.py），不要跳过校验\n"
            "       （会静默产出未修复的 .so，崩溃会照旧）。\n")
        return 1

    # 结构性检查：限定目标函数体 + 精确计数
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        sys.stderr.write("ERROR: 源文件找不到函数 " + TARGET_FN + "（Mesa 结构可能已变）\n")
        return 1
    nxt = src.find("\n}\n", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got != want:
            sys.stderr.write(
                "ERROR: 在 " + TARGET_FN + " 函数体内 `" + token + "` 出现 " + str(got)
                + " 处，期望 " + str(want) + " 处。\n"
                "       Mesa 结构已变 —— 守卫逻辑依赖这些点，请人工核对后重新生成补丁。\n")
            return 1

    src = src.replace(OLD, NEW, 1)

    for token in (MARKER,
                  "!operation->conv.depthwise",
                  "operation->conv.weight_tensor->dims[3] != input_tensor->dims[3]",
                  "return false;"):
        if token not in src:
            sys.stderr.write("ERROR: 打完补丁后自检失败，缺少: " + token + "\n")
            return 1
    # 关键结构自检：`return false;` 必须在守卫块内、且在 `supported = true;` 之前，
    # 否则守卫会 fail-open（漏放行 = 崩溃照旧）。
    g_at = src.index(MARKER, src.index(TARGET_FN))
    g_end = src.index("}", g_at)
    guard = src[g_at: g_end]
    if "return false;" not in guard:
        sys.stderr.write("ERROR: 守卫块内没有 return false —— 补丁形态不对\n")
        return 1
    st_true = src.find("supported = true;", g_end)
    st_true_early = src.rfind("supported = true;", 0, g_at)
    if st_true_early != -1:
        sys.stderr.write("ERROR: `supported = true;` 出现在守卫之前 —— "
                         "守卫会 fail-open，请重新生成\n")
        return 1
    if src.count(OLD) != 0:
        sys.stderr.write("ERROR: 锚点仍在源文件中，替换未生效\n")
        return 1
    if src.count(NEW) != 1:
        sys.stderr.write("ERROR: 打完补丁后新代码出现次数为 " + str(src.count(NEW)) + "（期望 1）\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-fsguard] 已加 filter 深度一致性守卫：不自洽卷积回退 CPU")
    print("  [och-fsguard] 预期效果：不崩 + 接管数下降 + 第 ④ 级逐字节比对通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

STRUCT_TOKEN = "och patch: filter 深度与输入通道一致性守卫"


def main():
    src = open(SRC, encoding="utf-8").read()

    # 生成前先确认结构特征在真实源码里确实是期望次数（防止期望值本身写错）
    fn_at = src.find(TARGET_FN)
    if fn_at < 0:
        die("源码里找不到 " + TARGET_FN)
    nxt = src.find("\n}\n", fn_at)
    fn_body = src[fn_at: nxt if nxt > 0 else len(src)]
    for token, want in STRUCT:
        got = fn_body.count(token)
        if got != want:
            die("结构特征期望值与真实源码不符: `" + token + "` 实际 "
                + str(got) + " 处，期望 " + str(want) + " 处")

    old, comment_line = extract_anchor(src)
    new = build_new(old, comment_line)

    if new == old:
        die("生成的新代码与原文相同（替换未生效）")
    if STRUCT_TOKEN not in new:
        die("生成的新代码缺少标记串")

    body = (TEMPLATE
            .replace("__OLD__", repr(old))
            .replace("__NEW__", repr(new))
            .replace("__STRUCT__", repr(STRUCT)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(OUT, 0o755)

    print("已生成:", OUT)
    print("--- 锚点（OLD）逐字预览 ---")
    print(old)
    print("--- 替换后（NEW）逐字预览 ---")
    print(new)


if __name__ == "__main__":
    main()
