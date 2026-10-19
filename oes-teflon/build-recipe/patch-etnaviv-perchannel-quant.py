#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能补丁：放开 **权重/偏置** 的 per-axis（per-channel）量化。

## 为什么需要（门 c）

`etnaviv_ml_operation_supported()` 的 CONVOLUTION 分支对 input/weight/bias/output
四个张量统一调用 `tensor_quantization_supported()`，而后者要求
`scales == NULL && zero_points == NULL` ⇒ **任何带 per-axis 参数的模型，
卷积族全部不被接管**（实测接管数归零）。

而下降路径**其实已经实现了 per-channel**，只是被那道门挡住：
`etna_ml_nn.c` 里权重按 `scales[oc]` 逐通道反量化，再用 `calc_quantization()`
沿所有通道求出**单一** scale，最后 `requantize_weights()` / `requantize_bias()`
把权重与偏置重量化到那个标量 scale（upstream 自己打印了 WARNING：
"hardware doesn't support it natively … with a loss of accuracy"）。

## ★ 为什么不能无脑把那个判断翻成 true

那张门对**四个**张量一律同判，但只有 **weight / bias** 的下降路径处理了 per-axis：

| 张量 | 下降时怎么取 scale | per-axis 是否已被处理 |
|---|---|---|
| weight | `scales[oc]` 反量化 → 折叠成标量（`nn.c:632-651`） | ✅ 是 |
| bias | `bias_tensor->scales[oc]` → `requantize_bias()` | ✅ 是 |
| input | `input_tensors[0]->scale`（**标量**，`nn.c:573`） | ❌ **否** |
| output | `output_tensors[0]->scale`（**标量**，`nn.c:580`） | ❌ **否** |

放行 input/output 的 per-axis 会让驱动读到**未设置的标量 scale** ⇒ 静默算错。
所以本补丁是**按张量区分**的：weight/bias 用新的宽松判定，input/output 保持严格。

## 代价（必须知道，不能装作没有）

per-axis → 单一 scale 的折叠**有精度损失**（upstream 原话 "with a loss of
accuracy"）：小权重量化到公共 scale 时可能被压到 0。因此本补丁只保证
"能接管、不崩溃"，**不保证数值达标**；门 e（cos ≥ 0.98 且 top-1 一致）
必须靠设备实测，不能由本补丁推断。
"""

import argparse
import sys

MARKER = "och patch: per-channel weight quantization (weight/bias only)"
COMPILE_MARK = "och-patch-perchannel-quant"

FN_GATE = "tensor_quantization_supported(struct pipe_tensor *tensor)"
FN_USE_INPUT = "etna_ml_operation_supported(struct pipe_ml_device *pdevice"
CALL_OLD = ("tensor_quantization_supported(input_tensor) &&\n"
            "             tensor_quantization_supported(weight_tensor) &&\n"
            "             tensor_quantization_supported(bias_tensor) &&\n"
            "             tensor_quantization_supported(output_tensor) &&")

STRUCT_GATE = (
    ("return tensor->scales == NULL && tensor->zero_points == NULL;", 1),
    ("Per-axis quantization not supported", 1),
)
STRUCT_USE = (
    ("tensor_quantization_supported(input_tensor)", 1),
    ("tensor_quantization_supported(weight_tensor)", 1),
    ("tensor_quantization_supported(bias_tensor)", 1),
    ("tensor_quantization_supported(output_tensor)", 1),
)

# ↓↓↓ 逐字锚点（勿手改；由 tools/gen-perchannel-quant-patch.py 从真实源码提取）↓↓↓
OLD_GATE = """static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per_axis_vs_per_tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}
"""
NEW_GATE = """static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per_axis_vs_per_tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}

/* och patch: per-channel weight quantization (weight/bias only)
 *
 * 上面的门对 input/weight/bias/output 一律同判，但只有 **weight/bias** 的下降路径
 * 真的处理了 per-axis：
 *   权重  : scales[oc] 逐通道反量化 -> calc_quantization() 求单一 scale
 *           -> requantize_weights() 重量化            (etnaviv_ml_nn.c)
 *   偏置  : bias_tensor->scales[oc] -> requantize_bias()
 * 而 input/output 的下降只读**标量** tensor->scale（input_tensors[0]->scale /
 * output_tensors[0]->scale），per-axis 张量的标量并未被设置 ⇒
 * 对它们放行会静默算错。
 *
 * 所以这里新增一个**只给 weight/bias 用**的宽松判定：
 * per-tensor 张量本来就满足 scales==NULL，原判定对它已为真，
 * 因此新判定对两种情形都放行。
 *
 * ⚠️ 代价：per-axis 折叠到单一 scale **有精度损失**（upstream 原话
 *    "with a loss of accuracy"），小权重可能被压到 0。
 *    本补丁只保证「能接管、不崩溃」，**不保证数值达标**；
 *    cos / top-1 必须由设备实测（门 e）判定。
 */
static bool
tensor_quantization_supported_per_axis(struct pipe_tensor *tensor)
{
   /* per-tensor（scales == NULL）与 per-axis（scales != NULL）**都**放行：
      两种情形在下降路径里都有对应处理（见上），此处无需再判。
      写成恒真而不是 `||` 拼式，是为了不留下「看起来像在判、其实没判」的误导代码。 */
   (void)tensor;
   return true;
}

/* 编译期标记：让 strings 门能硬判「这个补丁真的在二进制里」 */
__attribute__((used)) static const char och_perchannel_quant_marker[] = "och-patch-perchannel-quant";
"""
OLD_USE = CALL_OLD
NEW_USE = ("tensor_quantization_supported(input_tensor) &&\n"
           "             tensor_quantization_supported_per_axis(weight_tensor) &&\n"
           "             tensor_quantization_supported_per_axis(bias_tensor) &&\n"
           "             tensor_quantization_supported(output_tensor) &&")
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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
        sys.stderr.write("ERROR: 已包含本补丁，拒绝重复应用。\n")
        return 1

    for fn, struct in ((FN_GATE, STRUCT_GATE), (FN_USE_INPUT, STRUCT_USE)):
        b = _body(src, fn)
        if b is None:
            sys.stderr.write("ERROR: 找不到函数 %s（Mesa 结构可能已变）\n" % fn)
            return 1
        for token, want in struct:
            got = b.count(token)
            if got != want:
                sys.stderr.write("ERROR: 在 %s 内 `%s` 出现 %d 处，期望 %d 处。\n"
                                 % (fn, token, got, want))
                return 1

    for tag, old, new in (("gate", OLD_GATE, NEW_GATE),
                          ("use", OLD_USE, NEW_USE)):
        n = src.count(old)
        if n != 1:
            sys.stderr.write("ERROR: %s 锚点匹配到 %d 处，期望恰好 1 处。\n" % (tag, n))
            return 1
        if old == new:
            sys.stderr.write("ERROR: %s 替换未生效\n" % tag)
            return 1

    src = src.replace(OLD_GATE, NEW_GATE, 1)
    src = src.replace(OLD_USE, NEW_USE, 1)

    # 反向自检：gate 是**纯追加**（旧函数体仍保留在 NEW_GATE 里）⇒ 应恰好 1 处；
    # use 是**中间替换**（调用点被改写）⇒ 旧的四连应消失。
    if src.count(OLD_USE) != 0:
        sys.stderr.write("ERROR: use 锚点应在替换后消失\n")
        return 1
    if src.count(OLD_GATE) != 1:
        sys.stderr.write("ERROR: gate 旧函数体应恰好保留 1 处（纯追加型）\n")
        return 1
    for tag, new in (("gate", NEW_GATE), ("use", NEW_USE)):
        if src.count(new) != 1:
            sys.stderr.write("ERROR: %s 新代码出现次数不为 1\n" % tag)
            return 1
    for token in (MARKER, "tensor_quantization_supported_per_axis", COMPILE_MARK):
        if token not in src:
            sys.stderr.write("ERROR: 自检失败，缺少: %s\n" % token)
            return 1
    # ★ 关键不变量：input/output 必须**仍然**走严格判定。
    if src.count("tensor_quantization_supported(input_tensor)") != 1 or \
       src.count("tensor_quantization_supported(output_tensor)") != 1:
        sys.stderr.write("ERROR: input/output 被误改成宽松判定 —— 会读到未设置的标量 scale\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [perchannel] 已放开 weight/bias 的 per-axis 量化接管（input/output 仍严格）")
    print("  [perchannel] ⚠️ 折叠有精度损失，cos/top-1 必须设备实测")
    return 0


if __name__ == "__main__":
    sys.exit(main())