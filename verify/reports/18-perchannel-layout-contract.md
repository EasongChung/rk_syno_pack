# 18. per-channel 下降路径的**布局契约**违反（不止 depthwise 一处）

> 对象：`etnaviv_ml_nn.c` / `etnaviv_ml_nn_v7.c`（mesa-src-26.2.5，A311D 走 **v7**）
> 结论：**一行修法不足以修好**。它会把崩溃换成**静默的错误权重**。
> 本文每条结论都带源码行号，可独立复核。

## 0. 一句话

`requantize_weights()` 按 `[out][w][h][in]` **读**权重，
但它跑在 `transpose()` **之后**，而 `transpose()` 刚把权重改成 `[out][in][w][h]`。
⇒ 读侧布局与实际缓冲区**不一致**；depthwise 因为把 `input_channels` 强制成 1
而**尺寸也变小**，于是暴露成崩溃；**普通卷积尺寸不变，于是静默算错**。

## 1. 两个 write_core 期望的布局（v7）

| 位置 | 声明 | 推出布局 |
|---|---|---|
| `etnaviv_ml_nn_v7.c:226` `write_core_interleaved`：`weights_maps[kernel] = input + oc*W*H*in`，随后 `weights_maps[kernel][i]` 按 `block` 走 in ⇒ 块内**最内层是 in** | — | **`[out][in][W][H]`** |
| `etnaviv_ml_nn_v7.c:278` `write_core_sequential`：`(*weights_map)[in][W][H]`，`:315` 用 `weights_map[oc][z][x][y]` | — | **`[out][in][W][H]`** |

⇒ **两个 core 都要求 `[out][in][W][H]`**，而这正是 `transpose()` 的输出。

## 2. 权重在到达 `requantize_weights` 之前经历了什么

以我们的 depthwise 算子（`in_c=out_c=244`, `W=H=3`）为例：

| 步骤 | 源码 | 布局 | 字节数 |
|---|---|---|---|
| 原始 TFLite filter | — | `[1][h][w][out_c]` | 2,196 |
| `expand_depthwise`（`nn.c:612`，v7 因 `nn_core_version<8` 且 `oc>1` 触发） | `nn.c:327-355` | `[out][w][h][in_c]`，真实权重落在 `ic==c` | `244*3*3*244 = 535,824` |
| `transpose`（`nn.c:621-622`，因 `input_channels>1`） | `nn.c:377-411` | `[out][in][w][h]` | 535,824 |
| `requantize_weights` | `nn.c:230-…` | **读作 `[out][w][h][1]`** | `244*3*3*1 = 2,196` |

★ 第 3 行是 `nn.c:239-240`：`if (depthwise) input_channels = 1;`
—— 这个「1」**只改了读侧的解释，没改缓冲区**。

## 3. depthwise：崩溃可修，但**数据是错的**

`requantize_weights` 的读偏移（`input_channels` 已被置 1）：

    offset = oc*W*H + w*H + h            (W=H=3 ⇒ oc*9 + w*3 + h)

而缓冲区实际是 `[out][in][w][h]`，同一偏移落在：

    block = offset/9 = oc  ⇒  (out = oc/244 = 0, in = oc%244 = oc)
    ⇒ 读到的全部是 **output_channel 0** 那一组里的第 `oc` 个 in 通道

⇒ 产物 2,196 字节里装的是 **`(out=0, in=oc)` 的 244 个切片**，
而 depthwise 应当是 `(out=c, in=c)`。

**一行修法**（`if (poperation->conv.depthwise) operation->input_channels = 1;`）
的效果：

* ✅ 尺寸/stride 对上了（`write_core_sequential` 基址 `oc*W*H`，v7:391）⇒ **崩溃消失**
* ❌ **权重内容仍然是错的** ⇒ 输出静默错误

⇒ 这是本项目最该防的形态：**推理不报错，结果错**。

## 4. ★ 更要紧的发现：**普通 per-channel 卷积今天就已经静默算错**

非 depthwise 时 `input_channels` 不被置 1，`new_size` 不变（535,824）：

* `requantize_weights` 读作 `[out][w][h][in]`，偏移 `(oc*W + w)*H*in + h*in + ic`
* 缓冲区实际是 `[out][in][w][h]`，偏移 `(oc*in + ic)*W*H + w*H + h`
* ⇒ 两者**不一致**，但**尺寸相同** ⇒ 不崩溃、不越界、**直接算错**

⇒ 上机实测的 54 个接管节点里，**37 个非 depthwise CONV 现在就在用错权重**。
⇒ 即使把 depthwise 修好，④ 逐字节一致**也不可能通过**。

★ 为什么上游没发现：`tensor_quantization_supported()` 一直拒绝 per-axis，
**这条路径上游从未被执行过** —— 它是「门禁打开前就没人走过」的死代码。

## 5. 正确的修法（仍在一个函数 + 一行写回里）

`requantize_weights` 应当**按 `transpose()` 的实际输出去读**，
输出保持同一布局（depthwise 例外，折成 `in=1`）：

    unsigned in_dim = poperation->conv.depthwise ? 1 : operation->input_channels;
    uint8_t (*map_in)[operation->input_channels][W][H] = input;  // 真实 in 维
    uint8_t (*map_out)[in_dim][W][H]                             = output;
    for (oc, w, h, ic)
        map_out[oc][depthwise ? 0 : ic][w][h] =
            requant(map_in[oc][depthwise ? oc : ic][w][h]);

    new_size = output_channels * W * H * in_dim;

再保留那行写回（depthwise 时 `input_channels = 1`，v7/v8 都做）。

影响面：

| 情形 | 是否进 `requantize_weights` | 结果 |
|---|---|---|
| per-channel depthwise | 是 | 紧凑 + in=1 ⇒ sequential，**stride 与数据都对** |
| per-channel 普通卷积 | 是 | 布局保持 `[out][in][w][h]` ⇒ interleaved，**数据对**（现在错） |
| per-tensor depthwise | **否**（无 `scales`） | 不变 |
| per-tensor 普通卷积 | 否 | 不变 |

## 6. 建议的下一步：**先诊断，后修**

静态分析到此为止；建议先出一个**只加打印**的诊断补丁（och 方案 3），
一次构建就能把下面的量固定下来，验证本文第 2 节的表：

在 `etna_ml_lower_convolution` 的三个点各打一行
`depthwise / input_channels / output_channels / W / H / weight_tensor_size`
+ **权重前 16 字节**（便于离线比对排列）：

1. `expand/reorder` 之后
2. `transpose` 之后
3. `requantize_weights` 之后

外加在 `create_coeffs_v7` 的选路处打印最终选中的 `write_core_*`。

**预期（若本文成立）**：depthwise 依次为 `535,824 → 535,824 → 2,196`，
普通卷积为 `535,824 → 535,824 → 535,824`（但内容排列不同）。
任何一条不符，本文即被证伪。

★ 只加打印、不改控制流 ⇒ 符合本项目纪律（诊断轮产物标 NOT DELIVERABLE）。

## 7. 与精度损失的关系（不要混淆）

即使权重布局全对，per-channel 仍然**被折叠成单一 scale**
（`nn.c:638-641` upstream 自己写的 "with a loss of accuracy"）。
⇒ **门 e（cos ≥ 0.98 / top-1 一致）仍未被任何证据支持**，
布局修好只解决「崩溃」和「权重取错」，不解决「折叠掉精度」。
