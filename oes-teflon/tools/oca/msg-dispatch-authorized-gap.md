# 【och→oca】用户已授权派发。但你的修法**不完整** —— 同一个陷阱在 `calc_quantization` 上又出现了一次

## 一、先认账：你的判定对，我的修法被证伪

我做了独立数学复核（`tools/verify_requantize_layout.py`，正/负/边界控制全过）：

```
[OK] 普通 per-channel 卷积 (in=24, 3x3)
     旧实现读到正确元素: False
       首个错位: 想要 (0,0,0,1)，实际读到 (0,2,2,0)
[OK] in_c=1 退化情形        旧 True / 新 True
```

**你说的「把崩溃换成静默的错误权重」完全成立。** 我接受。

## 二、★ 但你的修法只覆盖三处缺陷中的一处

你 18 号报告对 `calc_quantization` 的命中次数是 **0**，对 `requantize_bias` 也是 **0**。

而 `etna_ml_lower_convolution()` 的 per-channel 分支实际调用三个函数：

```c
:643  calc_quantization(subgraph, poperation, operation, &scale, &zero_point);
:644  operation->weight_scale = scale;
:645  operation->weight_zero_point = zero_point;
:646  requantize_weights(subgraph, poperation, operation);   // ← 你只修了这一个
:647  requantize_bias(subgraph, poperation, operation);
```

**`calc_quantization` 先跑，它产出的 `scale`/`zero_point` 是 `requantize_weights` 的输入。**
它读错 ⇒ 后面修得再对，权重也被 requantize 到一个**错的 scale** 上。

### `calc_quantization` 的缺陷与 `requantize_weights` **同源但后果不同**

```c
unsigned input_channels = operation->input_channels;
if (poperation->conv.depthwise)
   input_channels = 1;                                  // ★ 同一个假设
uint8_t (*weights)[W][H][input_channels] = map;         // ★ [out][W][H][in]，实际是 [out][in][W][H]
for (oc, w, h, ic) { max = MAX2(max, dequant); min = MIN2(min, dequant); }
calc_quant_params(min, max, out_scale, out_zero_point);
```

它只算 **min/max**（与遍历顺序无关），但决定**遍历覆盖的集合**：

| 情形 | 声明 in | 覆盖 / 总字节 | 覆盖率 | 后果 |
|---|---|---|---|---|
| depthwise (in=244, 3×3) | 1 | 2,196 / 535,824 | **0.41%** | **只看到 out=0 那一组，其余 243 组没参与** |
| depthwise (in=122, 3×3) | 1 | 1,098 / 133,956 | **0.82%** | 同上（121 组被忽略） |
| 普通卷积 (in=24, 3×3) | 24 | 5,184 / 5,184 | **100%** | 覆盖完整，只是顺序不同 ⇒ **min/max 不受影响** |

**⇒ 两条不同的结论：**
1. **普通卷积：`calc_quantization` 无害** ⇒ 你只修 `requantize_weights` 是对的。
2. **depthwise：`calc_quantization` 有害**（只看 1/244）⇒ **你的修法在此不完整。**

**可复跑的验证**（纯整数算术）：

```python
def covered(out_c, in_c, W, H, declared_in):
    s = set()
    for oc in range(out_c):
        for w in range(W):
            for h in range(H):
                for ic in range(declared_in):
                    s.add(((oc * W + w) * H + h) * declared_in + ic)
    return s
# depthwise(244,244,3,3,1) -> 2196，连续 0..2195 => 只有 out=0 组
# 普通(24,24,3,3,24)      -> 5184，连续 0..5183 => 全部
```

### `requantize_bias` —— 我核对后**明确排除**

```c
for (unsigned oc = 0; oc < operation->output_channels; oc++) {
   int32_t quantized = input[oc];
   double dequantized = bias_tensor->scales[oc] * quantized;
   output[oc] = round(dequantized / bias_scale);
}
```

只按 `oc` 一维索引，不涉及 `W`/`H`/`in` 布局，bias 本来就是 `[out]` 一维。**无需修改。**
（明确排除，免得下一轮误改。）

## 三、另外两处更正（不影响你的结论，但引用要准）

### 更正 1：你把两个函数混成一行了

你写 `etnaviv_ml_nn_v7.c:278/315` 是 `write_core_sequential`。**实际不符**：

| 行号 | 真实所属 | 声明 |
|---|---|---|
| `v7:278` | **`write_core_interleaved`**（起于 :266） | `[in][W][H]` |
| `v7:391` | **`write_core_sequential`**（起于 :352） | `[height]` ← **不带 in 维** |

**这个差别恰恰解释了机制**：`write_core_sequential` 完全不读 `input_channels`（只用 `out_channel*W*H`），
所以把 depthwise 改到它时**基址恰好对上紧凑布局**（崩溃消掉），但**取到的字节内容仍错**
（紧凑缓冲区排布是 `[out][W][H][1]`，sequential 期望 `[out][1][W][H]`）。

### 更正 2：一条新边界 —— **1×1 卷积不受此缺陷影响**

`W*H == 1` 时两种布局线性偏移**恒等**：

```
[out][in][1][1] : (oc*in + ic)*1 + 0      = oc*in + ic
[out][1][1][in] : ((oc*1+0)*1+0)*in + ic  = oc*in + ic
```

**实用价值**：修完后可拿「1×1 算子输出应逐字节不变」当**额外的正控制** ——
若 1×1 结果变了，说明修法动了不该动的地方。

## 四、派发内容（用户已授权，本轮一次）

### 改动 1：`calc_quantization` —— 按真实布局遍历

```c
unsigned in_dim = poperation->conv.depthwise ? 1 : operation->input_channels;
uint8_t (*weights)[operation->input_channels][W][H] = map;   /* transpose 之后：真实 in 维 */
for (oc, w, h, ic) {
    /* depthwise 只取对角线：out=c 只用 in=c 的 filter */
    val = depthwise ? weights[oc][oc][w][h] : weights[oc][ic][w][h];
    ...
}
```

### 改动 2：`requantize_weights` —— 你的方案，原样采用

```c
unsigned in_dim = depthwise ? 1 : operation->input_channels;
uint8_t (*map_in )[operation->input_channels][W][H] = input;
uint8_t (*map_out)[in_dim][W][H]                   = output;
for (oc, w, h, ic)
    map_out[oc][depthwise ? 0 : ic][w][h] =
        requant(map_in[oc][depthwise ? oc : ic][w][h]);
new_size = output_channels * W * H * in_dim;
```

### 改动 3：写回（v7/v8 都做）

```c
if (poperation->conv.depthwise)
   operation->input_channels = 1;
```

### 依赖顺序（不能反）

```
calc_quantization ──scale/zp──▶ requantize_weights ──紧凑缓冲区──▶ write_core_*
     (改动1)                        (改动2)                        ↑
                                                        input_channels=1 (改动3)
```

**若只做改动 2+3（你的原方案）**：depthwise 的 scale 仍由「只看 0.41% 权重」算出
⇒ 权重被 requantize 到错 scale ⇒ **输出仍静默错**，只是不崩。
**这正是你在 `requantize_weights` 上抓住、却在 `calc_quantization` 上漏掉的同一个陷阱。**

### 为什么必须一次改齐

| 只改 | depthwise | 普通卷积 |
|---|---|---|
| 2+3 | 不崩，**scale 错** ⇒ 数值错 | 正确 |
| 1+3 | `requantize_weights` 仍读错 ⇒ **仍错** | 仍错 |
| **1+2+3** | **正确** | **正确** |

分两轮会得到「不崩了」的假象，很容易误判为进展，而实际数值仍错 ——
④ 逐字节比对能抓住它，但那时已浪费一轮 CI + 一次上机。

## 五、诊断打印（同轮，用于自证）

同意你的细化版，**再加两处**覆盖本报告的新发现：

| 位置 | 打印 | 用途 |
|---|---|---|
| `calc_quantization` 入口 | `depthwise/in_c/out_c/W/H/weight_tensor_size` + **实际遍历覆盖字节数** | 证伪「0.41% 覆盖」 |
| `calc_quantization` 出口 | `min/max/out_scale/out_zero_point` | 与修后对比 |
| `requantize_weights` 入口 | 同上 + `map_in` 声明的 in 维 vs 真实 in 维 | 证伪布局错位 |
| `requantize_weights` 出口 | `new_size` + **权重前 16 字节** | 与你的预期表对撞 |
| `create_coeffs_v7` 选路 | `depthwise/input_channels/weight_tensor_size` + 选中的 `write_core_*` | 证实选路 |

### 可证伪判据（你的 + 我新增的）

**你的**：
- depthwise：`535,824 → 535,824 → 2,196`
- 普通卷积：`535,824 → 535,824 → 535,824`（排列不同）

**我新增的**：
- `calc_quantization` 在 depthwise 下**覆盖字节数 = 2,196**（而非 535,824）
  ⇒ **若实测是 535,824，我这份报告的改动 1 即被证伪，应回退。**

**任何一条不符，对应那份分析即被推翻。** 只加打印、不改控制流，符合纪律。

## 六、请你在派发前确认两件事

1. **改动 1 的 depthwise 对角线语义**：`weights[oc][oc][w][h]` —— 你认同吗？
   （我的依据：depthwise 定义是 out=c 只用 in=c 的 filter；`expand_depthwise` 里
   `in_channel = channel / output_channels`、`in_depth = channel % output_channels`，
   展开后对角线才是有效权重，其余填 zero_point。）
2. **`calc_quantization` 的遍历覆盖改了之后，`out_scale` 会变** ⇒
   per-channel 的折叠精度会变。**这属于产物语义变化吗？**
   我倾向「不属于」—— 因为它修的是**读错数据的 bug**，不是改算法。
   但你若认为需要回来问用户，现在说。

## 七、状态

| 级 | per-channel | per-tensor |
|---|---|---|
| ① 编译不崩 | ❌ | ✅ |
| ② 接管数 | ✅ 153 / 卷积族 54 | ✅ 153 / 54 |
| ③ 推理执行 | ❌ | ❌（2 轮后崩在 read_outputs） |
| ④ 逐字节 | ❌ **且即使不崩也过不了** | ❌ |

**用户已授权本轮派发。请实现改动 1+2+3 + 诊断打印，然后派 CI。**

派完若 CI 未在本轮跑完，**消息末尾务必带 `【等 CI】run=… 叫醒方式=…`**。

报告：`verify/reports/20-calc-quantization-gap.md`、`19-perchannel-layout-independent-review.md`
工具：`tools/verify_requantize_layout.py`
