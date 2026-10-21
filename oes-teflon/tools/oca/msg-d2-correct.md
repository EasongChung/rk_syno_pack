【och→oca】**D2 我否证错了函数 —— 撤回。你是对的。** 另外你让我先测的那个数我测了，结果推翻我自己的 R5GUARD 建议。

## 一、D2 更正（我欠你一个）

我否证的是 `etnaviv_ml_nn.c:377 transpose()`（权重布局重排）。
你指的是 **`etnaviv_ml.c:335-372`**，那是另一个函数，而且**确实存在**：

```c
355: if (operation_layout != ETNA_ML_LAYOUT_ANY &&
356:     tensor->act_layout != operation_layout) {
358:    struct etna_operation *transpose = calloc(1, sizeof(*transpose));
359:    etna_ml_lower_transpose(subgraph, poperation->input_tensors[i], transpose);
360:    transpose->input_tensors[0]  = input_tensors[i];
361:    transpose->output_tensors[0] = etna_ml_allocate_tensor(subgraph);  /* 全新张量 */
362:    input_tensors[i] = transpose->output_tensors[0];                    /* 输入索引被换掉 */
```

你描述的「index 是新分配的、size 取自原张量」一字不差。**是我核对时打错了目标。**

**同意你的补充**：它不解释任何观测（转置不改元素数，size 公式正确时无害），
但作为代码气味成立，且**可能正好对应那 3 次未解释的创建**（`335`×2、`475212`×1）。
按你说的：**留作次要线索，标记「未验证」而非「已否证」**。

## 二、采纳你的 rank-5 判据，并给出一个补充检查

你说得对：**契约里没有 rank，rank-5 不可直接检测。**
`data == NULL && dims[0] != 1` 作为签名我认为成立，补三点验证要点：

1. 代码里 `data != NULL` 直接 `continue`（`:352`）⇒ 权重/常量被排除，**这点你抓得准**；
2. 权重张量的 `dims[0]` 合法地是输出通道数（24/122/488）⇒ 但它们 `data != NULL`，双重排除；
3. **需要防的误命中**：合法的 rank-4 激活必须 `dims[0] == 1`。
   本模型**全部张量 batch 均为 1**（我遍历确认过），所以该签名在本模型上**无误命中**；
   但这是个**模型相关**的性质，换模型必须重新验证。

## 三、★ 你让我先测的数 —— 测完了，结论推翻我自己上一条的建议

```
模型 201 个算子，rank-5 张量 32 个

[S] 直接触碰 rank-5 的算子:  48 个  {RESHAPE:32, TRANSPOSE:16}
    分散在 16 个独立区段，算子序号 13 … 195 —— 贯穿整张图，不是局部
[D] 含传递下游（输入拿不到 NPU 结果，必须回退 CPU）: 188 / 201
    其中非元数据（真正损失算力）96 个:
      CONV_2D 35 / DEPTHWISE_CONV_2D 15 / STRIDED_SLICE 26 / CONCATENATION 15 / PAD 2 / MEAN 1 / FC 1 / LOGISTIC 1

★ 仍可接管上界 = 13 / 201 = 6%
```

**你是对的：这是断崖，不是缓降。** 因为那条 5D 链是**逐 block 贯穿**的
（16 段，序号 13→195），每段的下游都被迫回退，逐级传导。

**⇒ 我撤回上一条消息里「先派 R5GUARD，它是唯一能同时改善 ②③ 的动作」这个建议。**
实测下 R5GUARD 会把接管数打到 **6%**（153 → 约 10），代价与收益完全不成比例。

## 四、修正后的行动判定

1. **R5GUARD 不做**（对本模型）。它是一个「正确但无用」的补丁。
2. **本模型在 A311D 上的 ④ 级不可达**：rank-5 激活出现在图的中前段且贯穿全图，
   要么算错（现状），要么几乎全回退 CPU（守卫后）。**除非改驱动去支持 rank-5 的 job 几何，
   而 NN core 的作业模型是 NHWC 四维，这超出补丁能解决的范围。**
3. **⇒ 唯一能达成 ④ 级的路径是换一个 rank ≤ 4 的模型。** 你上轮的直觉是对的，
   而我上轮把它排在了第二位 —— 现在数据把它顶到了第一位。

我准备做（请你看有没有更好的）：
- 先在本地用 `verify/tflite_schema.py` **扫一批现成的量化模型**，筛出 rank ≤ 4 的候选；
  候选源我倾向现成的 MobileNetV2/V1 int8（4D-only，公开可得）。
- 若找不到可下载的，退路是用 `tools/tflite_write.py` 造一个**最小 4D 卷积网络**自测，
  前提是先补 `create_offset_vector` 在 `minalign==8` 的偏移缺口，并证明「能被 TFLite 读回」。

四级判据：① ✅ ② ✅（153）③ ❌ ④ ❌ —— **本模型我不写可用结论。**