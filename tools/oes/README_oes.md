# tools/oes — OES(A311D) Photos concept INT8 管线

输入：`models/model_float16.tflite`（x86 Photos SPK 解出，12094724 B，TFL3，
输入 FLOAT32 NHWC `[1,395,395,3]`，输出 `[1,335]`；labels/thresholds 各 335 行）。

管线（见 `convert_to_int8.py`，CI 见 `.github/workflows/convert-concept-oes.yml`）：

```
.tflite --tf2onnx(opset13)--> ONNX --onnxruntime smoke--> --onnx2tf--> SavedModel
  --TFLiteConverter PTQ (representative_dataset, COCO语义等价的真实照片集)-->
  concept_int8.tflite（全整型 uint8 I/O）+ concept_int8_floatio.tflite（对照）
  --regress.py--> report.json（门限 top1>=0.95/top5>=0.99/cos>=0.98）
```

## 为什么这里不照搬 OEC（RKNN 线）

OEC 为 `rknn-toolkit2 rk3566 target` 在 ONNX 图首插了
`Transpose(NCHW[1,3,395,395] -> NHWC)`（否则 rknn 把 395 当通道数）。
TFLite/Teflon 原生布局就是 NHWC：强行转 NCHW 会多一个 delegate 大概率不支持的
Transpose 节点、破坏 PTQ per-channel 标定、增加 float/quant 边界。
实测本图已有 48 个原生 TRANSPOSE + 32 RESHAPE + 26 STRIDED_SLICE
（Inception 多分支），**禁止第 49 个**。脚本内以断言强制执行：
输入必须为 NHWC `[1,395,395,3]`，且不存在 `nchw_to_nhwc` 节点。

另：`tflite2onnx 0.4.1` 在此类多分支图上生成错图，禁用；统一走
`tf2onnx==1.16.1`（OEC 验证版本）。

版本钉死：`tensorflow-cpu==2.15.1` + `tf2onnx==1.16.1` + `onnx==1.16.1` +
`onnxruntime==1.16.3` + `protobuf==4.25.4`（与 OEC `convert-concept.yml` 同组合，
ubuntu-latest + Python 3.10 已验证可装）。

## labels / thresholds 的出处与已知限制（交付说明）

### 出处：厂商原件，未被我们改过

`labels.txt` / `thresholds.txt` / `model_float16.tflite` 与厂商包
`inbox/concept/` 下的同名文件**逐值完全一致**（剥 CRLF 差异后），
模型 **sha256 完全相同**（`1c4af48a97868d0767457ab66779c2ba`，12094724 B）。
预处理也逐字相同（`(v-127.5)*0.00784313725` + BILINEAR 395，
厂商注释写明来自 `npu_model_conf.json` 的 `concept.rgb_scale`）。

⇒ **不要重新标定 thresholds**：重标会偏离厂商标定语义，且解决不了根因
（模型在部分图上本来就只有 0.07~0.78 的分数）。

### ★ 已知限制：4 个类别的阈值不可达 —— 厂商原件即如此，非本路径引入

本模型输出端最后一个算子是 **`LOGISTIC`（sigmoid）**，全图**无 SOFTMAX**
⇒ 输出严格 ∈ (0,1)，**多标签**（每类独立阈值）。

阈值 335 条中，只有 4 条 > 1.0，且**恰好都是 2.00**：

| idx | 标签 | 阈值 |
|---|---|---|
| 60 | `child` | 2.00 |
| 178 | `monkey` | 2.00 |
| 208 | `person` | 2.00 |
| 280 | `swine` | 2.00 |

对 sigmoid 输出，阈值 2.00 **永远不可能达到** ⇒ 这 4 个类别在任何路径上
都**标不出来**。这是**厂商原件即如此**，不是本路径引入的既有行为。

**处理方式：照搬，不擅自修改厂商标定。** 修改会让「与厂商行为一致」这个
性质消失，是一个需要产品理由的决定。若业务上需要标注「人」，
需产品侧明确接受这 4 类不可达，或由厂商修正其阈值文件。

其余 331 条阈值 ∈ [0.50, 0.99]（中位数 0.85，148 条 ≥ 0.90），
与模型在**域内**样本上的输出匹配：真机实测微信截图 → `screenshot` **1.000**
≥ 阈值 0.97（通过），篮球比赛照片 → `basketball_game` 0.779 ≥ 0.50（通过）。

### 分数低不等于模型坏

域外/弱相关图上最高分只有 0.068~0.337，属模型的正常表现（argmax 仍语义正确）。
判定模型是否可用，应看**门 e（cos ≥ 0.98 且 top-1 与 float 参照一致）**，
而不是看某一张图的绝对分数。

## PadV2：源模型里根本没有，改回 PAD 才是数值等价的

源模型（`model_float16.tflite`）里**没有任何 PadV2**，只有 **8 个 PAD**，
且**共用同一个** `int32[4,2] = [0,0,1,1,1,1,0,0]` 的 **pads** 张量：

    op#114 PAD [1,395,395,3] -> [1,397,397,3]      op#164 PAD [1,50,50,244] -> [1,52,52,244]
    op#116 PAD [1,198,198,24]-> [1,200,200,24]     op#168 PAD [1,50,50,244] -> [1,52,52,244]
    op#118 PAD [1,99,99,24]  -> [1,101,101,24]     op#254 PAD [1,25,25,488] -> [1,27,27,488]
    op#121 PAD [1,99,99,122] -> [1,101,101,122]    op#258 PAD [1,25,25,488] -> [1,27,27,488]

注意 `[0,0,1,1,1,1,0,0]` 是 **pads**（每维前后补多少），
**不是** `constant_values`（每个位置填什么值）。`PAD` 恒定**零填充**，
所以源模型填充值是 **0**。

⇒ int8 模型里的 PadV2 是 `tflite→onnx→tf→tflite` 往返造出来的，
其 `constant_values` 是 rank-0 的 FLOAT32 `FLT_MAX` 垃圾常量。
**正确修法是改回 PAD（删掉 constant_values）**，即 `tflite_fix_padv2.py`。

⚠️ 若把 `constant_values` 设成 `[[0,0],[1,1],[1,1],[0,0]]`，
h/w 两侧填充值会从 0 变成 1 ⇒ **与源模型不等价**，top-1/cos 会漂。

`padv2_fill_gate.py` 是改写的前置安全门（三态）：
`ZERO` 放行 / `DEGENERATE` 告警放行 / **`NONZERO` 与 `UNKNOWN` 拒绝改写**。
没有它，一次「填充值非 0 被静默换成 0」的改写会推理不报错、结果却错。

