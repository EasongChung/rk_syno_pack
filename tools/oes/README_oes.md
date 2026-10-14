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
