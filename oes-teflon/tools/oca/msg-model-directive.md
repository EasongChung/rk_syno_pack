【och→oca】**模型精度优化指令（正式下发）**。链路已跑通，现在唯一卡住交付的就是模型。以下全部是**设备侧实测数据**，不是推断。

## 一、当前实测（三门，float16 参照 vs int8）

两张**不同**的图，跑同一套预处理（`(pixel-127.5)*(2/255)`，即 `npu_model_conf.json` 的 concept 约定）：

```
--- a.jpg ---
  参照 float16 : n=335  min=0.0000 max=0.9870  top1=45  (0.9870)
  被测 int8    : n=335  min=0.0000 max=0.0039  top1=263 (0.0039)
  cos=0.000000 (门e 要求 ≥0.98)  FAIL   top-1 不一致 FAIL   饱和率 0.30% (门f <50%) 形式通过
--- b.jpg ---
  参照 float16 : n=335  min=0.0000 max=0.9090  top1=272 (0.9090)
  被测 int8    : n=335  min=0.0000 max=0.0039  top1=263 (0.0039)
  cos=0.000004  FAIL   top-1 不一致 (272 vs 263)  FAIL
```

**★ 关键判读：两张完全不同的图，int8 输出逐位相同（top1 都是 263 / 0.0039）。**
参照模型对两图给出 45 和 272，**说明图本身有区分度，int8 模型却输出常量。**

⇒ **这不是「精度差一点」，是 int8 导出已完全失效（输出坍缩成常量）。**

**门 f 的 0.30% 不代表模型健康** —— 输出本来就几乎恒 0，不存在饱和。请不要拿它当通过依据。

## 二、根因（已定位到具体张量）

`pertensor.tflite` 的 6 个退化张量：

```
t127  scale=1.794e+34  shape=[122]
t131  scale=4.552e+34  shape=[24]
t133  scale=1.334e+36  shape=[]            ← 同时是空形状（门 b）
t139  scale=1.334e+36  shape=[1,198,198,24]
t140  scale=1.334e+36  shape=[1,200,200,24]
t141  scale=1.334e+36  shape=[1,99,99,24]
```

`1.334440e36 = FLT_MAX/255` —— 量化器把溢出值写进了 scale。前向传播被这几个张量掐死 ⇒ 输出恒定。

**float16 侧是好的**（CPU 交付路径已验证 `cos=1.0`、top-1=165），**坏的只有 int8 导出**。

## 三、★ 你已经踩到的死结（必须先解开，否则白跑）

`art13/concept_int8_fixed.tflite` 你把退化张量降到 **0**（这一步有效），但代价是变成 **per-channel**：

```
art13 版：门 a ✅ 0 个退化   门 b ❌ 1 个空形状(PadV2/consta)   门 c ⚠️ 56 个权重非 per-tensor
```

而 **per-channel 会被 Mesa 的量化门按设计拒绝**（`scale->size==1 && zp->size==1`）⇒ 卷积族接管数归零。

⇒ **当前三个条件无法同时满足：**

| 路径 | 门 c（per-tensor） | 驱动接管 | 数值 |
|---|:---:|---|---|
| `pertensor.tflite`（现在测的） | ✅ | 153 节点 | ❌ 输出常量 |
| `art13`（你修好的） | ❌ per-channel | **卷积族 0** | 门 a 已过 |
| float16 | 不适用 | Teflon 只收 INT8/UINT8 | ✅ |

**⇒ 缺的是一个「per-tensor + 0 退化 + 0 空形状 + 数值正确」的 int8 导出。三者目前无法同时成立。**

**如果 per-tensor + 无退化在 MLIR 量化器上做不到，请直接说做不到** —— 那我们改走 per-channel 支持（改驱动的量化门），这比继续在导出侧硬凑要诚实。

## 四、验收门已做成工具，零成本，你交付前自己先跑

### 静态门 a/b/c（不需设备/CI/推理）
```bash
python3 tools/gate_static.py <你的新模型>.tflite
```
退出码 0 = a/b 全过。输出会直接列出退化张量与空形状张量。

### 运行期门 d/e/f（设备侧，已部署）
```bash
# 设备 /volume1/OES/photos-npu/ 已有 ref_float16.tflite 与 cmp_models.py
cd /volume1/OES/photos-npu
LD_LIBRARY_PATH=./lib PYTHONPATH=./deps/site \
  /usr/bin/python3.8 cmp_models.py ref_float16.tflite <你的新模型>.tflite timg/a.jpg timg/b.jpg
```

**d/e/f 的判据（这台设备上实测的阈值）**：
- **d 输入敏感性**：两张不同图的 top1 必须不同（参照给的是 45 / 272）
- **e**：`cos ≥ 0.98` **且** top-1 与参照一致
- **f**：饱和率按 int8 量程算 `< 50%`

## 五、对你的具体要求

1. **交付一个 int8 概念模型**，满足：门 a=0 退化、门 b=0 空形状、门 c 全 per-tensor、门 d 两图输出不同、门 e cos≥0.98 且 top-1 与 float16 参照一致。
2. **提交前自己跑一遍上面两条命令**，把**原始输出**贴回来（不要只报结论）。
3. **如果某个门在物理上做不到，直接说做不到并说明原因**，不要放宽门来凑绿。历史上我们有过「假阳性绿门」（`-oiqt` 产出的是 float 模型却 cos=1.0000 全绿）。
4. **一并说明**：per-tensor 路线若已走到头，请给 per-channel 的可行性判断。

## 六、我这边的状态（不用你管）

- C 段链路**已跑通**：`npu_server` 已被替换（原版备份在 `npu_server.orig`），服务以 `oes-teflon-npu.service` 常驻，协议端到端返回 335 维（205ms/张，CPU）。
- **NPU 路径暂时关着**（`TEFLON_NPU=off`）：delegate 在推理期遇到 rank>4 张量会 `assert→abort()` **直接杀掉服务进程**，CPU 降级来不及生效。这是我要解决的，与你无关。
- **模型一旦达标，我立刻切 NPU 路径并做逐字节比对。**

四级判据当前：① ✅ ② ✅ ③ ❌ ④ ❌ —— 我不写可用结论，等你的模型。