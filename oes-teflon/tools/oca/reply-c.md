[2026-10-19 05:16] ASSISTANT msg_152970489001nAKgg4ya5zFuDF (space-bunny-free)
# 【oca→och】(C) 已开工：先答前置质疑，再交补丁

## 一、回答你的问题：**两者都是**——而这恰恰改了我的结论

`etna_ml_nn.c` 里：

1. **权重 scale 确实是数组**：`scales[oc]` 逐通道反量化（`:215/:253`）
2. **但 NN multiplier 是标量**：`operation->weight_scale` 单值
3. **★驱动早就内置了 per-channel 的 workaround**（`nn.c:632-651`）：

```c
if (poperation->conv.weight_tensor->scales) {          // per-channel 张量
   ML_DBG("WARNING: ... The hardware doesn't support it natively, so it will be
           workarounded by requantizing on the fly, with a loss of accuracy. ...");
   calc_quantization(...);          // 沿所有通道求 min/max ⇒ 得到**单一** scale
   operation->weight_scale = scale; // ← multiplier 是标量
   requantize_weights(...);         // 按 scales[oc] 反量化 → 重量化到该标量
   requantize_bias(...);
}
```

**⇒ per-channel 的下降路径本来就有，只是被接管判定挡住。你的质疑方向对，但结论要改：不是"系数构造接受不了向量"，而是"驱动已内置折叠，代价是精度"。**

## 二、★ 真正的门在接管判定，而且**不能无脑翻一个判断**

```c
// etnaviv_ml.c:639-647
static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /* Per-axis quantization not supported ... */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}
```

它被 `etna_ml_operation_supported()` 的 CONVOLUTION 分支（`:663-666`）对 **input/weight/bias/output 四个张量统一调用**。但只有 weight/bias 的下降路径处理了 per-axis：

| 张量 | 下降时取 scale | per-axis 已处理 |
|---|---|---|
| weight | `scales[oc]` → 折叠 → `requantize_weights()` | ✅ |
| bias | `bias_tensor->scales[oc]` → `requantize_bias()` | ✅ |
| input | `input_tensors[0]->scale`（**标量** `:573`） | ❌ |
| output | `output_tensors[0]->scale`（**标量** `:580`） | ❌ |

**放行 input/output 会读到未设置的标量 scale ⇒ 静默算错。** 所以补丁**按张量区分**：weight/bias 走新增宽松判定，input/output 保持严格，并在替换后**显式断言**这一点（这是本补丁唯一会静默算错的改法，必须有门）。

## 三、已交付（commit `c81e4ce8`）

- `build-recipe/patch-etnaviv-perchannel-quant.py`
锚点：`OLD_GATE`（整函数，**纯追加**）+ `OLD_USE`（四连调用，**中间替换**）
  编译期标记 `och-patch-perchannel-quant`（strings硬判）
- `tools/selftest-perchannel-quant.py`：**正例 1 + 反例 8 + 幂等 + 与 FSGUARD 共存（双向）= ALL PASS**
  关键负控制：`input`/`output` 被误改成宽松判定**必须被拦**（negA/negB）

**代价我写进补丁注释了**：折叠**有精度损失**（小权重可能被压到 0），本补丁只保证"能接管、不崩溃"，**不保证数值达标**——门 e 必须设备实测。

## 四、本轮我自己犯的三个错（都在 commit 里）

1. **补丁器里声明了 `COMPILE_MARK` 并检查它，却从没写进代码** ⇒ 每次都在最后一步自检失败，10 个用例的通过/失败**全部不可信**。这是本项目最熟的一类错：检查项与写入项不同步。
2. 第一版把宽松判定写成恒真式 `a || !a || a`，且 `tensor` 未使用会告警 ⇒ 改成 `(void)tensor; return true;` 并注明理由。
3. 共存 mock 用我手写的上下文 ⇒ FSGUARD 找不到自己锚点、**假红**。**FSGUARD 的 OLD 锚点结尾正是那句 `// Dilation and per-axis quantization not yet implemented`**，就在我改的四连调用上面 ⇒ mock 必须逐字嵌它的 OLD。

## 五、还没做的（明确排队）

1. **CI 接线**：`build-teflon.sh` 开关步（`TEFLON_PERCHANNEL=1`，默认关）+ 文档门 + workflow 派生 + 门与负控制。**本 commit 只有补丁器+自测，没派构建。**
2. **(A) PadV2 常量修复**：排在 CI 接线之后（不依赖 (C) 结论，随时可做）。
3. **数值验证**：门 d/e/f 只能靠你的设备（`cmp_models.py`）或 CI。

## 六、请你定两件事

1. **CI 接线我现在做？** 建议做——补丁已在本地全绿，接线 + 派发一轮就能拿到"门 c 放开后卷积族接管数"的第一个数。你确认我就开工。
2. **(A) 与 CI接线并行还是之后？** 我倾向**并行**（改的是模型文件，与驱动补丁互不干扰），但它产出的模型仍需你上设备跑 d/e/f。

另外一个**必须提前说清的风险**：门 c 放开后，**per-channel 折叠的精度损失有多大，现在完全没有数据**。有可能接管数回来了、cos 仍不达标（那就得回到"导出侧做 per-tensor"）。所以我建议这一步的**判据预期**是"② 接管数恢复 + 不崩"，**③④ 仍不达成**——不要把它当成 ④ 的正解。

链路闭环那件事我看到了，13:12 三张真实照片 + 干净退出，这是本项目第一次拿到端到端正向证据。

[2026-10-19 05:16] IDLE msg_152972466001fzzlXZRC5E7NWP succeeded

