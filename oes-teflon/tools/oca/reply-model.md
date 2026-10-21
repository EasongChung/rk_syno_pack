[2026-10-19 05:11] ASSISTANT msg_152921c6e001cmLSGCylDrrSIr (space-bunny-free)
工具已入库（`f362ef5b`），可直接零成本跑：

```bash
python3 oes-teflon/tools/gate_static_tflite.py <model.tflite> [--names PadV2,MaxPool2d,...]
```

---

# 【oca→och】退化 scale 的根因找到了：**在 float 源里，不在量化器**

## 一、先纠正一处因果方向`1.334441e36 = FLT_MAX/255` 不是"实测出的溢出范围"。我读 `QuantizationParameters` 的 `min/max` 数组：**这 6 个张量的 min/max 全是空的** ⇒ 它们**从未被校准观察到**，TF MinMax 走了兜底分支。这条标准行为**我们自己的代码里早就写着**（`tools/oes/input_sensitivity.py:16-18`、`transplant_scales.py:6,14-15`），只是没人在模型上验过。

## 二、★ 真正的坏点（一个，float 源里就有）

```
baseline_float_..._float32.tflite
  t133  model_32/tf.compat.v1.pad_1/PadV2/constant_values   FLOAT32  shape=[]   max|v| = 3.402823e+38
```

**`PadV2/constant_values`（paddings 常量）被压成了 rank-0、值 = FLT_MAX 的标量。**
输入 `[1,198,198,24]` → 输出 `[1,200,200,24]` 说明这是 SAME padding（各边 1），paddings 本应是 `[[0,0],[1,1],[1,1],[0,0]]`。

⇒ **一个坏点同时造成你两个症状**：
- 它**就是门 b 那个空形状**（rank-0）；
- 它让 TF MinMax 对该区域兜底 ⇒ **6 个张量连锁退化成 1.334441e36** ⇒ 前向被掐死 ⇒ 输出恒定 0.0039。

**旁证**：动态范围量化的 `concept_dyn` 里**根本没有这个张量** ⇒ 门 a 干净。所以这不是量化参数的问题，是**上游图里那个常量本身是垃圾**。

## 三、三变体对照（按**名字**对齐，索引会漂移）

| 变体 | 门 a | 那 6 处的 scale | 门 c |
|---|---|---|---|
| `concept_int8` | ❌ 6 个退化 | 全兜底（`zp0=127`，min/max 空） | ❌ |
| `concept_int8_fixed` | ✅ | PadV2/MaxPool2d = `4.73e-01 zp=-128 min=0 max=120.6`（合理） | ❌ |
| `concept_int8_fixed2` | ✅ |全部合理（`2.24e-3` / `9.41e-3` / `4.73e-1`） | ❌ per-channel |

⇒ **你已经把它修好了（`fixed`/`fixed2` 门 a 都过）。所以门 a 不是当前的拦路虎。**

## 四、当前真正的三个问题（按可解性排序）

1. **门 c（per-channel）是独立的第二问题**，不是 pad 引起：权重 per-channel ⇒ **偏置也跟着 per-channel**（`pseudo_qconst106` 是 `[122]` 的 INT32 偏置却带 122 个 scale）。Mesa 按设计只收 per-tensor ⇒ 卷积族 0接管。
2. **门 d/e（数值）我无法本地验证**：本机**没有 tensorflow / tflite_runtime**（Windows ARM64 无轮子），`onnxruntime` 跑不了 tflite。**所以你的要求 2（提交前自己跑出原始输出）我做不到**——我不会交一个"声称达标"的未验证模型，那正是你警告过的假绿（`-oiqt` 那次）。
3. float16 基线里还有个别常量**恰好饱和在 65504**（float16 上限），来源未查，可能同源。

## 五、回答你的问题 4（per-channel 可行性）：**可行，且我建议就走这条**

per-channel 需要改的只有 Mesa 那个量化门（`scale->size==1 && zp->size==1`），是**一处条件判断**，而我们已经在驱动侧改了四处。相比在导出侧硬凑 per-tensor（要把每个权重反量化再按单一 scale 重量化，还要同步偏置 scale，风险高、且无法本地验证），**改驱动门更可控**。

我的建议顺序：
```
① 门 c 走 per-channel + 改驱动量化门   ← 一处代码，比导出侧折叠风险低得多
② 数值门 d/e 在 CI 或设备上实测        ← 我这边只能静态验，数值必须跑
③ 门 b 的空形状：pad 常量已修（fixed/fixed2 都过），确认是否还有残留
```
**在 ① 落地前，任何"per-tensor + 数值正确"的模型都交付不了**——因为驱动不收 per-channel。

## 六、我能接着做的（等你点头）

- **(A)** 写一个 flatbuffer 补丁器：把 `PadV2/constant_values` 就地修成 `INT32 [4,2] = [[0,0],[1,1],[1,1],[0,0]]`（形状可从 in/out 推导），并把 `tflite_fix_padv2.py` 的逻辑收敛成可复跑的门。**静态门我能自己验，数值门必须你跑。**
- **(B)** 写 per-channel → per-tensor 的折叠器（纯 numpy，本地可做）——但**我无法验证它对不对**，只交付工具不交付"达标模型"。
- **(C)** 给驱动加 per-channel 量化门（改 `etnaviv_ml*.c` 的判定 + 一条 `TEFLON_QPERCHANNEL` 开关），带门与负控制。**这条最可能直接解开死结。**

## 七、状态

- 本轮**零 CI、零上机**；仓库新增1 个工具（`gate_static_tflite.py`），未改任何既有脚本或产物。commit `f362ef5b`。
- **我没有交付模型**，理由见第四节第 2 点：本机跑不了推理，交付未验证模型＝假绿。
- 四级判据不变：① ✅ ② ✅ ③ ❌ ④ ❌。

请挑一个方向（我建议 **C**，因为它同时解开"门 c"和"接管数归零"两个死结），我就开工。

[2026-10-19 05:11] IDLE msg_152925425001b4pcsC1KcQMkLO succeeded

