# 【och→oca】per-channel 上机结果：② 达成，①③ 崩在 depthwise 权重布局（根因已锁，修法已给出，等你判）

## 一、先说结论（四组对照，只换一个变量）

| 组 | 模型 | delegate | EXIT | supported | 卷积族 |
|---|---|---|---|---|---|
| A | per-channel | `9598375d` 旧 | 0 | **99** | CONV 39 un / DWCONV 17 un = **0 接管** |
| B | per-channel | **`14e099a7` 新** | **139** | **153** | CONV 37 sup + DWCONV 17 sup = **54 接管** |
| C | per-tensor | **`14e099a7` 新** | **0** | **153** | 54 接管，**编译全过** |
| D | per-tensor | `9598375d` 旧 | 139（2 graph） | — | — |

**A 组与报告 02 的基线逐项吻合**（supported 99 = TRANSPOSE 47 / RESHAPE 32 / CONCAT 16 / PAD 3 / LOG 1；
unsupported 103 = CONV 39 / STRIDED_SLICE 26 / DWCONV 17 / TRANSPOSE 16 / MEAN 1 / MAXPOOL 1 / unknown 1）
⇒ **测量方法已用已知结果自证**。

**B 的 153 − 99 = 54 = 37 CONV + 17 DWCONV**，剩下 2 个 CONV 正是 FSGUARD 该挡的（`39 − 37 = 2`），
与你 BUILD-INFO 里「ops #58/#153 改走 CPU，BY DESIGN」一致 ⇒ **FSGUARD 生效，未误伤**。

**⇒ 四级判据 ② 在 per-channel 路线上实测达成。**

## 二、根因：`requantize_weights` 压成 `input_channels=1` 布局，v7 不回写该字段

崩溃点（用未 strip 产物反汇编，不依赖源码版本）：

```
nm: 0000000000257800 t write_core_interleaved.isra.0
  257abc: 8b1a0021  add   x1, x1, x26
  257ac0: 387c6821  ldrb  w1, [x1, x28]     ← 越界读
  257ac4: 97fffb2f  bl    256780 <wb_stream_write>
```

`etnaviv_ml_nn.c` `etna_ml_lower_convolution()`：

```c
:607  if (operation->depthwise) {
:608     if (nn_core_version < 8) { :612 expand_depthwise(...); }   // 展开 [out][w][h][in]
      }
:632  if (weight_tensor->scales) {                                  // 只有 per-channel 进
:646     requantize_weights(...);                                    // ★ 内部把 depthwise 的 in 当 1
:649     if (nn_core_version >= 8 && depthwise)
:650        operation->input_channels = 1;                          // ★ v7 漏了
      }
```

`requantize_weights`：

```c
:237  unsigned input_channels = operation->input_channels;
:239  if (poperation->conv.depthwise) input_channels = 1;   // ★ 内部按 1 算
:242  new_size = output_channels * W * H * input_channels;  // ★ 生成紧凑布局
```

**⇒ 权重被压成 `[out][w][h][1]`，但 `operation->input_channels` 仍是 244。**

### 数字严丝合缝（日志原文）

| 观测 | 值 | 解释 |
|---|---|---|
| per-tensor `in_c=244 w=3 h=3 wt_size=535824` | 535,824 | = 244×244×9，`expand_depthwise` 展开布局，**与 in_c=244 自洽** |
| per-channel `in_c=244 w=3 h=3 wt_size=2196` | 2,196 | = 244×1×9，`requantize_weights` 紧凑布局，**与 in_c=244 矛盾** |
| `write_core_interleaved` 需要 | 535,824 | 实际 2,196 ⇒ **越界 243 倍** |

**同一个算子、同一个 `in_c=244`，两个模型的权重张量差了 244 倍，而索引 stride 共用** ——
这就是「per-tensor 能活、per-channel 必崩」的全部原因。

## 三、修法（一行，有上游同仓库依据）

```c
-      if (nn_core_version >= 8 && poperation->conv.depthwise)
+      if (poperation->conv.depthwise)
          operation->input_channels = 1;
```

**为什么正确（布局退化论证）**：`requantize_weights` / `expand_depthwise` 输出布局是 `[out][w][h][in]`，
而 `write_core_interleaved` 假设 `[out][in][w][h]`。**当 `in == 1` 时两种维度顺序退化等价**，歧义消失；
且 `create_coeffs_v7` 的选路 `else if (input_channels > 1)` 变假 ⇒ 走 `write_core_sequential`，
它的索引是 `weights_map[x][y]`（stride = `out_channel×W×H`），**对紧凑布局正确**，
而且它本来就带 depthwise 特判（`:395-397` `if ((depthwise || input_width > 64) && weight_height > 3) stride = 3;`）
—— **它本来就是为 depthwise 写的**。

**影响面逐条核过**：

| 情形 | 走 requantize | 改后 | 结论 |
|---|---|---|---|
| per-channel depthwise | 是 | in=1 ⇒ sequential | **修复** |
| per-channel 普通卷积 | 是 | in 不变 | 不变 |
| per-tensor depthwise | **否**（无 scales） | 仍 244 + 展开布局 | 不变 |
| per-tensor 普通卷积 | 否 | 不变 | 不变 |

## 四、★ 你需要判的三件事

1. **这个修法是否成立？** 我只有静态论证 + 数字吻合，没有跑过。你若看出漏洞，现在说比构建后说便宜。
2. **构建开关**：`TEFLON_PERCHANNEL=1` 是否需要与一个新开关（比如 `TEFLON_DWDEPTH=1`）分离？
   我倾向**直接改在 PERCHANNEL 补丁内部**（因为只有 per-channel 会进 requantize），
   这样不引入新开关、不改变产物语义之外的组合空间。
3. **另外给个诊断补丁的备选**：若你不信静态论证，可先只加打印
   （在 `create_coeffs_v7` 选路处打印 `depthwise / input_channels / weight_tensor_size / 选中的 write_core_*`），
   一次构建就能证伪。**这条只加打印、不改控制流，符合纪律。**

## 五、附带发现：per-tensor 的 NPU 真的跑起来了（但没拿到结果）

```
INVOKE 输入填充: bytes=468075 pattern=(i*31+7)&0xFF
teflon: invoked graph, took 23 ms
teflon: invoked graph, took 11 ms
Assertion `offset + length <= buffer->width0' failed.
  libteflon.so(etna_ml_subgraph_read_outputs+0x38c)
```

CPU 基准（`--no-delegate`）：

```
INVOKE n=1 min_ms=301.341 median_ms=301.341 out_bytes=335
out_hash=0x707496a009ff9690 out_min=-128 out_max=-127 out_mean=-127.9970
```

- **CPU `out_hash` 与报告 08/09 逐位一致** ⇒ 参照物稳定。
- **NPU 首次真正跑完 invoke（23 ms / 11 ms vs CPU 301 ms）** —— 这是本项目第一次。
- 第 3 轮崩在 `read_outputs+0x38c`，**与报告 09 的 ROFIX assert 同址同偏移**，
  是**已知的独立问题**（tensor `size=610000` vs resource `5000`），PERCHANNEL 没碰该路径。
- **没拿到 NPU 的 `out_hash`** ⇒ **③ 仍未达成，④ 仍未达成。**

## 六、状态与纪律

| 级 | per-channel | per-tensor |
|---|---|---|
| ① 编译不崩 | ❌ SIGSEGV | ✅ |
| ② 接管数 | ✅ 0→54 | ✅ 54 |
| ③ 推理执行 | ❌ | ❌（跑 2 轮后崩） |
| ④ 逐字节 | ❌ | ❌ |

**「② 达成」是本轮唯一新增的事实。①③④ 不得由它推断。**

**未派 CI、未改设备部署、未动在跑的 `npu_server`（PID 2660）。** 等你回修法判定，我再回来问用户授权。

报告：`verify/reports/17-perchannel-depthwise-rootcause.md`
