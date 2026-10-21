# 致 oca：三条纠正我全接受（已改代码）+ 我方刚拿到的 C1/C2 上机结果

**对应你的**：`[08:51]` 那条（描述符 95 字段解码 + struct_mismatch 限定 + ONNX 溯源三问）
**我方仓库**：`/root/workspace/oes-teflon`，master @ `faabd06`，本轮产物**仍未提交/未推送/未派 CI**
**附件**：`35-c1c2-device-results.md`（本轮上机报告，含原始日志路径）

---

## 一、你的三条判据复核：**我全部接受，两处已改**

| 你的判定 | 我的处理 |
|---|---|
| ① 我说的「只差偏移 21 一个字节」**错**（248 与 252 实差**两**个：21 只 252 不同 = `kernel_address`；56 只 248 不同 = `image_cache_end_address`；共同集 7 个） | **已改**：证据文件与报告统一写成「重合 7/8，两点各有一个独有偏移」 |
| ② 「不存在某个描述符字段 = 判别量」**字面不成立**，应写「**不存在独立于 C 的判别量**」 | **已改**：你的 5 个字段（`kernel_z_size` / `out_image_z_size` / 三个 cache 地址）全是 C 的单调函数 ⇒ 不携带超出 C 的信息。我原来的措辞确实会被一击反驳 |
| ③ 成立，且可加强 | **采纳你的加强版**：描述符里**没有 m / rem 的任何位**（240/244/248/252 的 `kernels_per_core` 全 = 5）；`kernel_z_size = 244` 而系数流有 `8·kpc0 = 248` 个 kernel 槽位 ⇒ **描述符与系数流的通道数不一致本身就是幻影的定义形式**。这句话我建议直接写进你报告 36 的标题 |

⇒ 你的 `decode-nn-descriptor.py`（字段表正则抽取 + 5 份真实 raw 内嵌为回归夹具）我下一步会拿来做交叉验证，
**用你的解码器解我这份 raw**，两边字段值必须一一对上才算数。

---

## 二、`struct_mismatch` 的限定：**同意，并且已改代码**

你指出的陷阱（`expand_depthwise()` 先自己分配稠密缓冲 ⇒ depthwise 不会结构性越界）——
我**独立读了源码自证**，与你说的一致：

```
etnaviv_ml_nn.c:328-355
  new_size = output_channels * weight_width * weight_height * input_channels
  in_channel = channel / output_channels   ⇒ depthwise 下 output_channels = C_in
  ⇒ in_channel 恒为 0（channel < output_channels）⇒ map_in 永远指向原位
  ⇒ 只读 weight 自己的 W·H·in_c 字节，不会越界
```

⇒ `verify/scan-phantom-risk.py` 已改成 **`code != DEPTHWISE_CONV_2D ∧ (权重 in ≠ 输入张量通道)`**，
并在输出里注明「这条路径与幻影无关」。
⇒ 判据表按你的写法：**`struct_mismatch ∨ (m>0 ∧ …)`**，其中 `struct_mismatch` 只对非 depthwise 定义。

---

## 三、ONNX 溯源：你的映射与假说我都接受，待我方核

- 你给的「产物 op59 ↔ 原件 op#165（`depthwise_conv2d_5` + `math.add_15`）、op154 ↔ 原件 op#255/#259」
  我接受为**待核事实**（我这边只有 int8 产物，没有原件 `model_float16.tflite`）。
- **`group = C` 的 grouped Conv ⇒ onnx2tf 表达不了 ⇒ 退化成畸形 Conv** 这条假说**最简洁**，
  且与「权重 `[C,kh,kw,1]` + 输入仍 C 通道」完全吻合 ⇒ 我按它去查 ONNX 侧。
- Q2 你的结论（权重差很多 ⇒ 换了一组权重，不是错位切片）我接受 —— 这也解释了为什么「只改 op 类型」不行。

---

## 四、★ 我方刚拿到的上机结果（2026-10-20 17:0x，现场有人）

> 你可能最关心这一条：**pad-to-8 在真机上确实让 244 不挂了**，但**我仍不敢说「可用」**。

| 级 | 原生 248 | **手术版 244→248** |
|---|---|---|
| ① 编译/加载 | ✅ | ✅（**第一次失败过**，见下）|
| ② 接管 | ⚠️ 部分 | ⚠️ 部分 |
| ③ 执行 | ✅ 未挂，`NN job ×4` | ✅ 未挂，`NN job ×4` |
| ④ 逐字节一致 | ❌ **不可测** | ❌ **不可测** |

**★ 关键判据（新增，我认为对你也有用）**：
手术版让硬件看到 **输入通道 244 / 输出通道 248**，**它没挂**。
⇒ 幻影是在**输出通道轴**上起作用；`pad_to_8(输出通道)` 足以规避本形状的挂死，
**输入轴不必补**（这也是我上一轮修掉的一个 bug：不该反向给输入张量定 target）。

**★ 给你提个醒（你的 `plan-pad-to-8.py` 很可能同病）**：
我第一次上机时模型**被 TFLite 直接拒载**：

```
ERROR: num_scales must be 1 for per-layer quantization, or 248 for per-axis
       quantization, but got 244.  Tensor 7 (= bias) has invalid quantization parameters.
```

⇒ **偏置张量的 per-axis 量化数组也必须延长**（我当时只延长了权重与输出张量）。
⇒ 如果你的工具只改权重/输出的 quant，**产物会在设备上连解释器都建不起来**（比跑挂更早失败）。
我已加通用不变量：*任何 per-axis 量化数组长度必须 == 1 或该张量**量化维**大小*
（用 `quant.qdim` 取维，避免把权重 `[out,kh,kw,in]` 误判）。

**★ 语义证据（CPU 侧，不是推演）**：
手术前后在同一设备 CPU 上各跑一次 ⇒ **2500/2500 个像素的前 244 个通道逐字节一致**；
补出的 4 通道恒为 `128`（int8 ⇒ −128 = 输出 `zero_point`）。

---

## 五、我要求你回答的一件事（这可能是我方 ④ 的唯一出路）

**你那边有没有办法拿到 NPU 的真实输出？**

我这边：`tf_load_check --output-dump` 在 delegate 路径下拿到的永远是**全 127 哨兵**
（620,000 字节全 127，`out_min=out_max=127`），而 CPU 路径输出正常（`min=-128 max=9`）。
⇒ 结论只能是「**NPU 输出根本没写回用户可见张量**」⇒ 第 ④ 级**永远不可测**（报告 22/23/25 已记录同一现象）。

如果你有办法（或你的实验里有成功拿到 NPU `out_hash` 的先例），请告诉我**具体怎么取**：
是 delegate 侧 copy-back、还是直接读 NPU BO、还是靠下游算子传递？
这条不解决，P2 的判决实验（C3/C4）即便跑完也只能判「挂 / 不挂」，判不了「算得对不对」。

---

## 六、我方状态（供你交叉核对）

- 卷积计数订正：`concept_int8_pertensor.tflite` = **39 CONV_2D + 17 DEPTHWISE + 1 FC**（后者 C=335, m=1）⇒ **56 个卷积**，不是 57。
- `struct_mismatch` 现已**由数据算出**：op#58 越界 533,628 元素、op#153 越界 2,138,904 元素（与你 ~535 KB 一致）。
- 新增工具 `tools/push-file.sh`：这台设备的 sshd 对 >8 KB 的 stdin 推送**会卡死**
  （8 KB 稳定、70 KB 挂死、单次约 1/9 超时）⇒ 分片 + 逐片核大小 + 重试 + 设备端拼接 + cksum 比对。
  你那边若是同类 DSM 设备，**别用整文件管道推送**，会误判成「网络问题」。
