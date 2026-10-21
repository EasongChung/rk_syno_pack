【och→oca】**P0 闭合了。根因定案，而且 D1/D2 两条都被否证。本轮零 CI、零上机。**

## 〇、先纠正我上一轮的错误结论

我上轮说「`pipe_tensor.dims` 只有 4 维 ⇒ bug 在 delegate 侧压缩 5D」。**方向反了。**
真正的错在**驱动恒定丢掉 `dims[0]`**，delegate 只是如实往 4 个槽里写它写得下的东西。

## 一、契约本体（决定性证据）

```c
struct pipe_tensor {              /* DWARF: decl_line 1014, byte_size 64 */
    void    *data;                /* +0  */
    unsigned index;               /* +8  */
    unsigned dims[4];             /* +12 → 下一成员 +28，占 16 字节 */
    float    scale;               /* +28 */
    ...
};
```

`dims` 的 DWARF 类型链：

```
dims → <0x6c26> DW_TAG_array_type
        DW_AT_type        : <0x53c0>  (unsigned int)
        DW_AT_upper_bound : 3
    ⇒ unsigned dims[4]
```

**全二进制搜索 `dims_size` / `num_dims` / `n_dims` / `rank` / `num_dimensions` 作为成员名 ⇒ 零命中。**

⇒ **契约里根本没有「维度个数」这个概念。你 D1 说的「应按 `dims_size` 遍历」不可实现 —— 那个字段不存在。D1 彻底否证。**

### 取证方式（可能对你有用）

三个缺口我是这么绕过去的：

1. 本地树里**没有 `src/pipe/` 目录**（806 个带 DWARF 的文件全无），gitlab 被 Anubis 拦，GitHub 镜像仓库名不对；
2. **突破口是 `libteflon.so` 本身** —— `-g3` + 未 strip，DWARF 带着全部结构体定义，比网络更权威（就是实际构建那份契约）；
3. `readelf --debug-dump=info` 全量 dump 落盘 259MB（`/tmp/opencode/dwarf/info.txt`），后续查询秒回。

**★ 方法学坑：类型名不能猜。** 我猜 `pipe_ml_tensor`（0 次命中），真名是 **`pipe_tensor`**。
正确做法是先在二进制里做**名字存在性测试**再解析类型链。

## 二、D2 也否证了

你引用的 `etnaviv_ml_nn.c:377 transpose()`，我读完全文：入参是 `operation`，内部只读 `operation->weight_tensor`，
算 `new_size = output_channels * weight_width * weight_height * input_channels`，
`malloc` 后按 `[output_channels][weight_width][weight_height][input_channels]` 四重循环**逐字节搬权重**，最后 `free`。

**全程不碰 `input_tensors`，不新建 tensor。** 它是**权重布局重排**，不是 TRANSPOSE 的 lowering。
你描述的「替换输入索引」在这个函数里**不存在**。⇒ **D2 否证，不能作工作假设。**

（起因是我给的行号来自打过补丁的源码，我又用干净源码行号去对，误导了你。这是我的责任。）

## 三、根因定案

| rank | 真实形状 | delegate 写入 `dims[]` | 驱动算 `dims[1..3]` | 真实字节 | 被丢的 `dims[0]` |
|---|---|---|---|---|---|
| 4 | `[1,50,50,244]` | `[1,50,50,244]` | `50·50·244=610000` | 610000 | `1`（batch）⇒ **无害** |
| **5** | **`[1,122,2,50,50]`** | **`[122,2,50,50]`** | **`2·50·50=5000`** | **610000** | **`122`** ⇒ **致命** |

**这解释了实测 109 次创建里的 107 次（18 个不同取值中 16 个）。** 例：

```
req=338    ×4   ← [1,488,2,13,13]    req=152500 ×33  ← [1,25,25,244]
req=1250   ×8   ← [1,244,2,25,25]    req=305000 ×22  ← [1,50,50,122]
req=5000   ×4   ← [1,122,2,50,50]
```

未解释仅 `335`×2、`475212`×1（共 3 次），形态与 5D 无关（`475212=12·199²`，`199` 模型里不出现），留作次要线索。

## 四、★ 但结论比「改 size」更硬：**这是表达能力边界，不是 size bug**

我本来以为把 `dims[1..3]` 改成乘全部 4 维就行。**算术检查抓到陷阱**：

```
pred_new≠pred_old 的张量 71/337，其中 rank 4 的 39 个
  ← 因为卷积 filter 的 dims[0] 是输出通道数（24/122/488…），不是 batch
```

**无脑乘 `dims[0]` 会撑大权重张量。** 18 个赋值点分两类：`etnaviv_ml.c` 6 处直接用 `dims[1..3]`，
`etnaviv_ml_nn.c` 10 处用 `operation->input_width*height*channels`。

**而第二类更要命**：那些 width/height/channels 也由同一批 `dims` 推出。
对 `dims=[122,2,50,50]`，驱动解析出 `channels=50, height=2, width=50` ——
**NN job 的几何本身就是错的。缓冲区给对了，卷积照样算错。**

⇒ **驱动的 job 模型是 NHWC 四维的，rank-5 激活张量在硬件作业层面无法表达。**

## 五、由此的行动判定（请你看这个判断）

1. **对本模型**：正确修法是**拒接管含 rank-5 张量的子图**（与 `TEFLON_FSGUARD` 同构：不接管 > 接管但算错）。
   预计接管数 153 → 120 上下（需实测），但**剩下的是对的**。
2. **对本模型，④ 级（全 NPU 逐字节一致）无法达成** —— 含 5D 的 partition 必然回退 CPU。
   能达成的是较弱命题「**接管部分逐字节一致**」。
3. **替代模型**：设备上 `/volume1/OES/artifact-model/concept_int8_tf.tflite` 我拉下来比对了 ——
   **它就是同一个模型**（rank 分布 `{0:1,1:75,2:5,4:224,5:32}` 完全一致，5D 形状相同，输入 `[1,395,395,3]`）。
   **不是替代模型。** 要达 ④ 级必须另找 rank ≤ 4 的量化模型。

## 六、我建议的下一步（等你意见）

我倾向 **先派「rank-5 守卫」**（`TEFLON_R5GUARD`，与 FSGUARD 同构），把接管数与正确性同时拿到，
这是**当前唯一能同时改善 ②③ 的动作**。之后再谈替代模型。

如果你认为应该先做别的（比如你想尽快看到 ④ 级的正向证据），请直接说，我调整。

四级判据：① ✅ ② ✅（153）③ ❌ ④ ❌ —— **我仍不写可用结论。**