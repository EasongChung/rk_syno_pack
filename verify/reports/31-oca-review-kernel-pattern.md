# 31. 独立复核：「NPU 挂死 = kernel pattern 0x6 档」根因假设

> 复核方：oca（驱动/Mesa 侧）　被复核方：och
> 输入：`verify/reports/11-contract-rootcause-2026-10-19.md`（主输入）、
> `22-layout-fix-device-verdict.md`、`24-gate-review-verdict.md`
> 本轮**未碰设备**（约束 §一.1），全部结论来自**本地 Mesa 源码逐行**（`文件:行号`）。
> ⚠️ **本地没有驱动源码**（`D:\clawtemp\oes-dsm-kernel` 为空）⇒ 涉及驱动侧的
> 结论一律标注「无法复核」，不猜。
>
> **本报告写完后，och 已并行提交了 `f731eecb`（NNDUMP 描述符逐字 dump）、
> `0648c0dc`（kernel_pattern 运行期强制）及其 workflow。**
> ⇒ §7 已按此修订：**本轮不再新增工具**（避免两个补丁打同一锚点），
> §6 的实验设计建议改为「在 och 已有工具上怎么用」。

## 0. 结论速览

| och 的结论 | 判定 | 一句话理由 |
|---|---|---|
| (a) 1×1 conv 的权重由 `write_core_6` 写出 | ⚠️ **需修正** | 条件是 `pointwise && output_channels > 8`（v7:492/541），**不是「所有 1×1」**；且**「0x6 档 ↔ write_core_6」在源码里不存在对应关系** |
| (b) pattern 分档表 | ⚠️ **需修正** | 表**逐字正确**（nn.c:1197-1217 已核），但**只在 PARTIAL cache 分支生效**（nn.c:1194）——走 FULL 时 pattern 是 0x0/0x0/0x0（nn.c:1190-1192） |
| (c) 6 个通道数的相关性表 | ⚠️ **需修正** | 数字对，**归因错**：判别量不是通道数，是**片上 SRAM 占用**（nn.c:1188 判据） |
| (d) 「不配对 ⇒ 算错值但不挂死」 | ❌ **站不住** | pattern 是 **70 bit 位图**（msb 6bit @nn.c:91 + low 32 @:96 + high 32 @:99），**不是布局标签**；该表**不依赖任何计算量** ⇒ 错 pattern 的后果是「硬件按位图去 SRAM 找不存在的 kernel」，**不是算错值** |
| (e) 已排除的项 | ⚠️ **无法复核 + 补一条** | 缺设备日志；另补：`kernels_per_core` **描述符值与系数流值可能不一致**（nn.c:1168 vs v7:193） |

**最重要的一条**：H 的**前提**（244 这次作业声明 0x6/0x7e/0x0）**尚未被证实**。
按源码复算，在**默认 buffer depth** 下 244 这次 1×1 conv **仍然走 FULL cache**
（total 453,952 < 524,288）⇒ pattern = **0x0/0x0/0x0** ⇒ **H 当场不成立**。
只有当驱动的 `nn_input_buffer_depth`/`nn_accum_buffer_depth` 较大时 244 才走 PARTIAL。
⇒ **必须先在真机上把实际数值打出来**（och 的 NNDUMP 已经能做，见 §7）。

---

## 1. 「我方结论逐条判定」

### 1.1 (a) 权重写出分派 —— ⚠️ 需修正

och 引的代码**逐字属实**（`etnaviv_ml_nn_v7.c:492-497` 试算、`542-546` 正式）：

```c
if (operation->pointwise && output_channels > 8)
   actual_size = write_core_6(subgraph, map, core, operation, zrl_bits);
else if (input_channels > 1)
   actual_size = write_core_interleaved(...);
else
   actual_size = write_core_sequential(...);
```

★ **但推论要改**：
1. 条件是 `pointwise && **output_channels > 8**`，**不是「所有 1×1 conv」**；
   `out_c ≤ 8` 的 1×1 走 interleaved/sequential。
2. **「pattern 0x6 档 ↔ write_core_6 的布局」这个对应关系在源码里不存在。**
   pattern 桶**只由 `out_image_z_size` 决定**（nn.c:1197-1217），
   与用哪个 write_core_* 写系数**毫无关系**。
   ⇒ 直接反例：**1×1、488 通道**同样走 `write_core_6`（v7:492），
   却拿到 **0x3e** 桶（nn.c:1205-1208）。
   ⇒ 两个「6」是**数字巧合**。

### 1.2 (b) 描述符的 pattern 分档 —— ⚠️ 需修正（**漏了一个前提**）

och 引的表**逐字属实**（我核对了 nn.c:1197-1217 的每一行）：

```c
if      (out_image_z_size >= 1024) { msb=0x13; low=0x80000;    high=0x0; }
else if (out_image_z_size >=  512) { msb=0x3d; low=0x0;         high=0x2aaaaaa0; }
else if (out_image_z_size >=  256) { msb=0x3e; low=0xffffaaaa; high=0x7fffffff; }
else if (out_image_z_size >=  160) { msb=0x6;  low=0x7e;        high=0x0; }
else                                { msb=0x3f; low=0xfffffffe; high=0xffffffff; }
```

★ **但这张表嵌在 `else` 分支里**（nn.c:1194-1217）：

```c
if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size) {
   map->kernel_caching_mode = SRAM_CACHE_MODE_FULL_CACHE;      /* nn.c:1189 */
   map->kernel_pattern_msb = 0x0;                               /* nn.c:1190 */
   map->kernel_pattern_low = 0x0;                               /* nn.c:1191 */
   map->kernel_pattern_high = 0x0;                              /* nn.c:1192 */
   ...
} else {
   /* Doesn't fit in the 512KB we have of on-chip SRAM */        /* nn.c:1195 */
   map->kernel_caching_mode = SRAM_CACHE_MODE_PARTIAL_CACHE;    /* nn.c:1196 */
   ... 上面那张表 ...
}
```

⇒ **只有「系数 + 图像缓存装不进 512KB 片上 SRAM」时才会声明 0x6/0x7e/0x0。**
⇒ 「244 ⇒ 0x6」**不是无条件成立的**。

### 1.3 (c) 相关性表 —— ⚠️ 数字对，归因错

och 的 6 行表（24/122/244/488/976/2048）**逐行属实**（按 nn.c:1197-1217 的桶），
**但归因要改**：

★ **判别量不是通道数，是片上 SRAM 占用**：

| 量 | 来源 | 与通道数的关系 |
|---|---|---|
| `coef_cache_size` | v7:558 = `max_core_size * cores_used` | ∝ **in_c × out_c** |
| `image_cache_size` | nn.c:1175-1181 | ∝ **in_c** × tile 面积 |
| `oc_sram_size` | nn.c:995 取自驱动；注释说 512KB（nn.c:1195） | 固定 |
| 判据 | **nn.c:1188** `0x800 + coef + image < oc_sram_size` | — |

⇒ 244 之所以落进 0x6 档，是因为它是**唯一**让该判据翻面的通道数。

★ **我按源码逐行复算了这条链**（模型见 §4）：
`etna_ml_calculate_tiling_v7`（v7:63-107）→ `calc_interleave_mode`（v7:41-60）
→ `calc_superblocks`（v7:19-39）→ `write_core_6` 输出尺寸（v7:186-263）
→ `coef_cache_size`（v7:558）→ `image_cache_size`（nn.c:1170-1181）→ nn.c:1188 判据。

**结果（1×1 conv，输入 50×50，oc_sram = 512KB = 0x80000）**：

| C | superblocks | coef | image | total | 走哪条 | pattern |
|---|---|---|---|---|---|---|
| 24 | 12 | 832 | 38,400 | 41,280 | **FULL** | 0x0/0x0/0x0 |
| 122 | 61 | 15,872 | 195,200 | 213,120 | **FULL** | 0x0/0x0/0x0 |
| **244** | 122 | 61,504 | 390,400 | **453,952** | **FULL** | **0x0/0x0/0x0** ⚠️ |
| 300 | 150 | 92,416 | 480,000 | 574,464 | PARTIAL | 0x3e |
| 488 | 244 | 242,112 | 780,800 | 1,024,960 | PARTIAL | 0x3e |
| 976 | 488 | 960,448 | 1,561,600 | 2,524,096 | PARTIAL | 0x3d |
| 2048 | 1024 | 4,210,752 | 3,276,800 | 7,489,600 | PARTIAL | 0x13 |

⇒ **在默认 buffer depth（in=32/accum=64）下，244 走 FULL、pattern 是 0x0/0x0/0x0。**
⇒ **H 的前提不成立。**
⇒ **但结论对驱动给的 `nn_input_buffer_depth`/`nn_accum_buffer_depth` 敏感**：
取 64/128 时 244 变成 PARTIAL ⇒ 0x6（见 §4.1 敏感性表）。

★ **Mesa 侧本来就会打印**实际数值：`nn.c:1183` 的
`ML_DBG("coefficients_size 0x%x (%d) image_size 0x%x (%d)\n", ...)`
⇒ **设备日志里应该已经有这一行 ⇒ 先 grep 它，零成本。**

### 1.4 (d) 「不配对 ⇒ 算错值但不挂死」 —— ❌ 站不住

**三条源码事实**：

1. **pattern 是 70 bit 位图，不是标签。**
   `kernel_pattern_msb` **6 bit**（nn.c:91）、`kernel_pattern_low` **32 bit**（nn.c:96）、
   `kernel_pattern_high` **32 bit**（nn.c:99）⇒ 6+32+32 = **70 bit**。
   解码（pattern = msb | low<<6 | high<<38）：

   | 桶 | 置位数 | 直觉解读 |
   |---|---|---|
   | 0x3f（<160） | **70/70 全置** | 「全缓存」 |
   | 0x6（[160,256)） | **8** | 「只缓存 8 个槽」 |
   | 0x3e（[256,512)） | 多数 | 手调 |
   | 0x3d（[512,1024)） | 少数 | 手调 |
   | 0x13（≥1024） | 3 | 手调 |

2. **该表不依赖任何计算量**：不看 `kernels_per_core`、不看 `superblocks`、
   不看系数实际大小、不看 tile。**只由 `out_image_z_size` 决定。**

3. ⇒ **它对 244 通道（244 个 kernel）声明「只有 8 个 kernel 槽在 SRAM 里」**，
   而描述符里**没有描述「其余 kernel 从 DDR 流式取」的字段**
   （nn.c:55-130 的全部字段里没有这类项）⇒ 硬件很可能去 SRAM 找**不存在的 kernel**
   ⇒ **卡住**。这与「前端 DMA 指针不再前进」**一致**。

⇒ **och 的 (d) 推理链缺一环**：「不配对 ⇒ 算错值」隐含了「pattern 描述的是布局」，
而源码里**没有任何字段或注释说 pattern 描述布局**。
⇒ 正确表述应是：**「pattern 与实际 kernel 流不配对 ⇒ 硬件按位图取不到 kernel ⇒ 挂死」**，
与 och 的结论方向一致，但**机制完全不同** ⇒ **不能用「122/488/976/2048 不挂」
论证「不配对只算错值」**。

★ **而且 (c) 表里 488 档 0x3e、976 档 0x3d、2048 档 0x13 都不挂**，
按上表它们的置位数分别是「多数/少数/3」⇒ **位图稀疏程度与是否挂死不成正比**
⇒ **「0x6 特殊」这个相关性仍然没有机制解释**。这正是需要实测的地方。

### 1.5 (e) 已排除的项 —— ⚠️ 无法复核 + **补一条**

> ## ⚠️ 勘误（第二轮，2026-10-20）：**本节的「补一条」是错的，已撤回。**
>
> 本节曾断言「描述符的 `kernels_per_core`（nn.c:1168）与系数流里的
> `kernels_per_core`（v7:193）可能不一致 ⇒ 描述符与系数流脱节」。
> **第二轮用 och 的 NNDUMP 实测数据证伪了它**：
>
> | | 描述符侧 | 系数流侧 |
> |---|---|---|
> | 公式 | `nn.c:1168` `DRoU(DRoU(out_c, ncc), sb)` | `v7:220` `kernels_in_superblock = DRoU(kpc_stream, sb)`，其中 `kpc_stream = DRoU(out_c, cores_used)`（v7:193）、`cores_used = MIN2(out_c, ncc)`（v7:190） |
> | 化简 | `DRoU(DRoU(out_c, ncc), sb)` | `DRoU(DRoU(out_c, MIN2(out_c,ncc)), sb)` |
>
> 当 `out_c >= ncc`（本模型全部 6 档都满足）⇒ `MIN2(out_c,ncc) == ncc`
> ⇒ **两个数是同一个公式、必然相等**。
> 实测：244 两者都是 5、122 都是 4、488/976/2048 都是 5 ✔
>
> ⇒ **描述符与系数流是自洽的**；`kpc_desc × sb` 只是**槽位容量**，
> 系数流按 `d,d,…,rem` 填进去（244：5+5+5+5+5+5+1 = 31）⇒ **不越读**。
>
> ⇒ **本节下面的「补一条」整体作废。**（保留原文是为了让「错在哪、怎么被抓住的」可追溯。）
> ⇒ 同时也说明：**「kpc 不一致 ⇒ 硬件取错系数」这条线是死的**，
> 详见 `verify/reports/32-reconciliation-h-refuted.md`（och 侧）。

缺设备日志，och 列的排除项我**无法逐条复核**（不下结论）。
★ **补一条 och 漏掉的**（**已撤回，见上方勘误**）：

**描述符的 `kernels_per_core` 与系数流里的 `kernels_per_core` 可以不一致。**

| 位置 | 公式 |
|---|---|
| 描述符字段 | `nn.c:1168`：`DIV_ROUND_UP(DIV_ROUND_UP(output_channels, nn_core_count), superblocks)` |
| 系数流 | `v7:193`：`DIV_ROUND_UP(output_channels, cores_used)`，`cores_used = MIN2(out_c, nn_core_count)` |

以 244 通道、`nn_core_count=8`、`superblocks=122` 为例：
描述符说 `DIV_ROUND_UP(31,122) = 1`，而系数流按 **31** 个 kernel/core 写。
⇒ **描述符与系数流对「有几个 kernel」的说法不一致**。
⇒ 这与 pattern 桶是**同一类病**：「描述符字段与实际系数流脱节」。

### 1.6 och 的 §2.2 判定（rank-5 是表达能力边界）—— ✅ 站得住

`struct pipe_tensor` 无维度数字段 ⇒ 驱动一律 `dims[1]*dims[2]*dims[3]`
（nn.c:1132-1133 佐证 `out_image_x_stride = out_image_x_size`）⇒ 恒丢 `dims[0]`。
⇒ rank-5 激活在硬件作业层面确实无法表达。
⇒ **同意**，且与 pattern 假设**互不冲突**（不同问题、不同位置）。

---

## 2. 问题 A：`write_core_6` 写出什么布局？对应哪个 pattern？

### 2.1 逐字读 `write_core_6`（v7:186-263）

* **头**：`append_bits(zrl_bits, 8)` + `append_bits(kernels_per_core, 16)`（v7:215-216）
  ⇒ 8 bit 的 zrl 位宽 + **16 bit 的每 core kernel 数**。
* **三层循环**：superblock（外）→ kernel（中）→ block（内，每 block `stride = MIN2(in_c, 6)` 个输入通道）
  （v7:218-254）。
  * `block == 0`（v7:233-242）：先写 `weights_maps[kernel][0]`，
    再写 **32 bit bias correction**（v7:236-238），再写 `[1 .. stride-1]`（v7:240-242）。
  * `block > 0`（v7:243-248）：写 `[block*stride .. +stride)`。
  * **最后一个 block** 追加 **32 bit `out_values_per_channel * out_channel`**（输出偏移，v7:249-252）。
* **权重基址**：`weights_maps[kernel] = input + out_channel * W * H * input_channels`（v7:226）
  ⇒ **权重本体的线性布局是 `[oc][ic][w][h]`**；`stride=6` 只是**写出时的分组**，
  **不改变布局**。

### 2.2 它对应哪个 pattern？

⇒ **没有对应。**
`write_core_interleaved`（v7:278）与 `write_core_sequential`（v7:391）读的布局是
`[out][in][W][H]` / `[out][w][h]`，与 `write_core_6` 的 `[oc][ic][w][h]` + 内插
bias/offset 的码流**都不同**。
而 pattern 桶**只由 `out_image_z_size` 决定**（§1.1）⇒ **无法与写出函数对应**。

★ **一句话回答 A**：**write_core_6 的布局与 kernel pattern 之间没有任何源码层面的
对应关系**；och 假设的地基（「0x6 档就是 write_core_6 的布局」）**不成立**。
⇒ 这不否定 H 的**现象层**关联（244 挂、122 不挂），但**推翻了「机制」层解释**。

---

## 3. 问题 D：pattern 0x6 / 0x7e / 0x0 的语义出处？

**结论：查不到出处 —— 这本身是一条结论。**

| 检索范围 | 结果 |
|---|---|
| Mesa `etnaviv_ml_nn.c` 全部字段定义（nn.c:55-130） | 只有字段名与位宽，**无任何语义注释** |
| Mesa `etnaviv_ml_nn_v7.c` / `_v8.c` | `kernel_pattern` 只在 nn.c 出现；**v8 路径没有这张表** |
| 上游 Linux `drivers/gpu/drm/etnaviv/`（torvalds/linux master） | `etnaviv_hwdb.c` 芯片表里**没有 A311D 条目**，也没有 NN pattern 字段 |
| `oes-dsm-kernel` 驱动 | **本地无源码** ⇒ 未能检索 |

⇒ **0x6/0x7e/0x0 是一个 Mesa 私有的、没有任何公开定义的魔数。**
★ **旁证**：v8 路径**没有**这张表 ⇒ 若它是硬件固定约定，v8 也该有 ⇒
**更像 v7 路径上的历史遗留**。

★ **证据缺口**：拿不到上游 Mesa git 历史（mesa gitlab 被 Anubis 挡；
`Mesa3D/mesa` 镜像 404）⇒ **无法给出「从哪个版本开始」的行号级证据**。

---

## 4. 问题 C：这组分档是历史设计还是某次改错？

**能给的证据（源码内）**：
1. 该表**不依赖任何计算量**（§1.4-2）⇒ 要么是硬件固定约定，要么是**某个更早的、
   参数不同的实现的残留**。
2. v8 路径没有这张表（§3）⇒ 削弱「硬件固定约定」的解释。
3. 0x3f 桶 = 70 位全 1、0x6 桶 = 8 位 —— 这种**手写魔数表**的形态，
   典型是「按某次实测手工调出来的」。

**倾向判断**：**更像历史遗留，而非 Mesa 某次改错** ——
但**这恰恰意味着它可能是对的**（硬件当时就长那样）。
⇒ 两种可能都必须被实验区分（§6）。

**证据缺口**：拿不到上游 git 历史 ⇒ 无法给出「从哪个版本开始」的行号。

### 4.1 复算模型（供复核）

模型逐行对应：v7:19-39 / v7:41-60 / v7:63-107 / v7:186-263 / v7:513-561 /
nn.c:1168 / nn.c:1170-1181 / nn.c:1185 / nn.c:1188 / nn.c:1197-1217。
**敏感性**（1×1，输入 50×50）：

| nn_core_count / in_buf / accum | C=122 | C=244 |
|---|---|---|
| 1 / 32 / 64 | FULL, 0x0 | **FULL, 0x0** |
| 1 / 64 / 128 | FULL, 0x0 | **PARTIAL, 0x6** |
| 2 / 32 / 64 | FULL, 0x0 | FULL, 0x0 |
| 8 / 32 / 64 | FULL, 0x0 | FULL, 0x0 |
| 8 / 64 / 128 | FULL, 0x0 | PARTIAL, 0x6 |

⇒ **结论对驱动常量敏感 ⇒ 必须实测。**

---

## 5. 问题 E：独立枚举 —— 还有什么能让「前端 DMA 指针不再前进」？

★ **我方漏掉的可能根因**（按是否与通道数相关标注）：

### 5.1 与通道数相关

| # | 候选 | 源码依据 | 怎么区分 |
|---|---|---|---|
| E1 | **片上 SRAM 装不下**（真正的越界原因） | nn.c:1188 判据；image_cache ∝ in_c（nn.c:1179）、coef ∝ in_c×out_c（v7:558） | NNDUMP 打印的 `img_cache` / `coef_cache` + raw 字节里的 `kernel_caching_mode` |
| E2 | **kernel pattern 桶** | nn.c:1197-1217，只在 PARTIAL 分支 | `TEFLON_KPAT` 强制做单变量对照 |
| E3 | **描述符 kpc vs 系数流 kpc 不一致** | nn.c:1168 vs v7:193 | NNDUMP 打印的 `kpc`（描述符侧）；系数流侧需另取 |
| E4 | `kernel_z_size = input_channels`（14 bit） | nn.c:1160 | 244 装得下（<16384）⇒ 排除 |
| E5 | `out_image_z_size = output_channels`（14 bit） | nn.c:1130 | 同上 |
| E6 | 描述符 `kernels_per_core` 字段宽度（7 bit） | nn.c:91 区段 | 244/8=31 装得下 ⇒ 排除 |
| E7 | 每 core SRAM 容量 vs 244×244 系数 | nn.c:1185/1193 | 实测 |

### 5.2 与通道数无关

| # | 候选 | 源码依据 |
|---|---|---|
| E8 | 系数 BO 大小/对齐 | v7:526 `bo_size`；v7:548 `align(actual_size, 64)` |
| E9 | zrl 压缩码流（`zrl_bits` 选大 ⇒ 非法码流） | v7:215 `append_bits(zrl_bits, 8)`；v7:462-510 `calculate_zrl_bits` |
| E10 | 作业提交顺序 / 子图依赖 | `etna_ml_subgraph` 的 op 顺序（nn.c:1303 逐 op 建 instruction） |
| E11 | 输入/输出 buffer 的尺寸与 offset | nn.c:1125-1127 |
| E12 | `in/out_image_x_stride`（16 bit） | nn.c:125-126、nn.c:1132-1133 |
| E13 | **hangcheck 判据本身**（见下） | 驱动侧，**本地无源码** |

★ **E13 值得单独强调（我方漏掉的一条）**：
och 给的判据是「DMA 地址变化 ≤ 16 ⇒ 前端还在动 ⇒ 放过」。
⇒ **「指针不动 ⇒ 连作业都没开始推进」这个推论过强**：
若硬件在**两个地址之间反复重取同一个 kernel**（例如 244 通道的系数在 SRAM 里放不下，
每轮都从同一处重取），DMA 指针可能在窗口内**来回**⇒ 被判「还在动」而**放过**；
反之完全不动才判挂。
⇒ **「挂死」这个观测本身可能包含了「判据把真挂放过了」与「把假挂判成挂」两种误差。**
⇒ 需要 `etnaviv_sched_timedout_job` 的完整上下文才能核（**本地无驱动源码，未能核**）。

---

## 6. 问题 F：实验设计复核

### 6.1 och 的设计 —— ✅ 已被 och 自己修好（`0648c0dc`）

och 拟用 `TEFLON_KPAT=6|3`（不设=原逻辑），
两组：`TEFLON_KPAT=6 + 122 通道 conv` 与 `TEFLON_KPAT=3 + 244 通道 conv`。

★ **我原稿指出过一个缺陷**：若 `TEFLON_KPAT=3` 只改 msb、low/high 沿用原表，
得到的 pattern 在原表任何桶里都不存在 ⇒ **不是单变量**。
★ **och 的实现（`0648c0dc` 第 98-103 行）已经避开了这一点**：
`TEFLON_KPAT=3` 强制的是**完整的 0x3f 组**
（`msb=0x3f, low=0xfffffffe, high=0xffffffff`）⇒ **单变量成立**。

⇒ **§6.1 的批评只对「设计草图」成立，对已入库的实现不成立。**

### 6.2 ★ 我建议的三条对照（**全部只用 1×1 conv ⇒ 不会冻死整机**）

★ **前置（最便宜，先做）**：先跑一次**只加诊断、不改行为**的 NNDUMP
（`TEFLON_NNDUMP=1`），把 244 那次作业的 `img_cache` / `coef_cache` /
`kpc` / `raw=` 原始描述符字节全部打出来。
⇒ **如果 raw 字节里 `kernel_caching_mode` 显示是 FULL（nn.c:1189），
  H 当场被证伪**（因为 FULL 分支写死 pattern 0x0/0x0/0x0，nn.c:1190-1192），
  后面两个对照实验都不用做。
（更省：设备日志里应该已经有 nn.c:1183 那一行 `coefficients_size ... image_size ...`
——**先 grep 它**，零成本。）

| # | 对照 | 改动 | 若 H 成立 | 若 H 被证伪 | 安全性 |
|---|---|---|---|---|---|
| **A1** | 同一个 244→244 **1×1** conv，**输入空间尺寸** 50×50 → 10×10 | 只改空间尺寸 | **不挂**（image 390,400→27,392 ⇒ total 90,944 ⇒ FULL ⇒ pattern 0x0） | **仍挂** ⇒ 与 pattern 无关 | ★ 只用 1×1，**不碰 depthwise** ⇒ 不会冻死整机 |
| **A2** | 1×1 conv **160→160** | 只改通道数（仍在 [160,256) 桶内） | **不挂**（按复算 160 走 FULL ⇒ 拿不到 0x6） | **挂** ⇒ H 被证伪（160 根本声明不到 0x6） | 同上 |
| **A3** | 244 通道 1×1，`TEFLON_KPAT=3`（强制 0x3f 整组） | 只改 pattern 三字段 | **不挂** ⇒ 是 0x6 这一档的位图在害 | — | 同上 |

★ **A1 是最干净的单变量**：通道数不变、权重不变、只改空间尺寸，
而它恰好把 `image_cache_size` 压到 SRAM 之下 ⇒ pattern 必然从 0x6 变 0x0。
★ **A2 是最强的判别**：160 与 244 **同在 [160,256) 桶**，
但按源码只有 244 会（可能）拿到 0x6 ⇒ 若两者都挂 ⇒ **桶不是判别量**。

### 6.3 其它实验设计注意点

* och 的 `TEFLON_KPAT=3 + 244 通道 conv` 若该 conv 后面还挂着 244 通道 depthwise，
  **会冻死整机** ⇒ 隔离模型必须**只含那一个 1×1 conv**。
* 强制 pattern ⇒ 描述符与系数流不配对 ⇒ 可能算错值 ⇒ 强制后的产物
  **不可交付**，也不能用来判 ④。
★ **A1/A2 都不需要任何新开关**（只改模型里的空间尺寸/通道数）⇒ 更安全。

---

## 7. 交付物与本轮新增

| 交付 | 状态 |
|---|---|
| `verify/reports/31-oca-review-kernel-pattern.md`（本报告） | ✅ **本轮唯一新增** |
| 复核用的复算模型（tiling→superblocks→coef→image→FULL/PARTIAL→pattern） | ✅ 逻辑已逐行对应源码，见 §4.1；**本轮未入库为工具**（避免与 och 的补丁同锚点冲突） |
| NN 描述符诊断（`img_cache`/`coef_cache`/`kpc`/`raw=`） | **och 已有**：`f731eecb`（`TEFLON_NNDUMP=1`）—— 覆盖 §1.3 与 §5.1 的全部需求 |
| kernel_pattern 运行期强制 | **och 已有**：`0648c0dc`（`TEFLON_KPAT=1` 编译期 + `TEFLON_KPAT=6|3` 运行期） |
| 组合 workflow（NNDUMP + KPAT） | **och 已有**：`534cbfa4`（`build-teflon-fix-rofix-perchannel-kpat.yml`） |

★ **本轮我没有新增任何补丁器** —— 我原本写的 KPATDIAG（打印 cache/pattern 决策数值
+ 运行期强制）与 och 的 `0648c0dc` **打同一个锚点**（nn.c:1197-1217 的分档表），
两个都入库会互相打挂 ⇒ **主动弃用我的版本，保留报告与实验设计建议**。

---

## 8. 明确的证据缺口（如实标注）

1. **无驱动源码**（`D:\clawtemp\oes-dsm-kernel` 为空）⇒ 驱动侧 hangcheck 判据、
   SRAM/buffer depth 等**未能复核**。
2. **无设备日志** ⇒ och 的隔离实验数据**未能复核**（我方表格按其转述引用）。
3. **无上游 Mesa git 历史** ⇒ 「这张表从哪个版本开始存在」**给不出行号级证据**。
4. **未复算 `coef_cache_size` 的精确值**（我用 bit 流量估算，zrl 压缩会偏小）⇒
   数值是**上界**；但判据方向（FULL vs PARTIAL）在 244 上**余量不小**
   （453,952 vs 524,288，差 70,336）⇒ 在默认 buffer depth 下结论稳，
   在 64/128 变体下翻面。**必须实测。**
5. **未验证 0x6 位图的确切硬件语义** ⇒ §1.4 的机制解释是**假设**，
   由 §6.2 的 A1/A2/A3 实验区分。
