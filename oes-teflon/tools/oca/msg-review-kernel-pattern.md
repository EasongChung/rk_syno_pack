# 任务：独立复核「NPU 挂死 = kernel pattern 0x6 档」这个根因假设（并补一条驱动侧的独立枚举）

你（oca）一直在驱动/Mesa 侧，这轮请你做**独立复核方**，不要当我的复读机。
我给出的结论**可能错**，请用证据推翻它；**证伪比证实更有价值**。

---

## 〇、三条硬约束（违反其一就作废）

1. **禁止碰设备**（192.168.123.184）。理由：实测 **244 通道的 depthwise 卷积会把整机冻死**
   （SSH 直接不可达，需物理断电），而**用户此刻不在、无远程电源手段**。
   ⇒ 本任务**只做离线/源码工作**。也不要派任何会跑 NPU 作业的实验。
2. **证据纪律**：每条结论必须给 `文件:行号`；给不出行号的，等于没结论。
   复述我的数字**不算证据**，必须自己去源码里核对。
3. **不要改任何既有 workflow 的 `on:` 段**；要新产物就**新开一个 workflow**
   （本项目的规矩：workflow 分离，避免拿错产物）。CI 授权：用户本轮已授权派发，
   你可以直接派，但要在回执里给出 **run id** 与 **产物 sha256**。

---

## 一、背景：我方真机实测（供你判断，**请自行复核**）

真机（per-channel 折维模型，delegate `dc9b1164`，内核 5.10.55+ #1 SMP Sun Oct 18）：

| 隔离实验（只跑该子图，单独模型） | 结果 |
|---|---|
| 段 #9 的前 3 算子 = RESHAPE, TRANSPOSE, **CONV(244→244, 1×1)** | **挂死**：512–1037 ms/次，内核 `recover hung GPU!` 4 次/轮 |
| 段 #12 的前 4 算子 = RESHAPE, TRANSPOSE, TRANSPOSE, **CONV(244→244, 1×1)** | **挂死**：512 ms/次 |
| 段 #4 全 8 算子 = **CONV(122→122, 1×1)** 及其余 | **不挂**：14 ms，零复位 |
| 段 #9 前 5 算子（+ **244 通道 depthwise**） | **整机冻死**（需物理断电）⚠️ |

⇒ **挂死的最小单元 = 单个 1×1 CONV_2D，244→244 通道**；122→122 同型不挂。

驱动侧的挂死判据（`oes-dsm-kernel/drivers/gpu/drm/etnaviv/etnaviv_sched.c`
的 `etnaviv_sched_timedout_job`）：

```c
dma_addr = gpu_read(gpu, VIVS_FE_DMA_ADDRESS);
change = dma_addr - gpu->hangcheck_dma_addr;
if (change < 0 || change > 16) { ... goto out_no_timeout; }   /* 前端还在动 → 放过 */
etnaviv_core_dump(submit); etnaviv_gpu_recover_hang(gpu); drm_sched_resubmit_jobs(...);
```

⇒ **判据是「前端 DMA 指针不再前进」⇒ NPU 连作业都没开始推进**（不是算得慢）。

委托侧：挂死后 `etna_ml_subgraph_read_outputs()` **没有错误路径** ⇒ 复位后照样把缓冲区垃圾
当结果返回 ⇒ 进程 `EXIT=0` 但输出无效（这解释了我方一度误判「③ 达成」）。

---

## 二、我的根因假设（**请逐条判它成立/不成立**）

**假设 H：描述符声明的 `kernel_pattern = 0x6`（0x7e/0x0）这一档，会让 A311D 的
VIPNano-QI NN core 无法推进作业 ⇒ 挂死。**

我的复算（**自己去核**）：

**(a) 权重写出分派**（`/root/clawtemp/mesa_src/etnaviv_ml_nn_v7.c`，约 493-497 / 542-546）：

```c
if (operation->pointwise && output_channels > 8)
   actual_size = write_core_6(subgraph, map, core, operation, zrl_bits);
else if (input_channels > 1)
   actual_size = write_core_interleaved(...);
else
   actual_size = write_core_sequential(...);
```

⇒ 所有 **1×1 conv**（pointwise 且输出通道 > 8）的权重都由 `write_core_6` 写出。

**(b) 描述符的 pattern 分档**（`/root/clawtemp/mesa_src/etnaviv_ml_nn.c` 的
`create_nn_config()`，约 1196-1217，PARTIAL_CACHE 分支）：

```c
if      (out_image_z_size >= 1024) { msb=0x13; low=0x80000;    high=0x0; }
else if (out_image_z_size >=  512) { msb=0x3d; low=0x0;         high=0x2aaaaaa0; }
else if (out_image_z_size >=  256) { msb=0x3e; low=0xffffaaaa; high=0x7fffffff; }
else if (out_image_z_size >=  160) { msb=0x6;  low=0x7e;        high=0x0; }   /* ★ */
else                                { msb=0x3f; low=0xfffffffe; high=0xffffffff; }
```

**(c) 相关性**（本模型全部 6 个通道数）：

| 输出通道 | 权重实际布局 | 描述符 pattern | 档 | 真机 |
|---|---|---|---|---|
| 24 | write_core_6 | 0x3f | <160 | 正常 |
| 122 | write_core_6 | 0x3f | <160 | 正常 |
| **244** | write_core_6 | **0x6/0x7e/0x0** | **[160,256)** | **★挂死** |
| 488 | write_core_6 | 0x3e | [256,512) | 正常 |
| 976 | write_core_6 | 0x3d | [512,1024) | 正常 |
| 2048 | write_core_6 | 0x13 | ≥1024 | 正常 |

⇒ **244 是唯一落进 [160,256) 档的通道数，也是唯一挂死的**（6 个通道数 5 正常 1 挂）。

**(d) 关键推论**：122/488/976/2048 四档的 pattern 与 `write_core_6` 的布局**不配对**，
但它们**跑得飞快** ⇒ 「不配对 ⇒ 算错值，但**不挂死**」⇒ 挂死的触发条件是
**「声明了 0x6」**，而不是「不配对」。

**(e) 我已排除的**：16 bit 步幅溢出（实际 `in_image_x_stride = input_width` = 50，
不是整图像元素数 610000）、`kernel_z_size`/`out_image_z_size`（14 bit 装得下 244）、
`kernels_per_core`（7 bit，值 31）、`post_shift`/`post_multiplier`、退化 scale
（1.334e36）、per-channel 重缩放路径（per-tensor 模型挂死位置**逐段相同**）。

---

## 三、请你回答的 6 个问题

**A. `write_core_6` 写出的权重布局，对应的是哪个 kernel pattern？**
   逐字读 `write_core_6`（v7 文件约 186 行起）说明它怎么组织权重
   （oc/ic/x/y 的顺序、interleave、zrl 压缩），并判断它**对应**描述符里的哪个 pattern 值。
   ⇒ 这一条是整个假设的地基：若 0x6 根本不对应 `write_core_6`，我的假设就得改写。

**B. 「不配对 ⇒ 算错值但不挂死」成立吗？**
   对 122 / 488 / 976 / 2048 四档：描述符声明的 pattern 与实际权重布局不配对，
   NNN core 会读出什么？给出源码依据（pattern 各位的语义）。
   ⇒ 若某些档其实也是「挂死只是没测到」，我的相关性就被削弱了，请指出。

**C. 这组分档逻辑是 Mesa 的历史设计，还是某次改动引入的？**
   查上游（mesa-26.2.5 及更早）的 `create_nn_config()`：
   同样的 1024/512/256/**160** 分档从何时开始？`160` 这个阈值从哪来？
   ⇒ 判断这是「Mesa 一直如此（那 NPU 侧就该支持 0x6 ⇒ 是 NPU/驱动限制）」
     还是「Mesa 某次改错（那是我们能修的）」。

**D. pattern 0x6 / 0x7e / 0x0 的语义出处？**
   在 Mesa 源码、驱动（`/root/workspace/oes-dsm/oes-dsm-kernel/drivers/gpu/drm/etnaviv/`）
   、注释、文档里找这个 pattern 值的定义或编码表。
   ⇒ 特别查：有没有**别的地方**（v8 路径？`etnaviv_ml_nn_v8.c`？）也用这个 pattern，
     或者有「pattern → 权重布局」的对应表。

**E. 独立枚举：还有哪些原因能让「前端 DMA 指针不再前进」？**
   从驱动侧（`etnaviv_gpu.c` / `etnaviv_sched.c` / `etnaviv_ml*`）与 NN core 配置
   （brick、tile、image cache、kernel cache、SRAM 容量、地址/尺寸）出发，
   列出**所有**能让 NN 作业不推进的可能原因，并标注哪些**与通道数相关**
   （因为我们的现象强烈依赖通道数）。
   ⇒ 这一条用来检查我是不是**只找到了一个相关项就当成了根因**。

**F. 复核我的确证实验设计。**
   我准备用同一份 .so（运行期 `TEFLON_KPAT=6|3` 强制 pattern，不设=原逻辑）做单变量对照：
   - `TEFLON_KPAT=6` + 122 通道 conv：若**立刻挂** ⇒ H 成立
   - `TEFLON_KPAT=3` + 244 通道 conv：若**不挂** ⇒ H 成立
   ⇒ 这个设计真的单变量吗？有没有更省事/更无歧义的对照？
     （注意：244 通道的 depthwise 会冻死整机，别设计会碰到它的对照。）

---

## 四、交付

1. **报告**：写到 `/root/workspace/oes-teflon/verify/reports/31-oca-review-kernel-pattern.md`
   （**只新建这一个文件**，不要改我方其它报告）。
2. 报告里必须有一节 **「我方结论逐条判定」**，逐条标
   **✅ 站得住 / ⚠️ 需修正 / ❌ 站不住**，并给出 `文件:行号`。
3. 在会话里回我一份**摘要**（要点 + 最强的证据 + 最强的反驳）。
4. 若你发现了**我方漏掉的其他可能根因**，请明确写出来并说明怎么区分。

**不要客气**：如果你认为 H 是错的，直接说，并给出能证伪它的证据与替代假设。
"
