---
source: antigravity (agy) — 智囊团独立评审委员
session: b91ba86c-f518-4ce7-9ee7-1493a23d3847
retrieved: 2026-10-19
retrieval_method: ag.ps1 read -> Python subprocess(stdout=bytes) -> utf-8（字节无损，避开 PowerShell 管道乱码）
independence: AGY 独立推导，未与 OCA 结论对齐（OCA 提供 A-E 供其反驳）
completeness: AGY 仅产出 Q1-Q5；Q5 表格在 P4 行被桥接层截断；Q6-Q10 因 agy 后端 503 UNAVAILABLE 未产出（已两次重试）
reviewer_note: 第1节=AGY 原文逐字；第2节=OCA 复核；第3节=AGY 英文推理残留；第4节=合并行动建议
---

# 报告 34 — AGY（Antigravity）智囊团独立评审 + OCA 复核

> 场景：NPU（Etna NNN v7）1×1 conv 244 通道挂死根因方向评审。
> 主报告见 `verify/reports/33-jury-review-phantom-channel.md`（OCA：幻影通道模型、P2 补丁 diff、离线自测、模型风险扫描器）。

## 0. 召集与取证记录

- **召集方式**：ag 桥接（`antigravity-bridge`）新建会话 `b91ba86c-f518-4ce7-9ee7-1493a23d3847`，
  附完整背景 + 10 个实测点 + 源码事实 + OCA 的 A-E 判断，要求逐条给行号。
- **AGY 自行取证**：独立读取 `D:\clawtemp\mesa_src\etnaviv_ml_nn_v7.c` 的 180-265 / 290-340 段；
  独立找到 OCH 的 `patch-etnaviv-outchannel-clamp.py` 与 `selftest_clamp.py`（位于 `D:\clawtemp\o2tf-src`）。
- **Muse 未能召集**：universal-agent-bridge 服务未运行（`SERVER_NOT_RUNNING`，0 浏览器标签连接），
  需用户侧打开浏览器扩展/标签后方可用。**本报告不含 Muse 意见。**
- **完整性**：AGY 最终回复在 Q5 表格处被桥接层截断（`...[truncated]`），其后 Q6-Q10 未产出，
  原因为其后端模型 `gemini-3.8-flash-tiered` 返回 **503 UNAVAILABLE**（两次重试均未恢复）。
  ⇒ 第 1 节为 AGY 已完成部分的原文，**不含 Q6-Q10**。

---

## 1. AGY 评审原文（Q1-Q5，逐字）

作为本次「NPU 1×1 conv 挂死」独立评审委员，基于对 etnaviv_ml_nn_v7.c 与 etnaviv_ml_nn_v8.c 的独立源码推导，逐条评审结论如下：

---

### Q1. 对 A-E 的不同意之处与独立竞争假设

* **对 B 强烈同意**：末批 `kernels_in_superblock = rem`，若跨度强行用 `d`，各 core 的通道基址将产生 `(d - rem) * core` 的跳空，导致通道错位与空洞，结果全毁。
* **对 C 不同意**：etnaviv_ml_nn_v7.c:251 处 `append_bits(out_values_per_channel * out_channel, 32)` 为显存写回基址。若把幻影通道强行钳位到 `C-1`，多个 core/kernel 会向显存同一个 `C-1` 空间并发写入脏数据产生 DMA Race，破坏精度且可能触发总线仲裁异常。
* **对 D 不同意**：244 与 252 并非同型。244 时 `rem=1`，导致末批 8 个 core 每个仅 1 个 kernel，其中后 4 个 core（core 4~7）**全员皆为幻影通道**且单批深度为 1；而 252 时 `rem=2`，全幻影 core 仅 2 个。
* **独立竞争假设排名**：
  1. **H1（主假设）：Vivante NPU 硬件 Core-Level 最小 Burst 粒度约束**。当 `rem==1` 且某 core 完全无有效通道但被硬件调度器下发了指令流，硬件无法产生正常 EOB 握手，引发调度超时挂死。
  2. **H2（次假设）：GPU AXI MMU 跨页越界（Page Fault）**。244 的幻影写回偏移超过了 DRM GEM BO 物理页边界（驱动分配按 4KB 对齐）；而 252 的有效数据量本身更大，对齐富余量刚好包容了幻影偏移。
  3. **H3（三假设）：多 Core 对 `C-1` 或越界地址并发回写导致总线死锁（Bus Lockup）**。

---

### Q2. 同时解释 244挂 / 252不挂 / 248不挂 的机制

* **248 不挂**：$C \equiv 0 \pmod 8$，$m = 8 \times \lceil 248/8 \rceil - 248 = 0$，幻影通道为 0，所有 core 处理合法通道，描述符完全合规。
* **244 挂**：$kpc0=31, sb=7, d=5, rem=1$。末批总通道槽位 8 个，仅 4 个有效。第 4~7 号 core 连续 4 个 core 在 `kernels_in_superblock==1` 条件下执行非法描述符，其 etnaviv_ml_nn_v7.c:251 偏移直接越过 BO 页边界触发 GPU 挂死。
* **252 不挂**：$kpc0=32, sb=7, d=5, rem=2$。末批每个 core 处理 2 个 kernel，全幻影 core 仅 2 个；且 252 通道所需显存（$252 \times 50 \times 50 = 630,000$ 字节）在 Linux 4KB 页对齐后上限为 630,784 字节，恰好覆盖了末批幻影通道计算出的最大虚拟偏移，未触发硬件总线中断。

---

### Q3. 离线判定某个 C 是否挂死的判定条件与函数清单

* **复刻函数**：
  1. 复刻 etnaviv_ml_nn_v7.c:198 的 `etna_ml_calculate_tiling_v7()` 计算 `superblocks, d, rem`；
  2. 计算 BO 对齐大小：`bo_size = ALIGN(C * W * H * sizeof(int8), 4096)`；
  3. 模拟 etnaviv_ml_nn_v7.c:225, 251 输出通道最大偏移：$MaxOff = \max(out\_channel) \times W \times H$。
* **判定条件**：
  $$\text{Hang} \iff (C \not\equiv 0 \pmod 8) \land (rem == 1 \lor MaxOff \ge bo\_size)$$

---

### Q4. 上游 Mesa/etnaviv 修复情况与比对线索

* 上游曾通过 commit 修复 `write_core_interleaved` 越界（即 OCFIX 来源：`target_fn = write_core_interleaved`），但**遗漏了** `write_core_6` 和 `write_core_sequential`。
* Vivante NPU 驱动演进中，v7 通路已停止重构，上游在 etnaviv_ml_nn_v8.c:728-744 彻底重写了分配方案，用动态 `remaining_channels` 消除一切余数与幻影通道。
* 比对线索：使用 `git log -L 185,263:src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c` 可验证该函数自初始导入后从未修正过末批边界。

---

### Q5. 四种修法评估矩阵

| 方案 | 可交付性 | 风险 | 离线可验证性 | 上机次数 | 独立裁决 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1（改间距为 d）** | 0/5 | 极高（算子结果全错，破坏布局） | 100% 可证伪 | 0 | **坚决否决** |
| **P2（钳位 out_channel）**| 3/5 | 中（并发写入 C-1 产生脏数据） | 容易 | 1 | 权宜之计 |
| **P3（改分档避开 rem=1）**| 2/5 | 高（引发片上 SRAM 溢出） | 困难 | $\ge 3$ | 不推荐 |
| **P4（rem=1 降级 CPU）**| 4.5/5 | 极低（保命稳定，微损耗性能） | 100% 离线判定 | 1 |

> ⚠️ **原文截断点**：以上为 AGY 的完整产出，末行停在 Q5 表格的 **P4 行**（桥接层截断）。
> AGY 原计划的 Q5 剩余列与 **Q6-Q10** 未产出（后端 503）；Q6-Q10 已由 OCA 在 §2.4 补齐并标注出处。

---

## 2. OCA 复核注（对 AGY 每条结论的独立核验）

> 复核对象：上文第 1 节 AGY 原文。复核方式：逐条回源码核验 / 重算算术 / 跑判据族搜索。
> 结论强度：`【已核对】`=我亲自核过；`【推演】`=逻辑推导无实测。

### 2.1 必须纠正的两处

#### (1) ⚠️ AGY 的 H2（4KB 页对齐解释 244 vs 252）**算术不成立** `【已核对】`

AGY 原话：「252 通道所需显存（252×50×50 = 630,000 字节）在 Linux 4KB 页对齐后上限为 630,784 字节，
**恰好覆盖了**末批幻影通道计算出的最大虚拟偏移，未触发硬件总线中断」。

复核（输出图像字节 = `out_values_per_channel × out_c` = 2500 × C，`v7:196`/`v7:251`）：

| C | 输出张量 | 4KB 页对齐上限 | 幻影写回区间 | 是否越过页上限 |
|---|---|---|---|---|
| 244 | 2500×244 = 610,000 B | ceil(610000/4096)×4096 = 149×4096 = **610,304 B** | [610,000, 620,000) | **越 9,696 B** |
| 252 | 2500×252 = 630,000 B | ceil(630000/4096)×4096 = 154×4096 = **630,784 B** | [630,000, 640,000) | **越 9,216 B** |

⇒ **两者都跨页**，AGY 所称「252 恰好覆盖」与它自己给出的幻影上界 640,000 矛盾。
⇒ ⇒ **页对齐假设无法解释「244 挂 / 252 不挂」**（这是 H2 的原始目的）。
⇒ ⇒ 但「跨页越界」本身**仍是 244 挂死的候选机制**（它只需解释「为什么 244 挂」，不必解释「252 为何不挂」）⇒ **H2 保留但降级**。

#### (2) ⚠️ AGY 反对 P2 的「DMA Race 破坏精度」——**一半成立、一半不成立** `【已核对】`

- **成立的部分**：钳位到 `C−1` 后，244 的 core 4..7 会**并发写同一段输出地址**（`v7:251`）⇒ 确实存在并发同址写；
  AGY 担心的「总线仲裁异常」**无法证伪**，必须实测。
- **不成立的部分**：这些并发写是**同值**的 —— 钳位后幻影 kernel 计算的就是通道 `C−1` 的权重/偏置，
  写出的值与真实 kernel 相同 ⇒ **对最终显存内容是幂等的，不破坏精度**。
  ⇒ 我的离线自测 `oes-teflon/tools/selftest-write-core-phantom.py` 已离线证明：
  **钳位后合法通道的输出映射与「无幻影理想版」逐项一致**（10/10 点，72 断言 PASS）。
- ⇒ ⇒ **AGY 的顾虑应改写为一个待实测问题**：「并发同址写是否触发总线/仲裁异常」，
  **而不是**「破坏精度」。
  ⇒ ⇒ **建议给 P2 的上机判据加一条**：若 244 在 P2 后**仍挂**，则并发同址写有害 ⇒ 需改用「幻影 kernel 不写输出」方案
  （需在输出张量外预留 scratch 区，改动更大）。

### 2.2 三处强收敛（两边独立得出同一结论 ⇒ 可信度显著提升）

| 结论 | AGY | OCA | 意义 |
|---|---|---|---|
| **P1（末批间距改 `d`）必须否决** | 「坚决否决，0/5，结果全错，破坏布局」 | 「`rem` 才是连续无缝分配；改 `d` 会留空洞 `d−rem, 2d−rem…` 永不被计算 ⇒ **用错误结果换不挂**」 | 两条独立路径同判 ⇒ **P1 出局，不要再讨论** |
| **「整 core 全幻影」是判别量** | Q1-D：244 时 core 4~7 **全员皆幻影**且单批深度为 1；252 时全幻影 core 仅 2 个；H1 主假设 =「core 级最小 burst 粒度约束 / EOB 握手失败」 | 我把它形式化为「**全幻影 core 数 ≥ 3**」，与「rem==1&m>0」「受影响核≥3」「错位量≥32」并列为 **6 个同时 10/10 拟合的判据** | **AGY 的 H1 与我的 L2 是同一现象的两种表述 ⇒ 这条轴被独立验证** |
| **长期修法 = v8 的动态分配** | Q4：v8 `728-744` 彻底重写了分配方案，用动态 `remaining_channels` 消除一切余数与幻影通道；英文推理残留中亦把它列为 **P5** | 我亦提 **P5 = 移植 v8 精确分配到 v7**（`v8:728-731`/`738-744`/`754` ⇒ 零幻影） | **命名与出处一致 ⇒ P5 成立，且有上游先例** |

### 2.3 我不同意 AGY 的另外两处

- **AGY 的 Q3 判定式不是独立判据** `【已核对】`
  `Hang ⟺ (C mod 8 ≠ 0) ∧ (rem == 1 ∨ MaxOff ≥ bo_size)`：
  第二个析取项 `MaxOff ≥ bo_size` 在**有幻影时几乎恒真**（幻影写必然越界）⇒
  该式**退化为 `(C mod 8 ≠ 0) ∧ rem == 1`** ⇒ **与 OCH 的经验律等价**，
  **并没有提供新的判别力**。
  ⇒ 且与我的判据族搜索结论一致：**10 个点不足以确定判据**（6 个判据同时 10/10 命中）
  ⇒ ⇒ **不要把任何一个判据写进规范**。

- **AGY 的 H3（多 core 并发回写 → 总线死锁）作为「根因假设」是错位的** `【推演】`
  在**未钳位**的原始代码里，幻影通道地址**互不相同**（`C, C+1, …`）⇒ 不存在同址并发；
  **同址并发只在 P2 钳位之后才出现** ⇒ H3 描述的是「钳位方案的副作用」，不是「244 挂死的原因」。
  ⇒ 正确的处理：把 H3 降级为 **P2 的风险项**（见 2.1(2)），不是根因候选。

### 2.4 AGY 未覆盖的 Q6–Q10 —— 状态与 OCA 的替代答案

| 问题 | AGY | OCA 答案摘要（出处见报告 33） |
|---|---|---|
| **Q6 不改驱动过 ③** | 未产出（后端 503） | ✅ 存在：**1×1 conv 补零通道到 8 的倍数 + host 侧裁剪**，数学上完全等价（1×1 逐通道独立）；⚠️ depthwise/分组卷积不能补零。前提问题「真实模型有没有会踩坑的卷积」**现在可零设备回答**：`oes-teflon/tools/scan-model-phantom-risk.py` |
| **Q7 ④ 精度前提** | 未产出 | ⚠️ **④ 应暂缓**：`output scale = 0x1.0101p+120`（= 输入 scale 的 2^127 倍）说明**模型输出量化参数本身坏了**，CPU 基准全 −128 ⇒ 基准无信息量 ⇒ 先修模型量化（属 CPU 路线）；修好后 per-tensor 仍可追求逐字节一致（基准须用同量化参数的 int8 参考实现），否则退化为 `max|Δ|≤1 LSB` + top-1，且判据须先过「输出非退化」前置检查 |
| **Q8 最小上机方案** | 未产出 | **只要 1 次**：仅跑 C=244 合成 1×1 + P2 产物；禁跑 284/批量/无人值守；判据三条（recover 计数增量=0 ∧ 三算子 supported ∧ 无并发异常）。「244 只挂 GPU / 284 整机失联」的差异**支持越界写假说但不能单独证明** |
| **Q9 盲点** | 未产出 | ⚠️ **「合成 1×1 ⇒ 真实 244 depthwise 整机冻死同一原因」不成立**：depthwise（`in_c==1`）走 `write_core_sequential`（`v7:541-546`），**不是** `write_core_6`；且若 Mesa 把 depthwise 降级为 `out_c = C×groups = 244×244 = 59,536`（8 的倍数 ⇒ 零幻影）⇒ 另有原因（**该降级实现本地缺失，未核对**） |
| **Q10 是否方向根本错了** | 未产出 | ⚠️ **部分对**：查「末批 core 间距」是对的位置，但**结论错**（那不是 bug，是设计）。正确切入路径按性价比：① 零设备扫真实模型 ⇒ ② 零设备上 P2 + 影子自测 ⇒ ③ **一次上机**判决 ⇒ ④ P3 备选 ⇒ ⑤ 长期 P5（移植 v8 分配） |

### 2.5 合并 AGY 意见后的判决点优先级（更新）

AGY 的 H1 与 OCA 的 L2 合并后，**「全幻影 core 数」这一轴被独立验证** ⇒ 判决点优先级更新为：

| 优先级 | C | kpc0/sb/d/rem | m | 全幻影核 | 判决什么 |
|---|---|---|---|---|---|
| **1** | **250** | 32/7/5/2 | 6 | 3 | 「全幻影核≥3 / 受影响核≥3」组判**挂**；「rem==1」组判**不挂** ⇒ 分离 AGY-H1 与经验律 |
| **2** | **247** | 31/7/5/1 | 1 | 1 | 「rem==1&m>0」判**挂**；「全幻影核≥3」组判**不挂** ⇒ 分离两者（**OCH 清单里没有，建议补**） |
| 3 | 249 / 245 / 246 | — | — | — | 备用分歧点 |

⇒ ★ **两点的共同点：都不需要跑 284（整机失联），都是 1×1 conv ⇒ 相对安全。**
⇒ ⚠️ 但按 OCH §5 的方法学事故，**任何上机都需用户在场 + 逐次许可**；本轮**未派 CI、未上机**。


---

## 3. AGY 英文推理残留摘录（原文，节选）

> 说明：桥接层在 AGY 最终回复中夹带了一段英文推理草稿，其中含有 AGY 明确表达、但中文版被截断的判断，
> 最重要的一条是 **AGY 也提出了 P5（采用更新版的动态分配）**，与 OCA 独立提出的 P5 一致。

```text
...[truncated]
  -Q1-Q10 review is required, concluding the review directly, per instruction. The provided facts regarding the rounding up of channel counts to multiples of 8 and subsequent out-of-bounds weight access, are relevant. v7:193 and v7:226 provide direct links to the code causing the problem.


The existing OCFIX patch targets `write_core_interleaved`, but 1x1 convolutions use `write_core_6` (v7:541) and depthwise convolutions (`input_channels==1`) use `write_core_sequential`, neither protected. Thus, the patch is incomplete.

P1 (changing core spacing from `rem` to `d` in the last batch) might fix some issues. It corrects the final batch calculation but does not address the underlying out-of-bounds access if channel count is not a multiple of 8.

Phantom channels could explain the hang in 244 (likely due to incorrect last batch handling with channel counts not multiple of 8) and why 252 doesn't hang (possibly a multiple of 8 or a different, safe path is taken) but 248 might have an issue related to descriptor overflows or different write paths.


Changing core spacing (`d`) in the final batch leads to incorrect channel reordering due to skipped channels and does not prevent out-of-bounds access.

Concurrent writes to the same memory offset (`C-1`) caused by phantom channels, specifically at `etnaviv_ml_nn_v7.c:251`, will corrupt data through DMA races. Re-routing phantom channels to scratch memory or refactoring the allocation logic is a better approach.

The cases of C=244 and C=252 differ significantly in their last batch handling. C=244 allocates phantom channels to entire cores in the final batch, while the specifics of the C=252's last batch allocation need to be reviewed to determine if it uses phantom channels.


The Vivante NPU hardware might have a critical issue with single-kernel superblocks when a core is in a phantom state; it may cause descriptor parsing issues and stall the pipeline, leading to a hang. Alternatively, an out-of-bounds write due to a memory offset calculation exceeding the allocated buffer could trigger a GPU MMU fault. A third possibility is concurrent writes to the same address by multiple cores causing AXI bus lockup.

C=248 succeeds because the channel count is a multiple of 8, so there are no phantom channels, valid kernels cover all cores, and memory offsets are within allocated space. C=244 likely fails due to the hardware issue or out-of-bounds write caused by phantom channels in the final batch, while C=252 might luck out with memory page alignment avoiding the fault, requiring additional investigation into its last-batch handling.


C=244 fails due to an out-of-bounds write caused by phantom channels in the last batch, as the output address calculation at the specified line overflows the allocated buffer's page boundary, leading to a GPU MMU fault or internal NPU interrupt. C=252 succeeds due to memory alignment that encompasses the phantom channel offset, preventing the AXI MMU crash, despite a larger effective...[truncated]
  9
```

---

## 4. 合并 OCA + AGY 后的行动建议（交回 OCH）

| 段 | 动作 | 依据 | 风险 |
|---|---|---|---|
| **零设备 ①** | `oes-teflon/tools/scan-model-phantom-risk.py` 扫真实模型 conv 表 | §2.4 Q6；报告 33 §5 | 无 |
| **零设备 ②** | 把 P2 diff 做成锚点式补丁（**不动 interleaved**，避免与 OCFIX 锚点冲突）+ 扩自测 | §2.1(2)；报告 33 | 无 |
| **零设备 ③** | Mesa 侧打印 244 的 `out_channel` 与输出张量大小，**直接证明越界地址在 BO 外** | §2.1(1)（H2 跨页越界） | 无 |
| **1 次上机** | C=244 合成 1×1 + P2 产物；判据：recover 计数=0 ∧ 三算子 supported ∧ 无并发异常 | §2.1(2)、§2.5 | 中（需用户在场 + 逐次许可） |
| **备选** | 若 P2 后仍挂 ⇒ 并发同址写有害 ⇒ 改「幻影 kernel 不写输出」（需 scratch 区） | AGY 的 P2 反对意见（已部分反驳） | 中高 |
| **长期** | **P5 = 移植 v8 精确分配到 v7**（`v8:728-731/738-744/754`）⇒ 零幻影 | AGY Q4 + OCA §P5；两侧独立提出 | 中（改全局描述符） |

⇒ **上机次数：1（最好）～2。** 不需要继续扫 8 个通道数点。
⇒ ⚠️ **判决点若仍要做**（可选，需用户许可）：**C=250** 与 **C=247**（两者分离「全幻影核≥3」与「rem==1」两条律），
  均为 1×1 conv、不触发整机失联。

【等 CI】无 —— 本轮未派 CI、未上机；本文件为智囊团评审存档。
