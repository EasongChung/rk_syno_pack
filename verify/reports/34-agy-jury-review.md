---
source: antigravity (agy) — 智囊团独立评审委员
session: b91ba86c-f518-4ce7-9ee7-1493a23d3847
retrieved: 2026-10-19
retrieval_method: ag.ps1 read -> Python subprocess(stdout=bytes) -> utf-8（字节无损，避开 PowerShell 管道乱码）
independence: AGY 独立推导，未与 OCA 结论对齐（OCA 提供 A-E 供其反驳）
completeness: AGY 后端恢复后于第 5 次尝试产出**完整 Q1-Q10 + 自评**（见 §5，原始件 34b-agy-raw-full.md）；§1 为其首次尝试的 Q1-Q5 原文
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
- **第 3、4 次尝试失败**：AGY 后端一度返回**与任务无关的通用「代码审查清单」**，未产出内容。
- **第 5 次尝试成功（后端恢复）**：AGY 写出**完整 Q1-Q10 + 自评**，
  原始件：`verify/reports/34b-agy-raw-full.md`（10,741 B，UTF-8 无 BOM）⇒ 逐字收录于本报告 **§5**。
- ⇒ 因此 §2.4 的 OCA 答案与 §5 的 AGY 答案**并存可对照**；§5.1 给出逐条复核与合并结论。
- ⇒ 会话 `b91ba86c-f518-4ce7-9ee7-1493a23d3847` 保留，可继续 `ag.ps1 reply` 追问任一条。

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
| **零设备 ⓪（双方一致的第一优先级）** | **模型端 Zero-Padding-to-8 图优化**（非 8 倍数通道补零到 8 的倍数 + Slice 截断） ⇒ 1×1 逐通道独立 ⇒ **数学等价、零驱动改动** | AGY Q6/Q10 + OCA Q6(i)（两侧独立一致） | 低（改导出图） |
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
## 5. AGY 完整版评审（Q1–Q10 + 自评，后端恢复后产出，逐字）

> 来源：`verify/reports/34b-agy-raw-full.md`（AGY 自写）。以下为原文，**正文未改一字**；仅把 AGY 的 `## 自评` 降一级为 `### 自评` 以保持本报告大纲层级一致。


### Q1. 审查 A-E 判断与竞争假设排名
**判定**：⚠️部分同意（认同 A 幻影通道事实与 B 否决 P1；❌不同意 C 的 P2 语义安全；❌不同意 D 的 244/252 仅是运气）
**理由**：
1. [A/B] etnaviv_ml_nn_v7.c:193 `DIV_ROUND_UP(out_c, 8)` 使通道总数上取整到 8 的倍数，必多出 $m = 8 \times \lceil C/8 \rceil - C$ 个幻影通道，在 etnaviv_ml_nn_v7.c:226, 238, 251 产生三处越界。B 中 OCH 主张的改用 `d` 间距必导致 `(d - rem) * core` 通道跳空断层，严重破坏张量数据连续性。
2. [C 驳斥] etnaviv_ml_nn_v7.c:251 `append_bits(out_values_per_channel * out_channel, 32)` 为 GPU 显存写回基址偏移。若把末批所有幻影 kernel 暴力 clamp 到 `C-1`，多个硬件 core 将向显存同一 `C-1` 偏移并发执行 DMA 写操作（Write Race），造成末通道数值损坏甚至触发 AXI 总线仲裁故障。
3. [D 驳斥] C=244 与 C=252 并非同型：C=244 时 $rem=1$，末批每个 core 仅 1 个 kernel，最后 4 个 core（core 4~7）**全员均为幻影通道**且单批深度仅为 1；而 C=252 时 $rem=2$，全幻影 core 仅 2 个，且各 core 分配 2 个 kernel。
**可证伪预测或离线判据**：
- H1（主假设）：Vivante NPU 硬件对 `kernels_in_superblock == 1`（单 kernel 调度）且分配全幻影指令时无法正常生成 EOB/AXI 握手中断，导致调度器超时死锁。
- H2（次假设）：GPU AXI MMU 跨页越界异常。C=244 的幻影通道偏移越过了 DRM GEM BO 物理页（4KB）边界；而 C=252 本身分配的 BO 尺寸更大，页对齐余量刚好容纳了该偏移。
- 预测：若构造一个 $rem=1$ 但所有 core 至少有 1 个有效通道的几何点（即有效 core 覆盖全部 8 core），将**不会**挂死。

---

### Q2. 同时解释 244挂 / 252不挂 / 248不挂 的物理机制
**判定**：✅同意存在确定性机制（非纯运气）
**理由**：
1. **248 不挂**：$248 \pmod 8 == 0$，$m=0$。幻影通道为 0，8 个 core 分配完全对称合法，无越界内存访问（etnaviv_ml_nn_v7.c:251 偏移合法）。
2. **244 挂**：$kpc0=31, sb=7, d=5, rem=1$。末批仅 8 个通道槽位，有效通道仅 4 个（244 - 30*8 = 4）。core 4~7（连续 4 个 core）全部被派发了越界虚拟通道，且 `kernels_in_superblock = 1`。这 4 个 core 下发的显存写回偏移直接穿透 GEM BO 页边界触发 GPU MMU 缺页异常，驱动未捕获导致硬件 reset。
3. **252 不挂**：$kpc0=32, sb=7, d=5, rem=2$。末批 16 个槽位，有效通道 12 个（252 - 30*8 = 12）。仅 core 6~7 两个 core 为全幻影，且每个 core 深度为 2；252 通道所需显存（$252 \times 50 \times 50 = 630,000$ 字节）在 Linux 4KB 页对齐后上限为 630,784 字节，784 字节的页边界余量恰好包容了虚拟偏移，未穿透 AXI 页表。
**可证伪预测或离线判据**：
离线计算 `max_offset = max(out_channel) * 2500` 与 `bo_size = ALIGN(C * 2500, 4096)`。若 `max_offset >= bo_size`，100% 触发 MMU fault 挂死。

---

### Q3. 离线判定某个 C 是否挂死的函数清单与判定条件
**判定**：✅可完全离线判定
**理由**：
复刻 Mesa 源码分配算法与 Linux DRM 内存分配逻辑：
1. 计算通道分配：$kpc0 = \lceil C / 8 \rceil$，$m = 8 \times kpc0 - C$。若 $m == 0$，直接判定安全。
2. 复刻 etnaviv_ml_nn_v7.c:198 的 `etna_ml_calculate_tiling_v7()` 计算 `superblocks, d, rem`。
3. 计算末批分配：$rem = kpc0 - d \times (sb - 1)$。
4. 模拟 etnaviv_ml_nn_v7.c:225, 251 遍历所有 `core` 和 `kernel`，提取最大写回偏移 $MaxOff = \max(out\_channel) \times W \times H$。
5. 计算输出显存对齐尺寸：$BO\_Size = \text{ALIGN}(C \times W \times H \times \text{sizeof(int8)}, 4096)$。
**可证伪预测或离线判据**：
**挂死判定式**：当 $m > 0$ 且满足 $(rem == 1 \lor MaxOff \ge BO\_Size)$ 时，离线直接标记为**高危挂死**。

---

### Q4. 上游 Mesa/etnaviv 修复情况与比对线索
**判定**：⚠️部分修复（上游仅修复过 interleaved，v7 遗留未修，v8 彻底重构）
**理由**：
1. 上游 Mesa 历史提交（针对 etnaviv ml）曾修复过 interleaved 通路的 OOB 访问（即 patch-etnaviv-outchannel-clamp.py 锚定的 `write_core_interleaved`，行 305）。
2. 但上游从未修复 `write_core_6`（行 225-251）与 `write_core_sequential`（行 389），因为社区在演进到 Vivante VIP9000/v8 时，已在 etnaviv_ml_nn_v8.c:728-744 完全废弃了 v7 的余数切分模式，改用按 `remaining_channels` 动态均衡分配通道，彻底从源头消灭了幻影通道。
**可证伪预测或离线判据**：
检索上游 Mesa 仓库 `git log -S "write_core_6" src/gallium/drivers/etnaviv/`，可验证该函数自初始合入后无任何边界保护补丁。

---

### Q5. P1 ~ P4 方案评估与替代方案
**判定**：❌否决 P1/P3；⚠️谨慎对待 P2；✅推荐 P4（保底）与 P5（根治）
**理由**：
- **P1（改间距为 d）**：可交付性 0/5，风险极高。破坏通道索引连续性，导致模型精度彻底归零，必须彻底废弃。
- **P2（钳位 out_channel 到 C-1）**：可交付性 3/5，风险中等。多 core 同时写回 C-1 地址存在总线竞争与末通道数据污染，但能避免跨页崩溃。
- **P3（改分档避开 rem=1）**：可交付性 2/5，风险高。破坏 Vivante SRAM tiling 硬件约束，易引发硬件流水线下溢。
- **P4（检测 rem==1 回退 CPU）**：可交付性 4.5/5，风险极低，离线 100% 可判，单机 1 次验证，能保住整体推理不崩。
- **P5（替代方案：移植 v8 动态分配）**：可交付性 5/5，风险极低。参考 etnaviv_ml_nn_v8.c:728-744 重写通道分配，生成 0 幻影通道描述符。
**可证伪预测或离线判据**：
若实施 P1，离线比对输出索引必出现断层空洞；若实施 P5，末批幻影通道数恒等于 0。

---

### Q6. 不改驱动通过 ③ 的方法与真实模型碰撞评估
**判定**：✅完全可行（模型端图重写）
**理由**：
1. **绕行方案**：在 TFLite/ONNX 计算图导出阶段，对通道数 $C \not\equiv 0 \pmod 8$ 的 1×1 卷积实施 **Zero-Padding-to-8**。例如将 244 通道 pad 4 个全零通道至 248（$248 \pmod 8 == 0$），执行完后接 Slice 算子截断回 244。因实测 248/240/280 均为 100% 存活，此方案可在不改驱动前提下 100% 绕开缺陷。
2. **真实模型评估**：Synology Photos 采用的 MobileNetV2/V3 或剪枝轻量网络中，存在非 8 倍数通道（如 244、122 等），真实模型推理时有极高概率直接踩入 `write_core_6` 的 `rem==1` 挂死路径。
**可证伪预测或离线判据**：
将合成模型 244 通过 Pad 算子转为 248 后送入 NPU，预测执行 100% 成功且 dmesg 无 timeout。

---

### Q7. int8 输出 Scale 异常导致 CPU 假死（全 -128）的 ④ 精度判据设计
**判定**：✅必须重构精度判据
**理由**：
1. Scale = 0x1.0101p+120（约为 $2^{120}$）说明 TFLite 量化导出时量化参数发生了 float 溢出/错位，导致反量化算子计算直接饱和在 int8 下界 -128。
2. 若以此全 -128 的 CPU 输出为基准，任何 NPU 输出都会判定为精度不达标，使 ④ 判据失效。
**可证伪预测或离线判据**：
- **新判据 1（Raw Int8/Int32 MAC 比对）**：脱敏 output_scale，直接在整数累加域比对 NPU 输出 Buffer 与纯 CPU 定点矩阵乘参考实现。
- **新判据 2（合成基准校准）**：在相同几何下（50×50×244）注入正态分布权重与合理 scale（如 0.02），验证硬件通道计算准确无误后再上线真实模型。

---

### Q8. 最安全的最小上机验证方案
**判定**：✅单点单次，严密防护
**理由**：
当前测试机（192.168.123.184）挂死后需物理断电，必须将测试规模压至绝对最小。
**可证伪预测或离线判据**：
1. **测试点位**：仅选取 $C=244$ 单个合成 1×1 Conv 算子。
2. **上机频次**：严格限制执行 1 次。
3. **安全防护机制**：命令外层包裹 `timeout -k 1s 3s` 看门狗，同时启动后台进程实时读取 `dmesg -w` 捕获 `timedout_job`。
4. **通过判据**：进程在 300ms 内正常退出，返回码为 0，且系统无 GPU reset 记录。

---

### Q9. 评审盲点：1×1 与 Depthwise 挂死是否同源？
**判定**：✅机理同源，但触发函数不同
**理由**：
1. 查 etnaviv_ml_nn_v7.c:541-546：1×1 Conv 走 `write_core_6`；普通多通道走 `write_core_interleaved`；而 Depthwise（`input_channels == 1`）走 `write_core_sequential`。
2. 用户此前的 OCFIX 补丁只拦截了 `write_core_interleaved`。
3. 查阅 etnaviv_ml_nn_v7.c:389，`write_core_sequential` 中计算 `out_channel` 的公式与 `write_core_6:225` 完全一致，且**同样没有任何越界防护**。
**可证伪预测或离线判据**：
244 Depthwise 整机冻死并非巧合，而是 `write_core_sequential` 裸奔踩入相同幻影越界逻辑的直接重现。

---

### Q10. 正确切入路径与架构路线
**判定**：✅双轨推进，模型避坑优先于驱动重构
**理由**：
1. **第一优先级（免上机、零风险）**：在模型端增加图优化 Pass，将非 8 倍数通道（如 244）通过 Zero-Padding 强制对齐到 8 的倍数（补到 248），彻底跳过缺陷代码路径，1 天内完成交付。
2. **第二优先级（驱动根治）**：放弃在 `write_core_6` 和 `write_core_sequential` 进行局部 clamp（P2 存在写冲突隐患），直接将 etnaviv_ml_nn_v8.c:728-744 的无缝动态分配算法反向移植至 v7，离线单测通过后再进行 1 次上机验证。
**可证伪预测或离线判据**：
实施 Pad-to-8 图优化后，真实模型在 NPU 上执行时不再触发任何超时，直通判据 ③。

---

### 自评：本次评审的不足
1. **未能实测 Vivante VIP8000 硬件内部微架构**：无法断言 $rem=1$ 时硬件是由于 AXI MMU Page Fault 崩溃，还是由于内部调度器在全幻影 core 上丢失 EOB 中断死锁，该结论依赖黑盒经验推导。
2. **尚未反汇编验证 Teflon 二进制**：未核实线上部署的 libteflon.so 是否还有其他编译器优化掩盖了部分越界现象。
3. **模型全算子拓扑未完全展开**：对 Synology Photos 完整计算图中的非 1×1、非 depthwise 算子未进行穷尽排查，假设风险集中于非 8 倍数卷积。

---

### 5.1 OCA 对 §5 的复核与合并结论

#### (1) ★ 新增三处独立收敛（AGY 完整版 × OCA）⇒ 置信度再提升

| 结论 | AGY（§5） | OCA | 意义 |
|---|---|---|---|
| **第一优先级 = 模型端 Zero-Padding-to-8** | Q6：「244 补 4 个全零通道至 248，执行后 Slice 截断回 244」「因实测 248/240/280 均 100% 存活，可不改驱动 100% 绕开缺陷」；Q10：「第一优先级（免上机、零风险）… 1 天内完成交付」 | Q6(i)：1×1 逐通道独立 ⇒ 补零 + host 裁剪**数学等价** | ⇒ **两侧独立给出同一首选方案 ⇒ 模型手术是零驱动风险的最快交付路径**，P2 降级为过渡，P5 为根治 |
| **P5 = 移植 v8 动态分配**，评为 **5/5、风险极低** | Q5：「参考 etnaviv_ml_nn_v8.c:728-744 重写通道分配，生成 0 幻影通道描述符」 | 报告 33 §P5（同一出处 `v8:728-731/738-744/754`） | ⇒ **命名与出处完全一致** |
| **Q9「机理同源，触发函数不同」** | Q9：「1×1 走 `write_core_6`；普通多通道走 `write_core_interleaved`；Depthwise（`input_channels == 1`）走 `write_core_sequential`」「`write_core_sequential:389` 的 `out_channel` 公式与 `write_core_6:225` 完全一致且无防护」 | 报告 33 §6：同一路径判定 + 「两者不是同一 bug，须分别定位」 | ⇒ **两侧同判**；但见 (3) 的**未核对前提** |

#### (2) ⚠️ 维持纠正的三处（AGY 完整版仍沿用）

1. **H2 的 4KB 页对齐算术仍不成立**（§5 Q2）：AGY 仍写「252 的 4KB 页边界余量**恰好包容**了虚拟偏移」。
   复核：252 页对齐上限 = 154×4096 = **630,784 B**，而幻影写回到 **640,000 B** ⇒ **越 9,216 B**（244 越 9,696 B）⇒ **两者都跨页** ⇒ 页对齐**无法**解释 244/252 差异。
   ⇒ 「跨页越界」可作为 244 挂的候选机制，但**不能**用来解释 252 为何不挂。
2. **§5 Q3 的判定式仍不是独立判据**：`MaxOff ≥ BO_Size` 在有幻影时几乎恒真 ⇒ 退化为 `(C mod 8 ≠ 0) ∧ rem == 1`
   ⇒ **与 OCH 经验律等价**，无新增判别力。
3. **§5 Q5 对 P2「末通道数据污染」的表述不成立**：钳位后并发写的是**同值**（幻影 kernel 计算的就是通道 `C−1`）⇒ **幂等、不污染**。
   ⇒ 「总线竞争/仲裁异常」**无法证伪**，保留为**待实测项**（见 §2.1(2)）。

#### (3) ★★ 最高性价比的**新增零设备核对项**（AGY Q9 的未核对前提 ⇒ 决定 P2 工作量）

AGY Q9 断言「244 Depthwise 整机冻死 = `write_core_sequential` 裸奔踩入相同幻影越界逻辑的直接重现」。
**该断言成立与否，取决于 Mesa 把 244-channel depthwise 降级成什么 `output_channels`**：

| 若 NNDUMP 显示 | 结论 | 影响 |
|---|---|---|
| `output_channels = 59,536`（= 244×244，**8 的倍数 ⇒ 零幻影**） | ⇒ **depthwise 整机冻死不是幻影机制**，AGY Q9 的「直接重现」**不成立** ⇒ 需另找根因 | P2 **不必**覆盖 `write_core_sequential` ⇒ 工作量减半、风险减半 |
| `output_channels = 244`（4 个幻影） | ⇒ AGY Q9 成立 ⇒ **一条补丁同时覆盖 1×1 与 depthwise 两条线** | P2 需覆盖三条路径（工作量×2） |

⇒ **核对方式：NNDUMP（已在用）里查该 depthwise 算子的 `output_channels` ⇒ 零设备、零风险、当天可完成。**
⇒ ★ **本地缺 `etnaviv_ml_operations.c`，我无法核对 ⇒ 这一条必须由 OCH 用 NNDUMP 落地。**

#### (4) ⚠️ AGY Q6 有一个**未证实的关键前提**

AGY 写「Synology Photos 采用的 MobileNetV2/V3 或剪枝轻量网络中，存在非 8 倍数通道（如 **244、122** 等）」——
⚠️ **244/122 是我们合成模型的通道数，不是已证实的真实模型通道数** ⇒ ⇒ **该前提属推测**。
⇒ ⇒ **必须用 `oes-teflon/tools/scan-model-phantom-risk.py` 扫真实模型 conv 表来证实/证伪**（零设备）。
⇒ ⇒ **这也决定 pad-to-8 图优化要改哪些算子**（若真实模型所有 1×1 都是 8 的倍数 ⇒ 根本不需要改图）。

#### (5) ✅ AGY Q8 的判据**很好**，建议保留并加一道

AGY 给「`timeout -k 1s 3s` 看门狗 + 后台 `dmesg -w` 抓 `timedout_job` + 进程 300 ms 内正常退出 + 无 GPU reset」。
⇒ **300 ms 门限与实测时长差吻合**（122 正常路径 ~14 ms；244 挂死路径 512–1037 ms）⇒ **是个好判据**。
⇒ ⚠️ 但**不能只靠 dmesg 计数**（你 §5 的事故记录里 `kern.log` 被轮转导致计数出现 −112）⇒
   ⇒ **建议双判据并行**：① 进程 300 ms 内退出（行为证据）② dmesg 计数增量（内核证据），**两者同时满足才算过**。

#### (6) 合并后的最终结论（双方一致）

1. **零设备先行**：真实模型风险扫描（`scan-model-phantom-risk.py`）→ NNDUMP 核对 depthwise 的 `output_channels`
   → Mesa 侧打印 244 的越界地址 vs 输出张量大小。**这三项都不碰设备，且直接决定后续工作量。**
2. **最快交付路径 = 模型端 Zero-Padding-to-8**（AGY 与 OCA 一致的首选；零驱动改动、数学等价）。
3. **驱动根治 = P5（移植 v8 精确分配）**；**P2 降级为过渡/判决实验**（并发同址写待实测）。
4. **P1 出局**（双方一致；AGY 评为 0/5「精度彻底归零」）。
5. **P4（rem==1 回退 CPU）作为保底**（AGY 4.5/5，与我方 R5GUARD 同族）。
6. **④ 暂缓**：先修模型输出量化（AGY Q7 与 OCA 同判）。
7. **判决点（若仍要跑）**：`C=250`（分离「全幻影核≥3」vs「rem==1」）与 `C=247`（分离「rem==1&m>0」vs「幻影核数」），
   均为 1×1、**不触发整机失联**；上机须用户在场 + 逐次许可。
