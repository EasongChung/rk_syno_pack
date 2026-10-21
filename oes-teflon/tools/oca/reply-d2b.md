[2026-10-19 01:37] ASSISTANT msg_151cee0c3001EbhPkv6KZP0fHi (space-bunny-free)
## 一、分派结果：Antigravity 答了（有效），但先踩了一个坑

**第一次发中文包 → 回了一句完全无关的胡话**（"Understand the agent's role as a personal assistant…"）。读回会话才发现根因：**多行中文经 PowerShell → `Start-Process` → `agentapi` 的 argv 被按 ANSI 代码页（GBK）错位**，模型收到的是坏字节。改成**纯 ASCII** 重发即正常。

⇒ 已写入记忆（`environment.md`）：**这条通道只能发纯 ASCII**，长文本走 Python 写无 BOM 文件 → `ReadAllText` 读成变量传入。第一个被污染的会话 `b9d1a43e…` 留档不删。

## 二、两份独立评审**收敛到同一个选择 (a)**

| | Muse | Antigravity |
|---|---|---|
| 选择 | (a) rank≤4 模型作验证载体，本模型标 unsupported | **同 (a)** |
| 核心理由 | 验证与能力边界解耦 | ① Gallium 驱动是**硬件执行管线，不是图编译器**；在驱动内做切分+多 Pass **颠覆职责** ② 15 个 site 要重写 ③ 边际成本倒挂 |
| 最可能被证伪处 | 载体无代表性 ⇒ 只证伪玩具；产品需求即该模型 | **产品/验收硬约束**：若该模型是法定基准且禁止重导出，"声明不支持"工程正确但**商业被一票否决** |
| 下一步测量 | 载体上跑 print-only，报接管率/site 覆盖/op-type 覆盖 | 跑标准 rank≤4 INT8 CNN（MobileNetV2/ResNet-18），测逐层 **max abs error + byte diff** |

两份**独立**（不同模型、不同通道）落在同一结论上，可信度比单一来源高。

## 三、★ Antigravity 提了一条我认为**应该插到最前面**的东西

> **推翻条件**：如果 rank≤4 模型**也**达不到 byte-exact —— 例如 NPU 定点乘加的舍入模式（RNE vs trunc-toward-zero）、累加器饱和溢出、非零 `zero_point` 硬件加速路径差异 —— 那么**"byte-exact"这个判据本身在这颗 NPU 上可能就不成立**，"用 rank≤4 载体验证驱动"的前提会整体塌掉。

⇒ **这意味着我们可能正在按错误顺序投资**：还没确认"这颗 NPU 能不能对任何 int8 模型做到逐字节一致"，就准备去找载体模型。

**我建议插一个更便宜的探针**（排在找模型之前）：拿**最小**的 rank-4 int8 用例（单个 conv，或一个极小的 rank≤4 模型）先测 byte-exact——
- 若**能** ⇒ 载体路线成立，按计划找模型；
- 若**不能**（且已排除我们四处修复的 bug）⇒ **④ 需要与你重新定义**（例如"NPU 输出 == CPU 参考在容差内"或"==同一个 int8 参考实现"），否则后面所有工作都在追一个达不到的门。

这条比"扫一批候选模型"便宜一个数量级。

## 四、两条我不采纳 Antigravity 的内容（标出来，不当证据）

1. **"Vivante/VeriSilicon NPU 核心"** —— 这是它**自己补的前提**，我的包里没提厂商。未经证实的前提一旦进入推理就会污染结论（这正是我们这四轮反复付学费的东西）。它后面的论证不依赖厂商，但这个事实本身不可信。
2. **(b) 的第 4 阶段"用 2D DMA 做 gather/scatter 拼接"** —— 属于它的架构发明，不是从我们的数据推出来的。可以当**候选思路**记录，不能当工作量估算。

（会话末尾有一段乱码 `屯BoJZ=…`，是它输出的编码残留，不引用。）

## 五、我更新后的行动排序（替换上一轮那版）

```
⑥ 窄 SIZEDIAG（1 轮）：销掉335×2 / 475212×1 + 按 per-subgraph/op-type 量化 rank-5 影响面
        └─ 顺带产出「驱动能力包络」：哪些 op 类型 × 哪些 rank 能正确 take-over（筛选载体的清单）
① ★ 由你拍板：④ 的对象是「驱动在契约内正确」还是「这个模型必须上 NPU」
        （两份外部评审都独立指到这一点；不定，后面全是空转）
② ★ 字节级可行性探针（新增，最便宜）：最小 rank-4 int8 用例能否 byte-exact
③ 若② 通过 ⇒ 按能力包络筛候选载体（公开 int8 模型，MobileNetV2/V1 方向）
④ guard 只作为 correctness 兜底（model-level early-out），不计接管数
```

## 六、状态

- 本轮**零 CI、零上机、仓库零改动**；新工具在 `D:\clawtemp\o2tf-src`（`ask_model.py`、`clone_packet.py`、`dwarf_contract_check.py`、`make_ag_msg_ascii.py` 等）。
- 记忆已更新：`preferences.md`（4 条分派规则）、`environment.md`（Antigravity ASCII 限制、pyelftools/WSL 备注、GBK/BOM 老坑）。
- 判据不变：① ✅ ② ✅（153）③ ❌ ④ ❌ —— 本模型你不写可用结论，我也不写。

要不要我现在就把 **DWARF 契约检查器**落到仓库 `oes-teflon/tools/`？它把"契约只有 4 槽、无 rank"这条**根因级事实**固化成可重跑的检查——换 Mesa 版本时它会立刻告诉你契约变了没有。

[2026-10-19 01:38] IDLE msg_151cf05a50016qeFhNMURPeCFU succeeded

