[2026-10-19 00:52] ASSISTANT msg_151a557c5001hED4comiDx7PWx (space-bunny-free)
已发出，等待浏览器侧握手与回复。同时把这一轮的复核结论落盘（供下一轮引用）：

**独立复核结果（我这边，不用你的转述）**

| 你的论断 | 我的复核 | 手段 |
|---|---|---|
| `unsigned dims[4]`、byte_size=64、decl_line=1014 | ✅ **完全一致** | pyelftools 解析 `art/libteflon.so` 的 DWARF |
| 契约无 `dims_size`/`rank`/`num_dims` | ✅ **零命中** | 同上，逐成员枚举 `pipe_tensor` / `pipe_ml_operation` |
| ⇒ D1 否证 | ✅ **成立** | `dims` 在 +12、下一成员 `scale` 在 +28 ⇒ 16 字节 = 4×4B（偏移算术即证，不依赖我那个类型串渲染） |

**两点我要更正/补充**

1. **D2 你否证的不是我说的那处。** 我的 D2 指 `etnaviv_ml.c:335-372`（`lower_operations` 里 `input_tensors[i] = transpose->output_tensors[0]` 换成新分配张量）与各 op 分支的 `operation->input_tensors[i] = input_tensors[i]`（新）对 `..._sizes[i] = poperation->input_tensors[i]->dims[1..3]`（旧）。你读的是 `etnaviv_ml_nn.c:377 transpose()`（权重重排，不碰 input_tensors）——那是另一个函数。**不过我同意它不解释任何观测**：转置不改变元素数，所以"拿旧张量的 size 填新张量"在 size公式正确时无害。它作为一个代码气味仍在，且**可能正好解释那 3 次未解释的创建**（`335`×2、`475212`×1）——建议留作次要线索，不要当已否证。`etna_ml_lower_transpose` 的实现我本地仍缺（`mesa_src` 是局部拷贝），这条保持"未验证"。

2. **rank-5 在现有契约下**不是**直接可检测的**——没有 rank 字段。但我看到一个可行替代判据：`data == NULL` 是驱动需要分配的激活张量（代码里对 `data != NULL` 直接 `continue`，那是权重/常量），而权重张量的 `dims[0]` 合法地是输出通道数（24/122/488）。所以 **`data == NULL && dims[0] != 1`**（batch 不是 1）就是 rank-5 被压进 4 槽的签名。这条需要用真机数据验证（哪些 idx命中、是否有合法 rank-4 激活误命中），我准备把它作为 R5GUARD 的判据起点，而不是"检测 rank-5"这个做不到的事。

**另外提醒一个你估计里的风险**：153 → 120 这种"缓降"通常是乐观的。rank-5 激活若出现在图中间，其下游全部被迫回退 CPU，接管数更可能是**断崖**。建议先量一个数：**多少个 partition 含 rank-5 张量、以及它们在图中的位置**，再决定投入。

Muse 的回复到后我会先做 `validate result` + `pair` 配对校验，再逐条对照我的判断（外部意见按"建议"处理，能落到可判别测量的才采纳），然后再发第二个问题给 Gemini。

[2026-10-19 00:52] IDLE msg_151a573620012xbXGfLI6OsNp6 succeeded

