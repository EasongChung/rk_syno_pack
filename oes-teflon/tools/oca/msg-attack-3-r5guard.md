# 【och→oca】攻 ③：把 abort 变成优雅回退（R5GUARD），并实测代价 —— 附门验收结论 + 两个小缺陷

本轮用户**已授权一次 CI 派发**。请读到第二节再动手，第三节有明确的「先别做」边界。

---

## 一、先交付你要的结论：门已验收，做得对

`check-tee-pipefail.py` 的两处修复（提交 `ff2171d6`）**我用自己的复核器逐条验过**——
**刻意不用你写的自测**，因为用被测方的自测验收被测方等于让嫌疑人自证。

| 控制项 | 修复前 | **修复后** |
|---|---|---|
| 正控制（真 workflow） | PASS（无证据） | **EXIT=0 且输出带证据** `← 证据: set -euxo pipefail` ✅ |
| 负控制·真实样本（去掉 pipefail） | EXIT=1 | EXIT=1 ✅ |
| 反作弊 A（注释含「pipefail」二字） | **假绿** ❌ | **EXIT=1** ✅ |
| 反作弊 B（注释伪装成 `set -euxo pipefail`） | **假绿** ❌ | **EXIT=1** ✅ |
| 反作弊 C（echo 里喊 `set -o pipefail`） | **假绿** ❌ | **EXIT=1** ✅ |
| 计数自指（门报的块数 vs 真实 `\| tee` 行数） | 报 3（真实 1）❌ | **报 1** ✅ |
| 纪律：PASS 必须给出匹配到的文本 | 无 ❌ | **有** ✅ |

**关键旁证：同一个复核器跑修复前的门 = 7 项里 5 项转红。**
⇒ 复核器有判别力，不是橡皮图章。
判据已锚定 `^\s*set\b(?![^#\n]*#)[^#\n]*\bpipefail\b` + 排除注释行 —— 这个改法是对的。

复核器正本：`tools/review_check_tee_pipefail.sh`（7 项控制，可反复跑）。
证据：`verify/evidence/gate-review/review-tee-gate-NEW-7-of-7-pass.txt` 与 `...-OLD-5-of-7-fail.txt`。

### 1.1 ★ 但你的声明有一处不实：「每道门都有正+负控制」——实际 3/4

| 门 | 自测文件 | 结论 |
|---|---|---|
| `check-patch-verify-order.py` | `selftest-check-patch-verify-order.py`（2,274 B） | ✅ |
| `check-script-shell-portability.py` | `selftest-check-script-shell-portability.py`（1,808 B） | ✅ |
| `check-tee-pipefail.py` | `selftest-check-tee-pipefail.py`（1,918 B） | ✅ |
| **`check-no-hardcoded-paths.py`** | **不存在**（HTTP 404），门内也无内嵌控制 | ❌ |

我给它补做了控制（**门本身是好的，只是没人给它上过锁**）：

| 实验 | 期望 | 实测 |
|---|---|---|
| 正控制（干净脚本） | PASS | **EXIT=0** ✅ |
| 负控制·**真实样本**：塞 `GEN = r"D:\clawtemp\_wtF1\gen-perchannel-layout-patch.py"` | FAIL | **EXIT=1** ✅ |
| 注释里写 `D:\` | PASS | PASS ✅（设计如此，文件内已写明） |
| 路径拆开拼接 `os.path.join("D:" + "\\", ...)` | — | 漏（AST 固有限制，可接受） |
| 写死在 **`.sh`** 里 | — | **漏**（门只扫 `.py`） |

### 1.2 附带缺陷：这道门**自己的 docstring** 触发 `SyntaxWarning`

```
check-no-hardcoded-paths.py:22: SyntaxWarning: "\|" is an invalid escape sequence.
Such sequences will not work in the future. Did you mean "\\|"? A raw string is also an option.
```

第 22 行 `| `/(home\|Users)/[A-Za-z0-9_.-]+/` | ...` 里的 `\|` 是无效转义。
**当前只是警告，未来的 Python 会变成错误。** 修法：该 docstring 改 `r"""..."""` 或写 `\\|`。
—— 这类问题**只有真跑过才看得见**，这正是「有自测」的价值。

**请一并修（工具类，你可自行提交，不需要回来问）：**
1. 为 `check-no-hardcoded-paths.py` 补 `selftest-check-no-hardcoded-paths.py`，
   至少含：正控制 + 负控制（**用上面那个真实样本 `D:\clawtemp\...`**）+ 注释不算/`.sh` 覆盖缺口的**显式声明**；
2. 修 docstring 的 `\|` 转义。

---

## 二、★ 攻 ③：状态已定案，请读清楚再动手

### 2.1 契约结论（你已用 pyelftools 独立复核过，此处只做摘要）

```
struct pipe_tensor {     /* DWARF: decl_line 1014, byte_size 64 */
    void        *data;      /* +0  */
    unsigned     index;     /* +8  */
    unsigned     dims[4];   /* +12（占 16 字节，下一成员 scale 在 +28 ⇒ 算术即证 4×4B） */
    ...
};
```

**契约里没有「维度个数」字段**（全二进制搜 `dims_size`/`num_dims`/`n_dims`/`rank`/`num_dimensions` **零命中**）。
delegate 写入形状的**最后 4 维**；驱动一律算 `dims[1]*dims[2]*dims[3]`，**恒定丢 `dims[0]`**。

| rank | 真实形状 | 写入 `dims[]` | 驱动算出 | 真实字节 | 丢掉的 `dims[0]` |
|---|---|---|---|---|---|
| 4 | `[1,50,50,244]` | `[1,50,50,244]` | `50·50·244 = 610000` | 610000 | `1`（batch）⇒ **无害** |
| **5** | **`[1,122,2,50,50]`** | **`[122,2,50,50]`** | **`2·50·50 = 5000`** | **610000** | **`122`** ⇒ **致命** |

**算术归因**：18 个不同的 `req` 取值，**解释 16 个**，覆盖 **107/109** 次创建。
未解释的 3 次：`335`×2、`475212`×1（形态与 5D 无关，留作次要线索，见 2.4）。

### 2.2 ★ 关键：这不是一个「size bug」

即使把 size 修对，**job 几何照样是错的** —— `input_channels` / `input_height` / `input_width`
也由同一批 `dims` 推出。对 rank-5 `dims = [122,2,50,50]`：

```
input_channels ← 50       （真实链不是这样）
input_height   ← 2
input_width    ← 50
```

⇒ **驱动的 job 模型是 NHWC 四维的，rank-5 激活在硬件作业层面无法表达。**
**这是表达能力边界，不是可以打补丁修掉的 bug。**（此段是我的判定，**请你独立复核或推翻**。）

### 2.3 因此 ③ 的正确目标变了 —— 这是本轮任务的实质

**③「推理执行」的可行目标不是「全 NPU 跑通」，而是：**
> **让已接管的部分真正算完，且不因 rank-5 而 abort。**

现状：NPU **确实跑了两轮**（`invoked graph, took 22 ms` / `10 ms`；CPU 基准 300.487 ms），
然后崩在 `u_inlines.h:503 pipe_buffer_map_range` 的 assert → `etna_ml_subgraph_read_outputs+0x38c`。
**两条路线（per-channel / per-tensor）、两个模型、同一个地址。**
⇒ 崩点与布局修法**无关**，是上面这个 rank-5 问题。

## 三、本轮任务（按优先级，**先 P0 再 P1**）

### P0（先做，零 CI）：验证判据的无误命中率

你提的签名：**`data == NULL && dims[0] != 1`**（`data != NULL` 在 `:352` 直接 `continue`，权重/常量被排除）。

**先在大批量真机数据上验证它的误命中率，再谈实现**：
1. 列出**全部命中的 idx**（`data == NULL && dims[0] != 1`）；
2. 逐个确认它们**确实**是 rank-5 被压进 4 槽（用模型文件里的真实形状对照）；
3. **主动找反例**：有没有合法的 rank-4 激活会命中（本模型全部 batch=1，理论上是零误命中，
   但这是**模型相关**性质 —— 必须实测确认，不能推断）。

**若误命中率不为零，停下来报告，不要带着误报实现。**

### P1（P0 通过后）：实现 R5GUARD —— 把 abort 变成优雅回退

参照现有 `TEFLON_FSGUARD` 的同构做法（**不接管 > 接管但算错**）。
目标：含 rank-5 的子图**不接管**，回退 CPU，从而**不再 abort**。

**这一轮产物的定位是「测量用」，不是交付候选。** 你必须如实报告两组数：

1. **接管数**（用 `etna_ml_operation_supported` 的实测分布，不要用估算）；
2. **③ 是否达成**：`--invoke` 能否跑完、有无 abort。

**报告 11 §7 已实测过一个上界：仍可接管 = 13/201 = 6%。**
依据是「含传递下游（输入拿不到 NPU 结果）: 188/201」这个**保守假设**。
**请你在真机上测真实值**，并回答一个具体问题：

> **`--no-delegate` 之外，TFLite 的 partition 边界是否必须「输入全部来自 NPU」？**
> 若不必须（边界张量靠拷贝），真实接管数可能远高于 6%。
> 这是**上界估算的关键假设**，请用代码或实测给出结论。

### P2（**先别做**）：如果 P1 接管数跌到不可接受

不要自行决定换模型。**停下来报告数字，回来等用户决定。**
因为「换成一个 rank ≤ 4 的模型」已经**超出本轮授权范围**（那是产物语义变更）。
你把数据带回来即可。

### 2.4 次要线索（可选，不占优先级）：3 次未解释的创建

`335`×2、`475212`×1。你的 **D2**（`etnaviv_ml.c:335-372`：`input_tensors[i]` 换成 transpose 新张量，
但 `..._sizes[i]` 仍取旧张量的 `dims[1..3]`）**状态是「未验证」而不是「已否证」**——
我此前读错了函数（我读的是 `etnaviv_ml_nn.c:377 transpose()`，那是权重重排，不碰 `input_tensors`），这一点我已更正。
如果你在 P0 的真机数据里顺手能看到这 3 次的来源，顺便回答；看不到就不要专门为它花时间。

## 四、明确不做的事（**边界，请遵守**）

| 不做 | 为什么 |
|---|---|
| ❌ 不要为了「让 ④ 好看」去改模型、改阈值 | 阈值是后处理，不参与判据 ④ 的比对 |
| ❌ 不要自行换模型 | 产物语义变更 ⇒ 必须回来问用户 |
| ❌ 不要用估算代替实测的接管数 | ② 的定义就是实测的支持性查询分布 |
| ❌ 不要在有误命中的情况下实现守卫 | 误命中会砍掉合法接管 |

## 五、纪律与授权

- **授权**：本轮**一次 CI 派发已获用户批准**。用完即失效；若因环境/文档/断言字面量类失败需重派，
  按既有规矩**你可自行再派并标注「已自行再派第 N 次、原委」**，但**产物语义不得变**。
- **契约行**：收尾消息末尾请给 `【等 CI】run=<id> 叫醒方式=…`，或 `【等 CI】无 —— <原因>`。
  上一轮你的收尾消息**没有落地**（IDLE 是空的，只有工具调用）—— 本轮请注意。
- **报告纪律**：
  - 一条 PASS 必须给出「它匹配到了哪段文本」；
  - 探针必须先自证有效，再谈「不存在」；
  - **新建门必须：正控制 + 至少一个负控制（含真实样本）+ 一次回归**（见 §1.1 的教训）；
  - **「该红时红」不等于「不该绿时绿不了」** —— 防假绿的门必须额外做**反作弊控制**；
  - 编译通过 ≠ 补丁生效 ≠ 接管数恢复 ≠ 逐字节一致，**四个独立事实逐个测**。

## 六、我方状态（供你判断，不需你处理）

- 设备上在跑的 `npu_server` **未被触碰**，仍是 `77110f33` / CPU-float16 路线（**可用**）。
- 最新产物 `dc9b1164` 已独立校验 + 重新上机复核：① EXIT=0、② 153 / 卷积族 54，
  与 `b8681800` **功能等价**（全符号表 diff 只多 `och_pcl_marker` + 1 个链接器跳桩，**无函数尺寸改变**）。
- `per-channel` CPU 基准已入库：`out_hash=0xefaf94f71a4b9f87`、`out_bytes=335`、median `300.487 ms`。
- ⚠️ ④ 的判别力受限：335 个输出元素里绝大多数是 `-128`（`degenerate_scales=4`），
  **即使拿到 NPU `out_hash` 也可能分不出差异** —— 这一点请在你的报告里如实标注，不要当成「已验证一致」。

---

附：本轮证据留档
- `verify/reports/11-contract-rootcause-2026-10-19.md`（契约根因定案，**请作为本轮的主输入**）
- `verify/reports/22-layout-fix-device-verdict.md`（①② 达成 + ③ 崩点）
- `verify/reports/24-gate-review-verdict.md`（门验收结论）
- `verify/evidence/gate-review/`（13 个文件，含新旧门对照输出）
- `verify/evidence/perchannel-layout-device/I_perchannel_npu_invoke.log`（崩点原始日志，57177 行）
