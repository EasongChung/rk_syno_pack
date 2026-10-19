# 21. per-channel 布局契约的**更正**与实现（含对报告 18 §4 的推翻）

> 前置：[18-perchannel-layout-contract.md](18-perchannel-layout-contract.md)
> 本报告**推翻 18 号报告的 §4**，并更正 och 的两处引用错误。
> 每条结论都带源码行号或可复跑的门编号，可独立复核。

## 0. 三句话

1. **18 号 §4 是错的**：非 depthwise 的 per-channel 卷积**原本就没有错**（§1）。
2. 真正的错**只在 depthwise**，且有**两重**：`calc_quantization` 只覆盖 0.41%，
   `requantize_weights` 把缓冲区压成 1/244（§2）。
3. och 的「1×1 不受影响」只对**非 depthwise** 成立；depthwise 1×1 仍然受影响（§3）。

---

## 1. ★ 推翻 18 号 §4：非 depthwise 路径经证明**无害**

18 号说「非 depthwise 时尺寸不变、只是元素错位 ⇒ 静默算错」。**这一半错了。**

两个索引函数：

| 实现 | 读/写线性下标 |
|---|---|
| 旧（声明 `[out][W][H][in]`） | `((oc*W + w)*H + h)*in + ic` |
| 新（声明 `[out][in][W][H]`） | `(oc*in + ic)*W*H + w*H + h` |

三条性质合起来 ⇒ **两者产生完全相同的输出**：

1. **都是双射**：`oc ∈ [0,out)`, `w ∈ [0,W)`, `h ∈ [0,H)`, `ic ∈ [0,in)`
   ⇒ 两式都恰好覆盖 `[0, out*W*H*in)` 且不重不漏。
2. **都是同下标读写**：`map_out[...] = f(map_in[同一坐标])`，函数内 `map_out`
   与 `map_in` 的坐标表达式逐字相同（nn.c:252/255 与 :258/260）。
3. **变换只用到 `scales[oc]`，而 `oc = p // (W*H*in)`**（两式的最外层都是
   `oc*W*H*in`）⇒ 同一下标 `p` 在两种下标下还原出**同一个** `oc`。

⇒ 置换自逆，**逐元素数值不变**。

**证据**：`tools/selftest-perchannel-layout-math.py` 用例 B，在
`out=in` 且 `W,H ∈ {(3,3),(5,3),(1,7),(3,5)}` 上断言 `旧输出 == 新输出`，全过。

★ 连带更正 18 号那句「即使 depthwise 修好，④ 逐字节一致也不可能通过」——
**理由不成立**。④ 过不了的原因仍然是 per-channel **折叠成单一 scale 的精度损失**，
与布局无关。

### 为什么 18 号会算错

因为它比的是「两个下标公式不同」，没意识到这里**同下标读写**且 `oc` 可由下标还原。
depthwise 之所以坏，恰恰是因为 `ic_n = 1` 让缓冲区**变小**、不再是双射 ——
**决定性因素是尺寸，不是排列。**

---

## 2. depthwise 的两重错（och 补的第二重是对的）

### 2.1 `calc_quantization`（nn.c:194-228）

depthwise 时 `nn.c:202-203` 把 `input_channels` 压成 1，于是只遍历
`out_c*W*H` = 2,196 个字节（占 535,824 的 **0.41%**），
读到的线性区间 `0..2195` 恰好是 **`(out=0, in=0..243)`** 那一组。
⇒ 折叠出来的 `out_scale` 只由 output-channel 0 的权重决定。

### 2.2 `requantize_weights`（nn.c:230-270）

同一行 `nn.c:239-240` 让 `new_size` 从 535,824 变成 2,196，
而 `operation->input_channels` 在 v7 上**没有被写回**（nn.c:649-650 的条件带
`nn_core_version >= 8`）⇒ `create_coeffs_v7`（v7:494）按 `in_c=244` 选
`write_core_interleaved` ⇒ 按 `oc*W*H*in` 取 244 倍 ⇒ **越界 ⇒ SIGSEGV**。

### 2.3 依赖顺序不可反

```
calc_quantization ──scale/zp──▶ requantize_weights ──缓冲区──▶ write_core_*
```

只改后两者 ⇒ 崩溃消失但 **scale 仍错** ⇒ 输出静默错误（最坏形态）。

### 2.4 `requantize_bias`（nn.c:272-293）—— **确实无需改**

只按 `oc` 一维索引，bias 本就是 `[out]` 一维。**och 的排除是对的。**

---

## 3. ★ 更正 och 的两处引用

### 3.1 「1×1 不受影响」只对非 depthwise 成立

och 的代数本身没错：`[out][in][1][1]` 与 `[out][1][1][in]` 的线性偏移恒等。
**但它证的是「布局置换等价」，不是「这一路径无害」。**

depthwise 的错不在布局，而在**声明的 in 维被写成 1**：

| 算子 | 正确位置 | 旧实现读到的位置 |
|---|---|---|
| depthwise `C=244, W=H=1` | `oc*C + oc` | `oc*1*1 + 0 = oc` |

⇒ depthwise 1×1 **仍然错**。och 若把「所有 1×1 算子输出不变」当正控制，
**会假报警**。

**证据**：数学自测用例 D（depthwise 16×1×1）断言 `旧 != 新`，通过。
**正控制请写成「非 depthwise 的 1×1 算子输出逐字节不变」**（用例 C 通过）。

### 3.2 行号归属

| 行号 | och 说 | 实际 |
|---|---|---|
| `v7:278` | `write_core_sequential` | **`write_core_interleaved`**（起于 :266） |
| `v7:391` | `write_core_interleaved` | **`write_core_sequential`**（起于 :353） |

18 号把两者写成一行，也错。三处正确归属：

| 位置 | 所属 | 声明 | 推出的布局 |
|---|---|---|---|
| `v7:226` | interleaved | `blocks = input + oc*W*H*in` | `[out][in][W][H]` |
| `v7:278` | interleaved | `(*weights_map)[in][W][H]` | `[out][in][W][H]` |
| `v7:391` | sequential | `(*weights_map)[height]`，基址 `oc*W*H` | `[out][w][h]`（**无 in 维**） |

★ 这恰好解释了机制：sequential **完全不读 `input_channels`**，
所以把 depthwise 改到它时基址**恰好**对上紧凑布局 ⇒ 崩溃消掉，
但取到的**字节内容仍错** ⇒ 静默错误。

---

## 4. 实现（`TEFLON_PERCHANNEL=1`，**不另开开关**）

补丁器：`build-recipe/patch-etnaviv-perchannel-layout.py`
（由 `tools/gen-perchannel-layout-patch.py` 从真源码生成，`OLD_*` 逐字来自源码）

| # | 改动 | 位置 |
|---|---|---|
| 1 | `calc_quantization` 按 `[out][in][W][H]` 遍历；depthwise 只取对角线 `ic == oc` | nn.c:194-228 |
| 2 | `requantize_weights` 按 `[out][in][W][H]` 读；depthwise 折成紧凑 `[out][W][H]` | nn.c:230-270 |
| 3 | 写回去掉 `nn_core_version >= 8`（v7 也要做） | nn.c:649-650 |
| 4 | `create_coeffs_v7` 两处选路各插一条 print-only 诊断 | v7:492-497 / :542-547 |

**对角线语义（och Q1）：✔ 认同**，依据有二：

- `expand_depthwise`（nn.c:342-348）把真实权重放在 `i % in_c == in_depth`
  即 `ic == oc`；
- `calc_quantization` 用 `scales[oc]` 去反量化，**per-channel 的 scale 属于
  out-channel c 的 filter**，与 `map[oc][oc]` 是唯一自洽的配对。

**非 depthwise 分支的行为变化：零**（§1 证明等价），只是把契约写明确。

**开关（och Q2）：不拆。** 门禁放开与布局修复都只在 per-channel 被准入时才走到；
拆开只会允许「开一个不启另一个」的坏组合。单一 `TEFLON_PERCHANNEL` = 单一语义。

---

## 5. 可证伪预测（上机时应逐条对上）

诊断前缀 `[och-pcl]`，开关 `TEFLON_PERCHANNEL_DIAG`（默认 1）。

| 断言 | 预期 | 若不符 |
|---|---|---|
| `CQ-OUT dw=1 visited=` | **2,196** | 若 535,824 ⇒ §2.1 被证伪 |
| `RW-OUT dw=1 new_size=` | **2,196** | 若 535,824 ⇒ §2.2 被证伪 |
| `SEL dw=1 input_channels=` | **1**，选 `write_core_sequential` | 若 244 ⇒ §4#3 没生效 |
| `CQ-OUT dw=0 visited=` | 5,184（覆盖整个缓冲区） | 若 2,196 ⇒ 非 depthwise 被误改 |

**CI 证明不了这些**（诊断是运行期的，要设备才触发）。
CI 只证明：补丁四处都落到源文件里（源码级硬门）+ 数值自测全过 + 能编过。

---

## 6. 本地已跑过的门

| 门 | 覆盖 | 结果 |
|---|---|---|
| `selftest-perchannel-layout-math.py` | 32 条断言：A depthwise 244×3×3 / B 非 depthwise 四档 / C 1×1 正控制 / D depthwise 1×1 更正 / E 边界与不变量破坏 | **PASS** |
| `selftest-perchannel-layout.py` | 24 条：锚点 `--verify`、应用、幂等拒、fail-closed、diag 0/1 两种模式 | **PASS** |
| `check-run-syntax.py` | 271 个 `run:` 块 `bash -n` | **PASS** |
| `check-log-asserts.py` | 84 条日志断言 | **PASS** |
| `run-embedded-gates.py` | 5 个内嵌门 + 43 文件 AST 名称检查 | **PASS** |

★ 数学自测在写成过程中抓到**我自己镜像里的两个真错误**：
① `transpose` 的读侧循环顺序写成 `d0,d3,d1,d2`（应为 `d0,d1,d2,d3`）；
② `ref` 的遍历顺序是 `oc,w,h,ic`，而输出缓冲区是 `oc,ic,w,h`。
**两个都会让这道门给出假绿** —— 门自己写错，是本项目栽得最多的一类。

---

## 7. 未做 / 未验证

- **未派 CI**（本轮先落地；派发需用户授权，och 称已授权）。
- **未上机**：①②③④ 判据本轮**无任何新增事实**。
- **门 e（cos ≥ 0.98 / top-1）仍无证据**：布局修好**不消除** per-axis → 单一 scale
  的折叠精度损失（upstream 原话 "with a loss of accuracy"，nn.c:638-641）。
- **非 depthwise 分支的行为等价是数学证明 + 数值自测**，不是上机实测。