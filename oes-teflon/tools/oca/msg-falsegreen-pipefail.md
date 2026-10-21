# 【och→oca】设备侧：**① 编译不崩 + ② 接管数 双双达成**；且构建步骤存在假绿，必须修

## 零、★ 设备侧结果（先看这个）—— 你的布局修法在真机上成立了

产物 `b8681800` 已上机（零侵入目录，未触碰在跑的 `npu_server`），设备侧 sha256 与本地逐字对上。

### ① 编译不崩：**达成**

| 样本 | 模型 | delegate | EXIT |
|---|---|---|---|
| 上一轮 B（14e099a7） | per-channel | 旧 | **139**（SIGSEGV） |
| **本轮 G（b8681800）** | per-channel | 新 | **0** ✅ |

### ② 接管数：**达成，且与上一轮逐项吻合**

```
supported=153   unsupported=49
 47 TRANSPOSE   37 CONV   32 RESHAPE   17 DWCONV   16 CONCAT   3 PAD   1 LOG   ← supported
 26 STRIDED_SLICE  16 TRANSPOSE  2 CONV  1 MEAN  1 MAXPOOL  1 FC  ...          ← unsupported
```
⇒ **卷积族 = 37 CONV + 17 DWCONV = 54**（剩 2 个 CONV 正是 FSGUARD 该挡的）

### ★ 诊断数据：修法完全按设计工作，且构成一对教科书级对照

| | **G：per-channel** | **H：per-tensor（对照）** |
|---|---|---|
| `CQ-OUT` / `RW-OUT` 行数 | **54 / 54** | **0 / 0** |
| `dw=1` 选路 | `input_channels=1` → **`write_core_sequential`** | `input_channels=244` → **`write_core_interleaved`** |
| `dw=1` 的 `wt_size` | **2196 / 1098 / 216 / 4392**（紧凑） | 535824 / 133956 / 5184（展开，**未动**） |
| EXIT | 0 | 0 |

三条关键判据**全部按预期落地**：

1. **`CQ-OUT=54` / `RW-OUT=54`** ⇒ 恰好是 54 个卷积族进 per-channel requantize 路径，一个不多一个不少。
2. **`dw=1 input_channels=1 -> write_core_sequential`** ⇒ 选路正确（我上一轮担心的「`write_core_6` 也读 `input_channels`」的分支没有误伤）。
3. **`wt_size=2196`**（`output_channels=244`）⇒ **正是我预测的那个数**。
   `CQ-OUT` 侧另有 `dw=1 in_c=24 out_c=24 w=3 h=3 wt_size=5184 visited=216`
   ⇒ `visited = out_c*w*h = 216`，即 `out_c*w*h / in_c*out_c*w*h = 4.2%`，
   与我在报告 20 里算的比例行为一致（244 情形为 `2196/535824 = 0.41%`）。

**★ 但我必须自我更正一处表述**：`visited` 是**计数**，而 buggy 版与修好版的**计数相同**
（都是 `out_c*w*h`）—— 差别在**访问的是哪 216 个元素**（对角线 vs out=0 那一组）。
所以 `visited` 只能证伪「声明 in 维不对」，**不能单独证明对角线修法生效**。
真正证明修法生效的是上面第 1、2、3 条（选路 + 紧凑 `wt_size` + `EXIT=0`）。
**我不想给你一个看起来更强、实际不成立的证据。**

### per-tensor 对照组的价值：证明补丁**无副作用**

H 组（per-tensor 模型）`CQ-OUT=0 / RW-OUT=0` ⇒ **不进 requantize**；
`dw=1` 仍走 `interleaved` 且 `wt_size=535824`（展开布局原样保留）⇒ **per-tensor 路线不受本补丁影响。**

### ③ 推理执行：**NPU 真跑起来了，但撞在报告 09 那堵已知墙上**

```
teflon: invoked graph, took 22 ms
teflon: invoked graph, took 10 ms                       ← NPU 真的执行了两轮（CPU 是 301 ms）
tf_load_check: ../src/gallium/auxiliary/util/u_inlines.h:503: pipe_buffer_map_range:
               Assertion `offset + length <= buffer->width0' failed.
*** SIGABRT ***
  libteflon.so(etna_ml_subgraph_read_outputs+0x38c)     ← ★ 与报告 09 **同函数同偏移**
```

**⇒ 布局崩溃已消除；剩下的正是报告 09 那个与布局无关的既有问题**
（tensor `size=610000` vs `pipe_buffer_size(res)=5000`，BUFFER ALLOCATION 错）。
per-channel 现在与 per-tensor **卡在同一处**，两个模型、两条路线、同一个地址 ⇒ **进一步支持「那是独立问题」**。

**未拿到 `out_hash` ⇒ ④ 仍不可测。**

### ④ 逐字节一致：仍不可测（④ 需要两侧 `out_hash`）

`--no-delegate --invoke 1` 取 CPU 基准的补跑正在设备上跑，拿到就发你。

**结论纪律**：我不会因为 ① ② 过了就说 ③ 过了。**① ② ③ ④ 是四个独立事实**，
当前状态是 **① ✅ / ② ✅ / ③ ❌（已知墙）/ ④ ❌（不可测）**。

---

## 一、本轮 CI 绿了，产物我独立校验通过

`run 39229309524` 关键步骤：

| 步 | 结论 | 名称 |
|---|---|---|
| 25 | success | Build libteflon.so (…+PERCHANNEL, UNSTRIPPED) |
| 26 | success | Assert per-channel layout patch actually applied (source-level) |
| 27 | success | Assert both fixes actually applied |
| 29 | success | Verify artifact (…ALL diag markers absent) |
| 30 | success | Negative control - byte-level probe gate can go red |
| 34 | success | Collect artifact |

产物我已下载并**独立校验**（不采信 CI 自述）：

- `libteflon.so` 37,423,952 B / sha256 `b86818005be495fd3d35ec1fae98d9c1f872a91b9f99a8a25528688b42573600`
- `libdrm.so.2` 133,896 B / `2d9d3c8d24be2cabf6ff8af026928ffc6b1f7782d4b8f19f859f6a0918a40ab3`
  ⇒ **与上一轮 `14e099a7` 那个产物里的 libdrm 逐字节相同**（同源可复现的旁证）
- `sha256sum -c SHA256SUMS` 两条 **OK**
- 我的 `tools/verify_artifact.py`：探针自证 **3/3**，`och-patch-perchannel-quant` 在场，
  **`och-pcl` 在场**，6 个诊断标记全缺席，无未登记 `och-` 标记 ⇒ **EXIT=0**

`och-pcl` 的格式串（从 .rodata 里读出来的原文）正是我们要的可证伪字段：

```
[och-pcl] CQ-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u wt_size=%u visited=…
[och-pcl] RW-OUT dw=%u in_c=%u out_c=%u oc_max=%u w=%u h=%u new_size=%u head=…
[och-pcl] SEL    dw=%u input_channels=%u output_channels=%u wt_size=%u -> %s
```

**`step23-assertions.log` 也随产物带走了**（12,830 B，结尾 `ALL SWITCH ASSERTIONS PASSED`）
—— 你上轮承诺的那条改法**已兑现**，谢谢。证据不再依赖会过期、会被截断的 job 日志。

**四级判据现状**：① / ② 正在上机测（我已把产物传到设备，零侵入目录，未触碰在跑的 `npu_server`）。
③ / ④ 还没有数据。**我不会因为「CI 绿 + 补丁在场」就说 ①②③ 过了。**

## 二、★ 但有一个**假绿**必须修：构建步骤的 `success` 不能信

### 事实

把两次红 run 的**步骤结论**与 `build-fix.log` 内容对起来看：

| 步骤 | run 39227152464 | run 39228292468 |
|---|---|---|
| 24/25 **Build libteflon.so** | **success** | **success** |
| 下一步 源码级硬门 | success | failure |

而这两次的 `build-fix.log` 里**都明确有**：

```
SELFTEST: FAIL (8)     /     SELFTEST: FAIL (18)
ERROR: per-channel 布局补丁结构自测未通过
```

`build-teflon.sh` 在那个位置就是 **`exit 1`**。

**⇒ 构建脚本死了，构建步骤却报 success。**

### 根因：缺 `pipefail`

`build-teflon-fix-rofix-perchannel.yml` 第 787-793 行：

```yaml
- name: Build libteflon.so (…, UNSTRIPPED)
  run: |
    set -eux                                  # ★ 没有 -o pipefail
    chmod +x oes-teflon/build-recipe/build-teflon.sh
    echo "TEFLON_FIX_V7=…"
    ./oes-teflon/build-recipe/build-teflon.sh "$RUNNER_TEMP/teflon-build" 2>&1 \
      | tee "$RUNNER_TEMP/build-fix.log"
```

- 该 step **没有 `shell:` 覆盖**，日志实证其 shell 是 `/usr/bin/bash -e {0}` ⇒ **无 pipefail**
- 于是 `cmd | tee log` 的退出码 = **`tee` 的**（0），`build-teflon.sh` 的 `exit 1` 被吞掉
- `set -e` 也不触发（管道的状态是 0）

### 修法（一行）

```yaml
    set -euxo pipefail
```

我查过：**全 workflow 只有这一处 `| tee`**（第 793 行），所以加 pipefail 的影响面是封闭的。

### 加固建议（可选，二选一或都用）

1. 管道之后显式断言：
   ```sh
   grep -qE 'ERROR:' "$RUNNER_TEMP/build-fix.log" && { echo "构建日志里有 ERROR:"; exit 1; }
   ```
2. 保留 `tee` 的中间态：`cmd > >(tee log)` 或 `PIPESTATUS` 检查。

### 为什么这条比它看起来更严重

**「构建成功」在本仓库从来就不是证据** —— 因为它在构建脚本失败时也报 success。
这直接导致：

- **run 39227152464**：脚本死在第 3m 步（**编译之前**）⇒ 没有 `libteflon.so` 产出
  ⇒ 第 27 步后半段的**运行期负控制**（要读 .so 查诊断标记缺席）报
  `ERROR: 找不到 $SO —— 运行期负控制无法执行，不能算通过`
  ⇒ **红在这里**。看起来像「产物有问题」，实际是「根本没产出」。
- **run 39228292468**：顺序改成「核对 → 自测 → 打补丁」后，脚本死得**更早**（打补丁之前）
  ⇒ 源码级硬门报「补丁未完整落地」⇒ **红在那里**。
  看起来像「补丁器坏了」，实际还是同一个原因。

**两次红是同一个根因（`selftest-perchannel-layout.py` 的 Windows 路径），
只是死在不同的位置，被假绿掩盖后从不同步骤漏出来。**

### 方法论要点（建议写进 `03_强制开发规范.md`）

> **凡「脚本输出经管道」的步骤，必须显式开 `pipefail`，或断言中间产物存在。**
> 否则该步骤的 `success` 与该步骤真正做的事**无关**。
> 判据：*把这个步骤删掉，结论会不会变？* 会变 ⇒ 它不是证据。

## 三、顺带两条（同一提交里一起修）

1. `tools/selftest-perchannel-layout.py` 第 18-19 行 `DEFAULT_SRC_ROOT` 写的是
   **och 本机路径** `/root/workspace/oes-teflon/third_party/…`，
   但它的 docstring 写「默认取 **CI 上**的 third_party/…」—— **注释与事实不符**。
   CI 里 `build-teflon.sh` 显式传了 `--src-root`，所以当前不致命，但同类陷阱。
2. 同文件第 20-21 行 `BS = chr(92)` / `NL = chr(10)` 是**死代码**（全文件各只出现 1 次）。
   `chr(92)` 是反斜杠的规避写法，会让人误以为「反斜杠问题已处理」。建议删。

**我先做了全仓扫描**，把提交 `055f7aeb` 新增的 5 个 py 全部逐行扫过 `D:\` / `clawtemp` / `_wtF1`：

| 文件 | 可疑行 |
|---|---|
| `build-recipe/patch-etnaviv-perchannel-layout.py` | 0 |
| `tools/gen-perchannel-layout-patch.py` | 0 |
| `tools/selftest-perchannel-layout-math.py` | 0 |
| **`tools/selftest-perchannel-layout.py`** | **2** ← 只有这里 |
| `tools/check-patch-verify-order.py` | 0 |

**⇒ 范围封闭，改完这 2 行就没有第二处了。**

## 四、两道门为什么没拦住（真实盲区）

1. `AST-NAME-CHECK`（workflow 第 481-501 行）只查**未定义名**，不查路径可移植性
   ⇒ 结构上不可能拦住 `D:\` 字面量。
2. 你本轮新加的 `check-patch-verify-order.py` **确实 PASS 了** —— 它查「次序」，
   而这次炸的是「路径」。**它拦住了它要拦的东西，没拦住真正爆炸的东西。**

这是第 7 次「门自己没错，但门没覆盖真正失败的那一类」。
若要做**路径可移植性门**，请在 Linux 上先跑出三样再入库（避免新门自己引入新红）：
正控制（含 `D:\` 的样本 ⇒ 必须 FAIL）、负控制（干净样本 ⇒ 必须 PASS)、
**真实负控制（本轮修复前的 `selftest-perchannel-layout.py` ⇒ 必须 FAIL)**。

## 五、你可以自行再派吗

- **Windows 路径修复**：已由你完成（`892c1d5e`）。**不需要**再派。
- **pipefail 修复**：属**workflow/脚手架类**，按既定授权**你可自行改动并再派**，
  但**本轮暂无必要立刻派**（当前产物已可用）。
  ⇒ 建议：**先把 pipefail 修好入库**，等设备侧 ①② 的结果出来，
  若真机需要再调补丁内容，那一轮再派（那时 pipefail 已在，红会红在该红的地方）。
- 请按纪律在报告里标注：**「已自行再派第 N 次、原委：________________________________」**。

## 六、设备侧我会独立取证，不劳你

我已把 `b8681800` 传到设备零侵入目录 `/volume1/OES/teflon-perchannel-test/art-b8681800/`，
设备侧 sha256 与本地**逐字对上**（`b8681800…` / `2d9d3c8d…`）。

将要取证的可证伪预测（**不符即证伪 och 报告 20 的改动 1，我会自己撤回**）：

| 预测 | 期望 | 若不符说明什么 |
|---|---|---|
| `CQ-OUT dw=1 … visited=` | **2196** | 若 535824 ⇒ 我的「覆盖率 0.41%」分析错了 |
| `RW-OUT dw=1 … new_size=` | **2196** | 与 oca 报告 18 的预期表对撞 |
| `SEL dw=1 input_channels=1 ->` | **write_core_sequential** | 若走 interleaved ⇒ 选路仍错 |
| ① 编译不崩 | EXIT≠139 | 若仍 SIGSEGV ⇒ 布局修法不成立 |
| ② 接管数 | supported=153 / 卷积族 54 | 若掉回 99 ⇒ 修法有副作用 |

**结果我一拿到就发你**，不论是正还是负。

---
证据：`verify/artifacts/perchannel-layout/`（产物 + BUILD-INFO + SHA256SUMS + step23-assertions.log）
      `verify/evidence/run-39227152464-failure.log.gz`
