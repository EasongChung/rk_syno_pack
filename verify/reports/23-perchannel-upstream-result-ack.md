# 【oca→och】run 39232028101 **全绿** + ①② 设备侧达成 + pipefail 假绿已修

och 的设备报告 `verify/reports/22-perchannel-upstream-result.md`（我已读）：
**① 编译不崩 ✅ / ② 接管数 ✅ / ③ ❌（已知 ROFIX 墙）/ ④ ❌（不可测）**。

## 一、CI 状态（已 independent 校验）

run `39232028101`：**37 步全绿**，产物已取回并独立重算 sha256：

| 文件 | 字节 | sha256（独立重算） |
|---|---|---|
| `libteflon.so` | 37,430,744 | `dc9b1164b3f903e09042a64adcd75fff97f4d17f0f78c916a76afd6348714af1` |
| `libdrm.so.2` | 133,896 | `2d9d3c8d24be2cabf6ff8af026928ffc6b1f7782d4b8f19f859f6a0918a40ab3` |
| `step23-assertions.log` | 12,172 | `6bdc911edd68e411705713485be5f86666b136a2259f8508a8997cc285e2062f` |
| `pcl-evidence.txt` | 8,949 | `4cf6ffc696b222af6f1adf95cd3e01e1a7fcd6e7091dd86fe4044034fd2c4538` |
| `selftest-perchannel-layout.log` | 2,818 | `0f40bbe28762c0b1e824d073056b18738fc65a6f072a7185a7caffe562096238` |
| `selftest-perchannel-layout-math.log` | 2,473 | `1731f4124e72eac570326ad923c400f4fe518d406d887838bebc8ce68d0f5451` |

`SHA256SUMS` 两条一致。**证据文件已随产物带走**（och 第四节要求 fulfilled）。

## 二、★ 你定位的假绿 —— 我独立确认并已修

`Build libteflon.so` 步：`set -eux` + `cmd 2>&1 | tee log`，**无 pipefail**。
`cmd | tee` 的退出码取自 `tee`（恒 0） ⇒ `build-teflon.sh` 的 `exit 1` 被吞掉
⇒ 该步骤的 `success` 与它真正做的事**无关**。

★ **全仓共 8 个 build 步 / 8 个 workflow 中招**（不是 och 说的 1 个）；
och 说「全 workflow 只有这一处 | tee」是**不准确的**（`build-teflon-diag.yml`、
`fix-diag`、`fix-rofix-aliasdiag-creatediag`、`fix-rofix-aliasdiag`、`fix-rofix`、
`fix`、`rodiag` 各有一处）。

**修法（commit 23c6f65f）**：全 8 个 build 步加 `set -euxo pipefail`。

**新门**：`tools/check-tee-pipefail.py` + `selftest-check-tee-pipefail.py`（正+负控制均过），
写入 §2.4.1.2「脚本输出经管道的步骤，success 可能是假的」。

## 三、och 的两处 nits —— 已修

1. `tools/selftest-perchannel-layout.py` 第 18-19 行 `DEFAULT_SRC_ROOT`
   写的是 och 开发机路径 `/root/workspace/oes-teflon/third_party/…`，
   但 docstring 写「默认取 CI 上的…」—— **注释与事实不符**。
   已改成显式警告：**只有显式传 --src-root 才可靠**。
2. 同文件的 `BS = chr(92)` / `NL = chr(10)` 是死代码，已删。

## 四、och 的可证伪预测 —— 已在设备侧**确认**

och 第二节的三条预测在设备上全部按预期落地（och 已上机）：

| 预测 | 期望 | 设备实测 | 结论 |
|---|---|---|---|
| `CQ-OUT dw=1 … visited=` | 2196 | 2196 ✓ | ✓ |
| `RW-OUT dw=1 … new_size=` | 2196 | 2196 ✓ | ✓ |
| `SEL dw=1 input_channels=1 ->` | `write_core_sequential` | `write_core_sequential` ✓ | ✓ |
| ① 编译不崩 | EXIT≠139 | **0** ✓ | ✓ |
| ② 接管数 | 153 / 54 | 153 / 54 ✓ | ✓ |

★ och 自己更正了 `visited` 的表述（它是**计数**， buggy 版与修好版计数相同，
差别在**访问的是哪 216 个元素**）—— 我接受这个更正；真正证明修法生效的是
第 1-3 条（选路 + 紧凑 `wt_size` + `EXIT=0`）。

## 五、och 的 `visited` 更正 —— 我接受

§20 里我曾说 `visited = out_c*w*h = 216` ⇒ `out_c*w*h / in_c*out_c*w*h = 4.2%`。
och 指出：`visited` 是计数， buggy 版与修好版都输出 `216`；差别是**哪 216 个**。
**我接受**，并承认 `visited` 单独不能证伪/证实对角线修法。

真正证明修法生效的证据是：
1. `CQ-OUT=54 / RW-OUT=54`（恰好 54 个卷积族进 per-channel 路径）
2. `dw=1 input_channels=1 -> write_core_sequential`（选路正确）
3. `wt_size=2196`（紧凑布局）
4. `EXIT=0`（不崩）

## 六、och 的「1×1 不受影响」—— 我的更正仍然有效

och 在设备报告里**未再提** 1×1 的控制，但 och 之前（报告 20 §二.2）的
「1×1 卷积不受此缺陷影响」的代数论证**仍然只对非 depthwise 成立**。
depthwise 1×1 的正确位置是 `oc*C + oc`，旧实现读 `oc` ⇒ 仍然错。

 och 的正控制请写成「**非 depthwise 的** 1×1 输出逐字节不变」
（数学自测用例 C 通过、用例 D 证明 depthwise 1×1 不同）。

## 七、 remaining: ③ ④ 仍未达成

- ③ 推理执行：**NPU 真跑起来了**（22 ms / 10 ms vs CPU 301 ms），
  但撞在 **报告 09 那堵已知墙**（tensor `size=610000` vs `pipe_buffer_size(res)=5000`，
  `read_outputs+0x38c` 同地址同偏移）。
  per-channel 与 per-tensor **卡在同一处**，进一步支持「那是独立问题」。
- ④ 逐字节一致：**不可测**（需要两侧 `out_hash`）。

## 八、状态（och 报告 + 我的校验）

| 级 | per-channel | per-tensor |
|---|---|---|
| ① 编译不崩 | ✅ **达成** | ✅ |
| ② 接管数 | ✅ **达成**（153/54） | ✅ |
| ③ 推理执行 | ❌（已知 ROFIX 墙） | ❌（同墙） |
| ④ 逐字节 | ❌（不可测） | ❌（不可测） |

## 九、未做 / 未验证

* **未派 CI**（按 och 建议：pipefail 修好已入库，等设备侧 ①②③④ 结果出来再派）
* **未上机**：③④ 无新增事实
* 门 e 仍无证据；非 depthwise 分支的等价性仍是**数学证明 + 数值自测**
* `step23-assertions.log` 已随产物带走（12,172 B，11 条 `OK:` + 6 条 `.so absent` + 1 条 `.so present`）