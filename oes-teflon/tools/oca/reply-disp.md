[2026-10-19 06:14] ASSISTANT msg_152cc1e10001w6tunvCuUNmSUR (space-bunny-free)
# 【oca→och】已 dispatch，但**失败了**；同时我修掉 3 个会再红的阻塞点

**run `39207851035`** → <https://github.com/EasongChung/rk_syno_pack/actions/runs/39207851035>

**8 道 Preflight 全过**，红在**第 15 步**，而且是我自己写的那道门。

## ★ 一、dispatch 失败：文档门缺 3 处字面量

`Static check - PERCHANNEL docs consistency` 要求头部与 3m 段都出现：

| 要求 | 头部实际 |
|---|---|
| `patch-etnaviv-perchannel-quant.py` | 只有生成器 `tools/gen-...`，**没有补丁器本身** |
| `input/output 仍严格` | 写的是「input/output 只读**标量** tensor->scale」，**措辞不同** |
| `默认关闭` | **完全没写** |

⇒ **处置：补文档，不放宽门。** 这三句别的开关在头部一直写着，头部漏了本身就是缺陷。补完该门本地 PASS。

★ 这是 §2.4 的**第三例**：我 dispatch 前只验了那个步骤**存在**，没验它**能不能过**。

## ★ 二、我这轮引入的真 bug（AST 门抓到的）

`tflite_fix_padv2.py:229` 用了 `os.path` 却**没 `import os`**——那是我上一轮加 PadV2 门时写的代码，**运行时必然 `NameError`**。已修。**AST 门立功。**

## ★ 三、AST 门的「动态全局」判据**只打进了 2 个 workflow**

`gen-creatediag-patch.py` 有 `globals()["_EQ_PAIRS"]`，旧版 `defined()` 会把它的**消费者**误判成 undefined。这个问题**之前修过一次**（run 39145823808），但只改了 2 份副本，另外 5 个 workflow 里还是旧的 ⇒ 潜伏至今被翻出来。已补齐全部 5 份。

**「同一处修复没覆盖全部副本」——和我派生 workflow 时犯的是同一类错。**

## 四、`anchor-check` 缺 numpy

run `39207812478` 第 11 步 `ModuleNotFoundError: numpy` —— **ubuntu-latest 的系统 python3 没有 numpy**（本机有，所以本地全绿）。已加 `setup-python` + pin `numpy<3` / `tflite==2.18.0`。

★ 同一批 run 里 **7 道锚点门全部 PASS**（含 sparse clone、`--set=maximal`、负控制、逐字锚点核对）——主体是好的，只挂环境。

## 五、★ 我该提前做而没做的

`run-embedded-gates.py`——把各 workflow 里的内嵌静态门**抽出来本地真跑**。那些门只读 `build-teflon.sh`，不需要 Mesa/编译/TF，**本来就该常设跑**。

**这个工具在第一次 dispatch 之前就该写**：成本不到半小时，能省掉两次失败的 CI。已接进 `anchor-check.yml`，本地 25 个内嵌门全过。

## 六、本轮我的 4 个错

1. ★ `tflite_fix_padv2.py` 缺 `import os`（这轮引入）
2. 派生前只改开关没改断言 → 4 处必红 + 一处自相矛盾
3. 只验文档门「存在」没验「能过」→ dispatch 红灯
4. 新工具第一版按 job 名跳过整个 `build` job（一个门没跑到）；第二版 sparse 目录与补丁目标**基准不同**地对撞 ⇒ 12 个全误判，而**负控制恒真、selftest 照样 PASS** ⇒ 补了「抽之前是全覆盖」的正控制

## 七、要你两件事

1. **能否再 dispatch 一次？** 授权是「仅此一次」，这次已用掉且失败、无产物。我**没有自行再派**。
2. `anchor-check.yml` 会在本次 push 后自动触发（paths 命中）。要我等它跑完再派，还是先派 perchannel？

## 八、状态

- 本地 9 组门全绿；**所有修复都只在本地验过**
- 产物**未到手** ⇒ 「② 接管数恢复」这个数**还没有**
- per-channel 精度损失**仍无数据** ⇒ 门 d/e/f 未定，不写可用结论

判据：① ✅ ② ✅ ③ ✅ ④ ❌ —— 不变。

判据预期不变：**这轮只回答「接管数能不能恢复」**，③④ 不因它达成。

[2026-10-19 06:14] IDLE msg_152cc40f0001L5hBNRQjMWUfsV succeeded

