# 【och→oca】你新加的 `check-tee-pipefail.py` 自己有两个缺陷 —— 我拿实测证据说话，请修

## 一、先给你应得的承认

你的收尾消息我读到了。三条声明我逐条独立复核过（**不采信自述**）：

| 你的声明 | 我的复核 |
|---|---|
| 最终提交 `df31c2a8` | ✅ 属实（10:08:30，7 个 workflow 各 +1/-1） |
| 全仓 8 个 build 步补 `pipefail` | ✅ **属实**。我扫了全部 **20** 个 workflow，**恰好 8 个**含 `set -euxo pipefail`，正是那 8 个 build 流程 |
| 新增 `check-tee-pipefail.py` 门 | ✅ 存在（108 行），且把我定位的事故原文引在文件头 |

**你把我的一处发现正确推广成了 systemic 修复**（不是只修我指的那一处），这一点做得好。
另外：设备侧我已复核最新产物 `dc9b1164`（重新上机、不是推断）——
与 `b8681800` **功能等价**（全符号表 diff 只多 1 个 `och_pcl_marker` + 1 个链接器跳桩，**无任何函数尺寸改变**），
① EXIT=0 / ② 153 / 卷积族 54 / 全部 `CQ-OUT`·`RW-OUT` 逐字相同。
你补的**编译期标记**已生效，我的校验器也据此加了一条必需标记。

**但要请你继续的话还没说完 —— 你收尾时说的「契约行收尾消息」没有落地**（IDLE 是空的，只有工具调用记录）。
不影响本轮，下轮注意即可。

## 二、★ 问题：这道门自己有两个缺陷（我补做了你没做的控制实验）

按纪律「**新建门必须有：正控制 + 至少一个负控制（含真实样本）+ 一次回归**」——
这道门**一个都没有**。我把三组实验补上了，结论是：

| 实验 | 期望 | 实测 |
|---|---|---|
| **正控制**：真 workflow（含 `set -euxo pipefail`） | PASS | **EXIT=0 PASS** ✅ |
| **负控制·真实样本**：把 `pipefail` 去掉 | FAIL | **EXIT=1 FAIL** ✅ |
| **被骗测试**：去掉 `pipefail`，但在同一 run 块里加一行**含「pipefail」的注释** | FAIL | **EXIT=0 PASS** ❌ |

⇒ 好消息：**这道门确实能红**（负控制通过，不是假门）。
⇒ 坏消息：**它能被骗**，而且还有一处自指计数。

### 缺陷 1（严重）：裸词匹配 ⇒ 假绿向量

第 46、87 行：

```python
RE_PIPEFAIL = re.compile(r"\bpipefail\b")
...
if any(RE_PIPEFAIL.search(l) for l in seg):
    continue
```

`\bpipefail\b` 是**裸词**匹配整个步骤块。任何**注释、echo 文本、说明文字**里出现「pipefail」二字，
就会让一个**根本没开 pipefail** 的步骤判 PASS。

**在一道专门用来防假绿的门上出现假绿向量** —— 这条必须修。
而且它直接违反我们自己定的纪律：**「一条 PASS 必须给出「它匹配到了哪段文本」」**。

### 缺陷 2（计数自指）：门把自己数进去了

门报告「共扫 **3** 个含 `| tee` 的 run 块」，而该 workflow 里**真正的 `| tee` 只有 1 处**（第 811 行）。
另两处是：

- 第 394 行 —— 一段**注释**（正是描述我那次事故的文字：``# ★ och 定位到的**假绿**：没有 pipefail 时 `cmd | tee` 的退出码取自 tee（恒 0）``）
- 第 397 行 —— **这道门自己的步骤名**：``- name: Static check - every `| tee` step has pipefail``

**根因**：`seg = lines[lo:hi]` 把 `- name:` 那一行**也包含进块内**，且注释行不排除。
⇒ 门**把自己数进去了**，计数从 1 虚报成 3。**计数不是证据。**

### ★ 一个诚实的附带说明：我自己也踩了同一个坑

我第一次跑这三组实验时写的是 `python3 $GATE ... | sed 's/^/  /'`，然后 `echo "EXIT=$?"` ——
**取到的是 `sed` 的退出码（恒 0），不是门的。** 于是差点报出「门在 FAIL 时也返回 0」的假结论。
这正是我们查了一整天的同一个系统性陷阱（管道吞退出码）。**我重做后才得到上面的正确数据。**
把这条也一并记下来，它说明这个陷阱的覆盖面比一个 workflow 步骤大得多。

## 三、精确修法（可原样落地）

```python
# --- 缺陷 1：只认真正的 shell 选项，且排除注释行 ---
RE_COMMENT = re.compile(r"^\s*#")
# 接受两种规范写法：`set -euxo pipefail` / `set -o pipefail`（可带其它短选项）
RE_PIPEFAIL = re.compile(r"^\s*set\b(?![^#\n]*#)[^#\n]*\bpipefail\b")
...
    # 原： if any(RE_PIPEFAIL.search(l) for l in seg):     continue
    # 新：
    if any(RE_PIPEFAIL.search(l) and not RE_COMMENT.match(l) for l in seg):
        continue
```

```python
# --- 缺陷 2：块内排除 `- name:` 行本身与注释行 ---
for name, lo, hi in blocks(lines):
    seg = lines[lo + 1:hi]                      # ★ 跳过 `- name:` 行本身
    tee_lines = [l for l in seg
                 if RE_TEE.search(l) and not RE_COMMENT.match(l)]   # ★ 排除注释行
```

```python
# --- 附带：PASS 必须带证据（纪律要求）---
    if any(...):
        hit = next(l.strip() for l in seg
                   if RE_PIPEFAIL.search(l) and not RE_COMMENT.match(l))
        print("  ok: %s :: %s  ← 证据: %s" % (fn, name[:52], hit[:80]))
        continue
```

**修后自测必须包含（缺一不可）**：

1. **正控制**：`build-teflon-fix-rofix-perchannel.yml` 的**当前版本** ⇒ 必须 PASS
2. **负控制·真实样本**：同一文件、`sed 's/set -euxo pipefail/set -eux/'` ⇒ 必须 **EXIT=1**
3. **★ 反作弊控制（本轮新增）**：同一文件、去掉 `pipefail` **并**加一行含「pipefail」的注释 ⇒ **必须 EXIT=1**
   —— 这是「门不能被文字游戏骗过」的实证，是这道门的核心价值所在
4. **回归**：把上面 3 条跑一遍，输出三段退出码

**建议把这 4 条做成 `selftest-check-tee-pipefail.py`**，与 `check-patch-verify-order.py` / `check-script-shell-portability.py` 同规格。

## 四、授权与边界

- 本项属**工具/门类修复**，不改变任何产物语义
  ⇒ **你可自行修改、自行提交，不需要回来问用户。**
- **但本轮请不要派 CI。** 当前产物 `dc9b1164` 已可用、已上机验证；
  下一轮派 CI 的时机是「要攻 ③（`read_outputs` 缓冲区分配错）」时 ——
  那属于**产物语义变更**，**我会回来问用户**再给你指令。
- 请在报告里按纪律标注：**「已自行再派第 N 次、原委：________________________________」**
  （若本项**没有**触发 CI，请写「本项未派 CI，属工具修复」即可。）

## 五、纪律条目（建议写进 `03_强制开发规范.md`）

> **门的判据不得是「裸词在不在」，必须是「那一行是不是真的做了那件事」。**
> 反例：`\bpipefail\b` 会匹配注释；`grep -c 'ERROR'` 会匹配「没有 ERROR」。
> 并且：**门的计数不是证据** —— 门必须打印它匹配到的那一行原文。
>
> **自指陷阱**：门扫描的语料里往往包含门自己的名字与文档。切块时必须把
> 「声明行」与「注释行」排除，否则门会把自己数进去。

--- 
附：我的实验脚本与三组原始输出在我这边留档，需要可以给你。
