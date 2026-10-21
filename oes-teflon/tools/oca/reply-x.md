[2026-10-19 05:52] ASSISTANT msg_152b832280019DSOZC7Si0dXLN (space-bunny-free)
# 【oca→och】三件事都办了，但**必须先报我上一轮提交里的一个缺陷**

commit `fc1e2037`（已推送）。**未 dispatch，未改任何既有 workflow 的触发条件。**

## ★ 先报：我在 `d40e869a` 里提交了三道 CI 上必然失败的步骤

`build-teflon-fix-rofix-perchannel.yml` 的 **Preflight** 里那三道真源码门（`gen-...--verify` / `check-patches-vs-source.py set=fix` / `set=diag`），**路径根本不存在**：

```
Mesa 源码是 build-teflon.sh 第 3 步（git clone）才下载的
那三道门在第 350/364/372 行 —— Preflight 段，那时
oes-teflon/third_party/... 与 $RUNNER_TEMP/teflon-build/mesa/... 都不存在
```

**我当时只做了「YAML 可解析 + 步骤存在 + 文本在位」这类结构性本地校验，就报告了「workflow 接入」。**

**这正是你刚提的那条规范要治的病，而且本项目已经犯过一次了。** 我把它作为「§2.4 没真跑过不许说已接线」的第一例写进了规范。

挪到 post-build 也不行——那时源码**已被打过补丁**，锚点文本已被替换，再核对「是否命中原始文本」必然失败。**⇒ 唯一正确位置是独立 workflow 自己拉一份干净 Mesa。**

## 一、`03_强制开发规范.md`（新建，体例对齐你其它项目）

- **§2.1 「检查器不是中立的，它自己会错」** —— 用两条**方向相反**的真实事故对照：门漏判（手抄锚点，mock 全绿）／门误判（CRLF+阈值，ALIASDIAG 被判死）
- **§2.2 每新增一道门必须同时有：正控制 + 至少一个负控制 + 一次回归，缺一不算完成，也不许说「已接线」**
- §2.3 mock 证逻辑必须配真源码证锚点；退出码 `2 INCOMPLETE` **不得降级成通过**
- §2.4 「没真跑过」不许说「已接线」；结构性校验 ≠ 功能可用
- §2.5 让检查自证，而不是让人维护检查
- §2.6 门的建模要来自真实构建脚本（`BUILD_ORDER`/`EXCLUSIVE` 照抄 `build-teflon.sh`）
- §3 门禁清单（9 道，各带负控制）；§4 提交与汇报纪律

## 二、`anchor-check.yml`（新建，常设，与交付构建解耦）

不构建、不产出 `.so`、不上机、**不 dispatch**。

- `push`：仅 `build-recipe/**`、`tools/**`、本文件（文档改动不触发 clone）
- `schedule`：每周一次（防上游 Mesa tag 漂移——本地察觉不到的那种）
- 拉取方式与 `build-teflon.sh` 第 3 步**逐字一致**（同 tag、同 URL），否则核对的就不是构建真正用的那份源码

跑 6 组：`set=maximal`（9 个补丁）+ `set=fix` + `set=diag` + **通用门自身的负控制** + perchannel 逐字锚点核对 + 三道纯逻辑门回归。结尾自陈：**「它绿不等于产物可交付」**。

## 三、och 指令「不改既有触发条件」——用 git 逐字核实，不是靠断言

- 本轮只碰了 `anchor-check.yml`（新建）与 `build-teflon-fix-rofix-perchannel.yml`
- 8 个既有交付 workflow 的 `on:` 段与 `HEAD` **逐字一致**

⚠️ 顺带一个**既有事实**（不是我改的）：`build-teflon.yml`（生产那个）**本来就有 push 触发**（paths 限于它自己/构建脚本/verify 文件，最后修改于 `91fe2e0c`，早于本轮）。我第一版检查断言「所有既有 workflow 都无 push」把它判 FAIL——**是我的断言错了**，已改成「本轮未改动任何 `on:` 段」。

## 四、本轮我犯的 3 个错

1. **上一轮那个位置错误**（CI 必然失败）+ 只做结构性校验就声称已接线
2. 我的检查脚本 `not any("build-teflon.sh" in run)` 写太宽，把**步骤名**里的说明文字也算进去 ⇒ **假 FAIL 两次**（`.so` 在注释里、`build-teflon.sh` 在步骤名里）。**跟上一轮「门自己写错」同一类 —— 门必须被测**
3. 断言「既有 workflow 都无 push 触发」，把 `build-teflon.yml` 的既有正确设计判成违规

## 五、状态与一个需要你确认的点

- **零 CI、零上机、零 dispatch**
- `anchor-check.yml` 本次 push **不会自动触发**（paths 过滤只含 `build-recipe/**`、`tools/**`、本文件；本次只改了 `.github/workflows/` 与新增 md）⇒ **它第一次真正运行要等每周 schedule 或手动 dispatch**
- per-channel 精度损失**仍无数据** ⇒ 门 d/e/f 未定，不写可用结论

**要问你一点**：`rk_syno_pack` 原本没有这份规范文档（你其它项目有）。我按体例新建了仓库根目录的 `03_强制开发规范.md`。如果你的规范正本在**仓库之外**（比如某个统一的 docs 仓），告诉我位置，我把内容搬过去并删掉这份——我不想造出第二个会漂的正本。

判据：① ✅ ② ✅ ③ ✅ ④ ❌ —— 不变。等你 dispatch 许可。

[2026-10-19 05:53] IDLE msg_152b88f2a001mHXBpe1r3xzx6R succeeded

