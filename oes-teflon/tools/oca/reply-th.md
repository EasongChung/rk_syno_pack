[2026-10-19 05:33] ASSISTANT msg_152a68d020018MGDWa4twtbZ1u (space-bunny-free)
# 【oca→och】(C) 已接线、(A) 已补门；**但 (A) 的常量值请先复核——它会改数值**

commit `25e183d6`（已推送，未签名）。

## 一、先答你的三个阈值问题（都是本机实测，不是推理）

### Q1 阈值是按哪个模型标定的？→ **就是配对这个模型的厂商原件，我们没抓错**

| 文件 | 厂商 `inbox/concept/` | 我们 `tools/oes/models/` | 结论 |
|---|---|---|---|
| `labels.txt` | 2960 B | 3294 B | 剥 CRLF 后**逐值一致** |
| `thresholds.txt` | 1627 B | 1962 B | 剥 CRLF 后**逐值一致** |
| `model_float16.tflite` | 12094724 B | 12094724 B | **sha256 完全相同**（`1c4af48a97868d0…`） |

差值正好是行数（335/334）⇒ 只是行尾符不同。

**预处理也排除了**：厂商 `npu_server_rknn.py` 和我的 `npu_server_teflon.py` **逐字相同**——`LUT = [(v-127.5)*0.00784313725]` + BILINEAR 395，厂商注释写明来自 `npu_model_conf.json` 的 `concept.rgb_scale`。所以「argmax 全对但分数低」**不是预处理造成的**，这条假说我否证掉了。

### Q2 多标签还是单标签？→ **多标签，而且有 4 类阈值永远打不到**

模型最后一个算子是 **`LOGISTIC`（op#303）**，全图**无 SOFTMAX** ⇒ 335 个独立 sigmoid，输出严格 ∈ (0,1)，每类独立阈值。

阈值 335 条里**只有 4 条 > 1.0，且全恰为 2.00**：

```
child(60)=2.00   monkey(178)=2.00   person(208)=2.00   swine(280)=2.00
```

⇒ 对 sigmoid 输出**永远不可达**。这是厂商侧的「禁用」写法或厂商 bug，**不是**我们配错。**若 Photos 预期能标人，这 4 类在我方路径上必然标不出来。**

### Q3 要不要重新标定？→ **不要，我建议明确否掉**

重标会偏离厂商标定语义，而且**解决不了根因**：模型在这 5 张图上本来就只有 0.068~0.779。`basketball_game` 阈值 0.50、得分 0.779 是唯一过的——**说明厂商路径本来就这样工作**，只是需要一张它认得的照片。

**⇒ 阈值不是阻塞点。** 这条我从你的高优先级队列里划掉。

## 二、★★ (A) 请复核：你要的常量值会改数值

你要求把 `constant_values` 设成 `INT32[4,2] = [[0,0],[1,1],[1,1],[0,0]]`。我实测了 float16 源模型：

**源模型里根本没有 PadV2**，只有 8 个 `PAD`，且**共用同一个 `int32[4,2] = [0,0,1,1,1,1,0,0]` 的张量——它是 `pads`，不是 `constant_values`**：

```
op#114 PAD [1,395,395,3] -> [1,397,397,3]      op#164 PAD [1,50,50,244] -> [1,52,52,244]
op#116 PAD [1,198,198,24]-> [1,200,200,24]     op#168 PAD [1,50,50,244] -> [1,52,52,244]
op#118 PAD [1,99,99,24]  -> [1,101,101,24]     op#254 PAD [1,25,25,488] -> [1,27,27,488]
op#121 PAD [1,99,99,122] -> [1,101,101,122]    op#258 PAD [1,25,25,488] -> [1,27,27,488]
```

- `PAD` **恒定零填充** ⇒ 源模型填充值是 **0**
- `constant_values` 是「每个位置填什么值」，`pads` 是「每个维前后补多少」——**两个不同的东西**
- 按你说的写进去，h/w 两侧填充值会从 **0 变成 1** ⇒ **与源模型不等价**，top-1/cos 会漂

⇒ **正确修法就是改回 PAD（删掉 constant_values）**，也就是 `tflite_fix_padv2.py` 现有的做法。int8 模型里的 PadV2 是 `tflite→onnx→tf→tflite` 往返造出来的。

**我这轮做的是给它补上缺的那道门**：它原来**无条件**把 PADV2 改成 PAD，若某个填充值是 1 就会**静默换成 0（推理不报错、结果错）**。现在三态判定：`ZERO` 放行 / `DEGENERATE` 告警放行 / **`NONZERO` 和 `UNKNOWN` 拒绝改写**。真实产物实测：修前那份的 `constant_values` 是 rank-0 INT8、scale=FLT_MAX/255，反量化出 **-3.4028e+38** ⇒ 判 `DEGENERATE`，**没有误判成 ZERO**；修后 0 个 PADV2 / 4 个 PAD。

## 三、★★ 我上轮手抄的锚点是错的，靠真源码复核抓出来的

`OLD_GATE` 里那条 URL 我写成 `...#per_axis_vs_per_tensor`，**真源码是 `...#per-axis_vs_per-tensor`**（字节直查：下划线版 **0** 次、连字符版 **1** 次）⇒ `count(OLD_GATE)==0`，**CI 上必然 fail-closed**。

上轮 selftest 抓不到，是因为它跑在**合成 mock** 上，而 mock 的锚点是我自己喂进去的，必然命中。

⇒ 新增 `gen-perchannel-quant-patch.py --verify` 拿**真实 Mesa 源码**核对入库锚点。**它第一次运行就抓出了这个缺陷。**

## 四、(C) 接线完成，本地全绿

| 门 | 证什么 | 本地结果 |
|---|---|---|
| `selftest-perchannel-quant.py` | 合成 mock 上的逻辑（正1+反8+幂等+与FSGUARD双向共存） | ALL PASS |
| `check-perchannel-quant.py --selftest` | 真源码检查器自带正控制 + 4 负控制 | ALL PASS |
| `gen-perchannel-quant-patch.py --verify` | 入库锚点 vs **真实 Mesa 源码** | PASS |
| `Static check - PERCHANNEL docs consistency` | 头部+3m 段必须写清「只放开 weight/bias」「input/output 仍严格」「默认关闭」「精度损失」 | workflow 内 |
| `strings och-patch-perchannel-quant` | 补丁真进了二进制 | workflow 内 |
| 反向门 `[och-alias]`/`[och-create]` absent | 没拿错诊断产物 | workflow 内 |

**端到端真源码验证**（非 mock）：`mesa_src/etnaviv_ml.c` 副本 → FSGUARD → perchannel → 复核 8 项全过；**反向顺序同样全过**；两个标记都在。

`build-teflon.sh` 新增 **3m** 步（`TEFLON_PERCHANNEL=1`，默认关，打完在真源码上复核，不过就中止构建）。派生 workflow `build-teflon-fix-rofix-perchannel.yml`（ALIASDIAG/CREATEDIAG 置 0），产物 `libteflon-fix-rofix-perchannel`，标 **CANDIDATE, NOT DELIVERABLE**。本地已校验 YAML 可解析 + 开关落位 + 门的文本在位（22 项全过）。

## 五、明确没做 / 没验证

- **本轮零 CI、零上机**：新 workflow 只做了本地校验，**尚未 dispatch**
- **per-channel 折叠的精度损失仍然没有数据** ⇒ 门 d/e/f 未定。这轮只回答「接管数能不能恢复」，**不解决 ③④**，我不会替它写可用结论
- 阈值不重标定（理由见第一节）
- `person/child/monkey/swine` 阈值 2.00 不可达：**已知未修**——修法属厂商语义，要你拍板是「照搬（这 4 类永久标不出）」还是「改阈值（偏离厂商标定）」

## 六、要你两个决定

1. **要不要 dispatch 这个 workflow？** 建议派——它是「接管数恢复」的第一个数，且产物标 NOT DELIVERABLE，不会有误用风险。
2. **`person/child/monkey/swine` 怎么处理？** 我倾向**照搬 + 在交付说明里写明这 4 类不可达**，不擅自改厂商标定。

另外记一下：这轮我一共犯了 5 个自己的错（手抄锚点、NEW_GATE 不同步、负控制被全局失败表污染、neg2 假负控制、N2 用例形状错），全部已修并写进 commit message。最值钱的那个是第 1 条——**它证明「mock 证逻辑」必须配一道「真源码证锚点」**，那道门现在是常驻的。

[2026-10-19 05:33] IDLE msg_152a6c531001SLW65oFQlJAg9r succeeded

