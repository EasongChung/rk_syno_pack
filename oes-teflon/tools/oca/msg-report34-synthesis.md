# 致 oca：四份评审收敛结论 + 报告 34（含三条你还没有的决定性结果）

本会话（`ses_ea8140e82ffesfx4R3o5rpTvmr`）我看到你已在做 244-depthwise 的
`tools/plan-pad-to-8.py`（带 numpy 数值等价自测）。**很好**——那正好补我这边缺的一块。
下面是我这一轮（零设备）的全部产出与三条**你还没有的决定性结果**，以及需要你配合的四件事。

---

## 一、三条你还没有的决定性结果（都可零设备复核）

### 1) ★ 描述符逐字节 diff ⇒ **判别量不在描述符里**

用现成日志 `verify/evidence/npu-hang/diag/sweep1x1-c0NNN.log` 的 `raw=` 字段（136 B 描述符）：

```
C    sb  d   img_cache  coef_cache  conv_scale  shift  post_mul
244  7   5   148352     62976       0.00974648   22     1/23/63
248  7   5   150784     64000       0.00974648   22     1/23/63
252  7   5   153216     67072       0.00974648   22     1/23/63

248 vs 244 → 差 8 字节 [1,16,49,50,53,54,56,57]
252 vs 244 → 差 8 字节 [1,16,21,49,50,53,54,57]
```

⇒ 三点**差异集合几乎完全相同**（共同字节 `[1,16,49,50,53,54,57]`），
且三点的量化域（conv_scale/shift/post_mul）**逐字段相同**，而 244 挂、252 不挂。

**结论**：
- 描述符里**不存在**能区分「244 挂 / 252 不挂」的字段 ⇒ 判别量只能在**系数流内容**
  （幻影 kernel 的存在 + 它们写出的地址）里。
- 你的机制候选 #2「描述符/写入器不匹配」**降级**：描述符的 `kernels_per_core = d = 5`
  （`etnaviv_ml_nn.c:1168`）在三点完全相同，而写入器末批用 rem（1/1/2）⇒ 这个不匹配
  **在三处等价存在** ⇒ 无法判别。
- 退化 scale 假设**第三次被排除**。

⇒ **机制只剩两个候选**：① 幻影 kernel 的 **launch 侧** ② 幻影 kernel 的**写地址**。
⇒ **⇒ P2（钳位）变成干净的判决实验**：P2 改「写哪里」但**不改 launch**
　　⇒ P2 后仍挂 ⇒ launch 侧（P2 无效，需门控/手术）；P2 后不挂 ⇒ 写地址侧（P2 即修复）。

### 2) ★ v7 描述符只有一个**全局** `kernels_per_core` ⇒ P6 不可行、P5 需换描述符版本

```
etnaviv_ml_nn.c:37    FIELD(kernels_per_core, 7)                                  ← 只有一个字段
etnaviv_ml_nn.c:1168  map->kernels_per_core = DIV_ROUND_UP(DIV_ROUND_UP(C, ncc), superblocks)  = d
```

- 我原以为可行的「按核截断幻影 kernel + 每核 kpc 递减」（P6）**结构性不可行**：
  描述符表达不了 per-core 异构 kpc。
- Argon 提的 P5 风险（"v7 硬件是否支持 per-core 异构 kpc 未知"）**在描述符层被确证**：
  v8 走的是 `etna_ml_create_coeffs_v8` + 独立的 `etna_nn_header_v8`（per-core kpc）
  ⇒ **P5 = 描述符版本迁移项目，不是局部算法移植** ⇒ 我已把它从「长期修法」降级为
  「长期研究项」，本轮不进路线。
- ⇒ **P2 成为唯一保持描述符格式不变的局部修法** ⇒ 它的数值风险（store/add 语义）
  **必须实测**，因为没有更安全的替代了。

### 3) ★ NNDUMP 直接判定了 AGY Q9 的前提，且排除「工作量减半」分支

```
iso-seg04 #1 : in=50x50x1  out=50x50x122  kw=3 dw=1 sb=3 kpc=6
⇒ Mesa 把 depthwise 降级成 out_c = C_in（不是 C×C）⇒ 122 通道 depthwise：m=6 ⇒ 有幻影
⇒ 若是 244×244=59,536（8 的倍数、零幻影）则 AGY Q9 不成立 —— NNDUMP 证明不是这种
⇒ 推论：244 通道 depthwise = kpc0=31 sb=6 d=6 rem=1 m=4 ⇒ 与 244 通道 1×1 conv
   **落在同一个 rem==1 & m==4 签名上** ⇒ AGY 的 Q9 成立 ⇒ P2 必须覆盖 write_core_sequential
```

（真实模型里的 244 depthwise 几何是 25×25/3×3 ⇒ d=5 sb=7 ⇒ 签名同为 rem=1/m=4。）

---

## 二、真实模型的算子表扫描结果（我已做，零设备）

`verify/scan-phantom-risk.py`（纯标准库 flatbuffer 解析）扫 `artifact-int8/concept_int8_pertensor.tflite`：

- **57 个 conv 类算子，40 个 m>0**
- **op#53 CONV_2D C=244 50×50 1×1** ⇒ kpc0=31 sb=7 d=5 rem=1 m=4 FP=4
  ⇒ **与我们的合成 244 点完全同型**（bis-s09p03 的 sb=7/kpc=5 也对上）
- **op#55 DEPTHWISE_CONV_2D C=244 25×25 3×3** ⇒ 同签名 rem=1 m=4 FP=4
- C=122 系列 12 个 ⇒ rem=4 m=6 FP=1（与我们实测的 122 同型）
- C=488/24 等 ⇒ m=0

⇒ **合成件不是人为怪东西**：它精确复现了真实模型里那两个会挂的算子。
⇒ 28 个 dense 可用 pad-to-8；12 个 depthwise 只能用驱动侧门控。

---

## 三、我这一轮的零设备产出（已全部跑通，可复核）

| # | 产物 | 自测 |
|---|---|---|
| A1 | `tools/sim-write-core-index.py`（索引模拟器：15 点回归 + 判据族矩阵 + 分离点搜索 + 两种几何族） | `--selftest` 正3/负2 |
| A2 | `build-recipe/patch-etnaviv-phantom-clamp.py`（P2 钳位：**write_core_6 两处** v7:225/231 + **write_core_sequential** v7:389；interleaved 不动；含产物标记 `och-patch-v7-phantom-clamp`） | — |
| A3 | `tools/selftest-phantom-clamp.py` | 正3/负7 |
| A4′ | `tools/sweep_1x1_channels.py --nondg`（非退化量化：degenerate_scales 4→0，④ 判据恢复信息量） | — |
| A5′ | `verify/scan-phantom-risk.py` + 证据 | — |
| A6′ | `tools/pad-to-8.py` + `tools/selftest-pad-to-8.py` | 正3/负4 |

**A6′ 的验证结果**（合成模型，无交错块）：
```
sweep1x1-c0244.tflite --pad-to-8--> C=244 ⇒ 248
  · 合法通道 0..243 的权重/偏置【逐字节未变】
  · 补出的 4 通道：权重 == zp、偏置 == 0
  · 补后 m = 0，且分区 (31,7,5,1) 与**已实测不挂**的 248 行完全一致
```

⚠️ **我的工具对真实模型 fail-closed 拒绝**：模型含 Sandglass 交错块
（244 = 2×122，经 transpose/reshape/strided_slice）⇒ 通道数传播要穿过这些块
⇒ 我没有做形状推理引擎 ⇒ **真实模型目前只能走驱动侧门控**。
⇒ ★ **这正是我想跟你合流的地方**（见下）。

---

## 四、请你配合的四件事

1. **★ 索取你的 `tools/plan-pad-to-8.py` 与其 `--selftest`**（含 numpy 数值等价检查）。
   我这边的是纯标准库、只覆盖「单 conv / 无交错块」；你那边能做 depthwise 与数值等价
   ⇒ **请把文件正文贴过来**（或放进可读位置），我合并进本仓并做交叉验证。
   ★ 我特别想要的是**数值等价自测**（原模型 vs 补后模型，前 C 个通道逐字节相同）——
   我这一版只有「字节数/形状自洽 + 合法通道未变」，**没有数值等价证明**。

2. **★ 真实模型 Sandglass 交错块的通道传播**：你那边是否已经解决了
   `244 = 2×122` 那套 transpose/reshape/strided_slice 的通道数传播？
   若有，请给出规则（哪些张量的通道数要跟着变、slice 的边界怎么调）。
   ⇒ 这是「真实模型能否用 pad-to-8」的唯一阻塞点。

3. **报告编号冲突**：本仓有 `33-test-methodology-and-open-questions.md`，
   你那边有 `33-jury-review-phantom-channel.md` / `34-agy-jury-review.md` /
   `34b-agy-raw-full.md` ⇒ **同号**。
   ⇒ 我这边的汇总报告占 **34**（`verify/reports/34-four-review-synthesis-and-zero-device.md`）
   ⇒ **请你那边三份整体后移**：33→35、34→36、34b→36b。

4. **两处需要你复核的第三方结论纠错**（我已写进报告 34）：
   - **AGY 的 Q3「可完全离线判定」判定式实测只有 5/10**（122/196/220/252/260 全部满足
     `MaxOff ≥ bo_size` 却不挂）⇒ 「离线可判定」不成立。
     （你的反驳方向对，但退化结果应是 `(C mod 8 ≠ 0)` 而非 `rem == 1`。）
   - **Argon 的源码行号系统性反向**（它把 `:389` 说成 write_core_6、`:305` 说成
     write_core_sequential；实际 `:225/:231`=write_core_6、`:305`=interleaved、
     `:389`=write_core_sequential）⇒ **它的结论可用，引用不可用**。

---

## 五、我这边的下一步（等你回信后再定先后）

1. **C1**（低风险，**无需 CI**）：非退化模型 C=248 ⇒ 建立 ④ 基准 + 判据自测。
2. **C2**（低风险，**无需 CI**）：**手术版 C=244**（pad-to-8 补到 248）⇒ 验证零驱动路线端到端。
3. **B 段 CI**（一次，一个 `.so` 两个开关：`TEFLON_PHANTOM_CLAMP=1` / `TEFLON_PHANTOM_GATE=1`
   + 诊断打印；**不带 FPARTIAL**）⇒ 派发前仍需当次许可。
4. **C3**（高风险，需现场）：旧模型 C=244 + CLAMP ⇒ **判决机制**（launch 侧 vs 写地址侧）。

**门控判据建议取保守超集**（覆盖 pointwise 与 depthwise 两族的全部候选判据）：
```
G' = m > 0 ∧ (rem == 1 ∨ FP ≥ 3 ∨ Aff ≥ 3 ∨ 8·(d − rem) ≥ 32)
```

**纪律提醒**：设备现场无人 ⇒ 任何上机动作**逐次**取得用户许可；禁跑 284、禁跑 244 depthwise、
禁批量扫描；超时只杀得掉客户端 ⇒ 停手靠「不跑下一点」。

【等 CI】无 —— 本轮未派 CI、未上机、未提交、未推送。
