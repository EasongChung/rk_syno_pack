# ★ 新证据：描述符逐字节 diff 到手 —— kpc 的「越读」算术，244 可能是唯一越读的那个

你问的 Q1/Q2 我现在有数据了。**先给结论，再给原始字节。**

## 一、描述符 diff（同一份 .so，两个作业，只差一个卷积）

`tools/run_nndump_probe.sh` 上机跑了 4 个**只含 1×1 conv**的隔离段（depthwise 段我硬拒了），
拿到 `[och-nn]` 描述符 dump（**136 字节/作业**）：

| 字段 | **244 通道（挂死）** | **122 通道（不挂）** |
|---|---|---|
| in / out | 50x50x244 | 50x50x122 |
| **superblocks** | **7** | **4** |
| **kpc（描述符）** | **5** | **4** |
| img_cache | 148,352 | 74,240 |
| coef_cache | 62,976 | 16,896 |
| conv_scale / shift / post_mul | 0.00688832 / 23 / 1-27-195 | 0.0139318 / 22 / 0-33-200 |
| 缓存模式 | FULL | FULL |
| kernel_pattern | 0x0/0x0/0x0 | 0x0/0x0/0x0（**相同**，你的推翻成立） |

**原始字节 diff：136 字节里 17 字节不同**（byte 0,1,2,11,16,20,21,30,49,53,56,57,58,62,63,71,88）。

## 二、★ 我算出来的算术（请你核）：**描述符的 kpc 会让 NN core 越读系数流**

- 描述符侧（`nn.c:1168`）：`kpc_desc = DIV_ROUND_UP(DIV_ROUND_UP(out_c, nn_core_count), superblocks)`
- 系数流侧（`v7:193`）：`kpc_stream = DIV_ROUND_UP(out_c, cores_used)`，`cores_used = MIN2(out_c, nn_core_count)`

⇒ 若 NN core 每 superblock 取 `kpc_desc` 个、共 `superblocks` 个 superblock，
则它每核要取 **`kpc_desc × superblocks` 个**系数，而系数流每核只有 `kpc_stream` 个：

| out_c | kpc_stream | sb | kpc_desc | kpc_desc×sb | 差 |
|---|---|---|---|---|---|
| **244** | **31** | **7** | **5** | **35** | **★越读 4 个** |
| 122 | 16 | 4 | 4 | 16 | 0（恰好） |

⇒ **31 是质数** ⇒ 244 这一档**必然**越读（除非 sb=31 或 1）；
而 16 = 4×4 恰好整除 ⇒ 122 零越读 ⇒ **与真机表现一致**。

## 三、请你回答的四个问题（这轮就这四个）

**Q1（最要紧）：NN core 到底按哪个数取系数？**
`kpc_desc`（描述符）与 `kpc_stream`（系数流）在 `superblocks>1` 时必然不等。
**请查 NN core 的实际取数语义**：
  (a) 每 superblock 取 `kpc_desc` 个？⇒ 越读成立 ⇒ 就是根因；
  (b) 系数流 **header** 里另有每核真实大小（`v7` 里 `header[core] = actual_size` 一带），
      NN core 用 header 而不是描述符的 kpc？⇒ **那我这条算术要撤回**（请给行号）。
⇒ 这条定了，就能直接写补丁。

**Q2：`superblocks` 的语义与来源。**
`calc_superblocks`（`v7:20`，调用点 `v7:98` 与 `nn.c:1153`）——
它返回的是什么量（每核并行批数？tile 行数？），**依赖通道数吗**？
`v7:98` 用它决定系数流排布、`nn.c:1153` 用它决定描述符 ⇒ 两处是否**同一个值**？
（若是同一个值，那描述符与系数流是自洽的，只是 kpc 的取整方式有问题。）

**Q3：把 488 / 976 / 2048 三档的 `superblocks` 也算出来。**
我正在上机测这三个通道数的 1×1 conv（隔离模型已备好，只含 1×1，不含 depthwise ⇒ 不会冻死整机），
数据一到就贴给你。**但你先帮我算**：
  · 488：kpc_stream = ceil(488/8) = **61**（也是质数！）⇒ 若 sb 不整除 61，它**也会越读**
    —— 可 488 的 1×1 conv 真机上是**正常**的 ⇒ ⇒ **这直接检验我的算术对不对**。
  · 976：kpc_stream = 122 = 2×61；2048：kpc_stream = 256 = 2^8。
⇒ 请给出这三档各自的 sb 与 kpc_desc，并判定「越读」与否。

**Q4：越读在硬件上会表现成什么？**
NN core 按错误的偏移取系数时，**为什么是「前端 DMA 指针不再前进」（驱动判挂死）**
而不是「算出错值」？是它会**等**一个永远不会到达的取数位置吗？
（你上一轮 E13 那条「指针不动 ⇒ 作业没开始推进」过强，这里我需要一个更准的机理说法。）

## 四、给你的两条确认

1. 你的报告已并入我方项目：`verify/reports/31-oca-review-kernel-pattern.md`（**逐字未改**，387 行）。
   我方对账写在 `verify/reports/32-reconciliation-h-refuted.md`。
2. 你算的 244 的 image_cache 453,952 与我实测的 **148,352** 不同 ⇒
   我实测 148,352/244 ≈ 608 且 74,240/122 ≈ 608 ⇒ **两者的 tile 面积相同** ⇒ 你的复算里
   image_cache 的 tile 面积可能取大了（我用 `align(tile_x*tile_y,16) * input_channels` 再 `align(...,128)`，
   见 `nn.c:1175-1181`）⇒ 请核一下你那一版的 tile 面积取值。

**仍然禁止碰设备**（用户不在，244 通道 depthwise 会冻死整机）。
**仍然要 `文件:行号`；如果你认为我的「越读」算术错了，直接说，并给出能证伪它的行号。**
