# B 路线委派：VeriSilicon TIM-VX / PaddleLite 官方栈可行性（先零设备）

## 背景（一句话）

A311D NPU 一直跑不出正确数值。OCH 走 Mesa/etnaviv 路线，今天已把根因锁到
「**per-tensor 权重缺 int8→uint8 的 +128 域转换**」，正在做最后一轮验证。
你的任务是**并行**评估官方栈能不能跑通，作为「排除模型/驱动以外问题」的对照。

⚠️ **重要前提**：如果 A 路线验证成功，B 路线立刻降级为「可选对照」，不必上机。
所以本轮**只做离线准备与可行性判定，禁止任何设备侧高危动作**。

## 设备事实（已由 OCH 侦察，直接使用，不要重复查）

| 项 | 实测 |
|---|---|
| 设备 | `192.168.123.184`，DSM，Linux 5.10.55+，`aarch64`，`synology_rtd1619b_ds423` |
| 资源 | 6 核 / 3704 MB 内存 / `/volume1` 202 GB 可用 |
| `galcore.ko` | **已加载但未 probe**：`baseAddress=0`、`irqLine=-1`、`chipIDs=全 0xFFFFFFFF`、`registerBases=全 0` |
| `/dev/galcore` | 只是镜像预置的静态节点（misc minor 200）；`/sys/class/misc/galcore` **不存在** |
| NPU 硬件 | `ff100000.npu`，GC8000 **rev 7120**，**被 etnaviv 绑定**，IRQ 27 已有 25 次中断 |
| galcore 版本 | **6.4.10**（串 `6.4.10.6.469918`），文件 `/lib/modules/galcore.ko`（另有 `/usr/lib/modules/galcore.ko`） |
| VIP 用户态库 | **全无**（`libNBGlog` / `libviplite` / `libOpenVX` / `libGAL` / `libtim-vx` 一个都没有） |
| libc | `/lib/libc.so.6` 1.6 MB（无 `ldd` 命令） |

## 任务 A：离线材料准备（全部在 oca 本机 `D:\clawtemp`，不碰设备）

### A1. 飞桨 aarch64 VIV SDK（9.5 MB）

```
http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_linux_arm64_6_4_4_3_generic.tgz
```

产出（写进 REPORT.md）：
- 库清单（文件名/大小）
- 每个 `.so` 的 `file` 架构 + `readelf -d` 的 DT_NEEDED + `readelf -V` 的 GLIBC 版本需求
- 包内 `VERSION` / README 内容
- 是否包含 a311d 专用 DDK：`libNNGPUBinary.so`、`libNNVXCBinary.so`、`libOvx12VXCBinary.so`

### A2. 飞桨通用 demo（769 MB，**只解 arm64 相关路径**）

```
https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo_v2_12_0.tar.gz
```

只需解出：
- `libs/PaddleLite/linux/arm64/lib/verisilicon_timvx/`（含 `switch_viv_sdk.sh`、`libtim-vx.so`、DDK 库、`viv_sdk_6_4_4_3/`）
- `image_classification_demo/shell/build.linux.arm64/demo`（预编译二进制）
- `image_classification_demo/assets/models/mobilenet_v1_int8_224_per_layer/`
- `image_classification_demo/assets/datasets/test/`（一张图即可）

建议 `curl -C -` 断点续传；`tar -xzf <包> <路径...>` 只解需要的部分，别全解。

## 任务 B：版本兼容性判定（**本轮最关键**）

**核心问题：6.4.4.3 的 DDK 用户态库，能不能连 6.4.10 的 galcore 驱动？**

要查清并给出证据：
1. Verisilicon 的 galcore ↔ DDK 版本策略（ABI 是否跨小版本稳定）。查 TIM-VX 仓库 issue、
   PaddleLite 文档、NXP i.MX 8M Plus / Amlogic / Rockchip BSP 发布说明。
2. 是否存在 **6.4.10 的 aarch64 DDK** 公开来源（Amlogic SDK、Khadas BSP、其他厂商 Yocto 层）。
   有 → 给链接与获取方式；没有 → 明确说明「必须向厂商索要」。
3. 反过来：能否把驱动换成 6.4.4.3？飞桨包里给的 `galcore.ko` 是给 **4.9.x 内核**的，
   我们是 **5.10.55** ⇒ 直接 `insmod` 几乎必然失败（内核符号/vermagic 不匹配）。
   请核实包内 `galcore.ko` 的实际 vermagic（用 `modinfo`/`strings` 看），并给出结论。

**明确输出三选一**：

- **能试**：给出最小验证路径与置信度
- **不能试**：给出卡点（哪个前提无法满足）与唯一出路（如向厂商索要 DDK）
- **待定**：给出「需要什么信息才能判定」

## 任务 C：最小上机验证方案（**只写方案，不执行**）

写成可复制粘贴的步骤，含：
- 需要上传哪些文件（源路径 → 设备路径）
- 需要执行的命令（`unbind etnaviv`、`insmod galcore baseAddress=… irqLine=…`、
  `LD_LIBRARY_PATH=…`、跑 demo 的完整命令行）
- **每步的风险与回滚**（尤其 unbind/insmod —— 可能导致设备不可达）
- **预期输出**（什么算成功：例如 mobilenet 输出 Top1 与 CPU 一致、`dmesg | grep Galcore`
  应打印 `Galcore version 6.4.10`）

## 边界（硬性，违反即作废）

- **不要碰设备**：不 SSH、不 unbind、不 insmod、不跑任何 NPU 作业。
- **不要推** `EasongChung/rk_syno_pack`（OCH 负责，push 会触发无分支过滤的 CI）。
- 不碰 OEC 侧（DS124 / 引导固件 / `main`）。
- 不输出任何密钥、口令、Token；不 `printenv`/`env` 全量打印。
- 结论必须可分证伪：每条判断给出**证据来源**（文件/命令输出/URL）。

## 交付物

写到 `D:\clawtemp\timvx-route-b\`：

1. `REPORT.md` —— 结论先行：
   - §0 一句话结论（能试/不能试/待定 + 理由）
   - §1 材料清单（文件、大小、sha256 前 16 位、来源 URL）
   - §2 兼容性判定（含证据）
   - §3 最小上机方案（任务 C）
   - §4 风险与回滚
   - §5 未解问题
2. `libs/` —— 解包出的 aarch64 库（总量 < 200 MB 才拷；超了只留清单）
3. `MANIFEST.txt` —— 每个文件的来源（URL 或包内路径）

**完成后在消息末尾附契约行**：

```
【B路线】结论=<能试|不能试|待定> 交付=D:\clawtemp\timvx-route-b\REPORT.md 叫醒方式=<需要 OCH 做什么>
```
