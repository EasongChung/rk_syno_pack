# L1 只读检查结果：B 路线终止（`/proc/misc` 无 galcore）

## 结论

按你给的判据 **「若 `/proc/misc` 里没有 galcore ⇒ B 路线直接终止（连 unbind 都不必做，设备零风险）」**，
B 路线**终止**。本轮不再推进 L2/L3/L4，也不会去 unbind etnaviv 或 insmod 任何东西。

## L1 六条实测（全部只读，零风险）

| # | 检查 | 实测 |
|---|---|---|
| 1 | `grep -i galcore /proc/misc` | **无匹配** ⇒ 模块没有注册 misc 设备 |
| 2 | `cat /sys/class/misc/galcore/dev` | `No such file or directory` |
| 3 | `lsmod` | `galcore 344064 0`（**已加载，引用计数 0**） |
| 4 | `/sys/module/galcore/parameters/` | 存在：`allMapInOne` `baseAddress` `chipIDs` `compression` `contiguousBase` … |
| 5 | `strings /usr/lib/libstdc++.so.6 \| grep -c GLIBCXX_3.4.2` | **0** |
| 6 | `dmesg \| grep -i galcore` | 只有 `calling viv_dev_init` / `returned 0 after 161 usecs`，**无 probe 记录** |

补充：`/sys/bus/platform/devices/ff100000.npu/driver` → **`etnaviv-gpu`**（硬件仍被 etnaviv 持有）。

## 判读

- 模块 `init` 返回 0（加载成功），但**从未 probe 硬件**（无 `baseAddress`/`irqLine`/`chipIDs` 初始化日志），
  所以 misc 设备没注册 ⇒ 用户态无 `/dev/galcore` 可用节点（现有 `/dev/galcore` 只是镜像预置的静态节点）。
- 这与之前的侦察一致：`baseAddress=0`、`irqLine=-1`、`chipIDs` 全 `0xFFFFFFFF`、`registerBases` 全 0。
- `libstdc++` 也没有 `GLIBCXX_3.4.21`（计数 0），即使驱动可用，demo 也要自带 libstdc++。
  但这一点已不重要——**P1 不成立即终止**。

## 对你的交付物

`D:\clawtemp\timvx-route-b\`（140.6 MB）**保留备查**，不删。
它已经完成了它的使命：证明「官方栈在这台设备上不是一条可立即走的路」，
而不需要任何设备侧高危动作。

## 什么条件下 B 会重启

只有一条：**厂商/上游给出能在 Linux 5.10.55 + `synology_rtd1619b_ds423` 上真正 probe 的 galcore.ko**
（版本需与用户态 DDK 对齐，6.4.4.3 或 6.4.10.2 二者之一）。
届时用你 `REPORT.md` §3 的 L1→L4 流程重跑即可，材料不用重做。

## 本轮不需要你再做任何事

A 路线（Mesa：per-tensor 权重 +128 域 + 图输出 NCHW→NHWC 收口）已到最后一公里，
正在等 CI 产物，随后上机验收。若 A 成功，B 永久降级为备查。

【B路线】结论=终止(设备侧无 misc 设备) 交付=已归档 D:\clawtemp\timvx-route-b\ 叫醒方式=无需叫醒
