# OEC CPU 分档 + eMMC52 兼容试验

> 本地兼容分支：compat/oec-bin-emmc52。不是正式版本，不自动合并或发布。

## 修改边界

- 内核基线 c17d76f7、pack 基线 4dea9d85；不换 U-Boot、GPT 或网口 PHY。
- 恢复 CPU rockchip,supported-hw、各频点两级掩码及 J/M 特殊 OPP。
- 不恢复 1800/1992MHz，不统一封顶 1416MHz，不改 NPU/SPK 55/53°C 门控。
- eMMC max-frequency=52000000，移除 HS200/HS400/Enhanced Strobe 声明。

## 分档不是全面硬件检测

Rockchip helper 从芯片信息推导版本档，并从 PVTM 等选择速度/电压档，再用
两级 opp-supported-hw 掩码筛选候选 OPP。它不是 PCB/eMMC/电源兼容性认证。
J/M 是源表中的分档名称，不能仅凭启动异常认定用户机器属于其中某档。

| 表 | 实际候选筛选结果 MHz | 50/55/75°C 对应最高频 MHz |
|---|---|---|
| 普通 | 408/600/816/1104/1416/1608 | 1416/816/408 |
| M | 408/600/816/1104/1416/1608，1608 使用 M 节点 | 1416/816/408 |
| J | 1008/1416 | 1416/1008/1008 |

J 档最低只能到 1008MHz，这是用户已接受的保护例外，不能声称仍实现 816/408MHz。
50/55/75°C 阈值、2°C 回滞、115°C critical 保留。新 helper 在绑定 cooling map
时根据实际 CPUFreq 表换算 state，不再统一假设 state1/3/5。

## 电压

普通六档的现有板级电压修复保持，普通高频 OPP 的 L0~L3 属性仍删除。
特殊节点沿用 Rockchip 原表：J 1008MHz=850mV、J 1416MHz=900mV、
M 1608MHz=1000mV（原表允许上限1150mV）；它们与普通档电压不同。
这些是配置来源，不是对每块板卡实际电压和稳定性的实测保证。

## 启动 hook 与构建

uInitrd 的 hook 仍请求 schedutil + max1608MHz；它不能把已筛掉的 OPP 加回来。
CPU cooling 注册时的 FREQ_QOS_MAX 同时取实际表的最高频（J 为1416MHz），
所以该用户上限请求不等于 J 可以运行1608MHz。此交互仍需实际新内核验收。

新 DTB 使用 rockchip,cpu-frequency-ceiling-khz 扩展，必须与本分支内核一起构建。
CI 检查实际 .config/System.map 和编译后的 DTB，不能拿旧内核只换新 DTB。
验证命令（CI/Linux）：

    python3 scripts/verify-oec-compat.py --selftest --source linux-5.10.x \
      --dtb output/dsm/boot-root/boot/rk3566-oec-box-wxy4-dsm.dtb \
      --config build/out/kernel-7.3/.config --system-map build/out/kernel-7.3/System.map

本地逻辑测试不替代完整内核编译或实机启动验证。云端构建需要先确认推送新分支。

## 用户测试

后续测试包需先备份并保留可启动的7.2恢复包，沿用 RKDevTool Loader 模式、
地址0整盘刷解压后的raw。只要求反馈“能启动 / 仍白灯”，不要求串口/SSH/日志。
合并兼容版即使能启动，也不能单凭结果判定 CPU 或 eMMC 中哪一项是根因。
