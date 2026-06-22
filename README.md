# syno rk3399 patchkit

这个目录是当前项目根目录，用来完成这条完整流程：

1. 编译 `u-boot`
2. 编译 RK3399 DSM 内核
3. 解包官方 `DSM_DS423_86009.pat`
4. 给 `initrd` 打补丁并替换内核刚编出来的 `syno_hddmon.ko`
5. 重新打包 `uInitrd`
6. 生成 Rockchip `update.img`

当前已经切到直接使用 `SynoXtract` 解官方加密 `pat`。

## 系统依赖

先安装宿主机依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential bc bison flex libssl-dev libelf-dev dwarves \
  cpio xz-utils patch curl e2fsprogs device-tree-compiler vim-common \
  dosfstools mtools libsodium-dev libmsgpack-dev
```

## 当前目录

- `build.sh`
  - 顶层入口
- `build/`
  - 构建输出和临时过程目录
- `scripts/`
  - 所有构建脚本
- `patches/`
  - `initrd` 补丁，按 `0001-*.patch` 顺序自动执行
- `linux-5.10.x/`
  - 内核源码
- `u-boot/`
  - U-Boot 源码
- `tools/`
  - `SynoXtract`
  - `linux_pack`
  - `prebuilts`
  - `rkbin`

## build 目录说明

`build/` 只放中间产物和输出，不放脚本。

当前会用到这些目录：

- `build/out/kernel`
  - 内核编译输出
- `build/DSM_DS423_86009.pat`
  - 本地官方 `pat`
- `build/pat-extract`
  - `pat` 解包结果
- `build/pat-rd`
  - 从 `rd.bin` 解出的原始 rootfs
- `build/pat-rd-patched`
  - 打完补丁后的 rootfs
- `build/rd.bin`
  - 重新打包后的 patched `rd.bin`
- `build/boot-patched`
  - 从官方 `pat` 重打出来的 `uInitrd`
- `output/dsm/boot-root`
  - 生成 `boot.img` 时临时组装的 boot 文件树

## 当前补丁内容

### linuxrc.syno.impl

- `synocfgen` 前增加 `sleep 5`
- 注释掉 `rmmod synobios`
- 设置 `red-led` trigger 为 `disk-activity`

### etc/rc

- 在 `exit 0` 前增加 `/sbin/getty 1500000 console`

补丁目录：

- [patches](/home/yxl/my_proj/syno/patches:1)

### root password

- 通过 `0003-etc.shadow-root-password.patch` 同时修改 `etc/passwd` 和 `etc/shadow` 中的 `root` 条目
- 通过 `0005-rk3399-load-extra-usb-modules.patch` 早期加载 RK3399 USB、DWC3 和 quota 模块
- `./build.sh pat` 会把自编内核的 `quota_tree.ko`、`quota_v1.ko`、`quota_v2.ko` 放进 initrd，并在 `linuxrc.syno.impl` 早期加载，保证 DS423 安装阶段新建的 ext4 quota 根分区可以挂载
- 内核内建 `mtdram`，启动时注册 8 个 RAM-backed
  Synology/Realtek junior 风格 MTD 分区。DS423 的 junior `updater` 会读取
  `/dev/mtd7` 的 FIS 信息，并按 `/dev/mtd0..7` 更新 `zImage/model.dtb/rd.bin`
  等内容；RK3399 没有 Synology 原生 junior flash 布局，所以这里提供安装阶段
  探测和写入使用的假 SPI NOR。该 RAM MTD 模拟为 16MiB、4KiB erase block，
  分区布局来自 `uboot_DS423.bin` 的 `mtdparts=RtkSFC`：
  `RedBoot/zImage/dtb/rd.gz/vendor/pstore/Misc Info/FIS directory`。

### syno_hddmon.ko

当前替换的是内核树里编出来、适配 `RK3399 + PCIe 转 SATA` 的版本：

- 如果不存在 Synology 原生盘位 GPIO/SMBus 电源控制
- 直接跳过 HDD monitor

模块产物：

- [build/out/kernel/drivers/hwmon/syno_hddmon.ko](/home/yxl/my_proj/syno/build/out/kernel/drivers/hwmon/syno_hddmon.ko:1)

## 构建入口

顶层只保留一个入口：

```bash
cd /home/yxl/my_proj/syno
./build.sh <target>
```

支持的 target：

- `uboot`
- `kernel`
- `pat`
- `updateimg`
- `all`

## 推荐流程

### 1. 编译 u-boot

```bash
./build.sh uboot
```

产物：

- `u-boot/rk3399_loader_v1.30.130.bin`
- `u-boot/uboot.img`
- `u-boot/trust.img`

### 2. 编译内核

```bash
./build.sh kernel
```

产物：

- `build/out/kernel-7.3/arch/arm64/boot/Image`
- `build/out/kernel-7.3/arch/arm64/boot/dts/rockchip/rk3399-nanopc-t4-dsm.dtb`

### 3. 解包官方 pat 并重打 rd.bin/uInitrd

先把官方 `pat` 放到：

- `build/DSM_DS423_86009.pat`

然后执行：

```bash
./build.sh pat
```

当前默认流程：

1. 使用 `tools/SynoXtract/synoxtract` 解 `pat`
2. 提取 `rd.bin`
3. 解开 `rd.bin`
4. 按 `patches/0001-*.patch` 顺序应用通用补丁（没有补丁时跳过）
5. 修正 `initrd` 里的 `model.dtb` SATA 盘位 PCIe 路径
6. 修正 `initrd` 里的 `synoinfo.conf` 盘位数量
7. 执行 `patches/0000-rk-initrd-fixes.sh` 注入 RK3399 运行时修补
8. 替换 `build/out/kernel-7.3/drivers/hwmon/syno_hddmon.ko`
9. 重新打包 `rd.bin`
10. 生成新的 `uInitrd`

默认会把 DS423 的 `/internal_slot@1..6/ahci/pcie_root` 改成当前
RK3399 + PCIe 转 SATA HBA 的路径：

- `SYNO_SATA_PCIE_ROOT=0000:00:00.0,00.0`

如果 PCIe HBA 枚举路径变化，可以在 `pat` 阶段覆盖：

```bash
SYNO_SATA_PCIE_ROOT=0000:00:00.0,00.0 ./build.sh pat
```

`patches/` 目录不再按 DSM 7.2/7.3 分版本。版本相关且容易冲突的
`linuxrc` / `webman` 修改已改为脚本注入，集中在
`patches/0000-rk-initrd-fixes.sh`。

关键产物：

- `build/rd.bin`
- `build/boot-patched/uInitrd`

### 3.1 只修补已有 initrd 文件

如果已经在 buildroot 或其他流程里拿到了 `rd.bin` / `uInitrd`，不需要走
`PAT` 下载和解包流程，可以直接修补源文件：

```bash
scripts/patch-initrd-file.sh /path/to/rd.bin /path/to/rd.bin.patched
```

这个入口会自己解开输入 initrd、执行同一套 RK3399 initrd 修补、再重新打包。

### 4. 生成 update.img

```bash
./build.sh updateimg
```

产物：

- `output/dsm/boot.img`
- `output/firmware/update.img`
- `output/dsm/rk3399-dsm-update.img`

默认不会在 `extlinux.conf` 固定 MAC 和序列号。如果没有手动传
`mac1=`/`sn=`/`custom_sn=`，U-Boot 会从 RK vendor storage 读取
LAN MAC 和 SN，并在启动内核前补成 DSM 需要的参数：

- `root=/dev/md0`
- `SYNO_FW_VERSION=M.115`

`SYNO_FW_VERSION` 要和当前 DS423 `PAT` 内的 `uboot_DS423.bin` 版本一致。
DS423 7.3.2-86009 里是 `M.115`。如果这里低于 `PAT` 内的版本，
官方 `updater` 会尝试更新 `/dev/mtd7`，在 RK3399 启动环境里会因为没有
Synology 原生 MTD flash 而安装失败。

如需强制从 `extlinux.conf` 传 MAC 或序列号，构建时传环境变量即可。
`SYNO_MAC1` 使用 12 位十六进制，不带冒号；`SYNO_CUSTOM_SN` 不设置时
默认跟随 `SYNO_SN`：

```bash
SYNO_MAC1=021132423001 SYNO_SN=RKG3399DS42301 SYNO_FW_VERSION=M.115 ./build.sh updateimg
```

### 5. 全流程

```bash
./build.sh all
```

顺序是：

1. `uboot`
2. `kernel`
3. `pat`
4. `updateimg`

## 脚本说明

- [build.sh](/home/yxl/my_proj/syno/build.sh:1)
  - 顶层入口，只转发到 `scripts/build-main.sh`
- [scripts/build-main.sh](/home/yxl/my_proj/syno/scripts/build-main.sh:1)
  - 主流程控制
- [scripts/build-uboot.sh](/home/yxl/my_proj/syno/scripts/build-uboot.sh:1)
  - 编译 `u-boot`
- [scripts/pack-updateimg.sh](/home/yxl/my_proj/syno/scripts/pack-updateimg.sh:1)
  - 编译内核、使用 `build/boot-patched/uInitrd` 组装 `boot.img/update.img`
- [scripts/common.sh](/home/yxl/my_proj/syno/scripts/common.sh:1)
  - `pat` 解包、`rd.bin` 解包、补丁和重打包公共逻辑
- [scripts/download_and_patch_pat.sh](/home/yxl/my_proj/syno/scripts/download_and_patch_pat.sh:1)
  - 处理 `pat`
- [scripts/patch_boot_a_img.sh](/home/yxl/my_proj/syno/scripts/patch_boot_a_img.sh:1)
  - 把新的 `uInitrd` 写回 `boot_a.img`

## 当前 pat 解包路线

当前优先使用：

- [tools/SynoXtract/synoxtract](/home/yxl/my_proj/syno/tools/SynoXtract/synoxtract:1)

已经验证可以直接解这份官方包：

- [build/DSM_DS423_86009.pat](/home/yxl/my_proj/syno/build/DSM_DS423_86009.pat:1)

并正确提取：

- `rd.bin`
- `zImage`
- `model.dtb`
- `VERSION`

## 一句话记法

日常只需要记这几个命令：

```bash
./build.sh uboot
./build.sh kernel
./build.sh pat
./build.sh updateimg
```

或者直接：

```bash
./build.sh all
```
