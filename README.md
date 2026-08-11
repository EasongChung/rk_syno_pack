# Synology DSM Rockchip 构建工具

本项目用于编译 Rockchip 平台的 DSM 内核、修补 DS423 PAT/initrd，并生成
可启动的 `boot.img`、Rockchip `update.img` 和 raw eMMC 镜像。

## 支持的平台

| `SOC` | 板卡 | 内核配置 | 默认 DTB |
| --- | --- | --- | --- |
| `rk3399` | FriendlyElec NanoPC-T4 | `rk3399_dsm_defconfig` | `rk3399-nanopc-t4-dsm.dtb` |
| `rk3566` | WXY/OECT Box | `rk3566_dsm_defconfig` | `rk3566-oec-box-wxy4-dsm.dtb` |
| `rk3568` | RK3568 EVB1 DDR4 V10 | `rk3568_dsm_defconfig` | `rk3568-evb1-ddr4-v10.dtb` |

未指定 `SOC` 时默认编译 `rk3399`。建议每条构建命令都显式传入
`SOC`，避免把不同平台的内核、DTB 和 initrd 混在一起。

## 系统依赖

在 Debian/Ubuntu 主机上安装：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential bc bison flex libssl-dev libelf-dev dwarves \
  cpio xz-utils patch curl e2fsprogs device-tree-compiler \
  dosfstools mtools gdisk gzip vim-common \
  libsodium-dev libmsgpack-dev
```

项目已包含 AArch64 交叉工具链、`SynoXtract` 和 Rockchip 打包工具，
通常不需要单独配置 `CROSS_COMPILE`。

## 快速开始

所有命令都在项目根目录执行：

```bash
cd /home/yxl/my_proj/syno
./build.sh --help
```

### RK3566 WXY/OECT

RK3566 使用 `tools/rkbin/rk3566/wxy-oect/` 中的板级 vendor
bootloader，不需要编译或替换 U-Boot。推荐按下面的顺序执行：

```bash
SOC=rk3566 JOBS="$(nproc)" ./build.sh kernel
SOC=rk3566 DSM_PAT_VERSION=7.4.1 ./build.sh pat
SOC=rk3566 ./build.sh updateimg
```

### RK3399

```bash
SOC=rk3399 JOBS="$(nproc)" ./build.sh all
```

也可以分步执行：

```bash
SOC=rk3399 ./build.sh uboot
SOC=rk3399 JOBS="$(nproc)" ./build.sh kernel
SOC=rk3399 DSM_PAT_VERSION=7.4.1 ./build.sh pat
SOC=rk3399 ./build.sh updateimg
```

### RK3568

```bash
SOC=rk3568 JOBS="$(nproc)" ./build.sh all
```

调试时建议分步执行，修改内核后只需重新运行：

```bash
SOC=rk3568 JOBS="$(nproc)" ./build.sh kernel
```

## 构建目标

```text
./build.sh [all|pat|uboot|kernel|updateimg]
```

| target | 作用 |
| --- | --- |
| `kernel` | 应用当前平台 defconfig，编译 `Image`、`Image.gz`、DTB 和全部模块 |
| `pat` | 下载或读取 DS423 PAT，解包并修补 initrd，生成 `rd.bin` 和 `uInitrd` |
| `uboot` | 编译当前平台 U-Boot；主要用于 RK3399 和 RK3568 |
| `updateimg` | 使用已有内核、DTB 和 `uInitrd` 生成启动及烧录镜像 |
| `all` | 依次执行 `uboot -> kernel -> pat -> updateimg` |

内核应通过 `./build.sh kernel` 编译，不要直接调用内核目录中的
`make`。构建脚本会选择对应 defconfig、输出目录和 DTB，并清理已失效的
模块产物。

## 编译内核

基本用法：

```bash
SOC=rk3566 JOBS=32 ./build.sh kernel
```

默认产物目录为 `build/out/kernel-7.3/`，主要文件包括：

```text
build/out/kernel-7.3/arch/arm64/boot/Image
build/out/kernel-7.3/arch/arm64/boot/Image.gz
build/out/kernel-7.3/arch/arm64/boot/dts/rockchip/<DTB_NAME>
build/out/kernel-7.3/drivers/hwmon/syno_hddmon.ko
```

覆盖内核源码、输出目录、配置或 DTB：

```bash
SOC=rk3566 \
KERNEL_SRC=/path/to/linux-5.10.x \
KERNEL_BUILD=/path/to/kernel-out \
KERNEL_DEFCONFIG=rk3566_dsm_defconfig \
DTB_NAME=rk3566-oec-box-wxy4-dsm.dtb \
JOBS=32 \
./build.sh kernel
```

如果使用自定义 `KERNEL_BUILD`，后续 `pat` 和 `updateimg` 也要传入同一个值。

## 修补 PAT 和 initrd

默认下载并处理 DS423 DSM 7.4.1-90080：

```bash
SOC=rk3566 ./build.sh pat
```

可用的版本选择：

| `DSM_PAT_VERSION` | PAT |
| --- | --- |
| `7.4.1` 或 `90080` | DSM 7.4.1-90080，默认 |
| `7.4` 或 `90075` | DSM 7.4-90075 |
| `7.3`、`7.3.2` 或 `86009` | DSM 7.3.2-86009 |

使用本地 PAT：

```bash
SOC=rk3566 \
DSM_PAT_VERSION=7.4.1 \
PAT_FILE=/path/to/DSM_DS423_90080.pat \
./build.sh pat
```

使用自定义下载地址：

```bash
SOC=rk3566 \
PAT_URL=https://example.com/DSM_DS423_xxxxx.pat \
PAT_FILE="$PWD/build/DSM_DS423_xxxxx.pat" \
./build.sh pat
```

主要产物：

```text
build/pat-extract/          PAT 解包目录
build/pat-rd/               原始 initrd rootfs
build/pat-rd-patched/       修补后的 initrd rootfs
build/rd.bin                修补后的 rd.bin
build/boot-patched/uInitrd  打包镜像使用的 initrd
```

`pat` 阶段会使用当前内核输出中的 `syno_hddmon.ko`，因此第一次完整构建时
应先执行 `kernel`，再执行 `pat`。

已有 `rd.bin` 时可以跳过 PAT 下载和解包：

```bash
SOC=rk3566 scripts/patch-initrd-file.sh \
  /path/to/rd.bin \
  /path/to/rd.bin.patched
```

如不需要替换 `syno_hddmon.ko`：

```bash
SOC=rk3566 INSTALL_SYNO_HDDMON=0 \
  scripts/patch-initrd-file.sh /path/to/rd.bin
```

## 编译 U-Boot

RK3399：

```bash
SOC=rk3399 ./build.sh uboot
```

主要产物：

```text
u-boot/rk3399_loader*.bin
u-boot/uboot.img
u-boot/trust.img
```

RK3568：

```bash
SOC=rk3568 ./build.sh uboot
```

RK3566 WXY/OECT 只能使用已验证的板级 vendor bootloader。打包脚本会自动
读取：

```text
tools/rkbin/rk3566/wxy-oect/MiniLoaderAll.bin
tools/rkbin/rk3566/wxy-oect/bootloader.bin
```

不要用通用 RK3566 U-Boot 覆盖该板 bootloader，除非已经准备好通过
MaskROM 恢复。

## 生成镜像

生成镜像前必须已有当前平台的内核、DTB 和
`build/boot-patched/uInitrd`：

```bash
SOC=rk3566 ./build.sh updateimg
```

通用产物：

```text
output/dsm/boot.img
output/firmware/update.img
output/dsm/<soc>-dsm-update_YYYYMMDD.img
```

RK3566 还会生成包含 vendor bootloader 和 GPT 的完整 raw 镜像，以及对应的
gzip 压缩文件：

```text
output/dsm/rk3566-dsm-raw_YYYYMMDD.img
output/dsm/rk3566-dsm-raw_YYYYMMDD.img.gz
```

在 MaskROM/Loader 模式烧写 raw 镜像会覆盖目标 eMMC，请先确认设备：

```bash
sudo rkdeveloptool db tools/rkbin/rk3566/wxy-oect/MiniLoaderAll.bin
sudo rkdeveloptool wl 0x0 output/dsm/rk3566-dsm-raw_YYYYMMDD.img
sudo rkdeveloptool rd
```

raw 镜像默认大小为 128 MiB，包含 32 MiB `boot` 分区和一个占用剩余空间的
`userdata` 分区。可在打包时覆盖：

```bash
SOC=rk3566 \
RAW_IMAGE_SIZE=256M \
BOOT_PART_SIZE=32M \
EXTRA_PARTS='rdnew:16M,userdata:0' \
./build.sh updateimg
```

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `SOC` | `rk3399`、`rk3566` 或 `rk3568`，默认 `rk3399` |
| `JOBS` | 内核并行编译任务数，默认 `nproc` |
| `DSM_PAT_VERSION` | PAT 版本选择，默认 `7.4.1` |
| `PAT_FILE` | 本地 PAT 路径 |
| `PAT_URL` | 自定义 PAT 下载地址 |
| `KERNEL_SRC` | 内核源码目录，默认 `linux-5.10.x` |
| `KERNEL_BUILD` | 内核输出目录，默认 `build/out/kernel-7.3` |
| `KERNEL_DEFCONFIG` | 覆盖当前平台默认 defconfig |
| `DTB_NAME` | 覆盖当前平台默认 DTB 文件名 |
| `CROSS_COMPILE` | 覆盖自动检测到的交叉工具链前缀 |
| `UBOOT_DEFCONFIG` | 覆盖当前平台 U-Boot defconfig |
| `SYNO_MAX_DISKS` | initrd 中的 DSM 内置盘位数量，默认 `6` |
| `SYNO_SATA_PCIE_ROOT` | RK3399/RK3568 SATA HBA 的 `pcie_root` |
| `SYNO_MAC1` | 固定 LAN1 MAC，12 位十六进制且不带冒号 |
| `SYNO_SN` | 固定 DSM 序列号 |
| `SYNO_CUSTOM_SN` | 固定 custom serial，默认跟随 `SYNO_SN` |
| `SYNO_FW_VERSION` | DSM bootarg 版本标记，默认 `M.115` |
| `SWIOTLB` | 内核 `swiotlb` bootarg，默认 `32768` |
| `COHERENT_POOL` | 内核 `coherent_pool` bootarg，默认 `4M` |
| `DEBUG_BOOTARGS` | 临时追加内核调试参数 |
| `OUTPUT_DATE` | 输出文件日期，默认 `YYYYMMDD` |

例如固定身份信息并重新打包：

```bash
SOC=rk3566 \
SYNO_MAC1=021132423001 \
SYNO_SN=RKG3566DS42301 \
SYNO_FW_VERSION=M.115 \
./build.sh updateimg
```

## 目录说明

```text
build.sh                 顶层构建入口
linux-5.10.x/            DSM/Rockchip 内核源码
u-boot/                  Rockchip U-Boot 源码
scripts/                 构建、PAT 和镜像脚本
patches/                 initrd 补丁和 overlay
tools/SynoXtract/        PAT 解包工具
tools/linux_pack/        Rockchip update.img 打包工具
tools/rkbin/             各 SoC/板卡的 rkbin 与分区描述
build/                   中间产物和内核输出
output/                  最终启动及烧录镜像
```

## 常见注意事项

- 修改内核后执行 `SOC=<soc> ./build.sh kernel`。
- 修改 initrd 补丁后执行 `SOC=<soc> ./build.sh pat`，再执行 `updateimg`。
- 切换 `SOC` 后应重新执行 `kernel` 和 `pat`，不要复用上一平台的 initrd。
- `updateimg` 只使用已有产物，不会自动重新编译内核或重新修补 PAT。
- RK3566 WXY/OECT 使用 vendor bootloader，正常流程不需要 `uboot` target。
