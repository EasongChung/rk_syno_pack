#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UBOOT_DIR="$ROOT_DIR/u-boot"

DEFCONFIG="${UBOOT_DEFCONFIG:-evb-rk3399}"
CROSS_COMPILE="${CROSS_COMPILE:-$ROOT_DIR/tools/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-}"

printf '[uboot] building %s_defconfig\n' "$DEFCONFIG"
# Keep the Rockchip build entrypoint intact and only provide the toolchain prefix.
(
  cd "$UBOOT_DIR"
  ./make.sh CROSS_COMPILE="$CROSS_COMPILE" "$DEFCONFIG"
)

test -f "$UBOOT_DIR/uboot.img"
test -f "$UBOOT_DIR/trust.img"
test -f "$UBOOT_DIR/rk3399_loader_v1.30.130.bin"
