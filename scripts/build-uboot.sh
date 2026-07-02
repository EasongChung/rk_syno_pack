#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UBOOT_DIR="$ROOT_DIR/u-boot"

SOC="${SOC:-rk3399}"

case "$SOC" in
	rk3399)
		DEFAULT_DEFCONFIG="evb-rk3399"
		DEFAULT_LOADER_PATTERN="rk3399_loader*.bin"
		NEED_TRUST=1
		;;
	rk3566)
		DEFAULT_DEFCONFIG="rk3566"
		DEFAULT_LOADER_PATTERN="rk356x_spl_loader*.bin"
		NEED_TRUST=0
		;;
	rk3568)
		DEFAULT_DEFCONFIG="rk3568"
		DEFAULT_LOADER_PATTERN="rk356x_spl_loader*.bin"
		NEED_TRUST=0
		;;
	*)
		printf '[uboot] unsupported SOC: %s\n' "$SOC" >&2
		exit 2
		;;
esac

DEFCONFIG="${UBOOT_DEFCONFIG:-$DEFAULT_DEFCONFIG}"
LOADER_PATTERN="${UBOOT_LOADER_PATTERN:-$DEFAULT_LOADER_PATTERN}"
CROSS_COMPILE="${CROSS_COMPILE:-$ROOT_DIR/tools/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-}"

printf '[uboot] building %s_defconfig for %s\n' "$DEFCONFIG" "$SOC"
# Keep the Rockchip build entrypoint intact and only provide the toolchain prefix.
(
  cd "$UBOOT_DIR"
  ./make.sh CROSS_COMPILE="$CROSS_COMPILE" "$DEFCONFIG"
)

test -f "$UBOOT_DIR/uboot.img"
if [ "$NEED_TRUST" -eq 1 ]; then
	test -f "$UBOOT_DIR/trust.img"
fi
for loader in "$UBOOT_DIR"/$LOADER_PATTERN; do
	if [ -f "$loader" ]; then
		printf '[uboot] loader: %s\n' "$loader"
		exit 0
	fi
done

printf '[uboot] missing loader matching %s/%s\n' "$UBOOT_DIR" "$LOADER_PATTERN" >&2
exit 1
