#!/bin/bash
set -euo pipefail

usage()
{
	cat <<'EOF'
Usage:
  ./scripts/pack-updateimg.sh [--kernel-only] [--pack-only]

Build the DSM RK3399 kernel, repack uInitrd, then generate a Rockchip update.img.

Inputs:
  ../linux-5.10.x                                  (default kernel source, overridable by KERNEL_SRC)
  arch/arm64/configs/rk3399_dsm_defconfig          (default kernel config target)
  ../build/pat-rd-patched                         (patched initrd rootfs)
  ../build/boot-patched/uInitrd                    (patched uInitrd from PAT)

Outputs:
  ../build/out/kernel-7.3/arch/arm64/boot/Image.gz
  ../build/out/kernel-7.3/arch/arm64/boot/dts/rockchip/rk3399-nanopc-t4-dsm.dtb
  ../build/boot-patched/uInitrd
  output/dsm/boot.img
  output/firmware/update.img
  output/dsm/rk3399-dsm-update.img

Options:
  --kernel-only   stop after building kernel Image and dtb
  --pack-only     skip kernel/initrd rebuild, only package existing artifacts

Environment:
  KERNEL_SRC      kernel source tree, defaults to ../linux-5.10.x
  KERNEL_BUILD    kernel out dir, defaults to ../build/out/kernel-7.3
  KERNEL_DEFCONFIG kernel defconfig target, defaults to rk3399_dsm_defconfig
  INITRD_ROOT     unpacked initrd root, defaults to ../build/pat-rd-patched
  PATCHED_UINITRD patched uInitrd path, defaults to ../build/boot-patched/uInitrd
  ROOT_MODULE_SRC initrd root module source dir, defaults to INITRD_ROOT/usr/lib/modules
  SYNO_MAC1       optional fixed DSM LAN1 MAC, 12 hex chars without ':'
  SYNO_SN         optional fixed DSM serial number, max 31 chars, no whitespace
  SYNO_CUSTOM_SN  optional fixed DSM custom serial, defaults to SYNO_SN when set
  SYNO_FW_VERSION fixed DSM uboot version marker, defaults to M.115 for DS423 86009
  SYNO_BOOT_LOGO  optional BMP copied to /logo.bmp in the FAT32 boot image
  CROSS_COMPILE   toolchain prefix, auto-detected when unset
  JOBS            parallel make jobs, defaults to nproc
EOF
}

die()
{
	echo "error: $*" >&2
	exit 1
}

log()
{
	echo "[dsm-pack] $*"
}

validate_syno_identity()
{
	if [ -n "$SYNO_MAC1" ]; then
		case "$SYNO_MAC1" in
			*[!0-9a-fA-F]*)
				die "SYNO_MAC1 must be 12 hex chars without ':'"
				;;
		esac

		if [ "${#SYNO_MAC1}" -ne 12 ]; then
			die "SYNO_MAC1 must be 12 hex chars without ':'"
		fi
	fi

	if [ -n "$SYNO_SN" ] && { [ "${#SYNO_SN}" -gt 31 ] || [[ "$SYNO_SN" =~ [[:space:]] ]]; }; then
		die "SYNO_SN must be 1-31 chars without whitespace"
	fi

	if [ -n "$SYNO_CUSTOM_SN" ] && { [ "${#SYNO_CUSTOM_SN}" -gt 31 ] || [[ "$SYNO_CUSTOM_SN" =~ [[:space:]] ]]; }; then
		die "SYNO_CUSTOM_SN must be 1-31 chars without whitespace"
	fi

	if [[ ! "$SYNO_FW_VERSION" =~ ^M\.[0-9][0-9][0-9]$ ]]; then
		die "SYNO_FW_VERSION must match M.NNN"
	fi
}

configure_kernel_if_needed()
{
	local config="$KERNEL_BUILD/.config"
	local defconfig="$KERNEL_SRC/arch/arm64/configs/$KERNEL_DEFCONFIG"

	if [ ! -f "$config" ]; then
		log "kernel .config missing, applying $KERNEL_DEFCONFIG"
		make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" "$KERNEL_DEFCONFIG"
		return 0
	fi

	if [ "$defconfig" -nt "$config" ]; then
		log "$KERNEL_DEFCONFIG is newer than .config, reapplying defconfig"
		make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" "$KERNEL_DEFCONFIG"
		return 0
	fi

	log "reusing existing kernel .config"
}

detect_cross_compile()
{
	local prefix
	local gcc
	local sdk_tc_dir

	sdk_tc_dir="$PROJECT_DIR/tools/prebuilts/gcc/linux-x86/aarch64"

	for gcc in \
		"$sdk_tc_dir"/gcc-*/bin/aarch64-rockchip*-gcc \
		"$sdk_tc_dir"/gcc-*/bin/aarch64-none-linux-gnu-gcc \
		"$sdk_tc_dir"/gcc-*/bin/aarch64-linux-gnu-gcc
	do
		if [ -x "$gcc" ]; then
			echo "${gcc%gcc}"
			return 0
		fi
	done

	for prefix in aarch64-rockchip1031-linux-gnu- aarch64-linux-gnu- aarch64-none-linux-gnu-; do
		if command -v "${prefix}gcc" >/dev/null 2>&1; then
			echo "$prefix"
			return 0
		fi
	done

	return 1
}

need_file()
{
	local file="$1"
	local hint="${2:-}"

	if [ -r "$file" ]; then
		return 0
	fi

	if [ -n "$hint" ]; then
		die "$file is missing; $hint"
	fi

	die "$file is missing"
}

root_module_source()
{
	local module="$1"
	local path

	for path in \
		"$ROOT_MODULE_SRC/$module" \
		"$KERNEL_BUILD/fs/btrfs/$module" \
		"$KERNEL_BUILD/lib/zstd/$module" \
		"$KERNEL_BUILD/drivers/target/$module" \
		"$KERNEL_BUILD/drivers/target/iscsi/$module" \
		"$KERNEL_BUILD/drivers/target/loopback/$module" \
		"$KERNEL_BUILD/drivers/vhost/$module"
	do
		if [ -f "$path" ]; then
			echo "$path"
			return 0
		fi
	done

	return 1
}

install_root_modules()
{
	local manifest="$ROOT_MODULE_SRC/rk3399.modules.dep"
	local dest="$BOOT_ROOT/boot/rk3399-root-modules"
	local depfile="$dest/rk3399.modules.dep"
	local line module src

	if [ ! -s "$manifest" ]; then
		log "no rootfs runtime modules to install"
		return 0
	fi

	mkdir -p "$dest"
	cp -f "$manifest" "$depfile"

	while IFS= read -r line; do
		[ -n "$line" ] || continue
		module="${line%%:*}"
		[ -n "$module" ] || continue
		src="$(root_module_source "$module")" || \
			die "root module $module is missing; rebuild kernel first"
		install -m 0644 "$src" "$dest/$module"
	done < "$manifest"
}

prune_stale_kernel_modules()
{
	local order="$KERNEL_BUILD/modules.order"
	local ko rel removed=0
	declare -A current_modules=()

	need_file "$order" "kernel modules were not built"

	while IFS= read -r rel; do
		[ -n "$rel" ] || continue
		current_modules["$rel"]=1
	done < "$order"

	while IFS= read -r -d '' ko; do
		rel="${ko#$KERNEL_BUILD/}"
		if [ -z "${current_modules[$rel]+x}" ]; then
			rm -f "$ko"
			removed=$((removed + 1))
		fi
	done < <(find "$KERNEL_BUILD" -type f -name '*.ko' -print0)

	if [ "$removed" -gt 0 ]; then
		log "removed $removed stale kernel module outputs"
	fi
}

KERNEL_ONLY=0
PACK_ONLY=0

while [ "$#" -gt 0 ]; do
	case "$1" in
		--kernel-only)
			KERNEL_ONLY=1
			;;
		--pack-only)
			PACK_ONLY=1
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			usage >&2
			exit 2
			;;
	esac
	shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_DIR/output/dsm"
BOOT_ROOT="$OUT_DIR/boot-root"
BOOT_IMG="$OUT_DIR/boot.img"
UPDATE_OUT="$OUT_DIR/rk3399-dsm-update.img"
FIRMWARE_DIR="$PROJECT_DIR/output/firmware"
CHIP_DIR="$PROJECT_DIR/tools/rkbin/rk3399"
PACK_TOOL_DIR="$PROJECT_DIR/tools/linux_pack"
KERNEL_SRC="${KERNEL_SRC:-$PROJECT_DIR/linux-5.10.x}"
KERNEL_BUILD="${KERNEL_BUILD:-$PROJECT_DIR/build/out/kernel-7.3}"
KERNEL_DEFCONFIG="${KERNEL_DEFCONFIG:-rk3399_dsm_defconfig}"
INITRD_ROOT="${INITRD_ROOT:-$PROJECT_DIR/build/pat-rd-patched}"
PATCHED_UINITRD="${PATCHED_UINITRD:-$PROJECT_DIR/build/boot-patched/uInitrd}"
ROOT_MODULE_SRC="${ROOT_MODULE_SRC:-$INITRD_ROOT/usr/lib/modules}"
SYNO_MAC1="${SYNO_MAC1:-}"
SYNO_SN="${SYNO_SN:-}"
SYNO_CUSTOM_SN="${SYNO_CUSTOM_SN:-$SYNO_SN}"
SYNO_FW_VERSION="${SYNO_FW_VERSION:-M.115}"
SYNO_BOOT_LOGO="${SYNO_BOOT_LOGO:-$PROJECT_DIR/assets/boot-logo/logo.bmp}"
RAW_INITRD="$OUT_DIR/uInitrd.raw"
LZMA_INITRD="$OUT_DIR/uInitrd.lzma"
KERNEL_IMAGE="$KERNEL_BUILD/arch/arm64/boot/Image.gz"
KERNEL_DTB="$KERNEL_BUILD/arch/arm64/boot/dts/rockchip/rk3399-nanopc-t4-dsm.dtb"
JOBS="${JOBS:-$(nproc)}"

command -v mkfs.vfat >/dev/null || die "mkfs.vfat is missing"
command -v mcopy >/dev/null || die "mcopy is missing"
command -v cpio >/dev/null || die "cpio is missing"
command -v lzma >/dev/null || die "lzma is missing"
command -v dd >/dev/null || die "dd is missing"
command -v xxd >/dev/null || die "xxd is missing"

validate_syno_identity

need_file "$CHIP_DIR/parameter-dsm.txt"
need_file "$CHIP_DIR/package-file-dsm"
need_file "$PROJECT_DIR/u-boot/rk3399_loader_v1.30.130.bin" "build u-boot first"
need_file "$PROJECT_DIR/u-boot/uboot.img" "build u-boot first"
need_file "$PROJECT_DIR/u-boot/trust.img" "build u-boot first"
need_file "$PACK_TOOL_DIR/afptool" "missing local Rockchip pack tool"
need_file "$PACK_TOOL_DIR/rkImageMaker" "missing local Rockchip pack tool"

if [ "$PACK_ONLY" -eq 0 ]; then
	need_file "$KERNEL_SRC/Makefile" "kernel source tree is missing"
	need_file "$KERNEL_SRC/arch/arm64/configs/$KERNEL_DEFCONFIG" "kernel defconfig is missing"

	if [ -z "${CROSS_COMPILE:-}" ]; then
		CROSS_COMPILE="$(detect_cross_compile)" || die "set CROSS_COMPILE to an aarch64 toolchain prefix"
	fi

	configure_kernel_if_needed
	log "building kernel Image and dtb"
	make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS" Image.gz dtbs
	log "building all kernel modules"
	make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS" modules
	prune_stale_kernel_modules

	if [ "$KERNEL_ONLY" -eq 1 ]; then
		log "built $KERNEL_IMAGE and $KERNEL_DTB"
		exit 0
	fi

	need_file "$INITRD_ROOT/linuxrc.syno" "initrd rootfs is missing"

	log "repacking DSM uInitrd"
	mkdir -p "$OUT_DIR" "$(dirname "$PATCHED_UINITRD")"
	rm -f "$RAW_INITRD" "$LZMA_INITRD" "$PATCHED_UINITRD"
	(
		cd "$INITRD_ROOT"
		find . -print | cpio -o -H newc -R 0:0 > "$RAW_INITRD"
	)
	lzma -9 -c "$RAW_INITRD" > "$LZMA_INITRD"
	mv "$LZMA_INITRD" "$PATCHED_UINITRD"
fi

need_file "$KERNEL_IMAGE" "build the DSM RK3399 kernel first"
need_file "$KERNEL_DTB" "build the DSM RK3399 dtb first"
need_file "$PATCHED_UINITRD" "build or restore DSM uInitrd first"
if [ -n "${SYNO_BOOT_LOGO:-}" ]; then
	need_file "$SYNO_BOOT_LOGO" "SYNO_BOOT_LOGO file not found"
fi

log "preparing DSM boot tree"
rm -rf "$BOOT_ROOT"
mkdir -p "$BOOT_ROOT/boot/extlinux"
install -D -m 0644 "$PATCHED_UINITRD" "$BOOT_ROOT/boot/uInitrd"

install -D -m 0755 "$KERNEL_IMAGE" "$BOOT_ROOT/boot/Image.gz"
install -D -m 0644 "$KERNEL_DTB" "$BOOT_ROOT/boot/rk3399-nanopc-t4-dsm.dtb"
if [ -n "${SYNO_BOOT_LOGO:-}" ]; then
	install -D -m 0644 "$SYNO_BOOT_LOGO" "$BOOT_ROOT/logo.bmp"
fi
install_root_modules

BOOTARGS=(
	root=/dev/md0
	netif_num=1
	syno_hw_version=DS423
	syno_fw_version="$SYNO_FW_VERSION"
	uio_pdrv_genirq.of_id=generic-uio
	vender_format_version=2
	vendor_format_version=2
	console=ttyS2,1500000
	earlycon=uart8250,mmio32,0xff1a0000
	fw_devlink=permissive
	swiotlb=1
	coherent_pool=1m
	vt.global_cursor_default=0
	fbcon=map:1
)

if [ -n "$SYNO_MAC1" ]; then
	BOOTARGS+=(mac1="$SYNO_MAC1")
fi
if [ -n "$SYNO_SN" ]; then
	BOOTARGS+=(sn="$SYNO_SN")
fi
if [ -n "$SYNO_CUSTOM_SN" ]; then
	BOOTARGS+=(custom_sn="$SYNO_CUSTOM_SN")
fi

{
	cat <<'EOF'
label DSM-rk3399
  kernel /boot/Image.gz
  initrd /boot/uInitrd
  fdt /boot/rk3399-nanopc-t4-dsm.dtb
EOF
	printf '  append'
	printf ' %s' "${BOOTARGS[@]}"
	printf '\n'
} > "$BOOT_ROOT/boot/extlinux/extlinux.conf"

log "building FAT32 boot image"
rm -f "$BOOT_IMG"
truncate -s 32M "$BOOT_IMG"
mkfs.vfat -F 32 -n DSMBOOT "$BOOT_IMG" >/dev/null
MTOOLS_SKIP_CHECK=1 mcopy -i "$BOOT_IMG" -s "$BOOT_ROOT"/* ::/

log "preparing firmware links"
mkdir -p "$FIRMWARE_DIR"
rm -f "$FIRMWARE_DIR/misc.img"
ln -rsf "$PROJECT_DIR/u-boot/rk3399_loader_v1.30.130.bin" "$FIRMWARE_DIR/MiniLoaderAll.bin"
ln -rsf "$PROJECT_DIR/u-boot/uboot.img" "$FIRMWARE_DIR/uboot.img"
ln -rsf "$PROJECT_DIR/u-boot/trust.img" "$FIRMWARE_DIR/trust.img"
ln -rsf "$CHIP_DIR/parameter-dsm.txt" "$FIRMWARE_DIR/parameter.txt"
ln -rsf "$BOOT_IMG" "$FIRMWARE_DIR/boot.img"

log "packing Rockchip update.img"
rm -rf "$FIRMWARE_DIR/update.raw.img" "$FIRMWARE_DIR/update.img"
(
	cd "$FIRMWARE_DIR"
	ln -rsf "$CHIP_DIR/package-file-dsm" package-file
	TAG="RK$(dd if=MiniLoaderAll.bin bs=1 skip=21 count=4 status=none | xxd -p -c 4 | xxd -r -p | rev)"
	"$PACK_TOOL_DIR/afptool" -pack ./ update.raw.img
	"$PACK_TOOL_DIR/rkImageMaker" -"$TAG" MiniLoaderAll.bin update.raw.img update.img -os_type:androidos
)

need_file "$FIRMWARE_DIR/update.img" "Rockchip update image pack failed"
cp -f "$FIRMWARE_DIR/update.img" "$UPDATE_OUT"
log "done: $UPDATE_OUT"
