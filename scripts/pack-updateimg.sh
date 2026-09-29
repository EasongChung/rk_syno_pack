#!/bin/bash
set -euo pipefail

usage()
{
	cat <<'EOF'
Usage:
  ./scripts/pack-updateimg.sh [--kernel-only] [--pack-only]

Build the DSM Rockchip kernel, repack uInitrd, then generate a Rockchip update.img.

Inputs:
  ../linux-5.10.x                                  (default kernel source, overridable by KERNEL_SRC)
  arch/arm64/configs/<soc>_dsm_defconfig           (default kernel config target)
  ../build/pat-rd-patched                         (patched initrd rootfs)
  ../build/boot-patched/uInitrd                    (patched uInitrd from PAT)

Outputs:
  ../build/out/kernel-7.3/arch/arm64/boot/Image
  ../build/out/kernel-7.3/arch/arm64/boot/dts/rockchip/<dtb>
  ../build/boot-patched/uInitrd
  output/dsm/boot.img
  output/firmware/update.img
  output/dsm/<soc>-dsm-update.img
  output/dsm/<soc>-dsm-raw.img                  (rkdeveloptool wl 0x0 image, when RAW_BOOTLOADER_BIN exists)

Options:
  --kernel-only   stop after building kernel Image and dtb
  --pack-only     skip kernel/initrd rebuild, only package existing artifacts

Environment:
  SOC             Rockchip SoC selector: rk3399, rk3566, rk3568; defaults to rk3399
  DTB_NAME        DTB file name under arch/arm64/boot/dts/rockchip
  LOADER_BIN      MiniLoaderAll.bin source path; auto-detected from u-boot when unset
  CONSOLE         kernel console bootarg, defaults from SOC
  EARLYCON        kernel earlycon bootarg, defaults from SOC
  KERNEL_SRC      kernel source tree, defaults to ../linux-5.10.x
  KERNEL_BUILD    kernel out dir, defaults to ../build/out/kernel-7.3
  KERNEL_DEFCONFIG kernel defconfig target, defaults from SOC
  INITRD_ROOT     unpacked initrd root, defaults to ../build/pat-rd-patched
  PATCHED_UINITRD patched uInitrd path, defaults to ../build/boot-patched/uInitrd
  ROOT_MODULE_SRC initrd root module source dir, defaults to INITRD_ROOT/usr/lib/modules
  SYNO_MAC1       optional fixed DSM LAN1 MAC, 12 hex chars without ':'
  SYNO_SN         optional fixed DSM serial number, max 31 chars, no whitespace
  SYNO_CUSTOM_SN  optional fixed DSM custom serial, defaults to SYNO_SN when set
  SYNO_FW_VERSION fixed DSM uboot version marker, defaults to M.115 for DS423 86009
  SYNO_BOOT_LOGO  optional BMP copied to /logo.bmp in the FAT32 boot image
  SWIOTLB         kernel swiotlb slot count, defaults to 32768 (64 MiB)
  COHERENT_POOL   kernel coherent_pool bootarg, defaults to 4M
  RAW_BOOTLOADER_BIN optional raw bootloader block for rkdeveloptool wl 0x0 image
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
	local config_id="$KERNEL_BUILD/.dsm-defconfig"
	local defconfig="$KERNEL_SRC/arch/arm64/configs/$KERNEL_DEFCONFIG"
	local applied_defconfig=""

	if [ -f "$config_id" ]; then
		read -r applied_defconfig < "$config_id" || true
	fi

	if [ ! -f "$config" ] || [ "$applied_defconfig" != "$KERNEL_DEFCONFIG" ]; then
		log "applying kernel config $KERNEL_DEFCONFIG"
		make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" "$KERNEL_DEFCONFIG"
		printf '%s\n' "$KERNEL_DEFCONFIG" > "$config_id"
		return 0
	fi

	if [ "$defconfig" -nt "$config" ]; then
		log "$KERNEL_DEFCONFIG is newer than .config, reapplying defconfig"
		make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" "$KERNEL_DEFCONFIG"
		printf '%s\n' "$KERNEL_DEFCONFIG" > "$config_id"
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

ensure_cross_compile()
{
	local gcc="${CROSS_COMPILE:-}gcc"

	if [ -n "${CROSS_COMPILE:-}" ] && command -v "$gcc" >/dev/null 2>&1; then
		return 0
	fi

	CROSS_COMPILE="$(detect_cross_compile)" || \
		die "set CROSS_COMPILE to a working aarch64 toolchain prefix"
	export CROSS_COMPILE
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

find_first_file()
{
	local pattern="$1"
	local file

	for file in $pattern; do
		if [ -f "$file" ]; then
			echo "$file"
			return 0
		fi
	done

	return 1
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

# Wire the OEC green LED to disk activity the way the stock firmware does:
# the AHCI software-activity path reads led_type/led_green from the slot node
# and pokes the LED class trigger "disk_syno" on every queued command. The
# factory synobios module (which normally flips the trigger's "activated"
# flag) cannot load against this kernel because it re-exports a symbol that
# is built in, so the trigger is activated at registration instead.
patch_kernel_disk_led()
{
	local dtsi="$KERNEL_SRC/arch/arm64/boot/dts/rockchip/rk3566-oec-box.dtsi"
	local dts="$KERNEL_SRC/arch/arm64/boot/dts/rockchip/rk3566-oec-box-wxy4-dsm.dts"
	local trig="$KERNEL_SRC/drivers/leds/trigger/ledtrig-disk-syno.c"

	if [ ! -f "$dts" ]; then
		return 0
	fi

	if grep -q 'led_green = <&green_led>' "$dts"; then
		log "disk-activity LED wiring already present in DTS"
	else
		need_file "$dtsi" "DTS include for this SoC is missing"
		log "wiring green LED to disk activity in DTS"

		if ! grep -q 'green_led: green-led' "$dtsi"; then
			perl -0pi -e 's/\t\tgreen-led \{/\t\tgreen_led: green-led {/' "$dtsi"
		fi
		if ! grep -q 'linux,default-trigger = "disk_syno"' "$dtsi"; then
			perl -0pi -e 's/(\t\tgreen_led: green-led \{\n)/$1\t\t\tlinux,default-trigger = "disk_syno";\n/' "$dtsi"
		fi

		perl -0pi -e 's/(\tinternal_slot\@\d+ \{\n\t\tprotocol_type = "sata";\n)/$1\t\tled_type = "trig_disk_syno";\n\t\tled_green = <&green_led>;\n/g' "$dts"
		perl -0pi -e 's/(\t\tahci \{\n)/$1\t\t\tsw_activity = <1>;\n/g' "$dts"

		grep -q 'green_led: green-led' "$dtsi" || die "$dtsi: failed to label green-led"
		grep -q 'linux,default-trigger = "disk_syno"' "$dtsi" || die "$dtsi: failed to add default trigger"
		grep -q 'led_type = "trig_disk_syno"' "$dts" || die "$dts: failed to add led_type"
		grep -q 'led_green = <&green_led>' "$dts" || die "$dts: failed to add led_green"
		grep -q 'sw_activity = <1>' "$dts" || die "$dts: failed to add sw_activity"
	fi

	# Dual-color disk activity blink: alternate green/blue on IO instead of
	# on/off blinking. Slots without led_blue fall back to the stock oneshot.
	local ahci_c="$KERNEL_SRC/drivers/ata/libahci.c"
	local ahci_h="$KERNEL_SRC/drivers/ata/ahci.h"
	local synolib_h="$KERNEL_SRC/include/linux/synolib.h"
	local alt_body

	if [ -f "$dtsi" ] && ! grep -q 'blue_led: blue-led' "$dtsi"; then
		perl -0pi -e 's/\t\tblue-led \{/\t\tblue_led: blue-led {/' "$dtsi"
		grep -q 'blue_led: blue-led' "$dtsi" || die "$dtsi: failed to label blue-led"
	fi

	if ! grep -q 'led_blue = <&blue_led>' "$dts"; then
		perl -0pi -e 's/(\t\tled_type = "trig_disk_syno";\n)/$1\t\tled_blue = <&blue_led>;\n/g' "$dts"
		grep -q 'led_blue = <&blue_led>' "$dts" || die "$dts: failed to add led_blue"
	fi

	if [ -f "$synolib_h" ] && ! grep -q 'DT_HDD_ALT_LED' "$synolib_h"; then
		perl -0pi -e 's/(\#define DT_HDD_GREEN_LED "led_green"\n)/$1\#define DT_HDD_ALT_LED "led_blue"\n/' "$synolib_h"
		grep -q 'DT_HDD_ALT_LED' "$synolib_h" || die "$synolib_h: failed to add DT_HDD_ALT_LED"
	fi

	if [ -f "$ahci_h" ] && ! grep -q 'syno_alt_phase' "$ahci_h"; then
		perl -0pi -e 's/(\tint\t+\(\*syno_set_blink\)\(struct ata_port\* ap, u32 state\);\n)/$1\tunsigned int\t\tsyno_alt_phase;\n/' "$ahci_h"
		grep -q 'syno_alt_phase' "$ahci_h" || die "$ahci_h: failed to add syno_alt_phase"
	fi

	if [ -f "$ahci_c" ] && ! grep -q 'DT_HDD_ALT_LED' "$ahci_c"; then
		log "wiring dual-color disk LED blink into libahci.c"
		alt_body="$(cat <<'NBU_ALT_EOF'
static int sw_activity_by_ledtrig_disk_syno(struct ata_port* ap, u32 state)
{
	int ret = -EINVAL;
	struct device_node *pSlotNode = NULL;
	struct device_node *pLedNode = NULL;
	struct device_node *pAltNode = NULL;
	struct led_classdev *led_cdev = NULL;
	struct led_classdev *alt_cdev = NULL;
	struct ahci_port_priv *pp = NULL;

	if (!ap) {
		goto Err;
	}

	for_each_child_of_node(of_root, pSlotNode) {
		if (ap->ops->syno_compare_node_info(ap, pSlotNode)) {
			break;
		}
	}
	if (!pSlotNode) {
		goto Err;
	}

	pLedNode = of_parse_phandle(pSlotNode, DT_HDD_GREEN_LED, 0);
	pAltNode = of_parse_phandle(pSlotNode, DT_HDD_ALT_LED, 0);
	of_node_put(pSlotNode);
	if (!pLedNode) {
		goto Err;
	}

	led_cdev = of_leddev_get(pLedNode);
	of_node_put(pLedNode);
	if (IS_ERR(led_cdev)) {
		goto Err;
	}

	if (pAltNode) {
		alt_cdev = of_leddev_get(pAltNode);
		of_node_put(pAltNode);
		if (IS_ERR(alt_cdev)) {
			alt_cdev = NULL;
		}
	}

	pp = ap->private_data;
	if (!pp) {
		goto Err;
	}

	if (SYNO_LED_BLINK_ON == state) {
		if (led_cdev->activated && alt_cdev) {
			pp->syno_alt_phase = !pp->syno_alt_phase;
			if (pp->syno_alt_phase) {
				led_set_brightness_nosleep(led_cdev, LED_OFF);
				led_set_brightness_nosleep(alt_cdev, alt_cdev->max_brightness);
			} else {
				led_set_brightness_nosleep(led_cdev, led_cdev->max_brightness);
				led_set_brightness_nosleep(alt_cdev, LED_OFF);
			}
		} else {
			ledtrig_syno_disk_activity_on(led_cdev);
		}
	} else if (SYNO_LED_BLINK_OFF == state) {
		if (led_cdev->activated) {
			led_set_brightness_nosleep(led_cdev, led_cdev->max_brightness);
		}
		if (alt_cdev) {
			led_set_brightness_nosleep(alt_cdev, LED_OFF);
		}
	}

	ret = 0;

Err:
	return ret;
}
#endif /* MY_ABC_HERE */
NBU_ALT_EOF
)"
		perl -0pi -e 'BEGIN { $nb = pop; } s/static int sw_activity_by_ledtrig_disk_syno\(struct ata_port\* ap, u32 state\)\n\{.*?\n\}\n#endif \/\* MY_ABC_HERE \*\//$nb/s' "$ahci_c" "$alt_body"
		grep -q 'DT_HDD_ALT_LED' "$ahci_c" || die "$ahci_c: failed to patch dual-color blink"
	fi

	if [ -f "$trig" ] && ! grep -q 'led_cdev->activated = true' "$trig"; then
		log "activating disk LED trigger by default in $trig"
		perl -0pi -e 's/(syno_disk_trig_activate\(struct led_classdev \*led_cdev\)\n\{\n)/$1\tled_cdev->activated = true;\n/' "$trig"
		grep -q 'led_cdev->activated = true' "$trig" || die "$trig: failed to enable default activation"
	fi
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
SOC="${SOC:-rk3399}"
OUT_DIR="$PROJECT_DIR/output/dsm"
BOOT_ROOT="$OUT_DIR/boot-root"
BOOT_IMG="$OUT_DIR/boot.img"
FIRMWARE_DIR="$PROJECT_DIR/output/firmware"
PACK_TOOL_DIR="$PROJECT_DIR/tools/linux_pack"
RAW_PACK_SCRIPT="$PROJECT_DIR/scripts/pack-raw-disk-img.sh"
KERNEL_SRC="${KERNEL_SRC:-$PROJECT_DIR/linux-5.10.x}"
KERNEL_BUILD="${KERNEL_BUILD:-$PROJECT_DIR/build/out/kernel-7.3}"
INITRD_ROOT="${INITRD_ROOT:-$PROJECT_DIR/build/pat-rd-patched}"
PATCHED_UINITRD="${PATCHED_UINITRD:-$PROJECT_DIR/build/boot-patched/uInitrd}"
ROOT_MODULE_SRC="${ROOT_MODULE_SRC:-$INITRD_ROOT/usr/lib/modules}"
SYNO_MAC1="${SYNO_MAC1:-}"
SYNO_SN="${SYNO_SN:-}"
SYNO_CUSTOM_SN="${SYNO_CUSTOM_SN:-$SYNO_SN}"
SYNO_FW_VERSION="${SYNO_FW_VERSION:-M.115}"
SYNO_BOOT_LOGO="${SYNO_BOOT_LOGO:-$PROJECT_DIR/assets/boot-logo/logo.bmp}"
SWIOTLB="${SWIOTLB:-32768}"
COHERENT_POOL="${COHERENT_POOL:-4M}"
RAW_INITRD="$OUT_DIR/uInitrd.raw"
LZMA_INITRD="$OUT_DIR/uInitrd.lzma"
KERNEL_IMAGE="$KERNEL_BUILD/arch/arm64/boot/Image"
JOBS="${JOBS:-$(nproc)}"

case "$SOC" in
	rk3399)
		DEFAULT_DTB_NAME="rk3399-nanopc-t4-dsm.dtb"
		DEFAULT_KERNEL_DEFCONFIG="rk3399_dsm_defconfig"
		DEFAULT_CONSOLE="ttyS2,1500000"
		DEFAULT_EARLYCON="uart8250,mmio32,0xff1a0000"
		LOADER_PATTERN="$PROJECT_DIR/u-boot/rk3399_loader*.bin"
		EXTLINUX_LABEL="DSM-rk3399"
		NEED_TRUST=1
		;;
	rk3566)
		DEFAULT_DTB_NAME="rk3566-oec-box-wxy4-dsm.dtb"
		DEFAULT_KERNEL_DEFCONFIG="rk3566_dsm_defconfig"
		DEFAULT_CONSOLE="ttyS2,1500000"
		DEFAULT_EARLYCON="uart8250,mmio32,0xfe660000"
		LOADER_PATTERN="$PROJECT_DIR/tools/rkbin/rk3566/wxy-oect/MiniLoaderAll.bin $PROJECT_DIR/u-boot/rk356x_spl_loader*.bin"
		EXTLINUX_LABEL="DSM-rk3566"
		NEED_TRUST=0
		NEED_UBOOT=0
		NEED_PARAMETER=1
		;;
	rk3568)
		DEFAULT_DTB_NAME="rk3568-evb1-ddr4-v10.dtb"
		DEFAULT_KERNEL_DEFCONFIG="rk3568_dsm_defconfig"
		DEFAULT_CONSOLE="ttyS2,1500000"
		DEFAULT_EARLYCON="uart8250,mmio32,0xfe660000"
		LOADER_PATTERN="$PROJECT_DIR/u-boot/rk356x_spl_loader*.bin"
		EXTLINUX_LABEL="DSM-rk3568"
		NEED_TRUST=0
		NEED_UBOOT=1
		NEED_PARAMETER=1
		;;
	*)
		die "unsupported SOC: $SOC"
		;;
esac

if [ "$SOC" = "rk3399" ]; then
	NEED_UBOOT=1
	NEED_PARAMETER=1
fi

DTB_NAME="${DTB_NAME:-$DEFAULT_DTB_NAME}"
KERNEL_DEFCONFIG="${KERNEL_DEFCONFIG:-$DEFAULT_KERNEL_DEFCONFIG}"
CONSOLE="${CONSOLE:-$DEFAULT_CONSOLE}"
EARLYCON="${EARLYCON:-$DEFAULT_EARLYCON}"
CHIP_DIR="${CHIP_DIR:-$PROJECT_DIR/tools/rkbin/$SOC}"
OUTPUT_DATE="${OUTPUT_DATE:-$(date +%Y%m%d)}"
UPDATE_OUT="$OUT_DIR/${UPDATE_BASENAME:-$SOC-dsm-update_$OUTPUT_DATE.img}"
RAW_UPDATE_OUT="$OUT_DIR/${RAW_UPDATE_BASENAME:-$SOC-dsm-raw_$OUTPUT_DATE.img}"
KERNEL_DTB="$KERNEL_BUILD/arch/arm64/boot/dts/rockchip/$DTB_NAME"
RAW_BOOTLOADER_BIN="${RAW_BOOTLOADER_BIN:-}"
if [ -z "$RAW_BOOTLOADER_BIN" ] && [ "$SOC" = "rk3566" ]; then
	RAW_BOOTLOADER_BIN="$PROJECT_DIR/tools/rkbin/rk3566/wxy-oect/bootloader.bin"
fi
LOADER_BIN="${LOADER_BIN:-}"
if [ -z "$LOADER_BIN" ]; then
	LOADER_BIN="$(find_first_file "$LOADER_PATTERN" || true)"
fi

if [ "$KERNEL_ONLY" -eq 0 ]; then
	command -v mkfs.vfat >/dev/null || die "mkfs.vfat is missing"
	command -v mcopy >/dev/null || die "mcopy is missing"
	command -v cpio >/dev/null || die "cpio is missing"
	command -v lzma >/dev/null || die "lzma is missing"
	command -v dd >/dev/null || die "dd is missing"
	command -v xxd >/dev/null || die "xxd is missing"
fi

validate_syno_identity

if [ "$KERNEL_ONLY" -eq 0 ]; then
	need_file "$CHIP_DIR/package-file-dsm"
	if [ "$NEED_PARAMETER" -eq 1 ]; then
		need_file "$CHIP_DIR/parameter-dsm.txt"
	fi
	need_file "$LOADER_BIN" "build u-boot first or set LOADER_BIN"
	if [ "$NEED_UBOOT" -eq 1 ]; then
		need_file "$PROJECT_DIR/u-boot/uboot.img" "build u-boot first"
	fi
	if [ "$NEED_TRUST" -eq 1 ]; then
		need_file "$PROJECT_DIR/u-boot/trust.img" "build u-boot first"
	fi
	need_file "$PACK_TOOL_DIR/afptool" "missing local Rockchip pack tool"
	need_file "$PACK_TOOL_DIR/rkImageMaker" "missing local Rockchip pack tool"
fi

if [ "$PACK_ONLY" -eq 0 ]; then
	need_file "$KERNEL_SRC/Makefile" "kernel source tree is missing"
	need_file "$KERNEL_SRC/arch/arm64/configs/$KERNEL_DEFCONFIG" "kernel defconfig is missing"

	ensure_cross_compile

	configure_kernel_if_needed
	patch_kernel_disk_led
	log "building kernel Image and dtb"
	make -C "$KERNEL_SRC" O="$KERNEL_BUILD" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS" Image Image.gz dtbs
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

need_file "$KERNEL_IMAGE" "build the DSM $SOC kernel first"
need_file "$KERNEL_DTB" "build the DSM $SOC dtb first"
need_file "$PATCHED_UINITRD" "build or restore DSM uInitrd first"
if [ -n "${SYNO_BOOT_LOGO:-}" ]; then
	need_file "$SYNO_BOOT_LOGO" "SYNO_BOOT_LOGO file not found"
fi

log "preparing DSM boot tree"
rm -rf "$BOOT_ROOT"
mkdir -p "$BOOT_ROOT/boot/extlinux" "$BOOT_ROOT/extlinux"
install -D -m 0644 "$PATCHED_UINITRD" "$BOOT_ROOT/boot/uInitrd"

install -D -m 0755 "$KERNEL_IMAGE" "$BOOT_ROOT/boot/Image"
install -D -m 0644 "$KERNEL_DTB" "$BOOT_ROOT/boot/$DTB_NAME"
if [ -n "${SYNO_BOOT_LOGO:-}" ]; then
	install -D -m 0644 "$SYNO_BOOT_LOGO" "$BOOT_ROOT/logo.bmp"
fi
install_root_modules

BOOTARGS=(
	root="${ROOT_DEVICE:-/dev/md0}"
	netif_num=1
	syno_hw_version="${SYNO_HW_VERSION:-DS423}"
	syno_fw_version="$SYNO_FW_VERSION"
	uio_pdrv_genirq.of_id=generic-uio
	vender_format_version=2
	vendor_format_version=2
	console="$CONSOLE"
	earlycon="$EARLYCON"
	fw_devlink=permissive
	swiotlb="$SWIOTLB"
	coherent_pool="$COHERENT_POOL"
)

case "$SOC" in
	rk3399)
		BOOTARGS+=(
			vt.global_cursor_default=0
			fbcon=map:1
		)
		;;
	*)
		;;
esac

if [ -n "$SYNO_MAC1" ]; then
	BOOTARGS+=(mac1="$SYNO_MAC1")
fi
if [ -n "$SYNO_SN" ]; then
	BOOTARGS+=(sn="$SYNO_SN")
fi
if [ -n "$SYNO_CUSTOM_SN" ]; then
	BOOTARGS+=(custom_sn="$SYNO_CUSTOM_SN")
fi
if [ -n "${DEBUG_BOOTARGS:-}" ]; then
	# Space-separated extra kernel arguments for one-off boot diagnostics.
	# Example: DEBUG_BOOTARGS="deferred_probe_timeout=5 initcall_debug ignore_loglevel loglevel=8"
	# shellcheck disable=SC2206
	DEBUG_BOOTARGS_ARRAY=($DEBUG_BOOTARGS)
	BOOTARGS+=("${DEBUG_BOOTARGS_ARRAY[@]}")
fi

{
	cat <<EOF
label $EXTLINUX_LABEL
  kernel /boot/Image
  initrd /boot/uInitrd
  fdt /boot/$DTB_NAME
EOF
	printf '  append'
	printf ' %s' "${BOOTARGS[@]}"
	printf '\n'
} > "$BOOT_ROOT/boot/extlinux/extlinux.conf"
cp -f "$BOOT_ROOT/boot/extlinux/extlinux.conf" "$BOOT_ROOT/extlinux/extlinux.conf"

log "building FAT32 boot image"
rm -f "$BOOT_IMG"
truncate -s 32M "$BOOT_IMG"
mkfs.vfat -F 32 -n DSMBOOT "$BOOT_IMG" >/dev/null
MTOOLS_SKIP_CHECK=1 mcopy -i "$BOOT_IMG" -s "$BOOT_ROOT"/* ::/

log "preparing firmware links"
mkdir -p "$FIRMWARE_DIR"
rm -f "$FIRMWARE_DIR/misc.img" "$FIRMWARE_DIR/parameter.txt" \
	"$FIRMWARE_DIR/uboot.img" "$FIRMWARE_DIR/trust.img"
ln -rsf "$LOADER_BIN" "$FIRMWARE_DIR/MiniLoaderAll.bin"
if [ "$NEED_UBOOT" -eq 1 ]; then
	ln -rsf "$PROJECT_DIR/u-boot/uboot.img" "$FIRMWARE_DIR/uboot.img"
fi
if [ "$NEED_TRUST" -eq 1 ]; then
	ln -rsf "$PROJECT_DIR/u-boot/trust.img" "$FIRMWARE_DIR/trust.img"
fi
if [ "$NEED_PARAMETER" -eq 1 ]; then
	ln -rsf "$CHIP_DIR/parameter-dsm.txt" "$FIRMWARE_DIR/parameter.txt"
fi
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

if [ -n "$RAW_BOOTLOADER_BIN" ] && [ -r "$RAW_BOOTLOADER_BIN" ]; then
	log "packing raw disk image"
	"$RAW_PACK_SCRIPT" "$RAW_BOOTLOADER_BIN" "$BOOT_IMG" "$RAW_UPDATE_OUT" >/dev/null
	log "done: $RAW_UPDATE_OUT"
elif [ -n "$RAW_BOOTLOADER_BIN" ]; then
	log "skip raw disk image: $RAW_BOOTLOADER_BIN is missing"
fi
