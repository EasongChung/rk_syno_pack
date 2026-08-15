#!/bin/bash
set -euo pipefail

die()
{
	echo "error: $*" >&2
	exit 1
}

usage()
{
	cat <<'EOF'
Usage:
  scripts/pack-raw-disk-img.sh <bootloader.bin> <boot.img> <output.img>

Create a raw eMMC image for rkdeveloptool:
  rkdeveloptool wl 0x0 <output.img>

The bootloader block is copied as-is. For WXY/OECT rk3566 boards this keeps
the closed-source SPL/U-Boot verified boot chain intact.

Optional:
  EXTRA_PARTS='name:size[,name:size...]'
    Add GPT partitions after boot. Size uses sgdisk syntax, for example
    EXTRA_PARTS='rdnew:16M,userdata:0'
    Defaults to userdata:0, matching tools/rkbin/rk3566/parameter-dsm.txt.
EOF
}

if [ "$#" -ne 3 ]; then
	usage >&2
	exit 2
fi

BOOTLOADER="$1"
BOOT_IMG="$2"
OUTPUT="$3"
BOOT_START_SECTOR="${BOOT_START_SECTOR:-32768}"
BOOT_PART_SIZE="${BOOT_PART_SIZE:-32M}"
IMAGE_SIZE="${RAW_IMAGE_SIZE:-128M}"
EXTRA_PARTS="${EXTRA_PARTS:-userdata:0}"

[ -r "$BOOTLOADER" ] || die "$BOOTLOADER is missing"
[ -r "$BOOT_IMG" ] || die "$BOOT_IMG is missing"
command -v sgdisk >/dev/null || die "sgdisk is missing"
command -v dd >/dev/null || die "dd is missing"
command -v truncate >/dev/null || die "truncate is missing"
command -v gzip >/dev/null || die "gzip is missing"

mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
truncate -s "$IMAGE_SIZE" "$OUTPUT"

dd if="$BOOTLOADER" of="$OUTPUT" bs=4M conv=notrunc status=none

sgdisk --clear \
	--set-alignment=1 \
	--new=1:"$BOOT_START_SECTOR":+"$BOOT_PART_SIZE" \
	--typecode=1:0700 \
	--change-name=1:boot \
	"$OUTPUT" >/dev/null 2>/dev/null

if [ -n "$EXTRA_PARTS" ]; then
	part_num=2
	IFS=',' read -r -a parts <<< "$EXTRA_PARTS"
	for part in "${parts[@]}"; do
		name="${part%%:*}"
		size="${part#*:}"
		[ -n "$name" ] || die "empty extra partition name"
		[ "$name" != "$size" ] || die "missing size for extra partition $name"

		if [ "$size" = "0" ] || [ "$size" = "grow" ]; then
			end=0
		else
			end="+$size"
		fi

		sgdisk \
			--new="$part_num":0:"$end" \
			--typecode="$part_num":8300 \
			--change-name="$part_num":"$name" \
			"$OUTPUT" >/dev/null 2>/dev/null
		part_num=$((part_num + 1))
	done
fi

dd if="$BOOT_IMG" of="$OUTPUT" bs=512 seek="$BOOT_START_SECTOR" conv=notrunc status=none
sgdisk --verify "$OUTPUT" >/dev/null
gzip -c -n "$OUTPUT" > "$OUTPUT.gz"

echo "$OUTPUT"
