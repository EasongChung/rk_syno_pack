#!/usr/bin/env bash
# Verify downloaded H0/H1 profile packages on a runner (read-only checks).
# Expects: verify/artifact/output/boot-hardening/{H0,H1}/...
set -euo pipefail

H0_GZ=a13e33d79d9bda1b23285a99982bd16e5cdb3c52ec7c83d6b2646c831076dde8
H0_RAW=c62d91c89a156099b57e8e634370c47ff4a2888e1cc7692277c00e1650c38d27
IMAGE=8023e24832bfe4eec2a4e7cf35cb3ffa8c497350e3fb126b70b9e4a083929025
UINITRD=173cfcfdc6e95482240f4bc6dd460c4c0fd9db1a8bba27375dc54a354ca4114c
DTB=2f3232a17733e28ba4d231bd26fb4139d8d348ac42f570e197363fe8aab7cb70
DTB_NAME=rk3566-oec-box-wxy4-dsm.dtb
# Locate the H0/H1 folders wherever download-artifact placed them.
H0_DIR=$(find verify/artifact -type d -name H0 | head -1)
ROOT=$(dirname "$H0_DIR")

need() { echo "FAIL: $*"; exit 1; }

[ -d "$ROOT/H0" ] && [ -d "$ROOT/H1" ] || need "H0/H1 folders missing (ROOT=$ROOT)"

echo "== H0 must be byte-identical to the official package"
(h0=$(find "$ROOT/H0" -name '*.img.gz'); echo "$H0_GZ  $h0" | sha256sum -c -)
h0gz=$(find "$ROOT/H0" -name '*.img.gz'); gzip -dc "$h0gz" > /tmp/h0.raw
echo "$H0_RAW  /tmp/h0.raw" | sha256sum -c -

echo "== H1 structure"
h1raw=$(find "$ROOT/H1" -name '*.img' ! -name '*.img.gz')
h1gz=$(find "$ROOT/H1" -name '*.img.gz')
gzip -dc "$h1gz" > /tmp/h1.raw
cmp /tmp/h1.raw "$h1raw" || need "H1 raw != decompressed gz"
[ "$(stat -c %s /tmp/h1.raw)" = "$((128*1024*1024))" ] || need "H1 raw size != 128MiB"

sgdisk -v /tmp/h1.raw | sed -n '1,40p'
sgdisk -i 1 /tmp/h1.raw | grep -qi 'legacy BIOS bootable: set' || need "p1 bootable flag not set"
sgdisk -i 2 /tmp/h1.raw | grep -qi 'legacy BIOS bootable: set' && need "p2 must not be bootable"

dd if=/tmp/h1.raw of=/tmp/boot.img bs=1M skip=16 count=32 status=none
echo "== H1 FAT type (must be FAT16)"
mdir -i /tmp/boot.img :: | head -5
fsck.fat -n /tmp/boot.img | grep -qi 'FAT16' || need "boot volume is not FAT16"

echo "== H1 frozen components must be unchanged"
mkdir -p /tmp/h1boot
mcopy -s -i /tmp/boot.img ::/boot/Image /tmp/h1boot/Image
mcopy -s -i /tmp/boot.img "::/boot/uInitrd" /tmp/h1boot/uInitrd
mcopy -s -i /tmp/boot.img "::/boot/$DTB_NAME" "/tmp/h1boot/$DTB_NAME"
echo "$IMAGE  /tmp/h1boot/Image" | sha256sum -c -
echo "$UINITRD  /tmp/h1boot/uInitrd" | sha256sum -c -
echo "$DTB  /tmp/h1boot/$DTB_NAME" | sha256sum -c -

echo "== H1 marker and contract, no H2"
mcopy -s -i /tmp/boot.img "::/boot/extlinux/extlinux.conf" /tmp/extlinux.conf
grep -q 'och_boot_profile=H1-fat16-bootable' /tmp/extlinux.conf || need "marker missing"
grep -qE '(^| )sn=[^ ]+' /tmp/extlinux.conf || need "sn bootarg missing"
mcopy -s -i /tmp/boot.img "::/boot/BOOT-CONTRACT.json" /tmp/BOOT-CONTRACT.json
python3 - <<'PY'
import json
m = json.load(open('/tmp/BOOT-CONTRACT.json'))
assert m['profile'] == 'H1', m['profile']
assert m['kernel_changed'] is False and m['CPU_EMMC_changes'] is False
assert m['components']['/boot/Image']
print('contract OK:', m['ARM64_Image']['file_size'], 'bytes kernel')
PY
[ ! -e "$ROOT/H2" ] || need "H2 must not exist"

echo "== identity values must not appear in reports"
if grep -rlE 'sn=[0-9A-Z]{10,}' "$ROOT" | grep -v extlinux.conf >/dev/null 2>&1; then
  need "identity-looking value in reports"
fi
echo "ALL CHECKS PASSED"
