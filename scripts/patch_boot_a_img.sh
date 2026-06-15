#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

need_cmd debugfs

BOOT_IMG="${1:-}"
[ -n "${BOOT_IMG}" ] || die "usage: $0 <boot_a.img> [patched_uInitrd]"
[ -f "${BOOT_IMG}" ] || die "boot_a image not found: ${BOOT_IMG}"

PATCHED_UINITRD="${2:-${PATCHED_BOOT_DIR}/uInitrd}"
[ -f "${PATCHED_UINITRD}" ] || die "patched uInitrd not found: ${PATCHED_UINITRD}"

TMP_IMG="${WORK_DIR}/$(basename "${BOOT_IMG%.img}")-patched.img"
cp -f "${BOOT_IMG}" "${TMP_IMG}"

msg "writing patched /boot/uInitrd into ${TMP_IMG}"
debugfs -w -R "write ${PATCHED_UINITRD} /boot/uInitrd" "${TMP_IMG}" >/dev/null 2>&1 || \
  die "failed to write patched uInitrd into boot image"

msg "done"
msg "patched boot image: ${TMP_IMG}"

