#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

need_cmd curl
need_cmd xz
need_cmd cpio
need_cmd patch
need_cmd fdtput
need_cmd fdtget

prepare_dirs

PAT_URL="${PAT_URL:-$(pat_url_for "${1:-${DSM_PAT_VERSION_DEFAULT}}")}"
PAT_FILE="${2:-${PAT_FILE:-${WORK_DIR}/$(basename "${PAT_URL}")}}"
FALLBACK_PAT_DIR="${3:-${FALLBACK_PAT_DIR_DEFAULT}}"
FALLBACK_RD_DIR="${4:-${FALLBACK_RD_DIR_DEFAULT}}"

mkdir -p "$(dirname "${PAT_FILE}")"

if [ ! -f "${PAT_FILE}" ]; then
  msg "downloading pat: ${PAT_URL}"
  curl -L --fail --output "${PAT_FILE}" "${PAT_URL}"
else
  msg "using existing pat: ${PAT_FILE}"
fi

extract_pat "${PAT_FILE}" "${FALLBACK_PAT_DIR}" "${FALLBACK_RD_DIR}"
apply_patches
repack_rd
cp -f "${PATCHED_RD_BIN}" "${PATCHED_BOOT_DIR}/uInitrd"

msg "done"
msg "patched rd.bin: ${PATCHED_RD_BIN}"
msg "boot payload for boot_a image: ${PATCHED_BOOT_DIR}/uInitrd"
