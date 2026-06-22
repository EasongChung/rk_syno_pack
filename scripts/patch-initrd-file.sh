#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/patch-initrd-file.sh <input-rd.bin> [output-rd.bin]

Patch a DSM initrd file directly. This does not download or extract a PAT file.

Environment:
  INITRD_WORK_DIR      temp work dir, defaults to build/initrd-file-patch
  INSTALL_SYNO_HDDMON  replace syno_hddmon.ko, defaults to 1
  SYNO_HDDMON_KO       replacement syno_hddmon.ko path
  SYNO_SATA_PCIE_ROOT  SATA pcie_root written into model.dtb
  SYNO_MAX_DISKS       DSM internal slot count, defaults to 6
EOF
}

main() {
  local input_file="${1:-}"
  local output_file="${2:-}"
  local initrd_work_dir="${INITRD_WORK_DIR:-${WORK_DIR}/initrd-file-patch}"
  local unpacked_dir="${initrd_work_dir}/unpacked"
  local patched_dir="${initrd_work_dir}/patched"

  case "${input_file}" in
    ''|-h|--help|help)
      usage
      [ -n "${input_file}" ] || exit 2
      exit 0
      ;;
  esac

  if [ -z "${output_file}" ]; then
    output_file="${input_file}.patched"
  fi

  need_cmd cpio
  need_cmd patch
  need_cmd fdtput
  need_cmd fdtget
  need_cmd xz

  rm -rf "${initrd_work_dir}"
  mkdir -p "${initrd_work_dir}"

  unpack_initrd_file "${input_file}" "${unpacked_dir}"
  patch_initrd_root "${unpacked_dir}" "${patched_dir}"
  repack_initrd_file "${patched_dir}" "${output_file}"

  msg "patched initrd file: ${output_file}"
}

main "$@"
