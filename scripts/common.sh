#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${PROJECT_DIR}/build"
BIN_DIR="${PROJECT_DIR}/tools"
PATCH_ROOT_DIR="${PROJECT_DIR}/patches"
PATCH_DIR=""
SYNOXTRACT_BIN="${PROJECT_DIR}/tools/SynoXtract/synoxtract"
KERNEL_BUILD_DIR="${KERNEL_BUILD:-${PROJECT_DIR}/build/out/kernel-7.3}"
SYNO_HDDMON_KO="${SYNO_HDDMON_KO:-${KERNEL_BUILD_DIR}/drivers/hwmon/syno_hddmon.ko}"
ROOT_MODULE_DEP_LINES=()

PAT_URL_DEFAULT="https://global.synologydownload.com/download/DSM/release/7.3.2/86009/DSM_DS423_86009.pat"
PAT_FILE_DEFAULT="${WORK_DIR}/DSM_DS423_86009.pat"
PAT_EXTRACT_DIR="${WORK_DIR}/pat-extract"
PAT_RD_DIR="${WORK_DIR}/pat-rd"
PATCHED_RD_DIR="${WORK_DIR}/pat-rd-patched"
PATCHED_RD_BIN="${WORK_DIR}/rd.bin"
PATCHED_BOOT_DIR="${WORK_DIR}/boot-patched"

FALLBACK_PAT_DIR_DEFAULT=""
FALLBACK_RD_DIR_DEFAULT=""

msg() {
  printf '[syno-rk3399-patchkit] %s\n' "$*"
}

die() {
  printf '[syno-rk3399-patchkit] error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

need_file() {
  [ -r "$1" ] || die "missing file: $1"
}

install_module_file() {
  local module_path="$1"

  need_file "${module_path}"
  install -m 0644 "${module_path}" \
    "${PATCHED_RD_DIR}/usr/lib/modules/$(basename "${module_path}")"
}

write_root_module_manifest() {
  local dep_file="${PATCHED_RD_DIR}/usr/lib/modules/rk3399.modules.dep"
  local dep_line

  if [ "${#ROOT_MODULE_DEP_LINES[@]}" -eq 0 ]; then
    rm -f "${dep_file}"
    return 0
  fi

  : >"${dep_file}"
  for dep_line in "${ROOT_MODULE_DEP_LINES[@]}"; do
    printf '%s\n' "${dep_line}" >>"${dep_file}"
  done
}

sync_root_modules_to_initrd() {
  if [ "${#ROOT_MODULE_DEP_LINES[@]}" -eq 0 ]; then
    msg "no rootfs runtime module overrides"
  else
    msg "writing rootfs runtime module manifest"
  fi
  write_root_module_manifest
}

prepare_dirs() {
  mkdir -p "${WORK_DIR}" "${PAT_EXTRACT_DIR}" "${PAT_RD_DIR}" \
    "${PATCHED_RD_DIR}" "${PATCHED_BOOT_DIR}"
}

apply_patch_series() {
  local patch_file found=0

  [ -n "${PATCH_DIR}" ] || die "PATCH_DIR is not selected"

  shopt -s nullglob
  for patch_file in "${PATCH_DIR}"/[0-9][0-9][0-9][0-9]-*.patch; do
    found=1
    msg "applying patch $(basename "${patch_file}")"
    patch -p0 <"${patch_file}"
  done
  shopt -u nullglob

  [ "${found}" -eq 1 ] || die "no patch files found in ${PATCH_DIR}"
}

read_version_key() {
  local key="$1"
  local file="$2"

  sed -n "s/^${key}=\"\\([^\"]*\\)\".*/\\1/p" "${file}"
}

select_patch_dir() {
  local patch_version="${DSM_PATCH_VERSION:-}"
  local version_file="${PAT_EXTRACT_DIR}/VERSION"
  local major minor

  if [ -z "${patch_version}" ]; then
    need_file "${version_file}"
    major="$(read_version_key majorversion "${version_file}")"
    [ -n "${major}" ] || major="$(read_version_key major "${version_file}")"
    minor="$(read_version_key minorversion "${version_file}")"
    [ -n "${minor}" ] || minor="$(read_version_key minor "${version_file}")"
    [ -n "${major}" ] || die "failed to detect DSM major version from ${version_file}"
    [ -n "${minor}" ] || die "failed to detect DSM minor version from ${version_file}"
    patch_version="${major}.${minor}"
  fi

  PATCH_DIR="${PATCH_ROOT_DIR}/${patch_version}"
  [ -d "${PATCH_DIR}" ] || die "patch directory not found for DSM ${patch_version}: ${PATCH_DIR}"
  msg "using DSM ${patch_version} patches: ${PATCH_DIR}"
}

patch_model_dtb_sata_pcie_root() {
  local pcie_root="${SYNO_SATA_PCIE_ROOT:-0000:00:00.0,00.0}"
  local max_disks="${SYNO_MAX_DISKS:-6}"
  local model_dtb slot slot_node ahci_node actual

  [ -n "${pcie_root}" ] || die "SYNO_SATA_PCIE_ROOT must not be empty"
  case "${pcie_root}" in
    *[[:space:]]*) die "SYNO_SATA_PCIE_ROOT must not contain whitespace" ;;
  esac
  case "${max_disks}" in
    ''|*[!0-9]*) die "SYNO_MAX_DISKS must be a positive integer" ;;
  esac
  [ "${max_disks}" -gt 0 ] || die "SYNO_MAX_DISKS must be greater than zero"

  msg "patching initrd model.dtb SATA slots 1-${max_disks} to ${pcie_root}"
  for model_dtb in \
    "${PATCHED_RD_DIR}/etc/model.dtb" \
    "${PATCHED_RD_DIR}/etc.defaults/model.dtb"; do
    need_file "${model_dtb}"

    for ((slot = 1; slot <= max_disks; slot++)); do
      slot_node="/internal_slot@${slot}"
      ahci_node="${slot_node}/ahci"

      fdtput -cp "${model_dtb}" "${ahci_node}"
      fdtput -t s "${model_dtb}" "${slot_node}" protocol_type sata
      fdtput -t s "${model_dtb}" "${ahci_node}" pcie_root "${pcie_root}"
      fdtput -t x "${model_dtb}" "${ahci_node}" ata_port "$((slot - 1))"

      actual="$(fdtget -t s "${model_dtb}" "${ahci_node}" pcie_root)"
      [ "${actual}" = "${pcie_root}" ] || \
        die "failed to patch ${ahci_node}/pcie_root in ${model_dtb}: got ${actual}"
    done
  done
}

set_synoinfo_kv() {
  local synoinfo="$1"
  local key="$2"
  local value="$3"

  if grep -q "^${key}=" "${synoinfo}"; then
    sed -i "s|^${key}=.*|${key}=\"${value}\"|" "${synoinfo}"
  else
    printf '%s="%s"\n' "${key}" "${value}" >>"${synoinfo}"
  fi
}

patch_synoinfo_disk_count() {
  local max_disks="${SYNO_MAX_DISKS:-6}"
  local synoinfo

  case "${max_disks}" in
    ''|*[!0-9]*) die "SYNO_MAX_DISKS must be a positive integer" ;;
  esac
  [ "${max_disks}" -gt 0 ] || die "SYNO_MAX_DISKS must be greater than zero"

  msg "patching initrd synoinfo disk count to ${max_disks}"
  for synoinfo in \
    "${PATCHED_RD_DIR}/etc/synoinfo.conf" \
    "${PATCHED_RD_DIR}/etc.defaults/synoinfo.conf"; do
    need_file "${synoinfo}"
    set_synoinfo_kv "${synoinfo}" maxdisks "${max_disks}"
    set_synoinfo_kv "${synoinfo}" max_sys_raid_disks "${max_disks}"
  done
}

extract_pat() {
  local pat_file="$1"
  local fallback_pat_dir="${2:-${FALLBACK_PAT_DIR_DEFAULT}}"
  local fallback_rd_dir="${3:-${FALLBACK_RD_DIR_DEFAULT}}"

  rm -rf "${PAT_EXTRACT_DIR}" "${PAT_RD_DIR}"
  mkdir -p "${PAT_EXTRACT_DIR}" "${PAT_RD_DIR}"

  msg "extracting pat: ${pat_file}"
  [ -x "${SYNOXTRACT_BIN}" ] || die "missing SynoXtract binary: ${SYNOXTRACT_BIN}"
  if ! "${SYNOXTRACT_BIN}" -i "${pat_file}" -d "${PAT_EXTRACT_DIR}"; then
    msg "SynoXtract returned non-zero, trying fallback pat directory"
  fi

  if [ ! -f "${PAT_EXTRACT_DIR}/rd.bin" ] || [ ! -f "${PAT_EXTRACT_DIR}/zImage" ]; then
    [ -n "${fallback_pat_dir}" ] || die "extractor did not produce rd.bin/zImage; set FALLBACK_PAT_DIR or pass fallback pat dir"
    [ -d "${fallback_pat_dir}" ] || die "fallback pat dir not found: ${fallback_pat_dir}"
    [ -f "${fallback_pat_dir}/rd.bin" ] || die "fallback rd.bin not found: ${fallback_pat_dir}/rd.bin"
    [ -f "${fallback_pat_dir}/zImage" ] || die "fallback zImage not found: ${fallback_pat_dir}/zImage"
    msg "using fallback extracted pat dir: ${fallback_pat_dir}"
    cp -r "${fallback_pat_dir}/." "${PAT_EXTRACT_DIR}/"
  fi

  if [ -n "${fallback_rd_dir}" ] && [ -d "${fallback_rd_dir}" ] && [ -f "${fallback_rd_dir}/linuxrc.syno.impl" ]; then
    msg "using fallback extracted rd root: ${fallback_rd_dir}"
    (
      cd "${fallback_rd_dir}"
      tar -cf - .
    ) | (
      cd "${PAT_RD_DIR}"
      tar -xf -
    )
  else
    msg "extracting rd.bin"
    (
      cd "${PAT_RD_DIR}"
      lzma -dc <"${PAT_EXTRACT_DIR}/rd.bin" 2>/dev/null | cpio -idm --quiet 2>/dev/null || true
    )
    [ -f "${PAT_RD_DIR}/linuxrc.syno.impl" ] || die "failed to unpack rd.bin"
  fi

  chmod u+r "${PAT_RD_DIR}/etc/shadow" "${PAT_RD_DIR}/etc.defaults/shadow" 2>/dev/null || true
}

apply_patches() {
  rm -rf "${PATCHED_RD_DIR}"
  mkdir -p "${PATCHED_RD_DIR}"
  (
    cd "${PAT_RD_DIR}"
    tar -cf - .
  ) | (
    cd "${PATCHED_RD_DIR}"
    tar -xf -
  )

  select_patch_dir
  (
    cd "${PATCHED_RD_DIR}"
    apply_patch_series
  )

  patch_model_dtb_sata_pcie_root
  patch_synoinfo_disk_count

  need_file "${SYNO_HDDMON_KO}"
  msg "replacing syno_hddmon.ko from kernel build"
  install -m 0644 "${SYNO_HDDMON_KO}" \
    "${PATCHED_RD_DIR}/usr/lib/modules/syno_hddmon.ko"
  msg "keeping factory synobios.ko"

  sync_root_modules_to_initrd
}

repack_rd() {
  msg "repacking rd.bin"
  rm -f "${PATCHED_RD_BIN}"
  (
    cd "${PATCHED_RD_DIR}"
    find . -print | cpio -o -H newc -R root:root --quiet | xz --format=lzma -9 >"${PATCHED_RD_BIN}"
  )
}
