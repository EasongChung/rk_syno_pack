#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${PROJECT_DIR}/build"
BIN_DIR="${PROJECT_DIR}/tools"
PATCH_ROOT_DIR="${PROJECT_DIR}/patches"
SYNOXTRACT_BIN="${PROJECT_DIR}/tools/SynoXtract/synoxtract"
KERNEL_BUILD_DIR="${KERNEL_BUILD:-${PROJECT_DIR}/build/out/kernel-7.3}"
SYNO_HDDMON_KO="${SYNO_HDDMON_KO:-${KERNEL_BUILD_DIR}/drivers/hwmon/syno_hddmon.ko}"
ROOT_MODULE_DEP_LINES=()

pat_url_for() {
  case "${1:-7.4}" in
    7.3|7.3.2|86009)
      printf '%s\n' 'https://global.synologydownload.com/download/DSM/release/7.3.2/86009/DSM_DS423_86009.pat'
      ;;
    7.4|90075)
      printf '%s\n' 'https://global.synologydownload.com/download/DSM/release/7.4/90075/DSM_DS423_90075.pat'
      ;;
    http://*|https://*)
      printf '%s\n' "$1"
      ;;
    *)
      printf '[syno-rk3399-patchkit] error: unknown DSM PAT version: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

DSM_PAT_VERSION_DEFAULT="${DSM_PAT_VERSION:-7.4}"
PAT_URL_DEFAULT="$(pat_url_for "${DSM_PAT_VERSION_DEFAULT}")"
PAT_FILE_DEFAULT="${WORK_DIR}/$(basename "${PAT_URL_DEFAULT}")"
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

abs_path() {
  local path="$1"
  local dir base

  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    *)
      dir="$(dirname "${path}")"
      base="$(basename "${path}")"
      printf '%s/%s\n' "$(cd "${dir}" && pwd)" "${base}"
      ;;
  esac
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

copy_initrd_root() {
  local src_dir="$1"
  local dst_dir="$2"

  [ -d "${src_dir}" ] || die "initrd source dir not found: ${src_dir}"

  if [ "$(cd "${src_dir}" && pwd)" = "$(mkdir -p "${dst_dir}" && cd "${dst_dir}" && pwd)" ]; then
    return 0
  fi

  rm -rf "${dst_dir}"
  mkdir -p "${dst_dir}"
  (
    cd "${src_dir}"
    tar -cf - .
  ) | (
    cd "${dst_dir}"
    tar -xf -
  )
}

unpack_initrd_file() {
  local initrd_file
  local dst_dir="$2"

  initrd_file="$(abs_path "$1")"

  need_file "${initrd_file}"
  rm -rf "${dst_dir}"
  mkdir -p "${dst_dir}"

  msg "unpacking initrd file: ${initrd_file}"
  (
    cd "${dst_dir}"
    if command -v lzma >/dev/null 2>&1; then
      lzma -dc <"${initrd_file}" | cpio -idm --quiet
    else
      xz --format=lzma -dc <"${initrd_file}" | cpio -idm --quiet
    fi
  )
  [ -f "${dst_dir}/linuxrc.syno.impl" ] || die "failed to unpack initrd file: ${initrd_file}"
  chmod u+r "${dst_dir}/etc/shadow" "${dst_dir}/etc.defaults/shadow" 2>/dev/null || true
}

repack_initrd_file() {
  local src_dir="$1"
  local initrd_file

  initrd_file="$(abs_path "$2")"

  [ -d "${src_dir}" ] || die "initrd root dir not found: ${src_dir}"
  mkdir -p "$(dirname "${initrd_file}")"

  msg "repacking initrd file: ${initrd_file}"
  rm -f "${initrd_file}"
  (
    cd "${src_dir}"
    find . -print | cpio -o -H newc -R root:root --quiet | xz --format=lzma -9 >"${initrd_file}"
  )
}

prepare_dirs() {
  mkdir -p "${WORK_DIR}" "${PAT_EXTRACT_DIR}" "${PAT_RD_DIR}" \
    "${PATCHED_RD_DIR}" "${PATCHED_BOOT_DIR}"
}

apply_patch_series() {
  local patch_file found=0

  shopt -s nullglob
  for patch_file in "${PATCH_ROOT_DIR}"/[0-9][0-9][0-9][0-9]-*.patch; do
    found=1
    msg "applying patch $(basename "${patch_file}")"
    patch -p0 <"${patch_file}"
  done
  shopt -u nullglob

  if [ "${found}" -eq 0 ]; then
    msg "no patch files found in ${PATCH_ROOT_DIR}, skipping patch series"
  fi
}

read_version_key() {
  local key="$1"
  local file="$2"

  sed -n "s/^${key}=\"\\([^\"]*\\)\".*/\\1/p" "${file}"
}

patch_model_dtb_rk3566_sata_slots() {
  local max_disks="${SYNO_MAX_DISKS:-6}"
  local model_dtb slot slot_node ahci_node rtk_ahci_node actual

  case "${max_disks}" in
    ''|*[!0-9]*) die "SYNO_MAX_DISKS must be a positive integer" ;;
  esac
  [ "${max_disks}" -gt 0 ] || die "SYNO_MAX_DISKS must be greater than zero"

  msg "patching initrd model.dtb RK3566 SATA slots 1-${max_disks}"
  for model_dtb in \
    "${PATCHED_RD_DIR}/etc/model.dtb" \
    "${PATCHED_RD_DIR}/etc.defaults/model.dtb"; do
    need_file "${model_dtb}"

    for ((slot = 1; slot <= max_disks; slot++)); do
      slot_node="/internal_slot@${slot}"
      ahci_node="${slot_node}/ahci"
      rtk_ahci_node="${slot_node}/rtk_ahci"

      fdtput -cp "${model_dtb}" "${ahci_node}"
      fdtput -cp "${model_dtb}" "${rtk_ahci_node}"
      fdtput -t s "${model_dtb}" "${slot_node}" protocol_type sata
      fdtput -t x "${model_dtb}" "${ahci_node}" ata_port "$((slot - 1))"
      fdtput -t x "${model_dtb}" "${rtk_ahci_node}" ata_port "$((slot - 1))"
      fdtput -d "${model_dtb}" "${ahci_node}" pcie_root 2>/dev/null || true

      actual="$(fdtget -t x "${model_dtb}" "${ahci_node}" ata_port)"
      [ "${actual}" = "$((slot - 1))" ] || \
        die "failed to patch ${ahci_node}/ata_port in ${model_dtb}: got ${actual}"
      actual="$(fdtget -t x "${model_dtb}" "${rtk_ahci_node}" ata_port)"
      [ "${actual}" = "$((slot - 1))" ] || \
        die "failed to patch ${rtk_ahci_node}/ata_port in ${model_dtb}: got ${actual}"
    done
  done
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

patch_model_dtb_sata_slots() {
  case "${SOC:-rk3399}" in
    rk3566)
      patch_model_dtb_rk3566_sata_slots
      ;;
    *)
      patch_model_dtb_sata_pcie_root
      ;;
  esac
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

patch_initrd_root() {
  local src_dir="${1:-${PAT_RD_DIR}}"
  local dst_dir="${2:-${PATCHED_RD_DIR}}"
  local install_syno_hddmon="${INSTALL_SYNO_HDDMON:-1}"

  copy_initrd_root "${src_dir}" "${dst_dir}"
  PATCHED_RD_DIR="${dst_dir}"
  chmod u+r "${PATCHED_RD_DIR}/etc/shadow" "${PATCHED_RD_DIR}/etc.defaults/shadow" 2>/dev/null || true

  (
    cd "${PATCHED_RD_DIR}"
    apply_patch_series
  )

  patch_model_dtb_sata_slots
  patch_synoinfo_disk_count
  "${PATCH_ROOT_DIR}/0000-rk-initrd-fixes.sh" "${PATCHED_RD_DIR}"

  if [ "${install_syno_hddmon}" = 1 ]; then
    need_file "${SYNO_HDDMON_KO}"
    msg "replacing syno_hddmon.ko from kernel build"
    install -m 0644 "${SYNO_HDDMON_KO}" \
      "${PATCHED_RD_DIR}/usr/lib/modules/syno_hddmon.ko"
  else
    msg "skipping syno_hddmon.ko replacement"
  fi
  msg "keeping factory synobios.ko"

  sync_root_modules_to_initrd
}

apply_patches() {
  patch_initrd_root "${PAT_RD_DIR}" "${PATCHED_RD_DIR}"
}

repack_rd() {
  repack_initrd_file "${PATCHED_RD_DIR}" "${PATCHED_RD_BIN}"
}
