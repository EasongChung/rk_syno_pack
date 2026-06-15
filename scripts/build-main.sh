#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_SRC_DEFAULT="$ROOT_DIR/linux-5.10.x"
KERNEL_BUILD_DEFAULT="$ROOT_DIR/build/out/kernel-7.3"
PAT_URL_DEFAULT="https://global.synologydownload.com/download/DSM/release/7.3.2/86009/DSM_DS423_86009.pat"

usage() {
  cat <<'EOF'
Usage:
  ./build.sh [all|pat|uboot|kernel|updateimg]

Targets:
  pat        download/extract/patch official DSM pat and generate patched rd.bin/uInitrd
  uboot      build u-boot/trust/loader artifacts locally
  kernel     build DSM kernel Image/dtb
  updateimg  generate Rockchip update.img using this project only
  all        run uboot -> kernel -> pat -> updateimg

Environment:
  PAT_URL         official PAT URL
  PAT_FILE        local PAT file path
  KERNEL_SRC      kernel source tree
  KERNEL_BUILD    kernel build output dir
  CROSS_COMPILE   aarch64 compiler prefix
  UBOOT_DEFCONFIG u-boot defconfig, defaults to evb-rk3399
  SYNO_SATA_PCIE_ROOT  SATA pcie_root written into initrd model.dtb
                       defaults to 0000:00:00.0,00.0
  SYNO_MAX_DISKS       DSM internal slot count written into initrd model.dtb
                       and synoinfo.conf, defaults to 6
  DSM_PATCH_VERSION    patch directory under patches/, defaults to PAT major.minor
EOF
}

msg() {
  printf '[build] %s\n' "$*"
}

target_pat() {
  msg "patching official pat"
  "$ROOT_DIR/scripts/download_and_patch_pat.sh" \
    "${PAT_URL:-$PAT_URL_DEFAULT}" \
    "${PAT_FILE:-$ROOT_DIR/build/DSM_DS423_86009.pat}" \
    "${FALLBACK_PAT_DIR:-}" \
    "${FALLBACK_RD_DIR:-}"
}

target_uboot() {
  "$SCRIPT_DIR/build-uboot.sh"
}

target_kernel() {
  msg "building kernel"
  KERNEL_SRC="${KERNEL_SRC:-$KERNEL_SRC_DEFAULT}" \
  KERNEL_BUILD="${KERNEL_BUILD:-$KERNEL_BUILD_DEFAULT}" \
  "$SCRIPT_DIR/pack-updateimg.sh" --kernel-only
}

target_updateimg() {
  msg "packing update.img"
  KERNEL_SRC="${KERNEL_SRC:-$KERNEL_SRC_DEFAULT}" \
  KERNEL_BUILD="${KERNEL_BUILD:-$KERNEL_BUILD_DEFAULT}" \
  "$SCRIPT_DIR/pack-updateimg.sh" --pack-only
}

main() {
  local target="${1:-all}"
  case "$target" in
    pat) target_pat ;;
    uboot) target_uboot ;;
    kernel) target_kernel ;;
    updateimg) target_updateimg ;;
    all)
      target_uboot
      target_kernel
      target_pat
      target_updateimg
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
