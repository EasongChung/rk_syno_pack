# WXY/OECT RK3566 vendor bootloader

This directory keeps the vendor bootloader files required by the WXY/OECT
RK3566 board.

Source:
- Copied from the ophub U-Boot repository:
  `u-boot/rockchip/wxy-oect`.
- The files in that directory are board-specific closed/vendor bootloader
  binaries for the WXY/OECT RK3566 board.
- The board did not boot reliably with the generic/self-built U-Boot path,
  so these files are kept as binary inputs.

Files:
- `MiniLoaderAll.bin`: loader used when packaging rk3566 `update.img`.
- `bootloader.bin`: complete bootloader block used by raw eMMC images written
  with `rkdeveloptool wl 0x0`.
- `u-boot-rockchip.bin`: vendor U-Boot payload extracted from the same source.
- `env.bin`: vendor environment block extracted from the same source.

Notes:
- Do not replace these files with generic rk3566 loader/U-Boot binaries unless
  the board has been retested from MaskROM recovery.
- The rk3566 raw image path intentionally copies `bootloader.bin` as-is to keep
  the vendor SPL/U-Boot chain intact.
- The rk3566 `update.img` path updates only `boot.img` plus the minimal package
  metadata, and does not include a self-built `uboot.img`.
