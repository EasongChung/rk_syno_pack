#!/usr/bin/env python3
"""Audit vendor ENV assets and a legacy GPT image without emitting identities.

CRC-valid assets are not proof of the backend/offset used by the running vendor
binary. H2 remains blocked until that and a complete raw boot are established.
"""
import argparse
import json
import re
import struct
import sys
import zlib
from pathlib import Path

from oec_boot_contract import regular_input, hash_file

KEYS = {"bootcmd", "boot_targets", "boot_prefixes", "mmc_boot", "scan_dev_for_boot_part",
        "kernel_addr_r", "ramdisk_addr_r", "fdt_addr_r", "scriptaddr"}


def env_block(data):
    if len(data) != 65536:
        return {"size": len(data), "crc_valid": False}
    crc = struct.unpack_from("<I", data)[0]
    good = zlib.crc32(data[4:]) & 0xffffffff == crc
    values = {}
    if good:
        for chunk in data[4:].split(b"\0"):
            if b"=" not in chunk: continue
            key, value = chunk.split(b"=", 1)
            key = key.decode("ascii", "replace")
            if key not in KEYS: continue
            text = value.decode("ascii", "replace")
            # Never echo a boot-control value that embeds identity/security data.
            values[key] = "<omitted: identity/security reference>" if re.search(
                r"\b(sn|custom_sn|mac[0-9]*|password|token|secret|authorization)\b", text, re.I) else text
    return {"size": len(data), "crc_valid": good, "control_variables": values}


def legacy_env(path):
    p = regular_input(path)
    with p.open("rb") as f:
        f.seek(512); header = f.read(512)
        if header[:8] != b"EFI PART": raise ValueError("legacy GPT header missing")
        hs = struct.unpack_from("<I", header, 12)[0]
        if not 92 <= hs <= 512: raise ValueError("invalid legacy GPT header size")
        check = bytearray(header[:hs]); check[16:20] = bytes(4)
        if zlib.crc32(check) & 0xffffffff != struct.unpack_from("<I", header, 16)[0]: raise ValueError("legacy GPT CRC mismatch")
        start, count, es = struct.unpack_from("<QII", header, 72)
        if not (0 < count <= 128 and es == 128): raise ValueError("invalid GPT geometry")
        f.seek(start * 512); entries = f.read(count * es)
        if zlib.crc32(entries) & 0xffffffff != struct.unpack_from("<I", header, 88)[0]: raise ValueError("legacy GPT entry CRC mismatch")
        report = {"file_bytes": p.stat().st_size,
                  "backup_lba": struct.unpack_from("<Q", header, 32)[0], "env_partition_present": False}
        report["backup_inside_file"] = (report["backup_lba"] + 1) * 512 <= p.stat().st_size
        for pos in range(0, len(entries), es):
            e = entries[pos:pos + es]
            if e[:16] == bytes(16): continue
            name = e[56:128].decode("utf-16-le").rstrip("\0")
            if name != "env": continue
            lba, last = struct.unpack_from("<QQ", e, 32)
            report.update(env_partition_present=True, env_start_lba=lba, env_size_bytes=(last-lba+1)*512)
            f.seek(lba * 512)
            report["env_copies"] = [env_block(f.read(65536)), env_block(f.read(65536))]
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor-dir", type=Path, required=True)
    p.add_argument("--legacy-image", type=Path)
    a = p.parse_args()
    asset = regular_input(a.vendor_dir / "env.bin")
    loader = regular_input(a.vendor_dir / "bootloader.bin")
    data = asset.read_bytes()
    report = {"env_asset_bytes": len(data), "env_asset_sha256": hash_file(asset),
              "two_64K_copies": [env_block(data[:65536]), env_block(data[65536:131072])],
              "exact_asset_in_loader": loader.read_bytes().find(data) >= 0,
              "H2_ready": False,
              "H2_blockers": ["actual vendor ENV backend and offsets not proven",
                              "asset is not evidence of runtime ENV selection",
                              "no complete-raw lab boot/marker observation yet"],
              "CPU_EMMC_white_LED_cause": "excluded by user test",
              "SN_empty_cause": "unproven; not attributed to DTB filename"}
    if a.legacy_image: report["legacy"] = legacy_env(a.legacy_image)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
