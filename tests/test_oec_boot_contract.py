#!/usr/bin/env python3
"""In-memory parser/contract tests; no formatter, device or credentials used."""
import json
import lzma
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from oec_boot_contract import (FatFS, parse_config, config_metadata, marked_config,
                               image_header, cpio_files, address_ranges_overlap,
                               lfn_checksum)
from make_oec_boot_profiles import safe_output
from probe_oec_boot_contract import content_status, failures
from audit_oec_boot_env import env_block
import zlib

CONFIG = b"label DSM-rk3566\n  kernel /boot/Image\n  initrd /boot/uInitrd\n  fdt /boot/rk3566-oec-box-wxy4-dsm.dtb\n  append root=/dev/sata1p1 cma=256M vender_format_version=2 vendor_format_version=2 sn=PUBLIC-TEST custom_sn=PUBLIC-TEST mac1=001122334455\n"


def fat_fixture(bits=16, long_name=False, cycle=False):
    data = bytearray(8 << 20); data[510:512] = b"\x55\xaa"
    struct.pack_into("<H", data, 11, 512)
    reserved, fatsz, spc, roots = (1, 32, 2, 512) if bits == 16 else (32, 128, 1, 0)
    data[13] = spc; struct.pack_into("<H", data, 14, reserved); data[16] = 2
    struct.pack_into("<H", data, 17, roots)
    if bits == 16:
        struct.pack_into("<H", data, 19, len(data) // 512)
        struct.pack_into("<H", data, 22, fatsz); data[54:62] = b"FAT16   "
    else:
        struct.pack_into("<I", data, 32, len(data) // 512)
        struct.pack_into("<I", data, 36, fatsz); struct.pack_into("<I", data, 44, 2)
        data[82:90] = b"FAT32   "
    fat_offset = reserved * 512
    root_offset = (reserved + 2 * fatsz) * 512
    start = root_offset + ((roots * 32 + 511) // 512) * 512
    file_cluster = 2 if bits == 16 else 3
    root_data = root_offset if bits == 16 else start
    for copy in range(2):
        base = fat_offset + copy * fatsz * 512
        for c in range(4):
            value = (0xffff if bits == 16 else 0x0fffffff)
            if cycle and c == file_cluster: value = file_cluster
            struct.pack_into("<H" if bits == 16 else "<I", data, base + c * (bits // 8), value)
    short = b"FILE    TXT"
    entry = bytearray(32); entry[:11] = short; entry[11] = 0x20
    struct.pack_into("<H", entry, 26, file_cluster); struct.pack_into("<I", entry, 28, 3)
    if long_name:
        e = bytearray(32); e[0] = 0x41; e[11] = 0x0f; e[13] = lfn_checksum(short)
        text = ("LongFile.txt\0".ljust(13, "\uffff")).encode("utf-16-le")
        e[1:11] = text[:10]; e[14:26] = text[10:22]; e[28:32] = text[22:26]
        data[root_data:root_data + 32] = e; root_data += 32
    data[root_data:root_data + 32] = entry
    content = start + (file_cluster - 2) * spc * 512
    data[content:content + 3] = b"abc"
    return data, root_data


def cpio_fixture(name="etc/VERSION", content=b'productversion="7.4.1"\n'):
    result = bytearray()
    for filename, payload in [(name, content), ("TRAILER!!!", b"")]:
        encoded = filename.encode() + b"\0"
        fields = [1, 0o100644, 0, 0, 1, 0, len(payload), 0, 0, 0, 0, len(encoded), 0]
        result.extend(b"070701" + b"".join(("%08x" % x).encode() for x in fields))
        result.extend(encoded)
        while len(result) % 4: result.append(0)
        result.extend(payload)
        while len(result) % 4: result.append(0)
    return lzma.compress(bytes(result), format=lzma.FORMAT_ALONE)


class Tests(unittest.TestCase):
    def test_fat16_standard_read(self):
        data, _ = fat_fixture(); fs = FatFS(data)
        self.assertTrue(fs.standard); self.assertEqual(fs.file("/file.txt"), b"abc")

    def test_legacy_fat32_inspectable_not_standard(self):
        data, _ = fat_fixture(32); fs = FatFS(data)
        self.assertFalse(fs.standard); self.assertEqual(fs.file("/FILE.TXT"), b"abc")

    def test_fat_long_name(self):
        data, _ = fat_fixture(long_name=True)
        self.assertEqual(FatFS(data).file("/LongFile.txt"), b"abc")

    def test_bad_lfn_checksum(self):
        data, pos = fat_fixture(long_name=True); data[pos - 32 + 13] ^= 1
        with self.assertRaises(ValueError): FatFS(data).file("/LongFile.txt")

    def test_lowercase_short_name(self):
        data, pos = fat_fixture(); data[pos + 12] = 24
        self.assertIn("/file.txt", FatFS(data).files())

    def test_chain_cycle(self):
        data, _ = fat_fixture(cycle=True)
        with self.assertRaises(ValueError): FatFS(data).file("/FILE.TXT")

    def test_fat_copies_mismatch(self):
        data, _ = fat_fixture(); data[(1 + 32) * 512 + 4] ^= 1
        with self.assertRaises(ValueError): FatFS(data)

    def test_truncated_file(self):
        data, pos = fat_fixture(); struct.pack_into("<I", data, pos + 28, 6000)
        with self.assertRaises(ValueError): FatFS(data).file("/FILE.TXT")

    def test_duplicate_casefold_name(self):
        data, pos = fat_fixture(); data[pos + 32:pos + 64] = data[pos:pos + 32]
        with self.assertRaises(ValueError): FatFS(data).file("/file.txt")

    def test_parent_path_refused(self):
        data, _ = fat_fixture()
        with self.assertRaises(ValueError): FatFS(data).file("../FILE.TXT")

    def test_config_and_metadata_no_identity_values(self):
        fields, args = parse_config(CONFIG); meta = config_metadata(CONFIG)
        self.assertEqual(fields["kernel"], "/boot/Image")
        self.assertNotIn(args["sn"], json.dumps(meta))
        self.assertNotIn(args["mac1"], json.dumps(meta))
        self.assertTrue(meta["identity"]["sn"]["set"])

    def test_marker_only_changes_marker(self):
        _, before = parse_config(CONFIG); _, after = parse_config(marked_config(CONFIG))
        self.assertEqual(after.pop("och_boot_profile"), "H1-fat16-bootable")
        self.assertEqual(before, after)

    def test_duplicate_parameter_refused(self):
        with self.assertRaises(ValueError): parse_config(CONFIG.replace(b"cma=256M", b"cma=256M cma=256M"))

    def test_missing_file_directive_refused(self):
        with self.assertRaises(ValueError): parse_config(CONFIG.replace(b"  initrd /boot/uInitrd\n", b""))

    def test_unsafe_boot_path_refused(self):
        with self.assertRaises(ValueError): parse_config(CONFIG.replace(b"/boot/Image", b"/boot/../Image"))

    def test_wrong_root_or_format_refused(self):
        for old, new in [(b"root=/dev/sata1p1", b"root=/dev/md0"), (b"vender_format_version=2", b"vender_format_version=1")]:
            with self.assertRaises(ValueError): parse_config(CONFIG.replace(old, new))

    def test_existing_marker_refused(self):
        with self.assertRaises(ValueError): marked_config(marked_config(CONFIG))

    def test_cpio_version_read(self):
        self.assertIn("etc/VERSION", cpio_files(cpio_fixture()))

    def test_cpio_parent_refused(self):
        with self.assertRaises(ValueError): cpio_files(cpio_fixture("../VERSION"))

    def test_cpio_truncated_refused(self):
        with self.assertRaises((ValueError, lzma.LZMAError)): cpio_files(cpio_fixture()[:-8])

    def test_image_footprint_and_static_overlap(self):
        data = bytearray(1024); data[56:60] = b"ARM\x64"
        struct.pack_into("<QQ", data, 8, 0, 31 << 20)
        self.assertEqual(image_header(data)["image_size"], 31 << 20)
        self.assertEqual(address_ranges_overlap([("kernel", 0x00280000, 31 << 20), ("script", 0x00c00000, 4096)]), ("kernel", "script"))
        self.assertIsNone(address_ranges_overlap([("kernel", 0x02000000, 31 << 20), ("script", 0x00c00000, 4096)]))

    def test_initrd_64mb_dictionary_is_supported(self):
        data = bytearray(cpio_fixture())
        struct.pack_into("<I", data, 1, 64 << 20)
        self.assertIn("etc/VERSION", cpio_files(bytes(data)))

    def test_procfs_zero_inode_size_does_not_mean_empty(self):
        result = content_status(b"example-value\n", inode_size=0)
        self.assertTrue(result["content_nonempty"])
        self.assertEqual(result["inode_size"], 0)
        self.assertNotIn("example-value", json.dumps(result))

    def test_procfs_whitespace_is_empty(self):
        self.assertFalse(content_status(b" \r\n\0")["content_nonempty"])

    def test_error_report_contains_counts_not_identities(self):
        report = failures(b"serial number checksum error serial=do-not-display\nUnknown symbol\n")
        self.assertEqual(report["serial_checksum_errors"], 1)
        self.assertNotIn("do-not-display", json.dumps(report))

    def test_env_crc_and_whitelist(self):
        body = b"bootcmd=run boot_fit;\0password=do-not-display\0\0".ljust(65532, b"\0")
        data = struct.pack("<I", zlib.crc32(body) & 0xffffffff) + body
        report = env_block(data)
        self.assertTrue(report["crc_valid"])
        self.assertNotIn("do-not-display", json.dumps(report))
        corrupt = bytearray(data); corrupt[-1] ^= 1
        self.assertFalse(env_block(corrupt)["crc_valid"])

    def test_bad_image_header(self):
        with self.assertRaises(ValueError): image_header(bytes(64))

    def test_output_device_refused(self):
        with self.assertRaises(ValueError): safe_output("/dev/mmcblk0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
