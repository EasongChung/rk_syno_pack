"""Read-only boot-contract parsers and checks; never modifies a device."""
import hashlib
import lzma
import re
import shlex
import stat
import struct
import zlib
from pathlib import Path, PurePosixPath

RAW_BYTES = 128 << 20
BOOT_START = 16 << 20
BOOT_BYTES = 32 << 20
BOOT_SHA = "4fdb15299148481ecf03f9330ee207c2dd26edbc96bbb72f802df72c84f88244"
RAW_SHA = "c62d91c89a156099b57e8e634370c47ff4a2888e1cc7692277c00e1650c38d27"
GZ_SHA = "a13e33d79d9bda1b23285a99982bd16e5cdb3c52ec7c83d6b2646c831076dde8"
DTB_NAME = "rk3566-oec-box-wxy4-dsm.dtb"
PINS = {
    "/boot/Image": "8023e24832bfe4eec2a4e7cf35cb3ffa8c497350e3fb126b70b9e4a083929025",
    "/boot/uInitrd": "173cfcfdc6e95482240f4bc6dd460c4c0fd9db1a8bba27375dc54a354ca4114c",
    "/boot/" + DTB_NAME: "2f3232a17733e28ba4d231bd26fb4139d8d348ac42f570e197363fe8aab7cb70",
}
CONFIGS = ("/boot/extlinux/extlinux.conf", "/extlinux/extlinux.conf")
IDENTITY_KEYS = ("sn", "custom_sn", "mac1")
LAYOUT = (("boot", 32768, 98303), ("userdata", 98304, 262110))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regular_input(path):
    p = Path(path).resolve()
    need(".secrets" not in p.parts and not (".ssh" in p.parts and p.name.startswith("id_")), "protected input refused")
    need(stat.S_ISREG(p.stat().st_mode), "input must be a regular file, never a device")
    return p


def gpt(path, with_entries=False):
    p = regular_input(path)
    size = p.stat().st_size
    need(size % 512 == 0, "unaligned raw size")
    with p.open("rb") as f:
        def read_at(off, n):
            f.seek(off); b = f.read(n)
            need(len(b) == n, "truncated raw")
            return b
        def header(lba):
            b = bytearray(read_at(lba * 512, 512))
            need(b[:8] == b"EFI PART", "GPT signature missing")
            hs = struct.unpack_from("<I", b, 12)[0]
            need(92 <= hs <= 512, "invalid GPT header size")
            crc = struct.unpack_from("<I", b, 16)[0]; b[16:20] = bytes(4)
            need(zlib.crc32(b[:hs]) & 0xffffffff == crc, "GPT header CRC mismatch")
            own, other = struct.unpack_from("<QQ", b, 24)
            start, count, es, ec = struct.unpack_from("<QIII", b, 72)
            need(own == lba and 0 < count <= 128 and es == 128, "invalid GPT geometry")
            entries = read_at(start * 512, count * es)
            need(zlib.crc32(entries) & 0xffffffff == ec, "GPT entry CRC mismatch")
            return other, entries
        alt, entries = header(1)
        need(alt == size // 512 - 1, "backup GPT must be at image end (not necessarily device end)")
        own, backup = header(alt)
        need(own == 1 and entries == backup, "GPT copies inconsistent")
    parts = []
    for i in range(0, len(entries), 128):
        e = entries[i:i + 128]
        if e[:16] == bytes(16): continue
        first, last, attrs = struct.unpack_from("<QQQ", e, 32)
        name = e[56:128].decode("utf-16-le").rstrip("\0")
        need(first <= last < alt, "partition out of bounds")
        parts.append((name, first, last, attrs))
    need(tuple(p[:3] for p in parts) == LAYOUT, "unexpected partition layout")
    return (parts, bytes(entries)) if with_entries else parts


def lfn_checksum(short):
    checksum = 0
    for c in short:
        checksum = (((checksum & 1) << 7) + (checksum >> 1) + c) & 255
    return checksum


class FatFS:
    """Read FAT16/FAT32, including long names; legacy FAT32 is inspectable."""
    def __init__(self, data):
        self.data = data
        need(len(data) >= 512 and data[510:512] == b"\x55\xaa", "FAT boot signature missing")
        self.bps = struct.unpack_from("<H", data, 11)[0]
        self.spc = data[13]
        reserved = struct.unpack_from("<H", data, 14)[0]
        self.nfat = data[16]
        roots = struct.unpack_from("<H", data, 17)[0]
        total = struct.unpack_from("<H", data, 19)[0] or struct.unpack_from("<I", data, 32)[0]
        f16 = struct.unpack_from("<H", data, 22)[0]
        self.bits = 16 if f16 else 32
        fsize = f16 or struct.unpack_from("<I", data, 36)[0]
        need(self.bps == 512 and self.spc in (1, 2, 4, 8, 16, 32, 64, 128), "invalid FAT sector/cluster size")
        need(reserved > 0 and self.nfat in (1, 2) and fsize > 0, "invalid FAT geometry")
        need(total * self.bps == len(data), "FAT volume/image length mismatch")
        signature = data[54:62] if self.bits == 16 else data[82:90]
        need(signature == (b"FAT16   " if self.bits == 16 else b"FAT32   "), "FAT type signature mismatch")
        root_sectors = (roots * 32 + self.bps - 1) // self.bps
        self.root_offset = (reserved + self.nfat * fsize) * self.bps
        self.root_bytes = root_sectors * self.bps
        self.start = self.root_offset + self.root_bytes
        self.clusters = (total - reserved - self.nfat * fsize - root_sectors) // self.spc
        need(self.clusters > 0, "no FAT data clusters")
        self.standard = (4085 <= self.clusters < 65525) if self.bits == 16 else (self.clusters >= 65525)
        self.cluster_bytes = self.spc * self.bps
        self.fat_offset = reserved * self.bps
        self.fat_bytes = fsize * self.bps
        need((self.clusters + 2) * (self.bits // 8) <= self.fat_bytes, "FAT too small for cluster count")
        if self.nfat == 2:
            need(data[self.fat_offset:self.fat_offset + self.fat_bytes] == data[self.fat_offset + self.fat_bytes:self.fat_offset + 2 * self.fat_bytes], "FAT copies differ")
        self.root = struct.unpack_from("<I", data, 44)[0] if self.bits == 32 else None

    def chain(self, c):
        out = bytearray(); seen = set()
        eoc = 0xfff8 if self.bits == 16 else 0x0ffffff8
        while c < eoc:
            need(2 <= c < self.clusters + 2 and c not in seen, "invalid or cyclic FAT chain")
            seen.add(c)
            off = self.start + (c - 2) * self.cluster_bytes
            need(off + self.cluster_bytes <= len(self.data), "FAT data out of bounds")
            out.extend(self.data[off:off + self.cluster_bytes])
            if self.bits == 16: c = struct.unpack_from("<H", self.data, self.fat_offset + c * 2)[0]
            else: c = struct.unpack_from("<I", self.data, self.fat_offset + c * 4)[0] & 0x0fffffff
        return bytes(out)

    def entries(self, cluster=None):
        data = self.data[self.root_offset:self.root_offset + self.root_bytes] if self.bits == 16 and cluster is None else self.chain(self.root if cluster is None else cluster)
        pieces = {}; checksum = None; names = set()
        for off in range(0, len(data), 32):
            e = data[off:off + 32]
            if len(e) < 32 or e[0] == 0: break
            if e[0] == 0xe5: pieces = {}; checksum = None; continue
            if e[11] == 0x0f:
                if e[0] & 0x40: pieces = {}; checksum = e[13]
                need(checksum == e[13] and 1 <= (e[0] & 31) <= 20, "invalid LFN sequence")
                pieces[e[0] & 31] = e[1:11] + e[14:26] + e[28:32]
                continue
            base = e[:8].decode("ascii").rstrip(); ext = e[8:11].decode("ascii").rstrip()
            if e[12] & 8: base = base.lower()
            if e[12] & 16: ext = ext.lower()
            short = base + ("." + ext if ext else "")
            if pieces:
                need(checksum == lfn_checksum(e[:11]) and sorted(pieces) == list(range(1, len(pieces) + 1)), "LFN checksum/order mismatch")
                name = b"".join(pieces[i] for i in sorted(pieces)).decode("utf-16-le").split("\0")[0].rstrip("\uffff")
            else: name = short
            pieces = {}; checksum = None
            if e[11] & 8 or name in (".", ".."): continue
            need(name and "/" not in name and "\\" not in name, "invalid FAT filename")
            need(name.casefold() not in names, "ambiguous FAT filename")
            names.add(name.casefold())
            c = struct.unpack_from("<H", e, 26)[0]
            if self.bits == 32: c |= struct.unpack_from("<H", e, 20)[0] << 16
            yield name, bool(e[11] & 16), c, struct.unpack_from("<I", e, 28)[0]

    def file(self, path):
        parts = PurePosixPath(path).parts
        need(parts and parts[0] == "/" and ".." not in parts, "absolute safe FAT path required")
        c = None
        for i, name in enumerate(parts[1:]):
            hits = [e for e in self.entries(c) if e[0].casefold() == name.casefold()]
            need(len(hits) == 1, "missing or ambiguous FAT path: " + path)
            _, directory, c, size = hits[0]
            if i < len(parts) - 2: need(directory, "non-directory in FAT path")
            else:
                need(not directory, "FAT path is a directory")
                data = self.chain(c) if size else b""
                need(len(data) >= size, "truncated FAT file")
                return data[:size]
        raise ValueError("file path required")

    def files(self):
        result = {}; seen = set()
        def walk(c, parent):
            if c is not None:
                need(c not in seen, "cyclic FAT directory")
                seen.add(c)
            for name, directory, child, size in self.entries(c):
                path = parent + "/" + name
                if directory: walk(child, path)
                else:
                    data = self.chain(child) if size else b""
                    need(len(data) >= size, "truncated FAT file")
                    result[path] = data[:size]
        walk(None, "")
        return result


def parse_config(data):
    fields = {}; labels = 0
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        pieces = line.split(None, 1)
        key = pieces[0]
        sep = len(pieces) == 2
        value = pieces[1] if sep else ""
        if key == "label": labels += 1
        if key in ("kernel", "initrd", "fdt", "append"):
            need(sep and key not in fields, "missing or duplicate extlinux directive: " + key)
            fields[key] = value.strip()
    need(labels == 1 and set(fields) == {"kernel", "initrd", "fdt", "append"}, "one complete boot label required")
    for key in ("kernel", "initrd", "fdt"):
        path = PurePosixPath(fields[key])
        need(fields[key].startswith("/boot/") and ".." not in path.parts and "$" not in fields[key], "unsafe or unresolved boot path")
    args = {}
    for token in shlex.split(fields["append"]):
        key, sep, value = token.partition("=")
        need(key not in args, "duplicate kernel parameter: " + key)
        args[key] = value if sep else None
    for key, value in {"root": "/dev/sata1p1", "cma": "256M", "vender_format_version": "2", "vendor_format_version": "2"}.items():
        need(args.get(key) == value, "incorrect required boot parameter: " + key)
    for key in IDENTITY_KEYS: need(bool(args.get(key)), "identity field absent: " + key)
    return fields, args


def config_metadata(data):
    fields, args = parse_config(data)
    return {"paths": {k: fields[k] for k in ("kernel", "initrd", "fdt")},
            "identity": {k: {"set": bool(args.get(k)), "length": len(args.get(k, ""))} for k in IDENTITY_KEYS},
            "profile": args.get("och_boot_profile"), "sha256": sha(data)}


def marked_config(data):
    _, args = parse_config(data)
    need("och_boot_profile" not in args, "baseline already has a profile marker")
    lines = data.decode().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("append "):
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            lines[i] = line.rstrip("\r\n") + " och_boot_profile=H1-fat16-bootable" + ending
    result = "".join(lines).encode()
    _, after = parse_config(result)
    need({k: v for k, v in after.items() if k != "och_boot_profile"} == args, "boot parameter semantics changed")
    return result


def image_header(data):
    need(len(data) >= 64 and data[56:60] == b"ARM\x64", "not an ARM64 Image")
    offset, size = struct.unpack_from("<QQ", data, 8)
    need(size >= len(data), "invalid ARM64 memory footprint")
    return {"text_offset": offset, "image_size": size, "file_size": len(data)}


def address_ranges_overlap(ranges):
    """Static range check only; it does not prove the running ENV addresses."""
    for i, (name, start, length) in enumerate(ranges):
        need(start >= 0 and length > 0, "invalid address range")
        for other, base, size in ranges[i + 1:]:
            if start < base + size and base < start + length:
                return name, other
    return None


def cpio_files(uinitrd):
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE, memlimit=128 << 20)
    raw = dec.decompress(uinitrd, max_length=(64 << 20) + 1)
    need(dec.eof and len(raw) <= 64 << 20, "initrd too large or truncated")
    files = {}; pos = 0
    while pos + 110 <= len(raw):
        need(raw[pos:pos + 6] in (b"070701", b"070702"), "invalid newc header")
        values = [int(raw[pos + 6 + i * 8:pos + 14 + i * 8], 16) for i in range(13)]
        size, namesize = values[6], values[11]
        pos += 110
        need(namesize > 0 and pos + namesize <= len(raw), "invalid cpio name size")
        name = raw[pos:pos + namesize - 1].decode(); pos = (pos + namesize + 3) & ~3
        if name == "TRAILER!!!": return files
        name = name[2:] if name.startswith("./") else name
        need(".." not in PurePosixPath(name).parts and pos + size <= len(raw), "unsafe/truncated cpio record")
        files[name] = raw[pos:pos + size]; pos = (pos + size + 3) & ~3
    raise ValueError("cpio trailer missing")


class Baseline:
    def __init__(self, raw):
        self.raw = regular_input(raw)
        need(self.raw.stat().st_size == RAW_BYTES and hash_file(self.raw) == RAW_SHA, "unapproved baseline raw")
        self.partitions = gpt(self.raw)
        with self.raw.open("rb") as f:
            f.seek(BOOT_START); self.boot = f.read(BOOT_BYTES)
        need(sha(self.boot) == BOOT_SHA, "baseline boot mismatch")
        self.fat = FatFS(self.boot); self.files = self.fat.files()
        for path, expected in PINS.items(): need(sha(self.fat.file(path)) == expected, "frozen component mismatch: " + path)
        self.config = self.fat.file(CONFIGS[0])
        need(self.config == self.fat.file(CONFIGS[1]), "configuration copies differ")
        fields, _ = parse_config(self.config)
        need(fields["kernel"] == "/boot/Image" and fields["initrd"] == "/boot/uInitrd" and fields["fdt"] == "/boot/" + DTB_NAME, "baseline boot paths changed")

    def manifest(self, profile):
        boot_config = marked_config(self.config) if profile == "H1" else self.config
        rd = cpio_files(self.fat.file("/boot/uInitrd"))
        versions = {}
        for line in rd.get("etc/VERSION", b"").decode().splitlines():
            key, sep, value = line.partition("=")
            if sep and key in ("productversion", "buildnumber", "unique"):
                versions[key] = value.strip().strip('"')
        modules = {k: sha(v) for k, v in rd.items() if k.startswith("usr/lib/modules/") and k.endswith(".ko")}
        image = self.fat.file("/boot/Image")
        banner = re.search(rb"Linux version 5\.10\.55[^\0]+", image)
        need(banner is not None, "kernel banner missing")
        return {"schema": 1, "profile": profile, "source_run": 44310500902,
                "raw_sha256": RAW_SHA, "components": PINS,
                "boot_configuration": config_metadata(boot_config),
                "source_configuration_sha256": sha(self.config),
                "ARM64_Image": image_header(image),
                "kernel_banner_sha256": sha(banner.group().strip()),
                "initrd_version": versions, "initrd_modules": modules,
                "kernel_changed": False, "CPU_EMMC_changes": False,
                "warning": "User testing has excluded CPU/eMMC experiments as this white-LED cause; identity root cause remains unproven."}


def same_region(a, b, start, end):
    with Path(a).open("rb") as fa, Path(b).open("rb") as fb:
        fa.seek(start); fb.seek(start)
        left = end - start
        while left:
            n = min(left, 1 << 20)
            need(fa.read(n) == fb.read(n), "undeclared raw region changed")
            left -= n


def verify_profile(raw, baseline, profile):
    parts, entries = gpt(raw, with_entries=True)
    if profile == "H0":
        need(hash_file(raw) == RAW_SHA, "H0 must be byte-identical")
        return {"profile": profile, "byte_identical": True, "legacy_fat32_clusters": baseline.fat.clusters}
    need(profile == "H1", "unsupported profile")
    need(parts[0][3] == baseline.partitions[0][3] | 4 and parts[1][3] == baseline.partitions[1][3], "incorrect GPT boot flags")
    _, original_entries = gpt(baseline.raw, with_entries=True)
    expected_entries = bytearray(original_entries)
    struct.pack_into("<Q", expected_entries, 48, baseline.partitions[0][3] | 4)
    need(entries == bytes(expected_entries), "GPT identifiers/types or other partition entries changed")
    same_region(raw, baseline.raw, 34 * 512, BOOT_START)
    same_region(raw, baseline.raw, BOOT_START + BOOT_BYTES, RAW_BYTES - 33 * 512)
    with Path(raw).open("rb") as f:
        f.seek(BOOT_START); fs = FatFS(f.read(BOOT_BYTES))
    need(fs.bits == 16 and fs.standard, "H1 FAT16 geometry not standard")
    expected = dict(baseline.files)
    for path in list(expected):
        if path.casefold() in (p.casefold() for p in CONFIGS): expected[path] = marked_config(baseline.config)
    contract_path = "/boot/BOOT-CONTRACT.json"
    result = fs.files()
    expected_names = {p.casefold() for p in expected} | {contract_path.casefold()}
    need({p.casefold() for p in result} == expected_names, "unexpected boot file inventory")
    for path, data in expected.items(): need(fs.file(path) == data, "packaged file changed: " + path)
    manifest = __import__("json").loads(fs.file(contract_path))
    need(manifest == baseline.manifest("H1"), "contract manifest mismatch")
    for path, expected_sha in PINS.items(): need(sha(fs.file(path)) == expected_sha, "component changed")
    need(fs.file(CONFIGS[0]) == fs.file(CONFIGS[1]), "H1 configuration copies differ")
    need(config_metadata(fs.file(CONFIGS[0]))["profile"] == "H1-fat16-bootable", "H1 marker absent")
    return {"profile": profile, "fat_bits": fs.bits, "data_clusters": fs.clusters,
            "same_partition_layout": True, "bootable": True, "components_frozen": True,
            "loader_userdata_unchanged": True, "raw_sha256": hash_file(raw)}
