#!/usr/bin/env python3
"""Optional post-boot lab probe. Read-only; outputs metadata, never SN/MAC.

Generate/ship separately from firmware. Do not run automatically on users'
NAS. Device nodes are observations, not proof of a specific crash cause.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def content_status(data, inode_size=None):
    value = data.strip(b" \t\r\n\0")
    return {"content_nonempty": bool(value), "content_bytes": len(value),
            "all_zero_text": bool(value) and set(value) <= {48, 58}, "inode_size": inode_size}


def failures(data):
    text = data.decode("utf-8", "replace")
    patterns = {"serial_checksum_errors": r"serial number checksum error",
                "missing_serial_tags": r"no serial tag found",
                "missing_checksum_tags": r"no checksum tag found",
                "unknown_symbol_events": r"Unknown symbol"}
    return {key: len(re.findall(pattern, text, re.I)) for key, pattern in patterns.items()}


def read(path):
    try: return Path(path).read_bytes()
    except OSError: return None


def node(path):
    try:
        info = os.stat(path)
        return {"exists": True, "block": stat.S_ISBLK(info.st_mode),
                "character": stat.S_ISCHR(info.st_mode),
                "major": os.major(info.st_rdev), "minor": os.minor(info.st_rdev)}
    except OSError: return {"exists": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path)
    a = p.parse_args()
    cmd = read("/proc/cmdline") or b""
    fields = {}
    # Only metadata is returned. Values are never included in the report.
    for key in ("sn", "custom_sn", "mac1", "root", "cma", "vender_format_version", "vendor_format_version"):
        values = re.findall(rb"(?:^|\s)" + key.encode() + rb"=([^\s]+)", cmd)
        fields[key] = {"count": len(values), "set": bool(values), "length": len(values[0]) if values else 0}
    marker = re.search(rb"(?:^|\s)och_boot_profile=([^\s]+)", cmd)
    marker_ok = marker is not None and marker.group(1) == b"H1-fat16-bootable"
    identity = {}
    for path in ("/proc/sys/kernel/syno_serial", "/proc/sys/kernel/syno_mac_address1"):
        data = read(path)
        if data is None: identity[path] = {"readable": False}
        else: identity[path] = content_status(data, Path(path).stat().st_size)
    root_version = {}
    data = read("/etc/VERSION") or b""
    for key in ("productversion", "buildnumber", "unique"):
        match = re.search(key.encode() + rb'="([^"\n]+)"', data)
        if match: root_version[key] = match.group(1).decode("ascii", "replace")
    result = {"expected_profile_marker_present": marker_ok, "bootargs_metadata": fields,
              "kernel_identity_fields": identity, "root_version": root_version,
              "nodes": {path: node(path) for path in ("/dev/synobios", "/dev/synoboot", "/dev/synoboot1", "/dev/synoboot2")},
              "project_model_matches": (read("/proc/device-tree/model") or b"").rstrip(b"\0\n") == b"synology_rk3566_oec_box_wxy4",
              "kernel_release": (read("/proc/sys/kernel/osrelease") or b"").decode().strip(),
              "CPU_OPP_names": sorted(p.name for p in Path("/proc/device-tree/cpu0-opp-table").glob("opp-*")),
              "UI_check_required": True, "full_compatibility_proven": False}
    if a.contract:
        contract = json.loads(a.contract.read_text())
        result["initrd_root_version_match"] = root_version == contract.get("initrd_version")
        version = read("/proc/version")
        result["kernel_banner_matches"] = version is not None and hashlib.sha256(version.strip()).hexdigest() == contract.get("kernel_banner_sha256")
    try:
        log = subprocess.run(["dmesg"], capture_output=True, timeout=10)
        result["error_counts"] = failures(log.stdout)
    except (OSError, subprocess.TimeoutExpired): result["error_counts"] = {"unavailable": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
