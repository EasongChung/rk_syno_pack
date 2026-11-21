#!/usr/bin/env python3
"""Build offline H0/H1 boot profiles from the pinned public raw, never a device.

H0 is the byte-identical reference. H1 rebuilds only the 32 MiB FAT volume as
FAT16, adds a non-identity profile marker/contract and marks GPT partition 1
bootable. Reliable dosfstools/mtools/gdisk are required; no formatter fallback.
"""
import argparse
import gzip
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

from oec_boot_contract import (Baseline, BOOT_BYTES, BOOT_START, CONFIGS,
                               GZ_SHA, RAW_BYTES, RAW_SHA, hash_file, marked_config,
                               need, regular_input, verify_profile)

REPO = Path(__file__).resolve().parents[1]
ALLOWED = (REPO / "output", REPO.parent / "payload", REPO.parent / "checks")


def safe_output(path):
    target = Path(path).resolve()
    need(any(target.is_relative_to(root.resolve()) and target != root.resolve() for root in ALLOWED),
         "output must be a dedicated subdirectory under repository output/payload/checks")
    need(not target.exists(), "output already exists; use a fresh directory, nothing is overwritten")
    return target


def tools():
    result = {}
    for name in ("mkfs.vfat", "mcopy", "fsck.fat", "sgdisk"):
        path = shutil.which(name)
        need(path is not None, "missing reliable build tool: " + name)
        result[name] = path
    help_text = subprocess.run([result["mkfs.vfat"], "--help"], capture_output=True, text=True)
    text = help_text.stdout + help_text.stderr
    need("-F" in text and "BusyBox" not in text, "formatter must support explicit FAT16 (-F 16); BusyBox is not suitable")
    return result


def run(args):
    env = dict(os.environ)
    env.pop("MTOOLS_SKIP_CHECK", None)
    result = subprocess.run(args, capture_output=True, env=env)
    need(result.returncode == 0, Path(args[0]).name + " failed, exit " + str(result.returncode))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def materialize(source, dest):
    source = regular_input(source)
    if source.name.endswith(".gz"):
        need(hash_file(source) == GZ_SHA, "gzip is not the approved baseline")
        with gzip.open(source, "rb") as src, dest.open("xb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
    else:
        need(source.stat().st_size == RAW_BYTES and hash_file(source) == RAW_SHA,
             "raw is not the approved baseline")
        with source.open("rb") as src, dest.open("xb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
    need(dest.stat().st_size == RAW_BYTES and hash_file(dest) == RAW_SHA, "baseline materialization mismatch")
    return source


def compress(raw, target):
    with raw.open("rb") as src, target.open("xb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            shutil.copyfileobj(src, gz, 1 << 20)
    import hashlib
    h = hashlib.sha256(); total = 0
    with gzip.open(target, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); total += len(chunk)
    need(total == raw.stat().st_size and h.hexdigest() == hash_file(raw), "gzip/raw mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-raw", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--profile", choices=("H0", "H1", "all"), default="all")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--check-tools", action="store_true")
    a = ap.parse_args()
    if a.check_tools:
        tools(); print("reliable FAT16 tooling available"); return
    need(a.baseline_raw and a.out, "baseline-raw and out are required")
    selected = ("H0", "H1") if a.profile == "all" else (a.profile,)
    native = tools() if "H1" in selected and not a.extract_only else None
    out = safe_output(a.out)
    out.mkdir(parents=True)
    source = materialize(a.baseline_raw, out / "baseline.raw.img")
    baseline = Baseline(out / "baseline.raw.img")
    if a.extract_only:
        tree = out / "tree"; tree.mkdir()
        for name, data in baseline.files.items():
            file = tree / name.lstrip("/"); file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        write_json(out / "BASELINE.json", baseline.manifest("H0"))
        print(json.dumps({"baseline_verified": True, "files": len(baseline.files),
                          "fat_bits": baseline.fat.bits, "clusters": baseline.fat.clusters,
                          "standard_fat_geometry": baseline.fat.standard,
                          "kernel_changed": False, "identity": baseline.manifest("H0")["boot_configuration"]["identity"]}, indent=2))
        return
    reports = []
    for profile in selected:
        folder = out / profile; folder.mkdir()
        raw = folder / ("rk3566-dsm-raw_" + profile + ("-reference" if profile == "H0" else "-fat16-bootable") + ".img")
        shutil.copyfile(baseline.raw, raw)
        if profile == "H1":
            tree = folder / "tree"; tree.mkdir()
            for name, data in baseline.files.items():
                file = tree / name.lstrip("/"); file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(marked_config(baseline.config) if name.casefold() in [p.casefold() for p in CONFIGS] else data)
            write_json(tree / "boot/BOOT-CONTRACT.json", baseline.manifest("H1"))
            boot = folder / "boot-H1.img"
            with boot.open("xb") as f: f.truncate(BOOT_BYTES)
            run([native["mkfs.vfat"], "-F", "16", "-n", "DSMBOOT", str(boot)])
            run([native["mcopy"], "-i", str(boot), "-s"] + [str(p) for p in sorted(tree.iterdir())] + ["::/"])
            run([native["fsck.fat"], "-n", str(boot)])
            with raw.open("r+b") as f:
                f.seek(BOOT_START)
                with boot.open("rb") as src: shutil.copyfileobj(src, f, 1 << 20)
                f.flush(); os.fsync(f.fileno())
            run([native["sgdisk"], "--attributes=1:set:2", str(raw)])
            run([native["sgdisk"], "--verify", str(raw)])
        report = verify_profile(raw, baseline, profile)
        gz = Path(str(raw) + ".gz")
        if profile == "H0" and source.name.endswith(".gz"):
            shutil.copyfile(source, gz)
            need(hash_file(gz) == GZ_SHA, "H0 gzip copy mismatch")
        else: compress(raw, gz)
        report["raw_sha256"] = hash_file(raw); report["gz_sha256"] = hash_file(gz)
        report["H2"] = "not generated; runtime ENV backend/location unproven"
        write_json(folder / "VALIDATION.json", report)
        (folder / "SHA256SUMS").write_text(report["raw_sha256"] + "  " + raw.name + "\n" + report["gz_sha256"] + "  " + gz.name + "\n")
        (folder / "README.md").write_text("# " + profile + " 启动兼容试验\n\nH0是原包对照；H1仅改FAT16、启动标志和非身份启动标记。CPU/eMMC实验已由用户实测排除为本次白灯根因，本包不叠加它们。\n\n内核/initrd/DTB冻结为项目20261119同代三件套；不换7.2底座，不改SPL/U-Boot，不绕版本或身份校验。SN不显示根因仍未证实，不归因于DTB名称。\n\n刷写会覆盖eMMC前128MiB，先备份并准备可启动的7.2恢复包。沿用RKDevTool Loader模式、地址0、目标eMMC整盘，选择解压raw。仅反馈能启动/仍白灯；能进DSM还需检查身份、安装升级与功能。此包未获实机验收，不自动刷机或正式发布。\n")
        reports.append(report)
    write_json(out / "RESULTS.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    try: main()
    except (ValueError, OSError, subprocess.SubprocessError, lzma.LZMAError) as error:
        print("FAILED: " + str(error), file=sys.stderr)
        sys.exit(1)
