#!/usr/bin/env python3
"""Verify the OEC CPU-bin/eMMC compatibility branch (stdlib only).

The model tests do not replace a kernel build or a boot test.  CI additionally
checks the compiled DTB, actual .config and System.map from this kernel build.
No input firmware, host setting or credential is modified by this tool.
"""
import argparse
import copy
import re
import struct
import sys
import unittest
from pathlib import Path

CAP_PROPERTY = "rockchip,cpu-frequency-ceiling-khz"
NO_LIMIT = 0xffffffff
REGULAR = {
    "opp-408000000": (408000000, (0xfb, 0xffff), 850000),
    "opp-600000000": (600000000, (0xfb, 0xffff), 850000),
    "opp-816000000": (816000000, (0xfb, 0xffff), 850000),
    "opp-1104000000": (1104000000, (0xfb, 0xffff), 900000),
    "opp-1416000000": (1416000000, (0xfb, 0xffff), 1025000),
    "opp-1608000000": (1608000000, (0xf9, 0xffff), 1100000),
}
SPECIAL = {
    "opp-j-1008000000": (1008000000, (0x04, 0xffff), 850000),
    "opp-j-1416000000": (1416000000, (0x04, 0xffff), 900000),
    "opp-m-1608000000": (1608000000, (0x02, 0xffff), 1000000),
}
EXPECTED = dict(REGULAR, **SPECIAL)
CAPS = {"map3": 1416000, "map2": 816000, "map0": 408000}
TRIPS = {"trip-point-0": 50000, "trip-point-2": 55000,
         "trip-point-1": 75000, "soc-crit": 115000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_opps(opps, version, speed):
    """Mirror opp/of.c's AND within each two-level hardware mask."""
    require(0 <= version < 32 and 0 <= speed < 32, "invalid bin index")
    result = [o for o in opps.values()
              if o[1][0] & (1 << version) and o[1][1] & (1 << speed)]
    require(result, "all OPPs filtered out")
    freqs = sorted(o[0] // 1000 for o in result)
    require(len(freqs) == len(set(freqs)), "duplicate selected OPP frequency")
    return freqs


def state_for_freq(freqs, cap):
    """Mirror the helper: descending cooling states, lowest-OPP fallback."""
    require(freqs and all(f > 0 for f in freqs), "empty/invalid frequency table")
    require(len(freqs) == len(set(freqs)), "duplicate frequency")
    require(freqs == sorted(freqs) or freqs == sorted(freqs, reverse=True),
            "unsorted frequency table")
    require(0 < cap <= NO_LIMIT, "invalid frequency cap")
    desc = sorted(freqs, reverse=True)
    for state, freq in enumerate(desc):
        if freq <= cap:
            return state, freq, False
    return len(desc) - 1, desc[-1], True


def validate_records(records):
    require(set(records) == set(EXPECTED), "candidate OPP set must contain six regular + three special nodes")
    for name, (hz, mask, uv) in records.items():
        require((hz, mask, uv) == EXPECTED[name], "unexpected OPP/mask/voltage: " + name)
    # The getter currently resolves SoC version 0/1/2. Also cover other bits in
    # the table and all defined speed bits, rather than hardcoding only one box.
    for version in range(8):
        for speed in range(16):
            freqs = selected_opps(records, version, speed)
            expected = [1008000, 1416000] if version == 2 else [408000, 600000, 816000, 1104000, 1416000, 1608000]
            require(freqs == expected, "unexpected filtered table for bin %d/%d" % (version, speed))
            states = [state_for_freq(freqs, cap)[0] for cap in CAPS.values()]
            require(states == ([0, 1, 1] if version == 2 else [1, 3, 5]), "bad cooling states")
            require(all(0 <= s < len(freqs) for s in states), "cooling-state overflow")
    return records


def clean_source(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def block(text, pattern):
    match = re.search(pattern + r"\s*\{", text)
    require(match is not None, "missing source block: " + pattern)
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i]
    raise ValueError("unclosed source block")


def blocks(text, pattern):
    """DTS allows the same node to be overridden multiple times."""
    return [block(text[match.start():], pattern)
            for match in re.finditer(pattern + r"\s*\{", text)]


def source_cells(text, key, optional=False):
    match = re.search(re.escape(key) + r"\s*=\s*(?:/bits/\s+64\s*)?<([^>]*)>", text)
    if match is None and optional:
        return None
    require(match is not None, "missing source property: " + key)
    return tuple(int(x, 0) for x in match.group(1).split())


def audit_source(kernel):
    directory = kernel / "arch/arm64/boot/dts/rockchip"
    soc = clean_source((directory / "rk3568.dtsi").read_text())
    board = clean_source((directory / "rk3566-dsm-common.dtsi").read_text())
    inherited = block(soc, r"cpu0_opp_table:\s*cpu0-opp-table")
    overlay = block(board, r"&cpu0_opp_table")
    require(re.search(r"rockchip,supported-hw\s*;", overlay) is not None,
            "hardware filtering not restored")
    deleted = set(re.findall(r"/delete-node/\s+(opp-[\w-]+)\s*;", overlay))
    records = {}
    for name in re.findall(r"\b(opp-[\w-]+)\s*\{", inherited):
        if name in deleted:
            continue
        original = block(inherited, re.escape(name))
        override = "\n".join(blocks(overlay, r"\b" + re.escape(name)))
        hz = (source_cells(override, "opp-hz", True) or source_cells(original, "opp-hz"))[0]
        mask = source_cells(override, "opp-supported-hw", True) or source_cells(original, "opp-supported-hw")
        voltage = source_cells(override, "opp-microvolt", True) or source_cells(original, "opp-microvolt")
        require(len(mask) == 2 and len(voltage) == 3, "invalid cells: " + name)
        require(voltage[0] == voltage[1] and voltage[0] <= voltage[2] <= 1200000, "invalid voltage range")
        for level in range(4):
            prop = "opp-microvolt-L%d" % level
            if re.search(re.escape(prop) + r"\s*=", original):
                require("/delete-property/ " + prop + ";" in override, "voltage override resurrected: " + name)
        require("/delete-property/ opp-supported-hw;" not in override, "OPP mask deleted")
        records[name] = (hz, mask, voltage[0])
    validate_records(records)
    zones = block(board, r"&soc_thermal")
    for name, cap in CAPS.items():
        node = block(zones, re.escape(name))
        require(source_cells(node, CAP_PROPERTY) == (cap,), "wrong source cap")
        require(re.search(r"cooling-device\s*=\s*<&cpu0\s+0\s+THERMAL_NO_LIMIT>", node) is not None, "invalid cap placeholder/bounds")
    for name in ("trip-point-0", "trip-point-1", "trip-point-2"):
        require(source_cells(block(zones, re.escape(name)), "temperature") == (TRIPS[name],), "wrong trip temperature")
    emmc = block(board, r"&sdhci")
    require(source_cells(emmc, "max-frequency") == (52000000,), "missing eMMC cap")
    for prop in ("mmc-hs200-1_8v", "mmc-hs200-1_2v", "mmc-hs400-1_8v", "mmc-hs400-1_2v", "mmc-hs400-enhanced-strobe"):
        require("/delete-property/ " + prop + ";" in emmc, "high-speed property not removed")
    helper = (kernel / "drivers/thermal/cpufreq_cooling.c").read_text()
    binding = (kernel / "drivers/thermal/thermal_of.c").read_text()
    header = (kernel / "include/linux/cpu_cooling.h").read_text()
    require("EXPORT_SYMBOL_GPL(cpufreq_cooling_get_state_for_freq)" in helper, "helper missing")
    require("cdev->ops != &cpufreq_cooling_ops" in helper, "non-CPU guard missing")
    require("cpufreq_cooling_get_state_for_freq" in binding and CAP_PROPERTY in binding, "helper not wired to OF binding")
    require("count != 1 || ret != 1" in binding and "ret || !prop" in binding, "invalid property guard missing")
    require("tbp->cpu_frequency_limit" in binding, "legacy-map guard missing")
    require(header.count("cpufreq_cooling_get_state_for_freq") == 2, "prototype/stub missing")
    mmc = (kernel / "drivers/mmc/host/sdhci-of-dwcmshc.c").read_text()
    require("host->mmc->f_max <= 52000000" in mmc and "MMC_CAP2_HS200 | MMC_CAP2_HS400" in mmc, "eMMC downgrade path missing")
    print("OK: source filtering, maps, guards and eMMC downgrade path")
    return records


def parse_dtb(data):
    require(len(data) >= 40, "short DTB")
    magic, total, off_struct, off_strings, _, _, _, _, len_strings, len_struct = struct.unpack_from(">10I", data)
    require(magic == 0xd00dfeed and 40 <= total <= len(data), "invalid DTB header")
    require(off_strings + len_strings <= total and off_struct + len_struct <= total, "invalid DTB bounds")
    strings = data[off_strings:off_strings + len_strings]
    nodes, stack = {}, []
    pos, end_struct = off_struct, off_struct + len_struct
    while pos + 4 <= end_struct:
        token = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        if token == 1:
            end = data.find(b"\0", pos, end_struct)
            require(end >= 0, "unterminated node")
            stack.append(data[pos:end].decode("ascii"))
            path = "/" + "/".join(x for x in stack if x)
            require(path not in nodes, "duplicate DTB node")
            nodes[path] = {}
            pos = (end + 4) & ~3
        elif token == 2:
            require(stack, "unbalanced DTB node")
            stack.pop()
        elif token == 3:
            require(stack and pos + 8 <= end_struct, "invalid property header")
            length, name_offset = struct.unpack_from(">II", data, pos)
            pos += 8
            name_end = strings.find(b"\0", name_offset)
            require(name_end >= name_offset >= 0 and pos + length <= end_struct, "invalid property bounds")
            name = strings[name_offset:name_end].decode("ascii")
            nodes["/" + "/".join(x for x in stack if x)][name] = data[pos:pos + length]
            pos = (pos + length + 3) & ~3
        elif token == 4:
            continue  # FDT_NOP is padding, never the end of the tree.
        elif token == 9:
            require(not stack, "unclosed DTB root")
            return nodes
        else:
            raise ValueError("invalid DTB token")
    raise ValueError("missing DTB end token")


def cells(node, key, count=None):
    require(key in node, "missing DTB property: " + key)
    value = node[key]
    require(len(value) % 4 == 0, "misaligned DTB property")
    result = struct.unpack(">" + str(len(value) // 4) + "I", value)
    require(count is None or len(result) == count, "wrong DTB cell count")
    return result


def audit_dtb(path):
    tree = parse_dtb(path.read_bytes())
    table = tree["/cpu0-opp-table"]
    require("rockchip,supported-hw" in table, "DTB filtering disabled")
    records = {}
    for name in EXPECTED:
        node = tree["/cpu0-opp-table/" + name]
        high, low = cells(node, "opp-hz", 2)
        hz = (high << 32) | low
        voltage = cells(node, "opp-microvolt", 3)
        require(voltage[0] == voltage[1] and voltage[0] <= voltage[2] <= 1200000, "bad voltage range")
        require(not any(k.startswith("opp-microvolt-L") for k in node), "per-bin voltage regression")
        records[name] = (hz, cells(node, "opp-supported-hw", 2), voltage[0])
    actual = {p.rsplit("/", 1)[-1] for p in tree if p.startswith("/cpu0-opp-table/opp-") and p.count("/") == 2}
    require(actual == set(EXPECTED), "unexpected extra or missing CPU OPP")
    validate_records(records)
    prefix = "/thermal-zones/soc-thermal"
    for name, temp in TRIPS.items():
        node = tree[prefix + "/trips/" + name]
        require(cells(node, "temperature", 1) == (temp,), "wrong DTB trip")
        require(cells(node, "hysteresis", 1) == (2000,), "wrong trip hysteresis")
    cpu_phandle = cells(tree["/cpus/cpu@0"], "phandle", 1)[0]
    trip_for_map = {"map3": "trip-point-0", "map2": "trip-point-2", "map0": "trip-point-1"}
    for name, cap in CAPS.items():
        node = tree[prefix + "/cooling-maps/" + name]
        trip_phandle = cells(tree[prefix + "/trips/" + trip_for_map[name]], "phandle", 1)
        require(cells(node, "trip", 1) == trip_phandle, "wrong CPU map-to-trip binding")
        require(cells(node, CAP_PROPERTY, 1) == (cap,), "wrong DTB CPU cap")
        require(cells(node, "cooling-device", 3) == (cpu_phandle, 0, NO_LIMIT), "invalid CPU cooling bounds")
    npu = tree["/npu@fde40000"]
    require("#cooling-cells" not in npu, "NPU cooling target resurrected")
    npu_handles = {int.from_bytes(npu[k], "big") for k in ("phandle", "linux,phandle") if k in npu}
    for p, node in tree.items():
        if p.startswith(prefix + "/cooling-maps/"):
            require(p.rsplit("/", 1)[-1] not in ("map4", "map5", "map6"), "NPU map resurrected")
            require(cells(node, "cooling-device")[0] not in npu_handles, "map references NPU")
    emmc = tree["/sdhci@fe310000"]
    require(cells(emmc, "max-frequency", 1) == (52000000,), "eMMC cap missing")
    require(cells(emmc, "bus-width", 1) == (8,), "eMMC bus width changed")
    require(not any(k.startswith("mmc-hs200") or k.startswith("mmc-hs400") for k in emmc), "eMMC high-speed declaration remains")
    print("OK: compiled DTB: nine candidate OPPs, 50/55/75/115 C, NPU out, eMMC 52 MHz")
    for version, label in [(0, "normal"), (1, "M"), (2, "J")]:
        freqs = selected_opps(records, version, 1)
        print(label, "frequencies(kHz)=", freqs, "caps(state/freq/fallback)=", [state_for_freq(freqs, cap) for cap in CAPS.values()])


def audit_build(config, system_map):
    text = config.read_text()
    required = ["THERMAL", "CPU_THERMAL", "CPU_FREQ_THERMAL", "THERMAL_OF", "ROCKCHIP_THERMAL",
                "THERMAL_GOV_STEP_WISE", "THERMAL_DEFAULT_GOV_STEP_WISE", "CPU_FREQ",
                "CPUFREQ_DT", "ARM_ROCKCHIP_CPUFREQ", "PM_OPP", "DEVFREQ_THERMAL"]
    for symbol in required:
        require(re.search(r"^CONFIG_" + symbol + r"=y$", text, re.M) is not None, "missing compiled config: " + symbol)
    require(re.search(r"^[0-9a-fA-F]+\s+T\s+cpufreq_cooling_get_state_for_freq$", system_map.read_text(), re.M) is not None,
            "frequency binding helper not in System.map")
    print("OK: actual kernel configuration and linked helper symbol")


class ModelTests(unittest.TestCase):
    def test_multiple_dts_overrides_are_merged(self):
        text = "opp-1104 { opp-hz = <1104>; }; opp-1104 { /delete-property/ opp-microvolt-L1; };"
        pieces = blocks(text, r"\bopp-1104")
        self.assertEqual(len(pieces), 2)
        self.assertIn("/delete-property/ opp-microvolt-L1;", "\n".join(pieces))

    def test_all_hardware_masks(self):
        validate_records(EXPECTED)

    def test_normal_and_m_keep_1608(self):
        for version in (0, 1):
            self.assertEqual(selected_opps(EXPECTED, version, 1)[-1], 1608000)

    def test_j_table(self):
        self.assertEqual(selected_opps(EXPECTED, 2, 1), [1008000, 1416000])

    def test_missing_specials_is_rejected(self):
        with self.assertRaises(ValueError): validate_records(REGULAR)
        with self.assertRaises(ValueError): selected_opps(REGULAR, 2, 1)

    def test_duplicate_frequency_rejected(self):
        bad = dict(EXPECTED); bad['duplicate'] = EXPECTED['opp-1608000000']
        with self.assertRaises(ValueError): selected_opps(bad, 0, 1)

    def test_wrong_mask_rejected(self):
        bad = dict(EXPECTED); hz, mask, uv = bad['opp-1608000000']
        bad['opp-1608000000'] = hz, (0xff, mask[1]), uv
        with self.assertRaises(ValueError): validate_records(bad)

    def test_speed_outside_defined_mask_is_empty(self):
        with self.assertRaises(ValueError): selected_opps(EXPECTED, 0, 16)

    def test_invalid_bin(self):
        for version, speed in [(-1, 0), (32, 0), (0, -1), (0, 32)]:
            with self.assertRaises(ValueError): selected_opps(EXPECTED, version, speed)

    def test_normal_caps(self):
        table = selected_opps(EXPECTED, 0, 1)
        self.assertEqual([state_for_freq(table, c) for c in CAPS.values()], [(1, 1416000, False), (3, 816000, False), (5, 408000, False)])

    def test_j_fallback(self):
        table = selected_opps(EXPECTED, 2, 1)
        self.assertEqual([state_for_freq(table, c) for c in CAPS.values()], [(0, 1416000, False), (1, 1008000, True), (1, 1008000, True)])

    def test_ascending_descending_equivalence(self):
        table = selected_opps(EXPECTED, 0, 1)
        for cap in (2000000, 1608000, 1500000, 1416000, 1200000, 816000, 408000, 1):
            self.assertEqual(state_for_freq(table, cap), state_for_freq(table[::-1], cap))

    def test_round_down(self):
        self.assertEqual(state_for_freq([408000, 816000, 1416000], 1000000), (1, 816000, False))

    def test_no_limit_frequency_selects_top(self):
        self.assertEqual(state_for_freq([408000, 1608000], NO_LIMIT), (0, 1608000, False))

    def test_empty_invalid_or_unsorted_table(self):
        for table in ([], [0], [-1], [408000, 408000], [816000, 408000, 1416000]):
            with self.assertRaises(ValueError): state_for_freq(table, 816000)

    def test_bad_cap(self):
        for cap in (0, -1, NO_LIMIT + 1):
            with self.assertRaises(ValueError): state_for_freq([408000], cap)

    def test_fdt_nop_does_not_end_tree(self):
        structure = struct.pack(">I", 1) + bytes(4) + struct.pack(">III", 4, 2, 9)
        total = 56 + len(structure)
        header = struct.pack(">10I", 0xd00dfeed, total, 56, total, 40, 17, 16, 0, 0, len(structure))
        self.assertEqual(parse_dtb(header + bytes(16) + structure), {"/": {}})

    def test_wrong_cell_shapes_rejected(self):
        for value, count in [(b"x", 1), (bytes(4), 2), (bytes(8), 1)]:
            with self.assertRaises(ValueError): cells({"cap": value}, "cap", count)
        with self.assertRaises(ValueError): cells({}, "cap", 1)

    def test_voltage_regression_rejected(self):
        bad = dict(EXPECTED)
        hz, mask, voltage = bad["opp-1608000000"]
        bad["opp-1608000000"] = hz, mask, 1050000
        with self.assertRaises(ValueError): validate_records(bad)

    def test_overdrive_regression_rejected(self):
        bad = dict(EXPECTED)
        bad["opp-1800000000"] = 1800000000, (0xf9, 0xffff), 1150000
        with self.assertRaises(ValueError): validate_records(bad)

    def test_single_opp_fallback(self):
        self.assertEqual(state_for_freq([1008000], 408000), (0, 1008000, True))

    def test_bad_dtb_is_rejected(self):
        for data in (b'', bytes(40), bytes(512)):
            with self.assertRaises(ValueError): parse_dtb(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--dtb', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--system-map', type=Path)
    args = parser.parse_args()
    if args.selftest:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests))
        if not result.wasSuccessful(): return 1
    if args.source: audit_source(args.source)
    if args.dtb: audit_dtb(args.dtb)
    require(bool(args.config) == bool(args.system_map), 'config and system-map must be provided together')
    if args.config: audit_build(args.config, args.system_map)
    require(args.selftest or args.source or args.dtb or args.config, 'no verification selected')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, struct.error) as error:
        print('FAILED:', error, file=sys.stderr)
        sys.exit(1)
