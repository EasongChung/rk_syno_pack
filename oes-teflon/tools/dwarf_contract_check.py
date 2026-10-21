#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipe_ml 契约检查器 —— 把「契约只有 4 槽、没有维度数」这条根因级事实固化成可重跑的检查。

## 为什么需要它

本项目的第三处缺陷根因（rank-5 张量被压进 4 槽、size 恒定丢 dims[0]）**完全建立在
契约形状之上**：

    struct pipe_tensor {
        void    *data;        /* +0  */
        unsigned index;       /* +8  */
        unsigned dims[4];     /* +12 → 下一成员 +28，占 16 字节 */
        ...
    };                         /* byte_size = 64 */

只要 `dims` 还是 4 个元素、且结构体里**没有维度数字段**，那么：

* etnaviv 用 `dims[1]*dims[2]*dims[3]` 算字节数**不是 bug，是该契约下唯一可能的实现**；
* 任何「按维度数遍历 size」的修法**都不可实现**（那个字段不存在）；
* rank-5 张量**在契约层就无法表达**，因此 driver 侧无法通过补丁支持它。

⇒ **一旦 Mesa 升级后 `dims` 变成 5 槽、或新增了维度数字段，本项目的根因分析必须重做。**
这个检查器就是用来第一时间发现那种变化的。

## 用法

    python3 tools/dwarf_contract_check.py <libteflon.so>
    python3 tools/dwarf_contract_check.py <libteflon.so> --json

退出码：
    0 = 契约与已知基线一致
    1 = **契约已漂移**（dims 槽数变了 / 出现维度数字段 / 结构体消失）—— 根因分析需重做
    2 = 用法错误或前置条件不满足（缺 readelf / 缺 DWARF / 找不到该 .so）

## 依赖

只需要 `readelf`（binutils）。不依赖 pyelftools —— 因为 DWARF 的**类型名与成员名都在
`.debug_str` 里是明文 ASCII**，而 `dims` 的数组长度能从 `DW_AT_upper_bound` 直接读到，
用 `readelf --debug-dump=info` 的文本输出解析已经足够，且输出稳定、可人工核对。

## 局限（如实声明）

* 依赖 `.debug_info` 存在 ⇒ **只对未 strip 的构建有效**。对 strip 过的 .so 会明确报退出码 2，
  而不是给出一个错误的「未漂移」结论。
* 本检查器验证的是**契约形状**，不验证 etnaviv 侧如何使用这个契约。
  「驱动恒定丢 dims[0]」这条行为结论由 `verify/reports/11-contract-rootcause-2026-10-19.md`
  的算术验证（107/109 次创建）支撑，不在本检查器范围内。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── 已知基线（首次由 libteflon.so 的 DWARF 实测确立，见报告 11）──────────────────
BASELINE = {
    "struct_name": "pipe_tensor",
    "byte_size": 64,
    "dims_member_offset": 12,
    "next_member_offset": 28,
    "dims_slots": 4,
    "has_dimension_count_member": False,
}

# 任何含有这些名字的成员，都意味着「契约带维度数」⇒ 根因分析需要重做
DIM_COUNT_NAMES = (
    "dims_size",
    "num_dims",
    "n_dims",
    "dims_num",
    "num_dimensions",
    "ndims",
    "rank",
    "dimensions_count",
    "dim_count",
)


def _die(msg: str, code: int = 2) -> "None":
    print(f"[契约检查] {msg}", file=sys.stderr)
    raise SystemExit(code)


def _structure_die(text: str, name: str) -> list[str] | None:
    """取出名为 `name` 的第一个 structure_type DIE 及其直属成员行。

    DWARF dump 的结构是：`<1><OFF>: ... (DW_TAG_structure_type)` 之后，
    成员以 `<2>`（深度 2）出现，直到下一个深度 <= 1 的 DIE 为止。
    """
    lines = text.split("\n")
    start = None
    pat = re.compile(rf"^ <\d+><([0-9a-f]+)>:.*\(DW_TAG_structure_type\)")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        # 该 DIE 的名字可能出现在接下来几行
        nm = None
        for j in range(i + 1, min(i + 10, len(lines))):
            # 必须用**贪婪** .*：DWARF 的名字行形如
            #   DW_AT_name : (indirect string, offset: 0x26979): pipe_tensor
            # 非贪婪会在 'offset: ' 处就截断，抓到 '0x26979): pipe_tensor'。
            mn = re.search(r"DW_AT_name\s+:.*:\s(\S+)\s*$", lines[j])
            if mn:
                nm = mn.group(1).strip()
                break
        if nm != name:
            continue
        start = i
        break
    if start is None:
        return None
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if re.match(r"^ <[01]><", ln):
            break
        out.append(ln)
    return out


def _members(struct_lines: list[str]) -> list[dict]:
    """把成员 DIE 解析成 [{name, offset, line, type_die}]。"""
    members: list[dict] = []
    for i, ln in enumerate(struct_lines):
        if "DW_TAG_member" not in ln:
            continue
        name = None
        for j in range(i + 1, min(i + 6, len(struct_lines))):
            mn = re.search(r"DW_AT_name\s+:.*:\s(\S+)\s*$", struct_lines[j])
            if mn:
                name = mn.group(1).strip()
                break
        off = None
        decl = None
        typ = None
        for j in range(i + 1, min(i + 10, len(struct_lines))):
            if off is None:
                mo = re.search(r"DW_AT_data_member_location: (\d+)", struct_lines[j])
                if mo:
                    off = int(mo.group(1))
            if decl is None:
                md = re.search(r"DW_AT_decl_line\s+:\s+(\d+)", struct_lines[j])
                if md:
                    decl = int(md.group(1))
            if typ is None:
                mt = re.search(r"DW_AT_type\s+:\s+<0x([0-9a-f]+)>", struct_lines[j])
                if mt:
                    typ = mt.group(1)
        if name:
            members.append({"name": name, "offset": off, "decl_line": decl, "type_die": typ})
    return members


def inspect(so_path: Path) -> dict:
    if shutil.which("readelf") is None:
        _die("找不到 readelf（binutils）。本检查器依赖它解析 DWARF。")
    if not so_path.is_file():
        _die(f"文件不存在: {so_path}")

    with tempfile.TemporaryDirectory(prefix="dwcontract-") as td:
        info = Path(td) / "info.txt"
        rc = subprocess.run(
            ["readelf", "--debug-dump=info", str(so_path)],
            stdout=info.open("w", encoding="utf-8", errors="replace"),
            stderr=subprocess.DEVNULL,
        )
        if rc.returncode != 0 or info.stat().st_size == 0:
            _die(
                f"从 {so_path} 取不到 DWARF。\n"
                "        本检查器**只对未 strip 的构建有效**；"
                "请传入带 -g3 且未 strip 的 libteflon.so。\n"
                "        （不会给出『未漂移』的误导结论。）"
            )
        text = info.read_text(encoding="utf-8", errors="replace")

    struct_lines = _structure_die(text, BASELINE["struct_name"])
    if struct_lines is None:
        # 名字变了 / 结构体消失 —— 本身就是漂移信号，但要区分「没有 DWARF」与「结构体没了」
        return {
            "so": str(so_path),
            "struct_found": False,
            "reason": f"未找到 struct {BASELINE['struct_name']}（可能被改名或删除）",
        }

    byte_size = None
    mb = re.search(r"DW_AT_byte_size\s+:\s+(\d+)", "\n".join(struct_lines[:8]))
    if mb:
        byte_size = int(mb.group(1))

    members = _members(struct_lines)
    dims = next((m for m in members if m["name"] == "dims"), None)

    dims_slots = None
    dims_upper = None
    if dims and dims["type_die"]:
        # 找到 dims 的数组类型 DIE，读 DW_AT_upper_bound
        pat = re.compile(rf"^ <\d+><{dims['type_die']}>:")
        for i, ln in enumerate(text.split("\n")):
            if pat.match(ln):
                for j in range(i + 1, min(i + 10, len(text.split("\n")))):
                    mu = re.search(r"DW_AT_upper_bound\s+:\s+(\d+)", text.split("\n")[j])
                    if mu:
                        dims_upper = int(mu.group(1))
                        dims_slots = dims_upper + 1
                break

    # 下一个成员的偏移 ⇒ dims 实际占用的字节数（与 DW_AT_upper_bound 互为交叉验证）
    next_off = None
    dims_byte_span = None
    if dims and dims["offset"] is not None:
        later = [m["offset"] for m in members
                 if m["offset"] is not None and m["offset"] > dims["offset"]]
        if later:
            next_off = min(later)
            dims_byte_span = next_off - dims["offset"]

    found_dim_count = sorted({n for n in DIM_COUNT_NAMES
                              if re.search(rf"DW_AT_name\s+:\s+.*?: {re.escape(n)}$",
                                           text, re.M)})

    return {
        "so": str(so_path),
        "struct_found": True,
        "byte_size": byte_size,
        "members": members,
        "dims_offset": dims["offset"] if dims else None,
        "next_member_offset": next_off,
        "dims_byte_span": dims_byte_span,
        "dims_upper_bound": dims_upper,
        "dims_slots": dims_slots,
        "dim_count_members_found": found_dim_count,
        "has_dimension_count_member": bool(found_dim_count),
    }


def compare(res: dict) -> list[str]:
    drift: list[str] = []
    if not res.get("struct_found"):
        drift.append(f"struct {BASELINE['struct_name']} 未找到：{res.get('reason','')}")
        return drift
    if res["byte_size"] != BASELINE["byte_size"]:
        drift.append(f"byte_size {res['byte_size']} ≠ 基线 {BASELINE['byte_size']}")
    if res["dims_offset"] != BASELINE["dims_member_offset"]:
        drift.append(f"dims 偏移 {res['dims_offset']} ≠ 基线 {BASELINE['dims_member_offset']}")
    if res["dims_slots"] != BASELINE["dims_slots"]:
        drift.append(f"dims 槽数 {res['dims_slots']} ≠ 基线 {BASELINE['dims_slots']}")
    # 交叉验证：偏移跨度应等于槽数 × 4 字节
    if res["dims_byte_span"] is not None and res["dims_slots"] is not None:
        if res["dims_byte_span"] != res["dims_slots"] * 4:
            drift.append(
                f"dims 偏移跨度 {res['dims_byte_span']}B 与槽数 "
                f"{res['dims_slots']}×4={res['dims_slots'] * 4}B 不自洽"
            )
    if res["has_dimension_count_member"] != BASELINE["has_dimension_count_member"]:
        drift.append(
            f"契约出现维度数字段 {res['dim_count_members_found']} —— "
            "**根因分析必须重做**（'按维度数遍历' 从不可实现变为可实现）"
        )
    return drift


def main() -> int:
    ap = argparse.ArgumentParser(description="pipe_ml 契约漂移检查")
    ap.add_argument("so", type=Path, help="未 strip 的 libteflon.so")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()

    res = inspect(a.so)
    drift = compare(res)

    if a.json:
        print(json.dumps({"result": res, "drift": drift}, ensure_ascii=False, indent=2))
        return 1 if drift else 0

    print("═" * 66)
    print(" pipe_ml 契约检查（根因级事实：契约只有 4 槽、无维度数）")
    print("═" * 66)
    if not res.get("struct_found"):
        print(f"  ✗ {res.get('reason')}")
        print("─" * 66)
        for d in drift:
            print(f"  ✗ 漂移：{d}")
        print("\n结论：★ 契约已漂移 —— 报告 11 的根因分析需要重做")
        return 1

    print(f"  文件            : {res['so']}")
    print(f"  struct          : {BASELINE['struct_name']}  byte_size={res['byte_size']}")
    print(f"  dims 偏移       : +{res['dims_offset']}  → 下一成员 +{res['next_member_offset']}"
          f"  （占 {res['dims_byte_span']} 字节）")
    print(f"  dims 槽数       : {res['dims_slots']}   "
          f"(DW_AT_upper_bound={res['dims_upper_bound']}, 4B/槽)")
    print(f"  维度数字段      : {res['dim_count_members_found'] or '无'}")
    print("─" * 66)
    print("  成员表：")
    for m in res["members"]:
        print(f"    +{str(m['offset']):>3}  {m['name']:<16} (行 {m['decl_line']})")
    print("─" * 66)
    if drift:
        for d in drift:
            print(f"  ✗ 漂移：{d}")
        print("\n结论：★ 契约已漂移 —— 报告 11 的根因分析需要重做")
        return 1
    print("  ✅ 与已知基线一致（4 槽、无维度数）")
    print("  ⇒ etnaviv 用 dims[1]*dims[2]*dims[3] 算 size 在该契约下是唯一可能的实现，")
    print("    不是 bug；rank-5 张量在契约层无法表达，driver 侧无法用补丁支持。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())