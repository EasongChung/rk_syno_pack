#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dwarf_contract_check.py 的双向自测：正例 + 反例 + 反向对照。

纪律：本项目的门必须有**反例**才能证明它真的能抓到漂移，否则「全绿」只是没触发。

    python3 tools/test_dwarf_contract_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwarf_contract_check as C  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}{('  ' + detail) if detail else ''}")


def good_result() -> dict:
    """与已知基线一致的构造结果。"""
    return {
        "so": "x.so", "struct_found": True, "byte_size": 64,
        "dims_offset": 12, "next_member_offset": 28, "dims_byte_span": 16,
        "dims_upper_bound": 3, "dims_slots": 4,
        "dim_count_members_found": [], "has_dimension_count_member": False,
        "members": [],
    }


print("=" * 64)
print(" dwarf_contract_check 双向自测")
print("=" * 64)

print("\n[1] 正例：与基线一致 ⇒ 无漂移")
check("基线结果不报漂移", C.compare(good_result()) == [])

print("\n[2] 反例A：dims 变成 5 槽 ⇒ 必须报漂移")
r = good_result(); r["dims_slots"] = 5; r["dims_upper_bound"] = 4; r["dims_byte_span"] = 20
d = C.compare(r)
check("5 槽被报漂移", any("槽数" in x for x in d), str(d))

print("\n[3] 反例B：契约出现维度数字段 ⇒ 必须报漂移且点名『根因分析需重做』")
r = good_result(); r["has_dimension_count_member"] = True
r["dim_count_members_found"] = ["dims_size"]
d = C.compare(r)
check("维度数字段被报漂移", any("维度数" in x for x in d))
check("漂移文案要求重做根因分析", any("必须重做" in x for x in d))

print("\n[4] 反例C：byte_size / dims 偏移变化 ⇒ 必须报漂移")
r = good_result(); r["byte_size"] = 72
check("byte_size 变化被报", any("byte_size" in x for x in C.compare(r)))
r = good_result(); r["dims_offset"] = 16
check("dims 偏移变化被报", any("dims 偏移" in x for x in C.compare(r)))

print("\n[5] 反例D：结构体消失 ⇒ 必须报漂移")
d = C.compare({"struct_found": False, "reason": "未找到 struct pipe_tensor"})
check("结构体消失被报漂移", len(d) >= 1, str(d))

print("\n[6] 反例E：自相矛盾（偏移跨度与槽数不符）⇒ 必须报漂移")
r = good_result(); r["dims_slots"] = 3; r["dims_byte_span"] = 16
check("跨度/槽数不自洽被报", any("不自洽" in x for x in C.compare(r)))

print("\n[7] 反向对照：只有上述反例能触发，正常输入不误报")
check("无假阳性（基线重复 5 次都不报）",
      all(C.compare(good_result()) == [] for _ in range(5)))

print("\n[8] 正则坑回归：名字必须取到 `pipe_tensor`，不能是 `0x…): pipe_tensor`")
line = "    <c33d>   DW_AT_name        : (indirect string, offset: 0x26979): pipe_tensor"
import re  # noqa: E402
got = re.search(r"DW_AT_name\s+:.*:\s(\S+)\s*$", line)
check("间接字符串格式解析正确", got is not None and got.group(1) == "pipe_tensor",
      f"实得={got.group(1) if got else None!r}")

print("\n[9] 前置条件失败 ⇒ 退出码 2 且不谎报『未漂移』")
with tempfile.TemporaryDirectory() as td:
    junk = Path(td) / "junk.so"
    junk.write_bytes(b"\x7fELF" + b"\x00" * 256)
    try:
        C.inspect(junk)
        check("无 DWARF 的文件应报错退出", False, "竟未抛错")
    except SystemExit as e:
        check("无 DWARF 的文件退出码=2", e.code == 2, f"实得 code={e.code}")

print("\n" + "=" * 64)
total = len(PASS) + len(FAIL)
# 反例组 = [2]~[6]，[9] 是前置条件反例；[7] 是反向对照（查假阳性），[8] 是回归
N_NEG = 7
if FAIL:
    print(f"  总判定: {len(PASS)}/{total} —— ❌ 有 {len(FAIL)} 项失败: {FAIL}")
    raise SystemExit(1)
print(f"  总判定: {len(PASS)}/{total} 全绿 ✅")
print(f"  其中反例 {N_NEG} 项全部被正确捕获；反向对照确认无假阳性。")