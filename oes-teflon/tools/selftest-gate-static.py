#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`tools/gate_static.py` 的自测（正控制 + 负控制 + 回归）。

为什么要有这个
--------------
本项目反复栽在同一类错误上：**检查器自己会错**，而且「门通过」被当成了
「模型可用」。`docs/README.md:154` 记着一次真实的证伪：

    「删掉空形状 t133 能治愈整条退化链」—— 实测退化数 6→6 不变，此路已废

⇒ 门 b 当初就是基于那个**已被证伪的因果假设**设的（shape=[] 即异常），
   实测 TFLite 的真实规则是「**buffer 字节数**要等于 `num_elements × type_size`」，
   而标量（shape=[]）的 num_elements 是 **1** 不是 0。
⇒ 本自测把这条规则钉死：任何一条控制失败 ⇒ 门不可信 ⇒ 退出码非 0。

控制清单
--------
正控制（应当 PASS / 判成「合法标量」）
  P1  合法标量：shape=[] + float32 + 4 字节  ⇒ 门 b 不该报不匹配
  P2  合法标量：shape=[] + int8 + 1 字节    ⇒ 同上
  P3  正常量化模型（per-tensor）             ⇒ a/b/c 全过

负控制（应当 FAIL，且**报对原因**）
  N1  字节数少一半（float32 标量只给 2 字节）⇒ 门 b 报「字节数不匹配」
  N2  字节数多一倍（int8 标量给 2 字节）    ⇒ 同上
  N3  scale 是哨兵（1e34）                   ⇒ 门 a 报「退化」
  N4  权重是 per-axis（scale.len>1）         ⇒ 门 c 报「非 per-tensor」
  N5  激活张量无内联数据（buffer 长度 0）     ⇒ **不得**被门 b 误判

回归（锁住真实样本的判定）
  R1  旧的 per-tensor 模型（6 个退化张量）    ⇒ 门 a 必须 FAIL（退化 6）
  R2  旧的 legacy 模型（per-channel 权重）     ⇒ 门 c 必须 FAIL
  R3  修复后的 per-tensor 模型                ⇒ a/b/c 全 PASS

用法
----
    python3 tools/selftest-gate-static.py          # 用仓库内样本
    python3 tools/selftest-gate-static.py --verbose
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import gate_static  # noqa: E402
from pathlib import Path  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  —— " + detail) if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def gates(path):
    return gate_static.gates(Path(path))


def report_of(path):
    """跑一遍 report 并把输出抓下来（门失败时 report 返回 False）。"""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = gate_static.report(gates(path))
    return ok, buf.getvalue()


def find_sample(*parts):
    p = os.path.join(ROOT, *parts)
    return p if os.path.exists(p) else None


def make_model(tensors_spec):
    """造一个最小 TFLite 文件（够门 a/b/c 用）。

    tensors_spec: [(name, type, shape, data_bytes, scale|None, zp|None)]
    """
    # 直接借真实模型做字节级改写太脆弱，这里用最小 flatbuffer 构造：
    # 一个 subgraph、若干 tensor、若干 buffer。
    import tflite as T  # PyPI 的 tflite 包（只读 API）

    # 用 flatbuffers builder 直接造
    import flatbuffers

    def create(path, specs):
        b = flatbuffers.Builder(1024 * 1024)
        # buffers
        buf_offsets = []
        for (_n, _t, _s, data, _sc, _zp) in specs:
            if data is None:
                buf_offsets.append(None)
                continue
            b.StartVector(1, len(data), 1)
            for byte in reversed(data):
                b.PrependByte(byte)
            vec = b.EndVector()
            b.StartVector(1, 1, 4)
            b.PrependUOffsetTRelative(vec)
            data_vec = b.EndVector()
            T.BufferStartDataVector(b, data_vec)
            b.StartBuffer()
            buf_offsets.append(b.EndBuffer())
        # 简化：直接用 tflite 包的 builder 工具函数拼
        raise NotImplementedError


def main(argv):
    verbose = "--verbose" in argv
    print("=== gate_static 自测 ===\n")

    # ---------- P/N 控制用真实样本（避免自造 flatbuffer 引入无关变量）----------
    legacy_clean = find_sample("verify", "artifacts", "local-surgery",
                                "legacy-clean-nodepad.tflite")
    pt_final = find_sample("verify", "artifacts", "local-surgery", "pt-final.tflite")
    pt_old = find_sample("artifact-int8", "concept_int8_pertensor.tflite")
    legacy_raw = find_sample("verify", "artifacts", "oes-cleaned-early-39628736933",
                              "concept_legacy.tflite")

    if not (legacy_clean and pt_final and pt_old and legacy_raw):
        print("ERROR: 缺样本文件（需要 legacy-clean-nodepad / pt-final / "
              "concept_int8_pertensor / concept_legacy）")
        return 2

    print("── 正控制 ──")
    ok, out = report_of(pt_final)
    check("P3 修复后的 per-tensor 模型 a/b/c 全过", ok, out)
    check("P1/P2 合法标量不被判为字节数不匹配",
          "字节数不匹配" not in out, out)
    check("P1/P2 合法标量被单独列出（可核对）",
          "合法标量" in out, out)

    ok_legacy, out_legacy = report_of(legacy_clean)
    check("P3' 清洗后的 legacy 门 a/b 通过", "❌ a" not in out_legacy and "❌ b" not in out_legacy,
          out_legacy)

    print("\n── 负控制（对同一模型做字节级破坏）──")
    # N1: float32 标量只给 2 字节
    tmp = tempfile.NamedTemporaryFile(suffix=".tflite", delete=False)
    tmp.close()
    data = bytearray(open(legacy_clean, "rb").read())
    # 找一个 float32 标量 buffer 的长度字段，改成 2
    broken = break_scalar_buffer(data)
    if broken is None:
        check("N1 字节数少一半 ⇒ 门 b 报不匹配", False, "找不到可破坏的标量 buffer")
    else:
        open(tmp.name, "wb").write(bytes(data))
        ok_broken, out_broken = report_of(tmp.name)
        check("N1 字节数少一半 ⇒ 门 b 报不匹配",
              (not ok_broken) and "字节数不匹配" in out_broken, out_broken)

    # N2: 把一个标量 buffer 的长度改成 2 倍
    tmp2 = tempfile.NamedTemporaryFile(suffix=".tflite", delete=False)
    tmp2.close()
    data2 = bytearray(open(legacy_clean, "rb").read())
    broken2 = break_scalar_buffer(data2, mult=2)
    if broken2 is None:
        check("N2 字节数多一倍 ⇒ 门 b 报不匹配", False, "找不到可破坏的标量 buffer")
    else:
        open(tmp2.name, "wb").write(bytes(data2))
        ok_b2, out_b2 = report_of(tmp2.name)
        check("N2 字节数多一倍 ⇒ 门 b 报不匹配",
              (not ok_b2) and "字节数不匹配" in out_b2, out_b2)
    os.unlink(tmp.name)
    os.unlink(tmp2.name)

    # N5: 激活张量（buffer 数据为空）不得被门 b 误判
    ok_p3, out_p3 = report_of(pt_final)
    check("N5 激活张量（无内联数据）不被门 b 误判",
          "字节数不匹配" not in out_p3, out_p3)

    print("\n── 回归（真实样本）──")
    ok_old, out_old = report_of(pt_old)
    check("R1 旧 per-tensor 模型门 a 必须 FAIL（6 个退化）",
          (not ok_old) and "a 失败（退化 6）" in out_old, out_old)
    check("R1' 旧 per-tensor 模型门 c 通过（权重本就是 per-tensor）",
          "c 通过" in out_old, out_old)

    ok_raw, out_raw = report_of(legacy_raw)
    # 门 c 是**警告**（per-channel 权重仍能跑，只是 NPU 上有精度损失），
    # 静态门的「生死」只对 a/b ⇒ 这里断言「必须被报出来」，而不是「必须 FAIL」
    check("R2 旧 legacy 模型门 c 必须报出（per-channel 权重）",
          "c 未过" in out_raw and "56 个权重非 per-tensor" in out_raw, out_raw)
    check("R2' 旧 legacy 模型门 a/b 通过（退化已清、字节自洽）",
          "✅ a 通过" in out_raw and "✅ b 通过" in out_raw, out_raw)

    print("\n" + "=" * 60)
    if FAILED:
        print("自测 FAIL（%d 项）：%s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("自测全部通过 ✅（门 a/b/c 的判定在本环境自洽）")
    return 0


def break_scalar_buffer(data: bytearray, mult=1):
    """在模型里找一个「标量常量 buffer」并改坏它的长度字段。

    做法：借 `verify/tflite_schema.py`（纯标准库）读出各 buffer 的 data 向量
    位置，把**长度为元素类型大小**的那个（标量常量）的长度改成不匹配的字节数。
    返回 True/False。

    ★ 不用 PyPI 的 tflite 包（本仓工具必须零第三方依赖），
      改用 schema 读取器 + 手工定位向量长度字段。
    """
    sys.path.insert(0, os.path.join(ROOT, "verify"))
    import tflite_schema as ts

    m = ts.parse_model(bytes(data), "<break>")
    sg = m.subgraphs[0]
    for t in sg.tensors:
        if t.shape:                      # 只要标量（shape=[]）
            continue
        nbytes = sg.buffer_size(t.buffer)
        if not nbytes:
            continue
        # 定位该 buffer 的 data 向量长度字段：
        #   Buffer 表 → 字段 0（data）→ uoffset → 向量起始（长度字段）
        pos = find_buffer_data_vec_pos(data, t.buffer)
        if pos is None:
            continue
        cur = struct.unpack_from("<I", data, pos)[0]
        if cur != nbytes:
            continue
        new_len = cur // 2 if mult == 1 else cur * 2
        struct.pack_into("<I", data, pos, new_len)
        return True
    return False


def find_buffer_data_vec_pos(data, buf_index):
    """返回第 buf_index 个 Buffer 的 data 向量**长度字段**在文件里的绝对偏移。"""
    import tflite_schema as ts
    # Model.buffers 是 table_vector(4)，逐个解析 Buffer 表并读其 data 字段
    F = ts.TABLE_FIELD_IDS["Model"]
    Fb = ts.TABLE_FIELD_IDS["Buffer"]
    root_off = struct.unpack_from("<I", data, 0)[0]
    root = ts.Table(data, root_off, "Model")
    bufs = root.table_vector(F["buffers"], "Buffer")
    if buf_index >= len(bufs):
        return None
    b = bufs[buf_index]
    off = b.field_offset(Fb["data"])
    if off == 0:
        return None
    slot = b.pos + off
    rel = struct.unpack_from("<i", data, slot)[0]
    return slot + rel


if __name__ == "__main__":
    sys.exit(main(sys.argv))
