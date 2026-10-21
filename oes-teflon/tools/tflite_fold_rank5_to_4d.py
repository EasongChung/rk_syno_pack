#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型手术：把 rank-5 块「折维」成等价的 4D 形式（报告 25 §4）。

## 为什么做这个

报告 11 定案：驱动契约 `unsigned dims[4]` 没有 rank 字段，delegate 写入形状的
**最后 4 维**，驱动一律算 `dims[1]*dims[2]*dims[3]` ⇒ **恒定丢掉 dims[0]**。

| rank | 形状 | 写入 dims | 驱动算出 | 真实字节 | 后果 |
|---|---|---|---|---|---|
| 4 | `[1,50,50,244]` | `[1,50,50,244]` | 610000 | 610000 | 无害（丢的是 batch=1） |
| 5 | `[1,122,2,50,50]` | `[122,2,50,50]` | 5000 | 610000 | **致命（丢的是 122）** |

本模型有 **16 个 rank-5 块**，形态完全一致：

    [1,C,H,W] --RESHAPE--> [1,2,C/2,H,W] --TRANSPOSE(交换1,2)--> [1,C/2,2,H,W] --RESHAPE--> [1,C,H,W]

## 手术：把最后两维折成一维

    旧:  [1,C,H,W] -> [1,2,C/2,H,W]   -> [1,C/2,2,H,W]   -> [1,C,H,W]     ← rank-5
    新:  [1,C,H,W] -> [1,2,C/2,H*W]   -> [1,C/2,2,H*W]   -> [1,C,H,W]     ← 全 4D

**等价性（索引算术，不靠“看起来对”）**

原张量按行主序展开：flat = `c·HW + p`（`c∈[0,C)`，`p∈[0,HW)`）。
`c = g·(C/2) + k`（`g∈{0,1}`，`k∈[0,C/2)`）⇒ flat = `g·(C/2)·HW + k·HW + p`。
这正是 `[1,2,C/2,HW]` 的 `(g,k,p)` 行主序下标 ⇒ **折维后的 reshape 与原来的 reshape 是同一次拆分**。
transpose(0,2,1,3) 后为 `(k,g,p)`；再 reshape 回 `[1,C,H,W]` 得通道号
`(k·2+g)·HW + p` ⇒ 通道 `k·2+g`。
**与原来的 5D 路径推出的通道映射完全相同**（`perm(c)=k·2+g`，双射已证）。

## 产出规模（预期值，写死后由自检核对）

| 项 | 之前 | 之后 |
|---|---|---|
| 张量数 | 337 | **337**（只改形状，不增删） |
| 算子数 | 201 | **201**（只改两个常量输入） |
| rank-5 张量 | 32 | **0** |
| 新增常量 | — | 3 个 reshape 形状 + 1 个 perm = 4 张量 / 4 buffer |

★ **不新增算子、不动权重、不动量化参数** ⇒ 风险面被压到最小。
★ 原模型只读，输出写到新路径。

用法：

    python3 tools/tflite_fold_rank5_to_4d.py <输入.tflite> <输出.tflite>
    python3 tools/tflite_fold_rank5_to_4d.py --selftest      # 只跑索引等价性证明
"""

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import flatbuffers  # noqa: E402
from tflite.Model import (Model, ModelStart, ModelAddVersion, ModelAddOperatorCodes,  # noqa: E402
                          ModelAddSubgraphs, ModelAddDescription, ModelAddBuffers,
                          ModelAddMetadataBuffer, ModelAddMetadata, ModelAddSignatureDefs,
                          ModelEnd)
from tflite.SubGraph import (SubGraphStart, SubGraphAddTensors, SubGraphAddInputs,  # noqa: E402
                             SubGraphAddOutputs, SubGraphAddOperators, SubGraphAddName,
                             SubGraphAddDebugMetadataIndex, SubGraphEnd)
from tflite.Tensor import (TensorStart, TensorAddShape, TensorAddType, TensorAddBuffer,  # noqa: E402
                           TensorAddName, TensorAddQuantization, TensorAddIsVariable,
                           TensorAddSparsity, TensorAddShapeSignature, TensorAddHasRank,
                           TensorAddVariantTensors, TensorEnd)
from tflite.Operator import (OperatorStart, OperatorAddOpcodeIndex, OperatorAddInputs,  # noqa: E402
                             OperatorAddOutputs, OperatorAddBuiltinOptionsType,
                             OperatorAddBuiltinOptions, OperatorAddCustomOptions,
                             OperatorAddCustomOptionsFormat, OperatorAddMutatingVariableInputs,
                             OperatorAddIntermediates, OperatorAddDebugMetadataIndex,
                             OperatorAddBuiltinOptions2Type, OperatorAddBuiltinOptions2,
                             OperatorAddLargeCustomOptionsOffset,
                             OperatorAddLargeCustomOptionsSize, OperatorEnd)
from tflite.OperatorCode import (OperatorCodeStart, OperatorCodeAddDeprecatedBuiltinCode,  # noqa: E402
                                 OperatorCodeAddCustomCode, OperatorCodeAddVersion,
                                 OperatorCodeAddBuiltinCode, OperatorCodeEnd)
from tflite.Buffer import (BufferStart, BufferAddData, BufferAddOffset, BufferAddSize,  # noqa: E402
                           BufferEnd)
from tflite.QuantizationParameters import (QuantizationParametersStart,  # noqa: E402
                                           QuantizationParametersAddMin,
                                           QuantizationParametersAddMax,
                                           QuantizationParametersAddScale,
                                           QuantizationParametersAddZeroPoint,
                                           QuantizationParametersAddDetailsType,
                                           QuantizationParametersAddDetails,
                                           QuantizationParametersAddQuantizedDimension,
                                           QuantizationParametersEnd)
from tflite.SignatureDef import (SignatureDefStart, SignatureDefAddInputs,  # noqa: E402
                                 SignatureDefAddOutputs, SignatureDefAddSignatureKey,
                                 SignatureDefAddSubgraphIndex, SignatureDefEnd)
from tflite.TensorMap import (TensorMapStart, TensorMapAddName, TensorMapAddTensorIndex,  # noqa: E402
                              TensorMapEnd)
from tflite.Metadata import MetadataStart, MetadataAddName, MetadataAddBuffer, MetadataEnd  # noqa: E402
from tflite.BuiltinOperator import BuiltinOperator as BO  # noqa: E402
from tflite.BuiltinOptions import BuiltinOptions as BOPT  # noqa: E402

FILE_IDENT = b"TFL3"
RESHAPE, TRANSPOSE = int(BO.RESHAPE), int(BO.TRANSPOSE)
OPT_NONE = int(BOPT.NONE)

# ------------------------------------------------------------------ 预期结构（写死，自检核对）
# 16 个块的三种尺寸：(C, H, W)
EXPECT_BLOCK_SHAPES = {244: (50, 50), 488: (25, 25), 976: (13, 13)}
EXPECT_N_BLOCKS = 16
EXPECT_RANK5_BEFORE = 32
EXPECT_TENSORS = 337
EXPECT_OPS = 201


# ================================================================== 索引等价性证明
def perm_channel(c, C):
    """原块对通道号的映射（5D 路径）。"""
    h = C // 2
    return (c % h) * 2 + (c // h)


def simulate_old(C, H, W):
    """按**原路径**算出：输出 flat 位置 -> 输入 flat 位置（不实际搬数据，只算下标）。"""
    HW = H * W
    # [1,C,H,W] --reshape--> [1,2,C/2,H,W]
    # 行主序 flat 下标 f = g*(C/2)*HW + k*HW + p   （与输入 f' = c*HW + p 相同，c=g*(C/2)+k）
    # transpose(0,2,1,3,4) -> [1,C/2,2,H,W]：flat = k*2*? ...直接用多维下标算更稳
    def idx_out_to_in(fo):
        # 输出 [1,C,H,W] 的多维下标
        co, p = divmod(fo, HW)
        hh, ww = divmod(p, W)
        # 逆推：co = k*2+g
        k, g = divmod(co, 2)
        # transpose 后 (k,g,p)；transpose 前 (g,k,p) -> reshape 前 c = g*(C/2)+k
        c = g * (C // 2) + k
        return c * HW + hh * W + ww
    return idx_out_to_in


def simulate_new(C, H, W):
    """按**新路径**（折维）算出同一映射。"""
    HW = H * W
    def idx_out_to_in(fo):
        co, p = divmod(fo, HW)                  # 输出 [1,C,H,W]
        # 新中间态 [1,C/2,2,HW]：flat = k*2*HW + g*HW + p  ⇒ co = k*2+g
        k, g = divmod(co, 2)
        # 折维前 [1,2,C/2,HW]：flat = g*(C/2)*HW + k*HW + p
        c = g * (C // 2) + k
        return c * HW + p
    return idx_out_to_in


def selftest():
    print("=" * 78)
    print("索引等价性证明：原 5D 路径 vs 新 4D 折维路径")
    print("=" * 78)
    bad = 0
    for C, (H, W) in sorted(EXPECT_BLOCK_SHAPES.items()):
        n = C * H * W
        a, b = simulate_old(C, H, W), simulate_new(C, H, W)
        diff = [fo for fo in range(n) if a(fo) != b(fo)]
        perm = sorted(perm_channel(c, C) for c in range(C)) == list(range(C))
        # 两条路径都必须是双射（否则会丢/重复元素）
        bij_a = sorted(a(fo) for fo in range(n)) == list(range(n))
        bij_b = sorted(b(fo) for fo in range(n)) == list(range(n))
        ok = (not diff) and perm and bij_a and bij_b
        print("  C=%-4d H=W=%-3d 元素 %-8d 两路径逐元素相同=%s 通道双射=%s 整体双射=%s/%s  → %s"
              % (C, H, H * W * C, not diff, perm, bij_a, bij_b, "✅" if ok else "❌"))
        if not ok:
            bad += 1
            print("     前 5 个不一致的 flat 下标: %s" % diff[:5])
    print()
    print("  结论: %s" % ("两条路径完全等价（%d 个尺寸全部通过）" % len(EXPECT_BLOCK_SHAPES)
                          if bad == 0 else "★ 存在不一致，手术方案必须推翻"))
    return 0 if bad == 0 else 1


# ================================================================== 读
def raw_vec(parent, fid):
    """直接取向量原始字节（不走 AsNumpy，本机没有 numpy）。

    vtableOffset = 4 + 2*field_id；`Vector(o)` 返回数据起始的绝对位置。
    """
    o = parent._tab.Offset(4 + 2 * fid)
    if o == 0:
        return None
    n = parent._tab.VectorLen(o)
    if n == 0:
        return b""
    a = parent._tab.Vector(o)
    return bytes(parent._tab.Bytes[a:a + n])


def vec(t, name):
    n = getattr(t, name + "Length")()
    return [getattr(t, name)(i) for i in range(n)]


def read_model(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[4:8] != FILE_IDENT:
        raise SystemExit("ERROR: 不是 TFLite（文件标识 %r）" % buf[4:8])
    m = Model.GetRootAs(buf, 0)
    if m.SubgraphsLength() != 1:
        raise SystemExit("ERROR: 只支持单 subgraph，实际 %d" % m.SubgraphsLength())
    sg = m.Subgraphs(0)

    codes = []
    for i in range(m.OperatorCodesLength()):
        c = m.OperatorCodes(i)
        codes.append({"dep": c.DeprecatedBuiltinCode(), "custom": c.CustomCode(), "ver": c.Version(),
                      "builtin": c.BuiltinCode()})

    buffers = []
    for i in range(m.BuffersLength()):
        b = m.Buffers(i)
        buffers.append({"data": raw_vec(b, 0),          # field 0 = data
                        "offset": b.Offset(), "size": b.Size()})

    tensors = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        if t.Sparsity() is not None or t.ShapeSignatureLength() or t.VariantTensorsLength():
            raise SystemExit("ERROR: 张量 %d 带 sparsity/shape_signature/variant 字段，本脚本不处理" % i)
        q = t.Quantization()
        qd = None
        if q is not None:
            if q.DetailsType() != 0:
                raise SystemExit("ERROR: 张量 %d 的 quantization.details_type != NONE，本脚本不处理" % i)
            qd = {"min": vec(q, "Min"), "max": vec(q, "Max"), "scale": vec(q, "Scale"),
                  "zp": vec(q, "ZeroPoint"), "qdim": q.QuantizedDimension()}
        tensors.append({"shape": vec(t, "Shape"), "type": t.Type(), "buffer": t.Buffer(),
                        "name": (t.Name().decode("utf-8") if t.Name() else None),
                        "quant": qd, "is_variable": bool(t.IsVariable()),
                        "has_rank": bool(t.HasRank())})

    ops = []
    for i in range(sg.OperatorsLength()):
        o = sg.Operators(i)
        if o.CustomOptionsLength() or o.MutatingVariableInputsLength() or o.IntermediatesLength():
            raise SystemExit("ERROR: 算子 %d 带 custom/mutating/intermediates 字段，本脚本不处理" % i)
        if o.BuiltinOptions2Type() != 0 or o.LargeCustomOptionsSize() != 0:
            raise SystemExit("ERROR: 算子 %d 带 BuiltinOptions2/LargeCustomOptions，本脚本不处理" % i)
        opts = None
        if o.BuiltinOptionsType() != OPT_NONE:
            opts = read_options(o)
        ops.append({"opcode_index": o.OpcodeIndex(), "inputs": vec(o, "Inputs"),
                    "outputs": vec(o, "Outputs"), "opts_type": o.BuiltinOptionsType(),
                    "opts": opts})

    subgraph = {"inputs": vec(sg, "Inputs"), "outputs": vec(sg, "Outputs"),
                "name": (sg.Name().decode("utf-8") if sg.Name() else None),
                "debug_md": sg.DebugMetadataIndex()}

    sigs = []
    for i in range(m.SignatureDefsLength()):
        s = m.SignatureDefs(i)
        sigs.append({"key": (s.SignatureKey().decode("utf-8") if s.SignatureKey() else None),
                     "subgraph_index": s.SubgraphIndex(),
                     "inputs": read_tensormaps(s, "Inputs"), "outputs": read_tensormaps(s, "Outputs")})

    metas = []
    for i in range(m.MetadataLength()):
        md = m.Metadata(i)
        metas.append({"name": (md.Name().decode("utf-8") if md.Name() else None), "buffer": md.Buffer()})

    mdl = {"version": m.Version(),
           "description": (m.Description().decode("utf-8") if m.Description() else None),
           "metadata_buffer": vec(m, "MetadataBuffer"),
           "codes": codes, "buffers": buffers, "tensors": tensors, "ops": ops,
           "subgraph": subgraph, "sigs": sigs, "metas": metas}
    return mdl


def read_tensormaps(parent, name):
    out = []
    for i in range(getattr(parent, name + "Length")()):
        tm = getattr(parent, name)(i)
        out.append({"name": (tm.Name().decode("utf-8") if tm.Name() else None),
                    "tensor_index": tm.TensorIndex()})
    return out


# ---- options 读取（union 的 field 4 得用 _tab 惯用法）----
# 字段清单**必须**与生成代码的读取器一一对应，漏一个就静默丢字段 ⇒ 用断言兜住。
OPT_FIELDS = {
    "PadOptions": [],                       # 空表
    "PadV2Options": [],
    "Conv2DOptions": ["Padding", "StrideW", "StrideH", "FusedActivationFunction",
                      "DilationWFactor", "DilationHFactor", "QuantizedBiasType"],
    "DepthwiseConv2DOptions": ["Padding", "StrideW", "StrideH", "DepthMultiplier",
                              "FusedActivationFunction", "DilationWFactor", "DilationHFactor"],
    "Pool2DOptions": ["Padding", "StrideW", "StrideH", "FilterWidth", "FilterHeight",
                      "FusedActivationFunction"],
    "ConcatenationOptions": ["Axis", "FusedActivationFunction"],
    "StridedSliceOptions": ["BeginMask", "EndMask", "EllipsisMask", "NewAxisMask",
                            "ShrinkAxisMask", "Offset"],
    "ReducerOptions": ["KeepDims"],
    "FullyConnectedOptions": ["FusedActivationFunction", "WeightsFormat", "KeepNumDims",
                             "AsymmetricQuantizeInputs", "QuantizedBiasType"],
}

_OPT_CLASSES = {}


def opt_class(type_name):
    if type_name not in _OPT_CLASSES:
        mod = __import__("tflite." + type_name, fromlist=[type_name])
        cls = getattr(mod, type_name)
        # ★ 自检：生成代码里的读取器必须被我上面列全，否则拒绝工作（防静默丢字段）
        readers = set()
        for x in dir(cls):
            if x.startswith("_") or "BufferHasIdentifier" in x:
                continue
            if x in ("Init",):
                continue
            readers.add(x)
        declared = set(OPT_FIELDS[type_name]) | {
            x for x in readers if x.endswith("IsNone") or x.endswith("Length") or x.endswith("AsNumpy")}
        missing = {r for r in readers if r not in declared
                   and not r.startswith(("Add", "Start", "End", "GetRootAs"))}
        if missing:
            raise SystemExit("ERROR: %s 的读取器 %s 不在 OPT_FIELDS 里 —— "
                             "拒绝在有未覆盖字段的情况下继续（会静默丢字段）" % (type_name, sorted(missing)))
        _OPT_CLASSES[type_name] = cls
    return _OPT_CLASSES[type_name]


def _int_names(mod):
    """模块里「名字 -> 整数」的映射。

    ★ 必须跳过下划线开头的键：Python 3.13+ 的模块会带 `__firstlineno__`（**是个 int**）
      与 `__static_attributes__` 等 dunder，它们会污染枚举名匹配 —— 本脚本第一版就栽在这，
      把 BuiltinOptions 类型 6 误判成了 `__firstlineno__`。
    """
    return {v: k for k, v in vars(mod).items()
            if not k.startswith("_") and isinstance(v, int) and not isinstance(v, bool)}


_OPT_NAMES = None


def opt_name(typ):
    global _OPT_NAMES
    if _OPT_NAMES is None:
        _OPT_NAMES = _int_names(BOPT)
    if typ not in _OPT_NAMES:
        raise SystemExit("ERROR: 未知 BuiltinOptions 类型 %d" % typ)
    return _OPT_NAMES[typ]


def read_options(op):
    name = opt_name(op.BuiltinOptionsType())
    cls = opt_class(name)
    o = op._tab.Offset(4 + 2 * 4)          # field 4 = builtin_options
    if o == 0:
        raise SystemExit("ERROR: 算子声明了 options 类型 %s 却没有 options 表" % name)
    x = op._tab.Indirect(o + op._tab.Pos)
    obj = cls()
    obj.Init(op._tab.Bytes, x)
    return {"__type__": name, **{f: getattr(obj, f)() for f in OPT_FIELDS[name]}}


# ================================================================== 写
def w_vec(b, kind, values):
    if kind == "i32":
        b.StartVector(4, len(values), 4)
        for v in reversed(values):
            b.PrependInt32(v)
    elif kind == "i64":
        b.StartVector(8, len(values), 8)
        for v in reversed(values):
            b.PrependInt64(v)
    elif kind == "i8":
        b.StartVector(1, len(values), 1)
        for v in reversed(values):
            b.PrependInt8(v)
    elif kind == "u8":
        b.StartVector(1, len(values), 1)
        for v in reversed(values):
            b.PrependUint8(v)
    elif kind == "f32":
        b.StartVector(4, len(values), 4)
        for v in reversed(values):
            b.PrependFloat32(v)
    else:
        raise SystemExit("ERROR: 未知向量类型 %s" % kind)
    return b.EndVector()


def w_vec_tables(b, offsets):
    """向量 of 表：这个 flatbuffers 版本没有 CreateVectorOfTables，手写。"""
    b.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        b.PrependUOffsetTRelative(off)
    return b.EndVector()


def build_options(b, opts):
    name = opts["__type__"]
    mod = __import__("tflite." + name, fromlist=[name])
    start = getattr(mod, name + "Start")
    end = getattr(mod, name + "End")
    start(b)
    for f in OPT_FIELDS[name]:
        getattr(mod, name + "Add" + f)(b, opts[f])
    return end(b)


def build_quant(b, q):
    if q is None:
        return None
    mn = w_vec(b, "f32", q["min"]) if q["min"] else None
    mx = w_vec(b, "f32", q["max"]) if q["max"] else None
    sc = w_vec(b, "f32", q["scale"]) if q["scale"] else None
    zp = w_vec(b, "i64", q["zp"]) if q["zp"] else None
    QuantizationParametersStart(b)
    if mn is not None:
        QuantizationParametersAddMin(b, mn)
    if mx is not None:
        QuantizationParametersAddMax(b, mx)
    if sc is not None:
        QuantizationParametersAddScale(b, sc)
    if zp is not None:
        QuantizationParametersAddZeroPoint(b, zp)
    QuantizationParametersAddDetailsType(b, 0)
    QuantizationParametersAddQuantizedDimension(b, q["qdim"])
    return QuantizationParametersEnd(b)


def build_buffer(b, bf):
    data = None
    if bf["data"] is not None:
        data = w_vec(b, "u8", bf["data"])
    BufferStart(b)
    if data is not None:
        BufferAddData(b, data)
    if bf["offset"]:
        BufferAddOffset(b, bf["offset"])
    if bf["size"]:
        BufferAddSize(b, bf["size"])
    return BufferEnd(b)


def build_tensor(b, t):
    shape = w_vec(b, "i32", t["shape"])
    name = b.CreateString(t["name"]) if t["name"] else None
    quant = build_quant(b, t["quant"])
    TensorStart(b)
    TensorAddShape(b, shape)
    TensorAddType(b, t["type"])
    TensorAddBuffer(b, t["buffer"])
    if name is not None:
        TensorAddName(b, name)
    if quant is not None:
        TensorAddQuantization(b, quant)
    TensorAddIsVariable(b, t["is_variable"])
    if t.get("has_rank"):
        TensorAddHasRank(b, t["has_rank"])
    return TensorEnd(b)


def build_op(b, o):
    opts_off = build_options(b, o["opts"]) if o["opts"] is not None else None
    ins = w_vec(b, "i32", o["inputs"])
    outs = w_vec(b, "i32", o["outputs"])
    OperatorStart(b)
    OperatorAddOpcodeIndex(b, o["opcode_index"])
    OperatorAddInputs(b, ins)
    OperatorAddOutputs(b, outs)
    OperatorAddBuiltinOptionsType(b, o["opts_type"])
    if opts_off is not None:
        OperatorAddBuiltinOptions(b, opts_off)
    OperatorAddCustomOptionsFormat(b, 0)
    return OperatorEnd(b)


def build_model(mdl):
    b = flatbuffers.Builder(4 << 20)

    # ---- 叶子 ----
    buf_offs = [build_buffer(b, x) for x in mdl["buffers"]]
    tens_offs = [build_tensor(b, x) for x in mdl["tensors"]]
    op_offs = [build_op(b, x) for x in mdl["ops"]]

    code_offs = []
    for c in mdl["codes"]:
        cc = b.CreateString(c["custom"]) if c["custom"] else None
        OperatorCodeStart(b)
        OperatorCodeAddDeprecatedBuiltinCode(b, c["dep"])
        if cc is not None:
            OperatorCodeAddCustomCode(b, cc)
        OperatorCodeAddVersion(b, c["ver"])
        OperatorCodeAddBuiltinCode(b, c["builtin"])
        code_offs.append(OperatorCodeEnd(b))

    sig_offs = []
    for s in mdl["sigs"]:
        def maps(key):
            offs = []
            for tm in s[key]:
                nm = b.CreateString(tm["name"]) if tm["name"] else None
                TensorMapStart(b)
                if nm is not None:
                    TensorMapAddName(b, nm)
                TensorMapAddTensorIndex(b, tm["tensor_index"])
                offs.append(TensorMapEnd(b))
            return w_vec_tables(b, offs)
        ins = maps("inputs")
        outs = maps("outputs")
        k = b.CreateString(s["key"]) if s["key"] else None
        SignatureDefStart(b)
        SignatureDefAddInputs(b, ins)
        SignatureDefAddOutputs(b, outs)
        if k is not None:
            SignatureDefAddSignatureKey(b, k)
        SignatureDefAddSubgraphIndex(b, s["subgraph_index"])
        sig_offs.append(SignatureDefEnd(b))

    meta_offs = []
    for md in mdl["metas"]:
        nm = b.CreateString(md["name"]) if md["name"] else None
        MetadataStart(b)
        if nm is not None:
            MetadataAddName(b, nm)
        MetadataAddBuffer(b, md["buffer"])
        meta_offs.append(MetadataEnd(b))

    # ---- 容器 ----
    # ★ 所有向量/字符串必须在 Start 之前建好：Builder 里处于 nested 状态时不许再建向量。
    sg = mdl["subgraph"]
    v_tensors = w_vec_tables(b, tens_offs)
    v_inputs = w_vec(b, "i32", sg["inputs"])
    v_outputs = w_vec(b, "i32", sg["outputs"])
    v_ops = w_vec_tables(b, op_offs)
    sg_name = b.CreateString(sg["name"]) if sg["name"] else None

    SubGraphStart(b)
    SubGraphAddTensors(b, v_tensors)
    SubGraphAddInputs(b, v_inputs)
    SubGraphAddOutputs(b, v_outputs)
    SubGraphAddOperators(b, v_ops)
    if sg_name is not None:
        SubGraphAddName(b, sg_name)
    if sg["debug_md"]:
        SubGraphAddDebugMetadataIndex(b, sg["debug_md"])
    sg_off = SubGraphEnd(b)

    v_codes = w_vec_tables(b, code_offs)
    v_subgraphs = w_vec_tables(b, [sg_off])
    v_buffers = w_vec_tables(b, buf_offs)
    v_mdbuf = w_vec(b, "i32", mdl["metadata_buffer"]) if mdl["metadata_buffer"] else None
    v_meta = w_vec_tables(b, meta_offs) if meta_offs else None
    v_sigs = w_vec_tables(b, sig_offs) if sig_offs else None
    desc = b.CreateString(mdl["description"]) if mdl["description"] else None

    ModelStart(b)
    ModelAddVersion(b, mdl["version"])
    ModelAddOperatorCodes(b, v_codes)
    ModelAddSubgraphs(b, v_subgraphs)
    if desc is not None:
        ModelAddDescription(b, desc)
    ModelAddBuffers(b, v_buffers)
    if v_mdbuf is not None:
        ModelAddMetadataBuffer(b, v_mdbuf)
    if v_meta is not None:
        ModelAddMetadata(b, v_meta)
    if v_sigs is not None:
        ModelAddSignatureDefs(b, v_sigs)
    root = ModelEnd(b)
    b.Finish(root, FILE_IDENT)
    return bytes(b.Output())


# ================================================================== 手术
def find_blocks(mdl):
    """找出全部 rank-5 块：RESHAPE -> TRANSPOSE -> RESHAPE。"""
    prod, cons = {}, {}
    for i, o in enumerate(mdl["ops"]):
        for t in o["outputs"]:
            prod.setdefault(t, []).append(i)
        for t in o["inputs"]:
            cons.setdefault(t, []).append(i)

    def is_reshape(i):
        c = mdl["codes"][mdl["ops"][i]["opcode_index"]]["builtin"]
        return c == RESHAPE

    def is_transpose(i):
        c = mdl["codes"][mdl["ops"][i]["opcode_index"]]["builtin"]
        return c == TRANSPOSE

    rank5 = {i for i, t in enumerate(mdl["tensors"]) if len(t["shape"]) == 5}
    blocks = []
    for i, o in enumerate(mdl["ops"]):
        if not is_reshape(i):
            continue
        t5a = o["outputs"][0]
        if t5a not in rank5:
            continue
        nxt = cons.get(t5a, [])
        if len(nxt) != 1 or not is_transpose(nxt[0]):
            continue
        j = nxt[0]
        t5b = mdl["ops"][j]["outputs"][0]
        if t5b not in rank5:
            continue
        nxt2 = cons.get(t5b, [])
        if len(nxt2) != 1 or not is_reshape(nxt2[0]):
            continue
        k = nxt2[0]
        blocks.append({"r1": i, "tr": j, "r2": k, "src": o["inputs"][0],
                       "t5a": t5a, "t5b": t5b, "dst": mdl["ops"][k]["outputs"][0]})
    return blocks, rank5


def const_i32(mdl, ti):
    """把常量张量读成 int32 列表；不是常量就报错（不猜）。"""
    t = mdl["tensors"][ti]
    bf = mdl["buffers"][t["buffer"]]
    if bf["data"] is None:
        raise SystemExit("ERROR: 张量 %d 不是常量（无数据），但它被当作常量使用" % ti)
    n = len(bf["data"]) // 4
    return list(struct.unpack_from("<%di" % n, bf["data"], 0))


def transform(mdl):
    blocks, rank5 = find_blocks(mdl)

    print("--- 结构校验 ---")
    if len(blocks) != EXPECT_N_BLOCKS:
        raise SystemExit("ERROR: 找到 %d 个块，期望 %d —— 拒绝手术" % (len(blocks), EXPECT_N_BLOCKS))
    if len(rank5) != EXPECT_RANK5_BEFORE:
        raise SystemExit("ERROR: rank-5 张量 %d 个，期望 %d" % (len(rank5), EXPECT_RANK5_BEFORE))
    if len(mdl["tensors"]) != EXPECT_TENSORS or len(mdl["ops"]) != EXPECT_OPS:
        raise SystemExit("ERROR: 张量/算子数不符（%d/%d，期望 %d/%d）"
                         % (len(mdl["tensors"]), len(mdl["ops"]), EXPECT_TENSORS, EXPECT_OPS))
    print("  块数 %d ✓   rank-5 张量 %d ✓   张量 %d / 算子 %d ✓"
          % (len(blocks), len(rank5), len(mdl["tensors"]), len(mdl["ops"])))

    # 每个块的形状与 perm 必须与预期一致（否则拒绝）
    by_C = {}
    for b in blocks:
        src = mdl["tensors"][b["src"]]["shape"]
        if len(src) != 4:
            raise SystemExit("ERROR: 块源张量 %s 不是 4D" % src)
        C = src[1]
        if C not in EXPECT_BLOCK_SHAPES:
            raise SystemExit("ERROR: 意外通道数 C=%d" % C)
        H, W = src[2], src[3]
        if (H, W) != EXPECT_BLOCK_SHAPES[C]:
            raise SystemExit("ERROR: C=%d 的空间尺寸 %s 与预期 %s 不符"
                             % (C, (H, W), EXPECT_BLOCK_SHAPES[C]))
        h = C // 2
        if mdl["tensors"][b["t5a"]]["shape"] != [1, 2, h, H, W]:
            raise SystemExit("ERROR: 块中间张量形状意外 %s" % mdl["tensors"][b["t5a"]]["shape"])
        if mdl["tensors"][b["t5b"]]["shape"] != [1, h, 2, H, W]:
            raise SystemExit("ERROR: 块中间张量形状意外 %s" % mdl["tensors"][b["t5b"]]["shape"])
        if mdl["tensors"][b["dst"]]["shape"] != src:
            raise SystemExit("ERROR: 块目标形状 != 源形状（不是纯置换）")
        perm_t = mdl["ops"][b["tr"]]["inputs"][1]
        pv = const_i32(mdl, perm_t)
        if pv != [0, 2, 1, 3, 4]:
            raise SystemExit("ERROR: 块 %d 的 transpose perm = %s，期望 [0,2,1,3,4]（不是交换 1,2 维）"
                             % (b["r1"], pv))
        s1 = const_i32(mdl, mdl["ops"][b["r1"]]["inputs"][1])
        if s1 != [1, 2, h, H, W]:
            raise SystemExit("ERROR: 块 %d 的 reshape1 形状常量 = %s，期望 %s"
                             % (b["r1"], s1, [1, 2, h, H, W]))
        s2 = const_i32(mdl, mdl["ops"][b["r2"]]["inputs"][1])
        if s2 != src:
            raise SystemExit("ERROR: 块 %d 的 reshape2 形状常量 = %s，期望 %s" % (b["r1"], s2, src))
        by_C.setdefault(C, []).append(b)
    print("  尺寸分组: %s" % {k: len(v) for k, v in sorted(by_C.items())})
    print("  每个块的 perm 常量均 = [0,2,1,3,4]（交换 1、2 维）✓")
    print("  每个块的 reshape 形状常量与张量声明一致 ✓")
    return blocks, rank5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.src or not args.dst:
        raise SystemExit("用法: %s <输入.tflite> <输出.tflite>  |  --selftest" % sys.argv[0])

    print("=" * 78)
    print("rank-5 → 4D 折维手术")
    print("=" * 78)

    rc = selftest()
    if rc != 0:
        raise SystemExit("ERROR: 索引等价性自检未通过，拒绝手术")
    print()

    mdl = read_model(args.src)
    blocks, rank5 = transform(mdl)

    # 组装手术：改形状 + 换常量
    print()
    print("--- 手术 ---")
    new_tensors = [dict(t) for t in mdl["tensors"]]
    new_ops = [dict(o) for o in mdl["ops"]]

    # 1) 新增常量：每个 C 一个 reshape 形状 [1,2,C/2,H*W]，全局一个 perm [0,2,1,3]
    def add_const(values, name):
        data = struct.pack("<%di" % len(values), *values)
        bi = len(mdl["buffers"]) + len(ADDED["buffers"])
        ADDED["buffers"].append({"data": data, "offset": 0, "size": 0})
        ti = len(new_tensors) + len(ADDED["tensors"])
        ADDED["tensors"].append({"shape": [len(values)], "type": 2, "buffer": bi,
                                 "name": name, "quant": None, "is_variable": False,
                                 "has_rank": True})
        return ti

    ADDED = {"buffers": [], "tensors": []}
    perm_ti = add_const([0, 2, 1, 3], "och.fold.perm")
    print("  新 perm 常量: [0,2,1,3] -> tensor %d" % perm_ti)

    shape_ti = {}
    for C, (H, W) in sorted(EXPECT_BLOCK_SHAPES.items()):
        shape_ti[C] = add_const([1, 2, C // 2, H * W], "och.fold.shape.C%d" % C)
        print("  新 shape 常量 C=%-4d: [1,2,%d,%d] -> tensor %d" % (C, C // 2, H * W, shape_ti[C]))

    for b in blocks:
        src = mdl["tensors"][b["src"]]["shape"]
        C, H, W = src[1], src[2], src[3]
        new_tensors[b["t5a"]]["shape"] = [1, 2, C // 2, H * W]
        new_tensors[b["t5b"]]["shape"] = [1, C // 2, 2, H * W]
        new_ops[b["r1"]]["inputs"] = [b["src"], shape_ti[C]]
        new_ops[b["tr"]]["inputs"] = [b["t5a"], perm_ti]
        # 名称里标注，便于设备侧对照
        new_tensors[b["t5a"]]["name"] = (new_tensors[b["t5a"]]["name"] or "") + "/och-fold4d"
        new_tensors[b["t5b"]]["name"] = (new_tensors[b["t5b"]]["name"] or "") + "/och-fold4d"

    mdl["tensors"] = new_tensors + ADDED["tensors"]
    mdl["buffers"] = mdl["buffers"] + ADDED["buffers"]
    mdl["ops"] = new_ops
    print("  改了 %d 个块的张量形状与常量引用" % len(blocks))
    print("  张量 %d -> %d   buffer %d -> %d   算子 %d（不变）"
          % (len(mdl["tensors"]) - len(ADDED["tensors"]), len(mdl["tensors"]),
             len(mdl["buffers"]) - len(ADDED["buffers"]), len(mdl["buffers"]), len(mdl["ops"])))

    if args.dry_run:
        print("\n--dry-run：不写文件")
        return 0

    blob = build_model(mdl)
    with open(args.dst, "wb") as f:
        f.write(blob)
    print("\n  写出 %s（%d 字节，原 %d 字节）" % (args.dst, len(blob), os.path.getsize(args.src)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
