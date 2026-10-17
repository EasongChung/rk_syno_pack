#!/usr/bin/env python3
"""按张量名 + 公式，把 legacy 的正确量化尺度移植到 MLIR per-tensor 模型。

## 背景
MLIR per-tensor 模型（`concept_int8_pertensor.tflite`）有 6 个退化张量
（scale=FLT_MAX/255=1.334441e+36），导致输出坍缩成常量：
  t127 INT32 [122] bias  (pseudo_qconst106)
  t131 INT32 [24]  bias  (pseudo_qconst110)
  t133 INT8  []    空形状常量 (pseudo_qconst112) <- PADV2 的 constant_values
  t139 INT8 [1,198,198,24]  (relu/add/conv)
  t140 INT8 [1,200,200,24]  (pad_1/PadV2)
  t141 INT8 [1,99,99,24]    (max_pool2d)

根因：TF MinMax 对**从未被观测**的张量保留哨兵 {min=0,max=FLT_MAX}
     -> scale=(FLT_MAX-0)/255 -> requant 乘数≈0 -> 输出=zp。
（已证伪的假设：删掉 t133 能治愈整条链 —— 实测 6->6 不变，
 见 CI run 39009507872。）

## 三条修法依据（均已用数据验证，不是猜）
1. **t139/t140/t141 按张量名移植**（本地 `check_name_match.py` 实测）：
   名字+类型+形状三者与 legacy 完全匹配，源值 = 0.4730178415775299 / zp=-128。
   （legacy 中这三个是同一个统一量化组，共享同一 scale，移植后仍统一。）
2. **bias 尺度按公式重算**：`bias_scale = in_scale * w_scale`。
   本地 `check_bias_formula.py` 实测：legacy 57/57、per-tensor 57/57 全部成立
   （rel_err ~1e-8）。t127/t131 的退化正是被退化输入"连坐"而来。
3. **t133 是空形状 + 被 PADV2->PAD 改写孤立**：给它一个合法形状与中性尺度，
   使其不再触发"空形状/退化"两条门。它是孤立张量，内核不读，不影响推理。

## 顺序很关键
先移植激活（1）-> 再重算 bias（2）—— 因为公式依赖移植后的 in_scale。

## 安全
- 复用已验证的 Pack 模式：`Pack(builder)` + `Finish(root, b'TFL3')` + 回读 revalidate
- 落盘前重新读回，断言 6 条门里的静态项（退化=0 / 无空形状 / conv 权重 per-tensor）
- `--dry-run` 也走完整 Pack + revalidate

用法：
  python transplant_scales.py --degen <MLIR per-tensor> --source <legacy> \\
      --out out/concept_int8_art13.tflite [--dry-run]
"""
import argparse
import os
import sys

FILE_IDENTIFIER = b'TFL3'
CONV, DWCONV, FC = 3, 4, 9
CODE_PAD, CODE_PADV2 = 34, 60
NEUTRAL_SCALE = 1.0
NEUTRAL_ZP = 0


def _attr(obj, snake, camel):
    return camel if hasattr(obj, camel) else snake


def _len(v):
    """安全长度：绝不能用 `v or []`。
    对象 API 的向量是 numpy 数组，长度 >1 时 `arr or []` 会抛
    `ValueError: truth value of an array ... is ambiguous`
    （CI run 39026333343 实测：t127/131/133 形状长度<=1 通过，
     t139 长度 4 崩溃）。"""
    return 0 if v is None else len(v)


def _shape_of(t):
    return [int(t.shape[j]) for j in range(_len(t.shape))]


def _first(seq):
    return seq[0] if seq is not None and len(seq) > 0 else None


def read_index(path):
    """用只读 API 建索引：name -> [(idx, type, shape, scale, zp)]。"""
    import tflite as T
    mo = T.Model.GetRootAsModel(open(path, 'rb').read(), 0)
    g = mo.Subgraphs(0)
    out = {}
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        q = t.Quantization()
        ns = 0 if q is None else q.ScaleLength()
        name = t.Name()
        name = bytes(name).decode('utf-8', 'replace') if name else ''
        out.setdefault(name, []).append({
            'idx': i, 'type': t.Type(),
            'shape': [t.Shape(j) for j in range(t.ShapeLength())],
            'ns': ns,
            'scale': q.Scale(0) if ns else None,
            'zp': (q.ZeroPoint(0) if q.ZeroPointLength() else 0) if ns else None,
        })
    return out, g


def transplant(degen, source, out_path, dry_run=False):
    from tensorflow.lite.python import schema_py_generated as fb
    import numpy as np

    raw = open(degen, 'rb').read()
    model = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(raw, 0))
    sub = model.subgraphs[0]
    codes = getattr(model, 'operatorCodes')
    tensors = sub.tensors
    ops = sub.operators

    src_by_name, _ = read_index(source)

    n_quant_before = sum(
        1 for t in tensors
        if _len(getattr(getattr(t, 'quantization', None), 'scale', None)) > 0)
    print('quantized tensors BEFORE transplant: %d / %d'
          % (n_quant_before, len(tensors)), flush=True)

    def eff(oc):
        bc = oc.builtinCode
        return bc if bc < 127 else oc.deprecatedBuiltinCode

    def qt_of(t):
        q = getattr(t, 'quantization', None)
        if q is None:
            return None
        return q

    def set_q(idx, scale, zp):
        """写 scale + zero_point。

        ⚠️ TF 对象树的字段名是**驼峰**（`zeroPoint`），不是 snake_case
        （`zero_point`）。只写 snake 版不会报错，但 Pack 时被静默忽略
        -> 内核以 "non-null scale but null zero_point" 拒绝
        （CI run 39029095469 实测：t127/131/133/139/140/141 全中）。
        这与早前 `operator_codes` -> `operatorCodes` 是同一类错。
        故两种拼写都设，不依赖对任一版本字段名的记忆。
        """
        t = tensors[idx]
        nq = fb.QuantizationParametersT()
        nq.scale = np.array([float(scale)], np.float32)
        for attr in ('zero_point', 'zeroPoint'):
            if hasattr(nq, attr):
                setattr(nq, attr, np.array([int(zp)], np.int64))
        nq.quantized_dimension = 0
        if hasattr(nq, 'quantizedDimension'):
            nq.quantizedDimension = 0
        t.quantization = nq

    # ---- 0. 先把 PADV2 改成 PAD（删掉退化的 constant_values 引用）----
    pad_idx = None
    for i, oc in enumerate(codes):
        if eff(oc) == CODE_PAD:
            pad_idx = i
            break
    ref_bt = ref_bo = None
    for op in ops:
        if eff(codes[op.opcodeIndex]) == CODE_PAD:
            ref_bt = op.builtinOptionsType
            ref_bo = op.builtinOptions
            break
    n_pad = 0
    if pad_idx is not None and ref_bt is not None:
        for op in ops:
            if eff(codes[op.opcodeIndex]) != CODE_PADV2:
                continue
            op.inputs = op.inputs[:2]
            op.opcodeIndex = pad_idx
            op.builtinOptionsType = ref_bt
            op.builtinOptions = ref_bo
            n_pad += 1
    print('PADV2->PAD: %d op(s)' % n_pad, flush=True)

    # ---- 1. 按名字移植激活尺度 ----
    fixed_by_name = {}
    for i, t in enumerate(tensors):
        q = qt_of(t)
        if q is None or _len(getattr(q, 'scale', None)) == 0:
            continue
        s = float(q.scale[0])
        if not (abs(s) > 1e10 or s <= 0):
            continue
        name = t.name.decode('utf-8', 'replace') if t.name else ''
        cands = src_by_name.get(name, [])
        same = [c for c in cands if c['type'] == t.type
                and c['shape'] == _shape_of(t)
                and c['scale'] is not None
                and not (abs(c['scale']) > 1e10 or c['scale'] <= 0)]
        if len(same) == 1:
            set_q(i, same[0]['scale'], same[0]['zp'])
            fixed_by_name[i] = (s, same[0]['scale'], same[0]['zp'])
            print('  t%-4d by NAME  %-11s %.6g -> %.6g zp %s -> %s'
                  % (i, name[:28], s, same[0]['scale'],
                     _first(getattr(q, 'zeroPoint', None)), same[0]['zp']),
                  flush=True)
        else:
            print('  t%-4d NO clean name match (cands=%d, same=%d) name=%r'
                  % (i, len(cands), len(same), name[:40]), flush=True)

    # ---- 2. 重算全部 conv bias 尺度（依据已验证公式）----
    def s_of(idx):
        q = qt_of(tensors[idx])
        if q is None or _len(getattr(q, 'scale', None)) == 0:
            return None
        return float(q.scale[0])

    n_bias = 0
    for op in ops:
        c = eff(codes[op.opcodeIndex])
        if c not in (CONV, DWCONV, FC):
            continue
        if len(op.inputs) < 3:
            continue
        s_in, s_w = s_of(op.inputs[0]), s_of(op.inputs[1])
        if s_in is None or s_w is None:
            continue
        pred = s_in * s_w
        old = s_of(op.inputs[2])
        if old is None:
            continue
        if abs(old - pred) / max(abs(pred), 1e-30) > 1e-3:
            set_q(op.inputs[2], pred, 0)
            n_bias += 1
            print('  bias t%-4d RECOMPUTED %.6g -> %.6g'
                  % (op.inputs[2], old, pred), flush=True)
    print('bias recomputed: %d' % n_bias, flush=True)

    # ---- 3. 修空形状/仍退化的孤立张量（t133 类）----
    n_orphan = 0
    for i, t in enumerate(tensors):
        shape = _shape_of(t)
        q = qt_of(t)
        # 「没有量化参数」与「量化参数退化」是两件事，**绝不能混**：
        # INT32 形状/axes/paddings 常量按 TFLite 惯例就没有量化参数，
        # 给它们硬塞 scale/zp 会损坏模型（本工具第一版就是这么错的：
        # 337/337 全量化 -> 产物不可用）。只有**已有 scale 且退化**
        # 的才需要修。
        has_sc = q is not None and _len(getattr(q, 'scale', None)) > 0
        degen = has_sc and (abs(float(q.scale[0])) > 1e10
                            or float(q.scale[0]) <= 0)
        if len(shape) == 0:
            t.shape = [1]
            if degen:
                set_q(i, NEUTRAL_SCALE, NEUTRAL_ZP)
            n_orphan += 1
            print('  t%-4d empty shape -> [1] (%s)'
                  % (i, 'scale neutralized' if degen else 'no scale, left alone'),
                  flush=True)
        elif degen and i not in fixed_by_name:
            set_q(i, NEUTRAL_SCALE, NEUTRAL_ZP)
            n_orphan += 1
            print('  t%-4d STILL degenerate -> neutral scale' % i, flush=True)
    print('orphan/degenerate neutralized: %d' % n_orphan, flush=True)

    # ---- 4. Pack + 回读校验 ----
    import flatbuffers
    builder = flatbuffers.Builder(1024 * 1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=FILE_IDENTIFIER)
    out = bytes(builder.Output())
    if out[4:8] != FILE_IDENTIFIER:
        raise RuntimeError('file_identifier lost: %r' % bytes(out[4:8]))
    _revalidate(out, expect_quant=n_quant_before)
    print('PACK OK: %d bytes' % len(out), flush=True)
    if dry_run:
        print('dry-run: not writing', flush=True)
        return 0
    with open(out_path, 'wb') as f:
        f.write(out)
    print('WROTE %s' % out_path, flush=True)
    return 0


def _revalidate(buf, expect_quant=None):
    """按 och 六条门的静态项逐条断言（a/b/c）+ 量化张量数不变。"""
    import tflite as T
    if bytes(buf[4:8]) != FILE_IDENTIFIER:
        raise RuntimeError('file_identifier bad')
    mo = T.Model.GetRootAsModel(buf, 0)
    g = mo.Subgraphs(0)
    codes = {}
    for i in range(mo.OperatorCodesLength()):
        oc = mo.OperatorCodes(i)
        bc = oc.BuiltinCode()
        codes[i] = bc if bc < 127 else oc.DeprecatedBuiltinCode()

    degen, empty, pc = [], [], []
    n_quant = 0
    bad_zp = []
    for t in range(g.TensorsLength()):
        tt = g.Tensors(t)
        q = tt.Quantization()
        ns = 0 if q is None else q.ScaleLength()
        nz = 0 if q is None else q.ZeroPointLength()
        if ns:
            n_quant += 1
            s = q.Scale(0)
            if abs(s) > 1e10 or s <= 0:
                degen.append((t, s))
            elif nz == 0:
                # 内核的真实判据（TFLite 的报错原文）：
                # "Quantization parameters has non-null scale but null
                #  zero_point" -> 直接拒载。
                # 只查 ScaleLength 会漏掉它（control_allq 的教训同源）。
                bad_zp.append(t)
        if tt.ShapeLength() == 0:
            empty.append(t)
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        if codes.get(op.OpcodeIndex()) not in (CONV, DWCONV, FC):
            continue
        if op.InputsLength() < 2:
            continue
        q = g.Tensors(op.Inputs(1)).Quantization()
        ns = 0 if q is None else q.ScaleLength()
        nz = 0 if q is None else q.ZeroPointLength()
        if ns != 1 or nz != 1:
            pc.append((i, ns, nz))
    print('REVALIDATE: tensors=%d  quantized=%d  a)degenerate=%d  '
          'b)empty_shape=%d  c)non-per-tensor-weights=%d  '
          'd)scale-without-zero_point=%d'
          % (g.TensorsLength(), n_quant, len(degen), len(empty), len(pc),
             len(bad_zp)), flush=True)
    if degen:
        raise RuntimeError('gate a FAIL: degenerate %s' % degen[:8])
    if empty:
        raise RuntimeError('gate b FAIL: empty shape %s' % empty[:8])
    if pc:
        raise RuntimeError('gate c FAIL: per-channel weights %s' % pc[:8])
    if bad_zp:
        raise RuntimeError(
            'kernel rejects "non-null scale but null zero_point": %s'
            % bad_zp[:8])
    if expect_quant is not None and n_quant != expect_quant:
        raise RuntimeError(
            'quantized tensor count CHANGED: %d -> %d. The transplant must '
            'only MODIFY scales, never ADD quantization params to tensors '
            'that had none (INT32 shape/axes constants).'
            % (expect_quant, n_quant))
    print('GATES a,b,c PASS (quantized=%d, unchanged=%s)'
          % (n_quant, expect_quant is not None), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--degen', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    print('=== degen=%s (%d B)' % (a.degen, os.path.getsize(a.degen)),
          flush=True)
    print('=== source=%s (%d B)' % (a.source, os.path.getsize(a.source)),
          flush=True)
    return transplant(a.degen, a.source, a.out, a.dry_run)


if __name__ == '__main__':
    sys.exit(main())