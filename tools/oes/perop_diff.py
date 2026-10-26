#!/usr/bin/env python3
"""逐张量对照两份 TFLite 模型，定位第一个失配点。

## 为什么需要它

int8 产物在**输出端**与浮点参照差 2.7 倍（报告 65 §3）。只比输出无法回答
关键问题：这是**量化本身的精度损失**，还是**转换链路的缺陷**？
两者处置完全不同 —— 前者要改量化方案（或放弃 per-tensor），后者是可修的 bug。

本工具把两份模型都跑同一张图，取出**全部**张量
(tf.lite.Interpreter 的 experimental_preserve_all_tensors=True)，
按**张量名**配对，按量化模型的算子顺序逐层计算 cos / 相对误差，
报出**第一个**跌破阈值的张量。

## 只比「激活」，不比常量（2026-10-26 修正）

量化会重写常量张量：浮点图里的 @c arith.constant20 在 int8 图里可能被折叠、
改名、或换成 INT32 的 scale/bias tensors。若把常量算进配对率分母，
会得到 48% 的**假告警**（run 40269559679 实测 art13）。

⇒ 分母与比对集合一律限定为**算子输出张量**（= 激活）。

## 为什么按名字配对可行

两份模型由同一条 onnx2tf 链路生成，激活张量命名一致
(Const_N / tf.nn.xxx_1/xxx 之类)。工具仍会打印配对率，
低于 --min-match 时以退出码 3 告警，避免「配对失败被当成模型正常」。

## 用法

  python perop_diff.py \
    --ref   out/baseline_float_concept_split_padsafe_float32.tflite \
    --quant out/concept_int8_art13.tflite \
    --image eval/oes-eval-0000.jpg \
    --json  out/perop_art13.json

  # 只列算子与激活名（不推理，不需要图片）
  python perop_diff.py --ref M.tflite --list-tensors

## 退出码

  0  正常（无论是否发现失配）
  2  参数/文件问题
  3  配对率过低（结论不可信）
"""

import argparse
import json
import os
import sys

import numpy as np


# TFLite builtin opcode -> 名字兜底表。
# 取自本仓 verify/tflite_schema.py 的 BUILTIN_OPERATOR_NAMES（v2.13.0 完整枚举），
# 已核对 34=PAD、60=PADV2（与 tools/oes/transplant_scales.py 的 CODE_PAD/CODE_PADV2 一致）。
# 为什么需要兜底：TF 各版本里 schema_fb.BuiltinOperatorNames 的容器类型不一致
# （dict / list），下标访问会抛异常，日志里就只剩 OP_34 这种不可读的编号。
OPCODE_FALLBACK = {
    0: 'ADD', 2: 'CONCATENATION', 3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D',
    9: 'FULLY_CONNECTED', 17: 'MAX_POOL_2D', 18: 'MUL', 19: 'RELU',
    20: 'RELU_N1_TO_1', 21: 'RELU6', 22: 'RESHAPE', 25: 'SOFTMAX',
    34: 'PAD', 39: 'TRANSPOSE', 43: 'SQUEEZE', 53: 'CAST', 54: 'PRELU',
    60: 'PADV2', 64: 'SELECT', 65: 'SLICE',
}


# --------------------------------------------------------------------------
# 图片预处理：与 tools/oes/photos_dump.py 保持同一口径
#   MEAN = 127.5, SCALE = 2/255  ⇒  归一化 = pixel/127.5 - 1.0
# 两份模型喂**同一份**归一化数组，差异只可能来自模型本身。
# --------------------------------------------------------------------------
def load_image(path, height, width):
    from PIL import Image

    img = Image.open(path).convert('RGB')
    if (img.size[1], img.size[0]) != (height, width):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr.reshape(1, height, width, 3)


def quantize_input(x, detail):
    """按目标模型自己的 (scale, zero_point) 把 float 输入转成 int8。

    关键：用**模型自带**的量化参数，而不是硬编码。这样即使两份模型的输入
    口径不同，也能各自拿到正确输入，不会把「喂错输入」误判成「模型有问题」。
    """
    scale, zero_point = detail['quantization']
    if scale == 0:
        raise SystemExit('input quantization scale == 0 (unexpected)')
    q = np.round(x / scale) + zero_point
    info = np.iinfo(np.int8)
    return np.clip(q, info.min, info.max).astype(np.int8)


def run_model(model_path, image_path):
    """跑一次推理，返回 (张量表, 输入详情, 输出详情)。

    experimental_preserve_all_tensors=True 是**必须**的：默认情况下 TFLite 会
    在算子执行后立即释放中间张量，get_tensor() 只能拿到输入/输出。
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(
        model_path=model_path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    h, w = int(in_det['shape'][1]), int(in_det['shape'][2])

    x = load_image(image_path, h, w)
    if in_det['dtype'] == np.int8:
        x = quantize_input(x, in_det)
    elif in_det['dtype'] == np.uint8:
        scale, zero_point = in_det['quantization']
        x = np.clip(np.round(x / scale) + zero_point, 0, 255).astype(np.uint8)
    else:
        x = x.astype(np.float32)

    interp.set_tensor(in_det['index'], x)
    interp.invoke()

    tensors, stats = {}, {}
    for det in interp.get_tensor_details():
        idx, name = int(det['index']), det['name']
        try:
            arr = interp.get_tensor(idx)
        except Exception:
            continue
        arr = np.asarray(arr)
        scale, zero_point = det['quantization']

        stats[name] = {
            'dtype': str(arr.dtype),
            'scale': float(scale) if scale is not None else 0.0,
            'zp': int(zero_point) if zero_point is not None else 0,
            'raw_min': float(np.min(arr)) if arr.size else 0.0,
            'raw_max': float(np.max(arr)) if arr.size else 0.0,
        }
        if arr.dtype in (np.int8, np.uint8) and zero_point is not None:
            # frac_at_zp：等于 zero_point 的元素占比。
            # 若某张量 frac_at_zp ≈ 1.0，说明它**整片塌成 0** —— 这是判断
            # 「requant 乘数≈0 / scale 过大」最直接的一个数。
            stats[name]['frac_at_zp'] = float(
                np.mean(arr == int(zero_point))) if arr.size else 0.0

        if arr.dtype in (np.int8, np.uint8) and scale and scale > 0:
            arr = (arr.astype(np.float32) - float(zero_point)) * float(scale)
        elif arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        tensors[name] = arr

    return tensors, stats, in_det, out_det


def parse_graph(model_path):
    """解析 flatbuffer，返回 (算子序列, 激活名列表)。

    算子序列 = [(op_index, op_type_name, [输出张量名])]
    激活名     = 所有算子的输出张量名集合（**排除**常量：常量不是任何算子的输出）

    注意：OperatorCodes 在 **Model** 上，不在 SubGraph 上
    （run 40269559679 就是在这里崩的：SubGraph 没有 OperatorCodesLength）。
    解析失败返回 (None, None)，不致命。
    """
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
    except Exception:
        return None, None

    with open(model_path, 'rb') as f:
        buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    sub = model.Subgraphs(0)

    names = {}
    for i in range(sub.TensorsLength()):
        names[i] = sub.Tensors(i).Name().decode('utf-8', 'replace')

    op_types = {}
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        # BuiltinOperatorNames 在不同 TF 版本里可能是 dict 也可能是 list，
        # 统一用下标 + 兜底，避免一次版本差异就整步崩掉。
        try:
            op_types[i] = schema_fb.BuiltinOperatorNames[code]
        except Exception:
            op_types[i] = OPCODE_FALLBACK.get(code, 'OP_%d' % code)

    def tshape(n):
        t = sub.Tensors(n)
        return tuple(t.Shape(j) for j in range(t.ShapeLength()))

    seq, activations = [], []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        outs = [names[op.Outputs(j)] for j in range(op.OutputsLength())]
        ins = [names[op.Inputs(j)] for j in range(op.InputsLength())]
        seq.append((i, op_types.get(op.OpcodeIndex(), '?'), outs, ins,
                    [tshape(op.Outputs(j)) for j in range(op.OutputsLength())]))
        activations.extend(outs)
    # 去重但保序
    seen, act = set(), []
    for n in activations:
        if n not in seen:
            seen.add(n)
            act.append(n)
    return seq, act


def cosine(a, b):
    fa, fb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na == 0 or nb == 0:
        # 两边全零 ⇒ 完全一致；只有一边全零 ⇒ 0
        return 1.0 if (na == 0 and nb == 0) else 0.0
    return float(np.dot(fa, fb) / (na * nb))


def candidate_names(name):
    """张量名可能被工具链融合成 'A;B;C'（onnx2tf 把 conv+add+relu 拼在一起）。

    量化会把融合拆开，于是量化侧是简单名 A / B / C。这里给出全部候选，
    由 match_best() 用数据（cos）而不是猜测来决定对应关系。
    """
    out = [name]
    for part in name.split(';'):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


def match_best(name, ref_t, qnt_t):
    """给参照侧的一个激活找量化侧对应张量，取 cos 最高的候选。

    为什么用「最高 cos」而不是「最后一个分量」：融合名的顺序没有文档保证，
    猜错会产生大量**假失配**（把 conv 的输出错配成 relu 的输出）。
    用数据选最优是保守做法 —— 它只会让失配更难被报出来，不会凭空造出失配。
    这是本工具已知的**偏乐观**偏差，已在报告里注明。
    """
    ra = ref_t[name]
    if not np.issubdtype(ra.dtype, np.floating):
        return None
    best = None
    for c in candidate_names(name):
        if c not in qnt_t:
            continue
        qa = qnt_t[c]
        if qa.shape != ra.shape:
            continue
        v = cosine(ra, qa)
        if best is None or v > best[1]:
            best = (c, v)
    return best


def _op_family(op_type):
    """把 PAD / PADV2 归一成同一族：padsafe 重写会把 PadV2 改成 Pad，
    语义等价，不该因为 opcode 不同就配不上。"""
    return 'PAD' if op_type in ('PAD', 'PADV2') else op_type


def align_struct(ref_seq, qnt_seq, ref_act, qnt_act):
    """结构对齐：按「算子类型 + 输出形状 + 已匹配的输入」推进。

    ## 为什么不能用逐位置对齐（本轮踩到的坑，必须写下来）

    逐位置对齐在 art13 上**通过了**我上一版的所有校验（143 个锚点一致率 100%、
    offset=+0、两侧激活数都是 201），但结果里仍有 2 行形状对不上：
        ref   [1, 99, 99, 122]  conv_1（stride 1 分支）
        quant [1, 50, 50, 122]  conv_2（stride 2 分支）
    原因是这两条并行分支在两张图里**发出顺序相反**。交换不改变下标，
    所以「锚点一致率」这类只统计相对偏移的校验**结构上就看不见它**；
    我上一版加的「同名锚点形状一致」自检更是空转（比的是同一个名字，
    两边必然一致）。那两行 cos=0.000000 就是由此产生的假失配。

    ⇒ 判据：位置对齐只能证明「没有整体平移」，不能证明「没有交换」。

    ## 本版做法

    在两张图上同时推进拓扑序，给 ref 的每个算子找那个
      · 算子类型相同（PAD/PADV2 归一）
      · 输出形状序列完全相同
      · 所有**激活输入**都已匹配到 qnt 侧对应张量
    的 qnt 算子；唯一候选才接受，多个候选留到下一轮（等更多输入被匹配）。

    独立校验：若某个 ref 张量名在 qnt 图里**也存在**，却被配到了别的张量，
    记 conflict。conflict 必须为 0，否则对齐不可信。

    返回 (pairs, info)。
    """
    info = {'mode': 'none', 'reason': '', 'pairs': 0, 'conflict': 0,
            'name_agree': 0, 'op_matched': 0, 'op_total': 0, 'conflict_sample': [],
            'by_order': 0}
    if not ref_seq or not qnt_seq:
        info['reason'] = '算子表不可用'
        return [], info

    ref_act_set = set(ref_act)
    qnt_act_set = set(qnt_act)

    def sig(op):
        return (_op_family(op[1]), tuple(op[4]))

    by_sig = {}
    for q in qnt_seq:
        by_sig.setdefault(sig(q), []).append(q)

    tmap, used, op_map = {}, set(), {}

    # 种子：两张图的第一个算子都读模型输入（形状相同），直接配
    r0, q0 = ref_seq[0], qnt_seq[0]
    if sig(r0) == sig(q0) and len(r0[3]) == len(q0[3]):
        for a, b in zip(r0[2], q0[2]):
            tmap[a] = b
        used.add(q0[0])
        op_map[r0[0]] = q0[0]

    for _ in range(80):
        progress = False
        for r in ref_seq:
            if r[0] in op_map:
                continue
            # 只比**激活输入**子序列：常量输入的个数本来就会变
            # （padsafe 把 PadV2(3 输入) 改写成 Pad(2 输入)，
            #   卷积有无 bias 也是 3/2 之别），按位置逐一对齐会把这些
            #   合法的元数差异当成「不是同一个算子」，实测把覆盖卡死在 2%。
            r_ins = [n for n in r[3] if n in ref_act_set]
            ok = []
            for q in by_sig.get(sig(r), []):
                if q[0] in used:
                    continue
                q_ins = [n for n in q[3] if n in qnt_act_set]
                if len(q_ins) != len(r_ins):
                    continue
                if any(tmap.get(a) != b for a, b in zip(r_ins, q_ins)):
                    continue
                ok.append(q)
            if len(ok) == 1:
                q = ok[0]
                for a, b in zip(r[2], q[2]):
                    tmap[a] = b
                used.add(q[0])
                op_map[r[0]] = q[0]
                progress = True

        if not progress:
            # 第二层：**同签名组内按发出顺序配**。
            # 到这一步剩下的都是「真歧义」—— 例如 6 个完全同形状的
            # STRIDED_SLICE 且输入是同一个张量，局部信息永远分不开。
            # 组内顺序配对是确定性的（不用数据、不看 cos），
            # 但它是一种**假设**，所以单独计数并在日志里披露。
            groups = {}
            for r in ref_seq:
                if r[0] in op_map:
                    continue
                groups.setdefault(sig(r), []).append(r)
            for s, rs in groups.items():
                free = [q for q in by_sig.get(s, []) if q[0] not in used]
                if len(free) != len(rs):
                    continue
                for r, q in zip(rs, free):
                    r_ins = [n for n in r[3] if n in ref_act_set]
                    q_ins = [n for n in q[3] if n in qnt_act_set]
                    if len(r_ins) != len(q_ins):
                        continue
                    if any(tmap.get(a) != b for a, b in zip(r_ins, q_ins)):
                        continue
                    for a, b in zip(r[2], q[2]):
                        tmap[a] = b
                    used.add(q[0])
                    op_map[r[0]] = q[0]
                    info['by_order'] += 1
                    progress = True

        if not progress:
            break

    qnt_by_idx = {q[0]: q for q in qnt_seq}
    pairs = []
    for r in ref_seq:
        q = qnt_by_idx.get(op_map.get(r[0]))
        if q is None:
            continue
        for a, b in zip(r[2], q[2]):
            pairs.append((a, b))

    conflict = [p for p in pairs if p[0] in qnt_act_set and p[0] != p[1]]
    info.update({
        'mode': 'struct',
        'pairs': len(pairs),
        'conflict': len(conflict),
        'conflict_sample': [c[0] for c in conflict[:5]],
        'name_agree': sum(1 for p in pairs if p[0] == p[1]),
        'op_matched': len(op_map),
        'op_total': len(ref_seq),
    })
    info['reason'] = ('结构对齐 %d/%d 算子；配对 %d 个激活；同名一致 %d；'
                      '组内顺序配 %d（假设项）；'
                      '★ 同名冲突 %d（必须为 0，否则对齐不可信）'
                      % (len(op_map), len(ref_seq), len(pairs),
                         info['name_agree'], info['by_order'], len(conflict)))
    return pairs, info


def align(ref_act, qnt_act, ref_t, qnt_t, ref_seq=None, qnt_seq=None,
          min_cover=0.90):
    """对齐入口：结构对齐优先；不达标则回退按名匹配（并说明原因）。"""
    pairs, sinfo = align_struct(ref_seq, qnt_seq, ref_act, qnt_act)
    cover = len(pairs) / float(len(ref_act)) if ref_act else 0.0
    sinfo['cover'] = cover
    if sinfo['mode'] == 'struct' and sinfo['conflict'] == 0 and cover >= min_cover:
        return pairs, sinfo

    why = ('结构对齐不达标（同名冲突 %d，覆盖 %.1f%% < %.0f%%）⇒ 回退按名匹配'
           % (sinfo['conflict'], cover * 100, min_cover * 100))
    fb = []
    for n in ref_act:
        m = match_best(n, ref_t, qnt_t)
        if m is not None:
            fb.append((n, m[0]))
    return fb, {'mode': 'name', 'reason': why, 'struct': sinfo,
                'pairs': len(fb), 'conflict': sinfo['conflict'],
                'cover': len(fb) / float(len(ref_act)) if ref_act else 0.0}


def compare(ref, qnt, pairs, cos_min, rel_max, ref_st=None, qnt_st=None):
    ref_st, qnt_st = ref_st or {}, qnt_st or {}
    rows, first_bad = [], None
    for name, qname in pairs:
        if name not in ref or qname not in qnt:
            continue
        ra, qa = ref[name], qnt[qname]

        # 形状必须一致。逐位置对齐下出现形状不一致，本身就是「对齐错了」的信号，
        # 要让它出现在结果里，而不是抛异常整步崩掉（reshape 会 ValueError）。
        if ra.shape != qa.shape:
            rows.append({'name': name, 'quant_name': qname, 'cos': 0.0,
                         'rel_max': 1.0, 'bad': True, 'masked': 0,
                         'shape_mismatch': True, 'shape': list(ra.shape),
                         'ref_stats': ref_st.get(name),
                         'qnt_stats': qnt_st.get(qname)})
            if first_bad is None:
                first_bad = rows[-1]
            continue

        # 哨兵剔除（2026-10-26 实测必要性）：
        # 这个模型的部分张量带有未初始化的 ±FLT_MAX 哨兵（float16 里是 ±65504）。
        # 一个 3.4e38 的元素就能把 cos 压成 0、把 rel 撑成 1.0，
        # 从而把「两边其实完全一致」误判成「全崩」——正控制组就踩了这个坑。
        # 规则：|值| > 1e30 的元素在两侧同时剔除，并记录被剔除个数。
        triple = np.abs(ra) > 1e30
        if qa.size == ra.size:
            triple = triple | (np.abs(qa.reshape(ra.shape)) > 1e30)
        masked = int(np.count_nonzero(triple))
        if masked:
            keep = ~triple
            ra_c, qa_c = ra[keep], qa.reshape(ra.shape)[keep]
        else:
            ra_c, qa_c = ra, qa

        c = cosine(ra_c, qa_c)
        denom = float(np.max(np.abs(ra_c))) + 1e-12 if ra_c.size else 1e-12
        rel = float(np.max(np.abs(ra_c - qa_c))) / denom if ra_c.size else 0.0
        bad = (c < cos_min) or (rel > rel_max)
        rows.append({'name': name, 'quant_name': qname, 'cos': c,
                     'rel_max': rel, 'bad': bad, 'masked': masked,
                     'shape': list(ra.shape),
                     'ref_stats': ref_st.get(name),
                     'qnt_stats': qnt_st.get(qname)})
        if bad and first_bad is None:
            first_bad = rows[-1]
    rows.sort(key=lambda r: (not r['bad'], r['cos'], -r['rel_max']))
    return rows, first_bad


def main(argv=None):
    ap = argparse.ArgumentParser(description='逐张量对照定位 int8 首失配点')
    ap.add_argument('--ref', required=True, help='浮点参照模型 .tflite')
    ap.add_argument('--quant', help='量化模型 .tflite')
    ap.add_argument('--image', help='输入图')
    ap.add_argument('--json', help='把结果写到这个 json')
    ap.add_argument('--cos-min', type=float, default=0.999)
    ap.add_argument('--rel-max', type=float, default=0.05)
    ap.add_argument('--min-match', type=float, default=0.80,
                    help='激活配对率下限，低于此值以退出码 3 告警')
    ap.add_argument('--min-cover', type=float, default=0.90,
                    help='结构对齐覆盖率达到此值才采用，否则回退按名匹配')
    ap.add_argument('--stats-rows', type=int, default=12,
                    help='打印前 N 个激活的原始统计（scale/zp/frac_at_zp）')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--list-tensors', action='store_true')
    args = ap.parse_args(argv)

    for p in (args.ref, args.quant, args.image):
        if p and not os.path.isfile(p):
            print('missing file: %s' % p, file=sys.stderr)
            return 2

    if args.list_tensors:
        seq, act = parse_graph(args.ref)
        if seq is None:
            print('schema 绑定不可用，无法解析', file=sys.stderr)
            return 2
        for i, ty, outs, ins, shp in seq:
            print('op#%-4d %-16s %-18s -> %s'
                  % (i, ty, str(shp)[:18], ', '.join(outs)[-60:] or '(none)'))
        print('--- 激活张量 %d 个 ---' % len(act))
        return 0

    if not (args.quant and args.image):
        print('--quant 与 --image 必须同时提供（除 --list-tensors 外）',
              file=sys.stderr)
        return 2

    print('=== ref   : %s' % args.ref)
    print('=== quant : %s' % args.quant)
    print('=== image : %s' % args.image)

    ref_t, ref_st, ref_in, _ = run_model(args.ref, args.image)
    qnt_t, qnt_st, qnt_in, q_out = run_model(args.quant, args.image)
    ref_seq, ref_act = parse_graph(args.ref)

    # 输入口径必须先对齐，否则后面所有比较都是错的（两个模型期望的
    # 归一化不同是最常见的坑：一边要 [-1,1]，一边要 [0,1]）。
    print('=== 输入: ref dtype=%s shape=%s quant=%s'
          % (ref_in['dtype'], list(ref_in['shape']), ref_in['quantization']))
    print('=== 输入: qnt dtype=%s shape=%s quant=%s'
          % (qnt_in['dtype'], list(qnt_in['shape']), qnt_in['quantization']))
    qnt_seq, qnt_act = parse_graph(args.quant)

    if ref_act is None or qnt_act is None:
        print('!! 无法解析算子表，退化为按全部张量比对', file=sys.stderr)
        ref_act = list(ref_t)
        qnt_act = list(qnt_t)

    print('=== tensors: ref=%d quant=%d ；激活: ref=%d quant=%d'
          % (len(ref_t), len(qnt_t), len(ref_act), len(qnt_act)))

    pairs, ainfo = align(ref_act, qnt_act, ref_t, qnt_t,
                         ref_seq=ref_seq, qnt_seq=qnt_seq,
                         min_cover=args.min_cover)
    rate = len(pairs) / float(len(ref_act)) if ref_act else 0.0
    low = rate < args.min_match
    print('=== 对齐方式: %s（%s）' % (ainfo['mode'], ainfo['reason']))
    print('=== 激活覆盖率: %d/%d = %.1f%%'
          % (len(pairs), len(ref_act), rate * 100))

    mism = [p for p in pairs
            if p[0] in ref_t and p[1] in qnt_t
            and ref_t[p[0]].shape != qnt_t[p[1]].shape]
    print('=== 对齐自检: 配上的张量里形状不一致 %d 个（应为 0；'
          '上一版逐位置对齐在这里漏了 2 个）' % len(mism))
    if mism:
        for a, b in mism[:5]:
            print('    ! %s  ref%s vs quant%s'
                  % (a[-40:], list(ref_t[a].shape), list(qnt_t[b].shape)))

    if low:
        # 关键：**不因为覆盖率低就提前退出**。
        # run 40270810573 就是在这里 return 3，导致三组一条对照数据都没打印。
        # 现在的规矩是：覆盖率低只降级「结论强度」，不影响「先把数据打出来」。
        print('!! 覆盖率低于 %.0f%% ⇒ 结论只能算**暂定**，但下面照常完整输出'
              % (args.min_match * 100), file=sys.stderr)
        miss = [n for n in ref_act if n not in dict(pairs)][:10]
        print('!! 未配对样例: %s' % miss, file=sys.stderr)

    rows, first_bad = compare(ref_t, qnt_t, pairs, args.cos_min, args.rel_max,
                              ref_st=ref_st, qnt_st=qnt_st)

    owner = {}
    for i, ty, outs, _ins, _shp in (qnt_seq or []):
        for o in outs:
            owner.setdefault(o, (i, ty))
    if first_bad:
        # 必须按**量化侧**张量名查生产者：ref 侧名字在 qnt 图里不存在。
        i, ty = owner.get(first_bad['quant_name'], (-1, '?'))
        print('\n=== 首个失配: op#%d %s -> %s' % (i, ty, first_bad['name']))
        print('    cos=%.6f  rel_max=%.4f  shape=%s'
              % (first_bad['cos'], first_bad['rel_max'], first_bad['shape']))
        rn, qn = first_bad['name'], first_bad['quant_name']
        print('    ref   统计: %s' % ref_st.get(rn))
        print('    quant 统计: %s' % qnt_st.get(qn))
        if qnt_st.get(qn, {}).get('frac_at_zp', 0) > 0.999:
            print('    ★ quant 侧 frac_at_zp≈1.0 ⇒ 该张量整片塌成 0，'
                  '不是精度损失，是 requant 乘数≈0 / scale 过大')
    else:
        print('\n=== 无失配：全部 %d 个激活张量都在阈值内' % len(rows))

    by_name = {r['name']: r for r in rows}
    print('\n=== 前 20 个可比激活（按 ref 激活顺序，看从第几步开始崩）')
    for name in [rn for rn, _ in pairs][:20]:
        rr = by_name.get(name)
        if rr is None:
            continue
        print('   %-44s cos=%-10.6f rel=%-8.4f masked=%-4d %s'
              % (name[:44], rr['cos'], rr['rel_max'], rr.get('masked', 0),
                 '✗' if rr['bad'] else '✓'))

    # 这一段是判「塌零」还是「量化精度」的直接证据：
    #   frac_at_zp ≈ 1.0  -> 整片塌成 0（requant 乘数≈0 / scale 过大）
    #   scale 比邻居大一个量级 -> 标定区间被离群值撑开
    #   两边 min/max 都正常而 cos 低 -> 值域没错，是**内容**算错了
    print('\n=== 前 %d 个激活的原始统计（ref 激活顺序）' % args.stats_rows)
    for name, qname in pairs[:args.stats_rows]:
        rr = by_name.get(name)
        if rr is None:
            continue
        rs = rr.get('ref_stats') or {}
        qs = rr.get('qnt_stats') or {}
        print('   op#%-3s %-40s cos=%.6f %s'
              % ('?', name[-40:], rr['cos'], '✗' if rr['bad'] else '✓'))
        print('      ref: min=%11.5f max=%11.5f' % (rs.get('raw_min', 0.0),
                                                    rs.get('raw_max', 0.0)))
        print('      qnt: scale=%-12.7g zp=%-5d min=%-6.0f max=%-6.0f at_zp=%.4f'
              % (qs.get('scale', 0.0), qs.get('zp', 0),
                 qs.get('raw_min', 0.0), qs.get('raw_max', 0.0),
                 qs.get('frac_at_zp', float('nan'))))

    print('\n=== 最差 %d 个激活张量（✗ = 超阈值 cos<%.4f 或 rel>%.3f）'
          % (args.top, args.cos_min, args.rel_max))
    print('   %-46s %-10s %-10s %s' % ('张量名', 'cos', 'rel_max', '判定'))
    for r in rows[:args.top]:
        print('   %-46s %-10.6f %-10.4f %s'
              % (r['name'][:46], r['cos'], r['rel_max'],
                 '✗' if r['bad'] else '✓'))

    out_scale, out_zp = q_out['quantization']
    print('\n=== 图输出: scale=%.8g zero_point=%d dtype=%s'
          % (out_scale, out_zp, q_out['dtype']))
    bad_n = sum(1 for r in rows if r['bad'])
    print('=== 超阈值激活数: %d / %d' % (bad_n, len(rows)))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({
                'ref': args.ref, 'quant': args.quant, 'image': args.image,
                'tensors_ref': len(ref_t), 'tensors_quant': len(qnt_t),
                'activations_ref': len(ref_act), 'activations_quant': len(qnt_act),
                'matched': len(pairs), 'match_rate': rate,
                'align_mode': ainfo['mode'], 'align_info': ainfo,
                'low_coverage': low,
                'cos_min': args.cos_min, 'rel_max_threshold': args.rel_max,
                'first_bad': first_bad, 'bad_count': bad_n, 'rows': rows,
            }, f, indent=1)
        print('=== json -> %s' % args.json)

    # 覆盖率不足时以 3 退出（提示结论暂定），但**数据已经完整打印过了**
    return 3 if low else 0


if __name__ == '__main__':
    sys.exit(main())
