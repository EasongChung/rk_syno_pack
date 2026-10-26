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

    tensors = {}
    for det in interp.get_tensor_details():
        idx, name = int(det['index']), det['name']
        try:
            arr = interp.get_tensor(idx)
        except Exception:
            continue
        arr = np.asarray(arr)
        scale, zero_point = det['quantization']

        if arr.dtype in (np.int8, np.uint8) and scale and scale > 0:
            arr = (arr.astype(np.float32) - float(zero_point)) * float(scale)
        elif arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        tensors[name] = arr

    return tensors, in_det, out_det


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
            op_types[i] = 'OP_%d' % code

    seq, activations = [], []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        outs = [names[op.Outputs(j)] for j in range(op.OutputsLength())]
        seq.append((i, op_types.get(op.OpcodeIndex(), '?'), outs))
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


def compare(ref, qnt, names, cos_min, rel_max):
    rows, first_bad = [], None
    for name in names:
        ra, qa = ref[name], qnt[name]
        if ra.shape != qa.shape:
            continue
        if not np.issubdtype(ra.dtype, np.floating):
            continue
        c = cosine(ra, qa)
        denom = float(np.max(np.abs(ra))) + 1e-12
        rel = float(np.max(np.abs(ra - qa))) / denom
        bad = (c < cos_min) or (rel > rel_max)
        rows.append({'name': name, 'cos': c, 'rel_max': rel, 'bad': bad,
                     'shape': list(ra.shape)})
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
        for i, ty, outs in seq:
            print('op#%-4d %-24s -> %s' % (i, ty, ', '.join(outs) or '(none)'))
        print('--- 激活张量 %d 个 ---' % len(act))
        return 0

    if not (args.quant and args.image):
        print('--quant 与 --image 必须同时提供（除 --list-tensors 外）',
              file=sys.stderr)
        return 2

    print('=== ref   : %s' % args.ref)
    print('=== quant : %s' % args.quant)
    print('=== image : %s' % args.image)

    ref_t, _, _ = run_model(args.ref, args.image)
    qnt_t, _, q_out = run_model(args.quant, args.image)
    _, ref_act = parse_graph(args.ref)
    qnt_seq, qnt_act = parse_graph(args.quant)

    if ref_act is None or qnt_act is None:
        print('!! 无法解析算子表，退化为按全部张量比对', file=sys.stderr)
        ref_act = list(ref_t)
        qnt_act = list(qnt_t)

    print('=== tensors: ref=%d quant=%d ；激活: ref=%d quant=%d'
          % (len(ref_t), len(qnt_t), len(ref_act), len(qnt_act)))

    # 配对率分母 = 参照模型的**激活**数（常量不参与，见文件头注释）
    common = [n for n in ref_act if n in qnt_t]
    rate = len(common) / float(len(ref_act)) if ref_act else 0.0
    print('=== 激活配对率: %d/%d = %.1f%%' % (len(common), len(ref_act), rate * 100))

    if rate < args.min_match:
        print('!! 激活配对率低于 %.0f%% ⇒ 结论不可信' % (args.min_match * 100),
              file=sys.stderr)
        miss = [n for n in ref_act if n not in qnt_t][:10]
        print('!! 未配对样例: %s' % miss, file=sys.stderr)
        return 3

    rows, first_bad = compare(ref_t, qnt_t, common, args.cos_min, args.rel_max)

    owner = {}
    for i, ty, outs in (qnt_seq or []):
        for o in outs:
            owner.setdefault(o, (i, ty))
    if first_bad:
        i, ty = owner.get(first_bad['name'], (-1, '?'))
        print('\n=== 首个失配: op#%d %s -> %s' % (i, ty, first_bad['name']))
        print('    cos=%.6f  rel_max=%.4f  shape=%s'
              % (first_bad['cos'], first_bad['rel_max'], first_bad['shape']))
    else:
        print('\n=== 无失配：全部 %d 个激活张量都在阈值内' % len(rows))

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
                'matched': len(common), 'match_rate': rate,
                'cos_min': args.cos_min, 'rel_max_threshold': args.rel_max,
                'first_bad': first_bad, 'bad_count': bad_n, 'rows': rows,
            }, f, indent=1)
        print('=== json -> %s' % args.json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
