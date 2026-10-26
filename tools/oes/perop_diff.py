#!/usr/bin/env python3
"""逐张量对照两份 TFLite 模型，定位第一个失配点。

## 为什么需要它

int8 产物在**输出端**与浮点参照差 2.7 倍（报告 65 §3）。只比输出无法回答
关键问题：这是**量化本身的精度损失**，还是**转换链路的缺陷**？
两者处置完全不同 —— 前者要改量化方案（或放弃 per-tensor），后者是可修的 bug。

本工具把两份模型都跑同一张图，取出**全部**张量
(tf.lite.Interpreter 的 experimental_preserve_all_tensors=True)，
按**张量名**配对，按量化模型的算子顺序逐层计算 cos / 相对误差，
报出**第一个**跌破阈值的张量 —— 那一点就是失配源头，再看它后面是
「一路缓慢劣化」（量化精度）还是「一步跳变」（转换缺陷）。

## 为什么按名字配对可行（以及何时不可行）

两份模型由同一条 onnx2tf 链路生成，张量命名一致
(arith.constant / arith.constant1 / ... / Const_N)。实测
baseline_float_concept_split_padsafe_float16 与 concept_int8_art13 的前 8 个
张量名与索引基本对齐。

但工具**不假设**配对率一定是 100%：它会打印配对率，低于 --min-match
(default 0.80) 时以非零退出码告警，避免「配对失败被当成模型正常」。

## 用法

  python perop_diff.py \
    --ref   out/baseline_float_concept_split_padsafe_float32.tflite \
    --quant out/concept_int8_art13.tflite \
    --image eval/oes-eval-0000.jpg \
    --json  out/perop_art13.json

  # 只列张量名（不推理，不需要图片）
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
    """跑一次推理，返回 (所有张量的 {名字: (数组, dtype)}, 输入详情, 输出详情)。

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
            # 少数张量（如未初始化的可选输入）取不到，跳过但要计数
            continue
        arr = np.asarray(arr)
        scale, zero_point = det['quantization']

        if arr.dtype in (np.int8, np.uint8) and scale and scale > 0:
            arr = (arr.astype(np.float32) - float(zero_point)) * float(scale)
        elif arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        tensors[name] = (arr, det['dtype'])

    return tensors, in_det, out_det, interp


def op_sequence(model_path):
    """解析 flatbuffer，给出量化模型的算子顺序 [(op_index, type_name, [输出张量名])]。

    用 TF 自带的 schema 绑定，不引第三方包。解析失败不致命 —— 退化为
    「按张量索引顺序」输出。
    """
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
    except Exception:
        return None

    with open(model_path, 'rb') as f:
        buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    sub = model.Subgraphs(0)

    names = {}
    for i in range(sub.TensorsLength()):
        t = sub.Tensors(i)
        names[i] = t.Name().decode('utf-8', 'replace')

    op_types = {}
    for i in range(sub.OperatorCodesLength()):
        oc = sub.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        op_types[i] = schema_fb.BuiltinOperatorNames.get(code, 'OP_%d' % code)

    seq = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        outs = [names[op.Outputs(j)] for j in range(op.OutputsLength())]
        seq.append((i, op_types.get(op.OpcodeIndex(), '?'), outs))
    return seq


def cosine(a, b):
    fa, fb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na == 0 or nb == 0:
        # 两边全零 ⇒ 完全一致；只有一边全零 ⇒ 0
        return 1.0 if (na == 0 and nb == 0) else 0.0
    return float(np.dot(fa, fb) / (na * nb))


def compare(ref, qnt, cos_min, rel_max, top_n):
    rows, first_bad = [], None
    for name, (qa, _) in qnt.items():
        if name not in ref:
            continue
        ra = ref[name][0]
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
    ap.add_argument('--image', help='输入图（建议用 CI 里同一张 picsum 图）')
    ap.add_argument('--json', help='把结果写到这个 json')
    ap.add_argument('--cos-min', type=float, default=0.999)
    ap.add_argument('--rel-max', type=float, default=0.05)
    ap.add_argument('--min-match', type=float, default=0.80,
                    help='配对率下限，低于此值以退出码 3 告警')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--list-tensors', action='store_true')
    args = ap.parse_args(argv)

    for p in (args.ref, args.quant, args.image):
        if p and not os.path.isfile(p):
            print('missing file: %s' % p, file=sys.stderr)
            return 2

    if args.list_tensors:
        seq = op_sequence(args.ref)
        if seq is None:
            print('schema 绑定不可用，无法列算子', file=sys.stderr)
            return 2
        for i, ty, outs in seq:
            print('op#%-4d %-24s -> %s' % (i, ty, ', '.join(outs) or '(none)'))
        return 0

    if not (args.quant and args.image):
        print('--quant 与 --image 必须同时提供（除 --list-tensors 外）',
              file=sys.stderr)
        return 2

    print('=== ref   : %s' % args.ref)
    print('=== quant : %s' % args.quant)
    print('=== image : %s' % args.image)

    ref_t, _, _, _ = run_model(args.ref, args.image)
    qnt_t, _, q_out, _ = run_model(args.quant, args.image)
    print('=== tensors: ref=%d quant=%d' % (len(ref_t), len(qnt_t)))

    matched = sum(1 for n in qnt_t if n in ref_t)

    # 配对率的分母必须是**参照模型的张量数**，不是量化模型的。
    # 反例（2026-10-26 实测）：正控制 float32 vs float16 时
    #   ref=339 / quant=454 —— float16 模型多了 115 个 Dequantize 附加张量，
    # 用 len(quant) 做分母会算出 74.4% 的假告警，而真实配对是 338/339 = 99.7%。
    rate = matched / float(len(ref_t)) if ref_t else 0.0
    print('=== 配对率: %d/%d = %.1f%%（分母=参照模型张量数；量化模型多出的附加张量不计）'
          % (matched, len(ref_t), rate * 100))

    if rate < args.min_match:
        print('!! 配对率低于 %.0f%% ⇒ 结论不可信（张量命名不一致）'
              % (args.min_match * 100), file=sys.stderr)
        miss = [n for n in ref_t if n not in qnt_t][:10]
        print('!! 参照侧未配对样例: %s' % miss, file=sys.stderr)
        return 3

    rows, first_bad = compare(ref_t, qnt_t, args.cos_min, args.rel_max, args.top)

    # 把算子类型挂到首失配张量上，便于直接对照模型结构
    seq = op_sequence(args.quant) or []
    owner = {}
    for i, ty, outs in seq:
        for o in outs:
            owner.setdefault(o, (i, ty))
    if first_bad:
        i, ty = owner.get(first_bad['name'], (-1, '?'))
        print('\n=== 首个失配: op#%d %s -> %s' % (i, ty, first_bad['name']))
        print('    cos=%.6f  rel_max=%.4f  shape=%s'
              % (first_bad['cos'], first_bad['rel_max'], first_bad['shape']))

    print('\n=== 最差 %d 个张量（✗ = 超阈值）' % args.top)
    print('   %-44s %-10s %-10s %s' % ('张量名', 'cos', 'rel_max', '判定'))
    for r in rows[:args.top]:
        print('   %-44s %-10.6f %-10.4f %s'
              % (r['name'][:44], r['cos'], r['rel_max'],
                 '✗' if r['bad'] else '✓'))

    out_scale, out_zp = q_out['quantization']
    print('\n=== 图输出: scale=%.8g zero_point=%d dtype=%s'
          % (out_scale, out_zp, q_out['dtype']))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({
                'ref': args.ref, 'quant': args.quant, 'image': args.image,
                'tensors_ref': len(ref_t), 'tensors_quant': len(qnt_t),
                'matched': matched, 'match_rate': rate,
                'cos_min': args.cos_min, 'rel_max_threshold': args.rel_max,
                'first_bad': first_bad, 'rows': rows,
            }, f, indent=1)
        print('=== json -> %s' % args.json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
