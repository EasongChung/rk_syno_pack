#!/usr/bin/env python3
"""精度回归：ref float 模型 vs quant INT8 模型（同一预处理、同一 eval 集）.

指标：top1/top5 一致率、余弦相似度、分数漂移；失败样例落盘。
门限：top1>=0.95 & top5>=0.99 & cos_mean>=0.98 & drift<=0.03（未命中则 exit 3，
但 report.json 照常产出供人工复核）。

用法：
  python regress.py --ref tools/oes/models/model_float16.tflite \\
      --quant out/concept_int8.tflite --eval-dir eval --topk 1,5 --out report.json
"""
import argparse
import glob
import io
import json
import os
import sys

NET = 395
MEAN = 127.5
SCALE = 0.00784313725


def load_interp(path):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from convert_to_int8 import make_interpreter
        return make_interpreter(path)['interp']
    except ImportError:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=path)
        it.allocate_tensors()
        return it


def run_once(interp, image_bytes):
    from PIL import Image
    import numpy as np
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    shape = [int(x) for x in inp['shape']]
    net = shape[2] if len(shape) == 4 else NET
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(
        (net, net), Image.BILINEAR)
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    if inp['dtype'] == np.uint8:
        t = arr.reshape(shape)
    elif inp['dtype'] == np.int8:
        # int8 输入（TFLiteConverter 全整数量化产物，in_type=9）：
        # 必须按该张量的 scale/zero_point 把 float 归一化输入量化成 int8，
        # 否则会把 float32 塞进 int8 tensor 直接报类型错（exit=1）。
        isc, izp = inp.get('quantization', (0.0, 0))
        f = ((arr.astype(np.float32) - MEAN) * SCALE).reshape(shape)
        if isc:
            t = np.clip(np.round(f / isc) + izp, -128, 127).astype(np.int8)
        else:
            t = f.astype(np.int8)
    else:
        t = (((arr.astype(np.float32) - MEAN) * SCALE)
             .reshape(shape).astype(np.float32))
    interp.set_tensor(inp['index'], t)
    interp.invoke()
    v = interp.get_tensor(out['index']).flatten().astype(float)
    q = out.get('quantization', (0.0, 0))
    if isinstance(q, tuple) and q[0]:
        v = (v - q[1]) * q[0]
    return v.tolist()


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(y * y for y in b)) + 1e-9
    return dot / (na * nb)


def topk(v, k):
    return sorted(range(len(v)), key=lambda i: -v[i])[:k]


def assert_int8(path):
    """断言模型确为 INT8 量化（HOST-VERDICT §1 教训：-oiqt 曾产出 float 冒充过门）。

    条件：输入 dtype 为 uint8/int8，且带量化参数的张量数 > 0。
    不满足则 sys.exit(4)（区别于门限未命中的 exit 3）。
    """
    import sys
    import numpy as np
    import tflite
    raw = open(path, 'rb').read()
    model = tflite.Model.GetRootAsModel(raw, 0)
    g = model.Subgraphs(0)
    tin = g.Tensors(g.Inputs(0))
    in_ok = tin.Type() in (3, 9)  # UINT8 / INT8
    nq = 0
    for i in range(g.TensorsLength()):
        q = g.Tensors(i).Quantization()
        if q is not None and q.ScaleLength() > 0:
            nq += 1
    print('int8-guard: %s in_type=%d quantized_tensors=%d' % (
        path, tin.Type(), nq), flush=True)
    if not in_ok or nq == 0:
        print('NOT-INT8: %s is NOT a quantized model '
              '(in_type=%d, quantized=%d). Refusing to score it as INT8.' % (
                  path, tin.Type(), nq), flush=True)
        sys.exit(4)


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--quant', required=True)
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--topk', default='1,5')
    ap.add_argument('--out', default='report.json')
    ap.add_argument('--require-int8', action='store_true',
                    help='quant 侧必须是真 INT8（输入 uint8/int8 且量化张量数>0），'
                         '否则 exit 4。防 float 模型冒充 INT8 过绿门。')
    args = ap.parse_args()
    if args.require_int8:
        assert_int8(args.quant)
    ks = [int(x) for x in args.topk.split(',')]
    files = sorted(glob.glob(os.path.join(args.eval_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert files, 'eval empty: %s' % args.eval_dir
    ri, qi = load_interp(args.ref), load_interp(args.quant)
    agree = {k: 0 for k in ks}
    coss, drifts, fails = [], [], []
    for f in files:
        b = open(f, 'rb').read()
        rv = run_once(ri, b)
        qv = run_once(qi, b)
        assert len(rv) == 335 and len(qv) == 335, 'dims must be 335'
        c = cos(rv, qv)
        coss.append(c)
        drifts.append(sum(abs(x - y) for x, y in zip(rv, qv)) / len(rv))
        row = {'file': os.path.basename(f), 'cos': round(c, 4)}
        ok1 = True
        for k in ks:
            if k == 1:
                hit = topk(rv, 1)[0] == topk(qv, 1)[0]
            else:
                hit = (len(set(topk(rv, k)) & set(topk(qv, k))) / k) >= 0.8
            row['agree@%d' % k] = bool(hit)
            agree[k] += 1 if hit else 0
            ok1 = ok1 and hit
        if c < 0.95 or not ok1:
            row['ref_top5'] = [(i, round(rv[i], 3)) for i in topk(rv, 5)]
            row['q_top5'] = [(i, round(qv[i], 3)) for i in topk(qv, 5)]
            fails.append(row)
    n = len(files)
    # 双门限（2026-10-16 och 决策）：
    #   通用门 = 既有约定（top1>=0.95 / top5>=0.99 / cos>=0.98 / drift<=0.03），
    #             不因单次结果改动，是"能否交付"的最终依据。
    #   场景门 = 针对 oes-eval（picsum 真实照片，图像多样性高）的实测上限，
    #             用于区分"未达标"与"该场景下的合理上限"，供产品侧参考。
    # 两套结论都输出，任何人都不需要再猜"用哪套"。
    gate_general = {'top1': 0.95, 'top5': 0.99, 'cos_mean': 0.98,
                    'drift': 0.03}
    gate_scene = {'top1': 0.90, 'top5': 0.97, 'cos_mean': 0.93,
                  'drift': 0.06}
    report = {
        'n': n, 'ref': args.ref, 'quant': args.quant,
        'agree': {('top%d' % k): round(agree[k] / n, 4) for k in ks},
        'cos_mean': round(float(np.mean(coss)), 4),
        'cos_p5': round(float(np.percentile(coss, 5)), 4),
        'drift_mean': round(float(np.mean(drifts)), 5),
        'gate_general': gate_general,
        'gate_scene': gate_scene,
        'fails': sorted(fails, key=lambda r: r['cos'])[:20],
    }

    def _hit(g):
        return (report['agree'].get('top1', 0) >= g['top1']
                and report['agree'].get('top5', 0) >= g['top5']
                and report['cos_mean'] >= g['cos_mean']
                and report['drift_mean'] <= g['drift'])

    general_hit = _hit(gate_general)
    scene_hit = _hit(gate_scene)
    report['gate_hit_general'] = bool(general_hit)
    report['gate_hit_scene'] = bool(scene_hit)
    # 兼容旧字段：仍以通用门为准
    report['gate_hit'] = bool(general_hit)
    json.dump(report, open(args.out, 'w'), indent=2)
    print(json.dumps(report, indent=2)[:3000], flush=True)
    print('n=%d agree=%s cos_mean=%.4f drift=%.5f' % (
        n, report['agree'], report['cos_mean'], report['drift_mean']),
        flush=True)
    print('GATE-GENERAL %s  (top1>=0.95 top5>=0.99 cos>=0.98 drift<=0.03)'
          % ('HIT' if general_hit else 'MISS'), flush=True)
    print('GATE-SCENE   %s  (top1>=0.90 top5>=0.97 cos>=0.93 drift<=0.06)'
          % ('HIT' if scene_hit else 'MISS'), flush=True)
    if general_hit:
        print('GATE HIT (general)', flush=True)
        sys.exit(0)
    if scene_hit:
        print('GATE MISS on general, but HIT on scene line '
              '(report kept, exit 3) -- 需产品侧裁决以哪套为准',
              flush=True)
        sys.exit(3)
    print('GATE MISS on both lines (report kept, exit 3)', flush=True)
    sys.exit(3)


if __name__ == '__main__':
    main() 