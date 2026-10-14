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


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--quant', required=True)
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--topk', default='1,5')
    ap.add_argument('--out', default='report.json')
    args = ap.parse_args()
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
    report = {
        'n': n, 'ref': args.ref, 'quant': args.quant,
        'agree': {('top%d' % k): round(agree[k] / n, 4) for k in ks},
        'cos_mean': round(float(np.mean(coss)), 4),
        'cos_p5': round(float(np.percentile(coss, 5)), 4),
        'drift_mean': round(float(np.mean(drifts)), 5),
        'gate': {'top1': 0.95, 'top5': 0.99, 'cos_mean': 0.98, 'drift': 0.03},
        'fails': sorted(fails, key=lambda r: r['cos'])[:20],
    }
    gate_hit = (report['agree'].get('top1', 0) >= 0.95
                and report['agree'].get('top5', 0) >= 0.99
                and report['cos_mean'] >= 0.98
                and report['drift_mean'] <= 0.03)
    report['gate_hit'] = bool(gate_hit)
    json.dump(report, open(args.out, 'w'), indent=2)
    print(json.dumps(report, indent=2)[:3000], flush=True)
    print('n=%d agree=%s cos_mean=%.4f drift=%.5f gate_hit=%s' % (
        n, report['agree'], report['cos_mean'], report['drift_mean'],
        gate_hit), flush=True)
    if not gate_hit:
        print('GATE MISS (report kept, exit 3)', flush=True)
        sys.exit(3)
    print('GATE HIT', flush=True)


if __name__ == '__main__':
    main()
