#!/usr/bin/env python3
"""前哨验证：TFLite(float 原版) vs ONNX(tf2onnx 产物)数值等价性。

背景：一切下游验证（smoke finite、SavedModel-vs-ONNX 余弦、PTQ 回归）都以
ONNX 为基准；但 ONNX 与 TFLite 原版的等价性从未被验证过（OEC 只比了
onnxruntime-vs-RKNN，同源循环论证）。若此处余弦≈0，说明 tf2onnx 前端
本身对该 Inception 多分支图有损，后续一切工作失去基准，必须先换前端。

用法：
  python compare_tflite_onnx.py --tflite model_float16.tflite --onnx concept_mid.onnx \\
      --eval-dir eval --num 50 --out compare.json
门限：cos_mean>=0.99 且 top1>=0.9，否则 exit 3（fail fast，不再浪费 PTQ）。
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


def preprocess(image_bytes):
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(image_bytes))
    try:
        img.draft('RGB', (NET, NET))
    except Exception:
        pass
    img = img.convert('RGB').resize((NET, NET), Image.BILINEAR)
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
    return ((arr - MEAN) * SCALE).reshape(1, NET, NET, 3)


def run_tflite(interp, tensor):
    import numpy as np
    ide = interp.get_input_details()[0]
    interp.set_tensor(ide['index'], tensor.astype(np.float32))
    interp.invoke()
    ode = interp.get_output_details()[0]
    return interp.get_tensor(ode['index']).flatten().tolist()


def run_onnx(sess, tensor):
    return sess.run(None, {sess.get_inputs()[0].name: tensor})[0].flatten().tolist()


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(y * y for y in b)) + 1e-9
    return dot / (na * nb)


def topk(v, k):
    return sorted(range(len(v)), key=lambda i: -v[i])[:k]


def run(tflite_path, onnx_path, eval_dir, num, out_path):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_to_int8 import make_interpreter
    import onnxruntime as ort
    files = sorted(glob.glob(os.path.join(eval_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    assert files, 'eval empty'
    ti = make_interpreter(tflite_path)['interp']
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    coss, agree1, agree5, rows = [], 0, 0, []
    for f in files:
        t = preprocess(open(f, 'rb').read())
        rv = run_tflite(ti, t)
        qv = run_onnx(sess, t)
        c = cos(rv, qv)
        coss.append(c)
        a1 = topk(rv, 1)[0] == topk(qv, 1)[0]
        a5 = len(set(topk(rv, 5)) & set(topk(qv, 5))) / 5 >= 0.8
        agree1 += 1 if a1 else 0
        agree5 += 1 if a5 else 0
        if c < 0.99:
            rows.append({'file': os.path.basename(f), 'cos': round(c, 4),
                         'tflite_top5': [(i, round(rv[i], 3)) for i in topk(rv, 5)],
                         'onnx_top5': [(i, round(qv[i], 3)) for i in topk(qv, 5)]})
    n = len(files)
    rep = {'n': n,
           'cos_mean': round(float(np.mean(coss)), 4),
           'cos_min': round(float(np.min(coss)), 4),
           'top1': round(agree1 / n, 4), 'top5': round(agree5 / n, 4),
           'gate': {'cos_mean': 0.99, 'top1': 0.9},
           'mismatch_top20': rows[:20]}
    rep['gate_hit'] = bool(rep['cos_mean'] >= 0.99 and rep['top1'] >= 0.9)
    json.dump(rep, open(out_path, 'w'), indent=2)
    print(json.dumps(rep, indent=2)[:2500], flush=True)
    if not rep['gate_hit']:
        print('FRONTEND MISMATCH: tflite vs onnx differ', flush=True)
        return False
    print('FRONTEND OK: onnx faithful to tflite', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tflite', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--num', type=int, default=50)
    ap.add_argument('--out', default='compare.json')
    args = ap.parse_args()
    ok = run(args.tflite, args.onnx, args.eval_dir, args.num, args.out)
    if not ok:
        print('(exit 3)', flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
