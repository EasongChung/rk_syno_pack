#!/usr/bin/env python3
"""首层发散定位：float ONNX vs QDQ ONNX 逐节点中间激活比对。

找 cos 从哪一层开始塌。若对照试验证明改写无辜、校准方法换了也崩，
这个定位就是下一步的指南针。

匹配策略：float 图节点输出名 -> QDQ 图同名张量；找不到则试
'_dequantized' 等常见后缀；再找不到则跳过并计入覆盖率。
按 float 图拓扑序报告首个 cos<0.99 的张量 + 全部 cos<0.9 清单。

用法：
  python diverge.py --ref work/concept_split.onnx \\
      --quant work/concept_ort_qdq.onnx --eval-dir eval --num 10 \\
      --out diverge.json
"""
import argparse
import glob
import os

import numpy as np


def load_images(eval_dir, num):
    from PIL import Image
    import io
    files = sorted(glob.glob(os.path.join(eval_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    assert files, 'eval empty'
    out = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((395, 395), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        t = (((arr - 127.5) * 0.00784313725).reshape(1, 395, 395, 3)
             .astype(np.float32))
        out.append((os.path.basename(f), t))
    return out


def qdq_tensor_names(qdq_path):
    import onnx
    m = onnx.load(qdq_path)
    names = set()
    for v in list(m.graph.input) + list(m.graph.output) + \
            list(m.graph.value_info):
        names.add(v.name)
    for n in m.graph.node:
        names.update(n.output)
    return names


def main():
    import json
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--quant', required=True)
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--num', type=int, default=10)
    ap.add_argument('--out', default='diverge.json')
    args = ap.parse_args()
    import onnx
    fm = onnx.load(args.ref)
    order = []  # (node_name, op, out_name) topo order
    for n in fm.graph.node:
        for o in n.output:
            order.append((n.name, n.op_type, o))
    avail = qdq_tensor_names(args.quant)
    pairs = []  # (node_name, op, float_name, q_name)
    skipped = 0
    for node_name, op, o in order:
        if o in avail:
            pairs.append((node_name, op, o, o))
            continue
        hit = None
        for cand in (o + '_dequantized', o + '_DequantizeLinear',
                     o + '_dequant'):
            if cand in avail:
                hit = cand
                break
        if hit is None:
            skipped += 1
            continue
        pairs.append((node_name, op, o, hit))
    print('float nodes=%d matched=%d skipped=%d' % (
        len(order), len(pairs), skipped), flush=True)
    sref = ort.InferenceSession(args.ref, providers=['CPUExecutionProvider'])
    sq = ort.InferenceSession(args.quant, providers=['CPUExecutionProvider'])
    rn = sref.get_inputs()[0].name
    qn = sq.get_inputs()[0].name
    imgs = load_images(args.eval_dir, args.num)
    # 一次 run 取全部候选（省时间）；若某名非法则整体失败→回退逐个试
    qnames = sorted({q for _, _, _, q in pairs})
    try:
        for _, t in imgs[:1]:
            sq.run(qnames, {qn: t})
        batched = True
    except Exception as e:
        print('batched fetch failed (%s), falling back per-tensor' % str(e)[:200],
              flush=True)
        batched = False
    agg = {}
    for fname, t in imgs:
        rv = dict(zip(
            [o for _, _, o, _ in pairs],
            sref.run([o for _, _, o, _ in pairs], {rn: t})))
        if batched:
            qv = dict(zip(qnames, sq.run(qnames, {qn: t})))
        else:
            qv = {}
            for _, _, _, q in pairs:
                try:
                    qv[q] = sq.run([q], {qn: t})[0]
                except Exception:
                    pass
        for node_name, op, o, q in pairs:
            a = np.asarray(rv[o], dtype=np.float64).flatten()
            if q not in qv:
                continue
            b = np.asarray(qv[q], dtype=np.float64).flatten()
            if a.shape != b.shape:
                agg.setdefault((node_name, op, o), []).append(('shape', a.shape,
                                                              b.shape))
                continue
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            c = float(np.dot(a, b) / denom)
            agg.setdefault((node_name, op, o), []).append(c)
    report = {'n_images': len(imgs), 'matched': len(pairs),
              'skipped': skipped, 'first_collapse': None, 'low': []}
    for node_name, op, o in [(n, p, x) for n, p, x, _ in pairs]:
        vals = agg.get((node_name, op, o), [])
        if not vals:
            continue
        if isinstance(vals[0], tuple):
            report['low'].append({'node': node_name, 'op': op, 'out': o,
                                  'issue': 'shape %s vs %s' % (vals[0][1],
                                                               vals[0][2])})
            if report['first_collapse'] is None:
                report['first_collapse'] = {'node': node_name, 'op': op,
                                            'out': o, 'issue': 'shape'}
            continue
        cm = float(np.mean(vals))
        if cm < 0.9:
            report['low'].append({'node': node_name, 'op': op, 'out': o,
                                  'cos_mean': round(cm, 4)})
        if cm < 0.99 and report['first_collapse'] is None:
            report['first_collapse'] = {'node': node_name, 'op': op, 'out': o,
                                        'cos_mean': round(cm, 4)}
    json.dump(report, open(args.out, 'w'), indent=2)
    print('coverage matched=%d/%d' % (len(pairs), len(order)), flush=True)
    print('first_collapse=%s' % (report['first_collapse'],), flush=True)
    print('low(cos<0.9): %d tensors' % len(report['low']), flush=True)
    for r in report['low'][:20]:
        print('  %s %s cos=%s' % (r['op'], r['out'][:80], r.get('cos_mean')),
              flush=True)


if __name__ == '__main__':
    main()
