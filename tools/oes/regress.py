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
    """加载 TFLite 解释器，兼容全整数量化（int8 I/O）模型。

    注意：不能用 convert_to_int8.make_interpreter 加载 int8 输入的模型
    —— 它内部用 FLOAT32 试跑 set_tensor，对全整数量化模型会报
    "Cannot set tensor: Got value of type FLOAT32 but expected type INT8"
    （CI run 38908301589 实测），且它抛 RuntimeError 而非 ImportError，
    原 fallback 因此不生效。故先探测输入 dtype，是整型就直接用
    tf.lite.Interpreter（不做试跑）。
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import tflite as _t
        raw = open(path, 'rb').read()
        mo = _t.Model.GetRootAsModel(raw, 0)
        g = mo.Subgraphs(0)
        tin = int(g.Tensors(g.Inputs(0)).Type())
        if tin in (3, 9):                       # UINT8 / INT8
            import tensorflow as tf
            # 必须禁用默认 delegate：XNNPACK 对该模型的 prepare 会失败
            # （CI run 38909737996 实测 "failed to create XNNPACK
            #  runtimeNode number 202 ... failed to prepare"），
            # 导致 Interpreter 构造整体失败。
            it = tf.lite.Interpreter(
                model_path=path,
                experimental_op_resolver_type=tf.lite.experimental
                .OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            it.allocate_tensors()
            print('load_interp: int8-IO model (type=%d), '
                  'no-delegate Interpreter' % tin, flush=True)
            return it
    except Exception as e:
        print('load_interp probe failed (%s); falling back' % str(e)[:90],
              flush=True)
    try:
        from convert_to_int8 import make_interpreter
        return make_interpreter(path)['interp']
    except Exception:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=path)
        it.allocate_tensors()
        return it


def load_onnx(path):
    """加载 ONNX QDQ 模型，返回 (sess, input_name, output_name)。

    与 TFLite 侧共用同一套预处理、同一个 ref、同一批图，
    确保「同集对照」是真的对照。

    背景（och 第11轮指出的最大风险）：此前 ORT QDQ 的 0.9621 是
    `--eval-num 50` 评的，TFLite 的 0.8879 是 regress 200 张评的，
    **两个数不在同一集合上，不能相减**。本函数让 regress 能直接
    评 ONNX 侧，从而消除这个不可比性。
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    i0 = sess.get_inputs()[0]
    o0 = sess.get_outputs()[0]
    print('load_onnx: %s in=%s%s out=%s%s'
          % (os.path.basename(path), i0.name, i0.shape, o0.name, o0.shape),
          flush=True)
    return sess, i0.name, o0.name


def run_once_onnx(sess, in_name, out_name, image_bytes):
    """与 run_once 逐字一致的预处理，只是执行器换成 ORT。"""
    from PIL import Image
    import numpy as np
    shape = sess.get_inputs()[0].shape
    net = int(shape[2]) if len(shape) == 4 else NET
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(
        (net, net), Image.BILINEAR)
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    f = ((arr.astype(np.float32) - MEAN) * SCALE).reshape([1, net, net, 3])
    o = sess.run([out_name], {in_name: f})[0]
    return o.flatten().astype(float)


DEBUG = {'n': 0}


def preprocess_input(inp, image_bytes):
    """按输入张量的 dtype / 量化参数，生成喂给 set_tensor 的数组。

    **单一真相源**：regress 与导出脚本（export_tensor_bytes.py）都调用
    本函数，保证「CI 侧参考字节」与「设备侧输入字节」完全同源。
    och 第18轮明确要求：宁可多传一个文件，不要多一个变量 ——
    重新实现一遍预处理正是最可能出现「两边都以为自己对齐了」的环节。
    """
    from PIL import Image
    import numpy as np
    shape = [int(x) for x in inp['shape']]
    net = shape[2] if len(shape) == 4 else NET
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(
        (net, net), Image.BILINEAR)
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    if inp['dtype'] == np.uint8:
        return arr.reshape(shape)
    if inp['dtype'] == np.int8:
        # int8 输入（TFLiteConverter 全整数量化产物，in_type=9）：
        # 必须按该张量的 scale/zero_point 把 float 归一化输入量化成 int8，
        # 否则会把 float32 塞进 int8 tensor 直接报类型错（exit=1）。
        isc, izp = inp.get('quantization', (0.0, 0))
        f = ((arr.astype(np.float32) - MEAN) * SCALE).reshape(shape)
        if isc:
            return np.clip(np.round(f / isc) + izp, -128,
                           127).astype(np.int8)
        return f.astype(np.int8)
    return (((arr.astype(np.float32) - MEAN) * SCALE)
            .reshape(shape).astype(np.float32))


def run_once(interp, image_bytes):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    t = preprocess_input(inp, image_bytes)
    interp.set_tensor(inp['index'], t)
    interp.invoke()
    raw = interp.get_tensor(out['index']).flatten().astype(float)
    # 反量化。不能用 isinstance(q, tuple) 判断：TF 各版本里
    # details['quantization'] 可能是 tuple / list / ndarray，
    # 判断失败会导致整型输出不被反量化，直接拿 int8 原始值(-128..127)
    # 去和参考模型的 sigmoid 输出([0,1]) 比，cos 会低到 ~0.002
    # （CI run 38913214531 实测）。故直接按下标解包。
    q = out.get('quantization', None)
    v = raw
    if q is not None:
        try:
            qsc = float(q[0])
            qzp = float(q[1])
            if qsc:
                v = (raw - qzp) * qsc
        except (TypeError, IndexError, ValueError):
            pass
    if DEBUG['n'] < 2:
        DEBUG['n'] += 1
        print('DBG in: dtype=%s shape=%s quant=%s type=%s' % (
            inp['dtype'], shape, inp.get('quantization'), type(
                inp.get('quantization'))), flush=True)
        print('DBG in: fed range=[%.1f,%.1f] first8=%s' % (
            float(np.min(t)), float(np.max(t)),
            np.ravel(t)[:8].tolist()), flush=True)
        print('DBG out: dtype=%s quant=%s type=%s' % (
            out['dtype'], q, type(q)), flush=True)
        print('DBG out: raw  range=[%.1f,%.1f] first8=%s' % (
            raw.min(), raw.max(), np.round(raw[:8], 3).tolist()), flush=True)
        print('DBG out: deq  range=[%.4f,%.4f] first8=%s' % (
            v.min(), v.max(), np.round(v[:8], 4).tolist()), flush=True)
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
    ap.add_argument('--quant-onnx', action='store_true',
                    help='quant 侧是 ONNX QDQ 模型（而非 TFLite）。'
                         '与 TFLite 侧共用同一 ref / 同一批图 / 同一预处理，'
                         '用于消除「ORT 用 50 张、TFLite 用 200 张」'
                         '导致的不可比（och 第11轮）。'
                         'ONNX 侧不走 --require-int8：那是 TFLite 专有检查。')
    args = ap.parse_args()
    if args.require_int8 and args.quant_onnx:
        sys.exit('--require-int8 is TFLite-only; drop it with --quant-onnx')
    if args.require_int8:
        assert_int8(args.quant)
    ks = [int(x) for x in args.topk.split(',')]
    files = sorted(glob.glob(os.path.join(args.eval_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert files, 'eval empty: %s' % args.eval_dir
    ri = load_interp(args.ref)
    if args.quant_onnx:
        sess, in_name, out_name = load_onnx(args.quant)
    else:
        qi = load_interp(args.quant)
    agree = {k: 0 for k in ks}
    coss, drifts, fails = [], [], []
    for f in files:
        b = open(f, 'rb').read()
        rv = run_once(ri, b)
        qv = (run_once_onnx(sess, in_name, out_name, b)
              if args.quant_onnx else run_once(qi, b))
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