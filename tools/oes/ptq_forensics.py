#!/usr/bin/env python3
"""PTQ 崩坏法医：三段判别，定位激活 scale 高估 ~46 倍的来源。

## 为什么要它
MLIR PTQ 与 ORT QDQ 两条独立量化路线全崩（cos≈0.0000），pre-rewrite 对照双份复制
证明与图改写无关。法医查到 ORT 产出的激活 scale 与真实分布差约 46 倍：

    Relu__53:0   CI 覆盖上界 812.4   真实幅度 [0, 17.7]   ->  45.8x
    Relu__68:0   CI 覆盖上界 1633.0  真实幅度 [0, 27.5]   ->  59.5x

och 验算：uint8/int8 的 zp 语义差异只解释约 2 倍，余下约 23 倍必须来自图集内容。
且偏差随深度单调收敛（46x -> 2.4x -> 1.28x），是离群图被深层放大的特征，
而非量化器版本 bug 的特征（后者应全局一致）。

## 三段设计（第1段即判决点）
  第1段 float 图逐图 max（float 侧，与量化无关）
        -> 若某图 max 冲到 ~800：H1 离群图，可指名道姓
        -> 若全图 max 只有 ~30    ：H2 量化器在本图上失效
  第1b MinMax + 全量校准集，dump 激活 scale/zp/覆盖上界
  第2段 entropy + 同一批图，对比第1b
  第3段 MinMax + 清洗后图集（剔除离群图），验证清洗效果

## 为什么要用子模型
ORT 1.31 拒绝按名取中间张量（连 trivial 图也拒，是环境行为非图问题）。
唯一可靠办法是 onnx.utils.extract_model 切子模型，本地已验证。

用法：
  python ptq_forensics.py --float-onnx work/concept_split.onnx \
      --calib-dir calib --num-calib 300 --outdir out
产物：out/ptq_forensics.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 监控点：跨深度分布的 5 个 ReLU 输出（含 CI 崩坏最重的 Relu__53/68）
MONITOR = ['Relu__36:0', 'Relu__53:0', 'Relu__68:0', 'Relu__74:0',
           'Relu__96:0']
VERDICT_KEY = 'Relu__53:0'   # CI 已知覆盖上界 812、真实约 17.7
OUTLIER_K = 3.0              # 清洗阈值：mag > K * median 判离群


def calib_files(calib_dir, num):
    """复刻 ort_quant.Reader 的选取顺序，必须一致，否则第1段不是判决点。"""
    files = sorted(glob.glob(os.path.join(calib_dir, '*')))
    return [f for f in files
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]


class ListReader:
    """按文件列表流式提供校准数据（不缓存全量，避免 CI 内存峰值）。

    预处理逐字复制 ort_quant.Reader（NET/MEAN/SCALE 从该模块 import，
    保证与服务端、与 ort_quant 主流程逐位一致）。
    """

    def __init__(self, files, input_name):
        from ort_quant import NET
        self.files = files
        self.input_name = input_name
        self.size = NET
        self.i = 0

    @staticmethod
    def _load(path, size):
        from PIL import Image
        from ort_quant import MEAN, NET, SCALE
        img = Image.open(path).convert('RGB').resize(
            (NET, NET), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(),
                            dtype=np.uint8).astype(np.float32)
        return ((arr - MEAN) * SCALE).reshape(1, NET, NET, 3)

    def get_next(self):
        # 必须直接返回 dict/None，不能 yield（ORT 内部对返回值调 .keys()）。
        if self.i >= len(self.files):
            return None
        d = {self.input_name: self._load(self.files[self.i], self.size)}
        self.i += 1
        return d

    def rewind(self):
        self.i = 0

    def __len__(self):
        return len(self.files)


def build_submodel(float_onnx, points, out_path):
    """切子模型（一次多输出），返回图输入名。"""
    import onnx
    from onnx import utils as onnx_utils
    inp = onnx.load(float_onnx).graph.input[0].name
    try:
        onnx_utils.extract_model(float_onnx, out_path, [inp], points)
    except TypeError:          # 老签名：直接返回模型对象
        em = onnx_utils.extract_model(float_onnx, [inp], points)
        onnx.save(em, out_path)
    m = onnx.load(out_path)
    m.ir_version = 8           # 与本地已验证形态保持一致
    onnx.save(m, out_path)
    return inp


def stage1_per_image(float_onnx, files, workdir):
    """float 侧逐图统计监控张量的最大激活，定位离群图。"""
    import onnxruntime as ort
    sub = os.path.join(workdir, 'forensics_sub.onnx')
    iname = build_submodel(float_onnx, MONITOR, sub)
    sess = ort.InferenceSession(sub, providers=['CPUExecutionProvider'])
    real_in = sess.get_inputs()[0].name

    per = []
    for f in files:
        from PIL import Image
        from ort_quant import MEAN, NET, SCALE
        img = Image.open(f).convert('RGB').resize((NET, NET), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(),
                            dtype=np.uint8).astype(np.float32)
        x = ((arr - MEAN) * SCALE).reshape(1, NET, NET, 3).astype(np.float32)
        outs = sess.run(MONITOR, {real_in: x})
        pm = {p: float(np.abs(o).max()) for p, o in zip(MONITOR, outs)}
        per.append({'file': os.path.basename(f),
                    'mag': max(pm.values()), 'per_point': pm})

    agg = {}
    for p in MONITOR:
        v = np.array([r['per_point'][p] for r in per])
        agg[p] = {'median': float(np.median(v)),
                  'p90': float(np.percentile(v, 90)),
                  'p99': float(np.percentile(v, 99)),
                  'max': float(v.max())}
    mag = np.array([r['mag'] for r in per])
    order = np.argsort(mag)[::-1]
    return {
        'n_images': len(per),
        'mag_median': float(np.median(mag)),
        'mag_p90': float(np.percentile(mag, 90)),
        'mag_p99': float(np.percentile(mag, 99)),
        'mag_max': float(mag.max()),
        'per_point': agg,
        # 完整逐图记录（清洗需要全量，不能只留 top-N）
        'per_file': [{'file': per[i]['file'], 'mag': float(mag[i])}
                     for i in range(len(per))],
        'top_outliers': [{'file': per[i]['file'], 'mag': float(mag[i]),
                          'dominant': max(per[i]['per_point'],
                                          key=per[i]['per_point'].get)}
                         for i in order[:8]],
    }


def quantize(float_onnx, out_onnx, files, method):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    m = {'minmax': CalibrationMethod.MinMax,
         'entropy': CalibrationMethod.Entropy}[method]
    import onnx
    iname = onnx.load(float_onnx).graph.input[0].name
    quantize_static(float_onnx, out_onnx, ListReader(files, iname),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    reduce_range=False,
                    nodes_to_exclude=[],
                    calibrate_method=m)
    assert os.path.exists(out_onnx), 'quantize produced nothing: %s' % method


def dump_act_scales(qdq_path):
    """dump 监控张量的激活 scale / zero_point / 实际覆盖上界。

    ONNX QuantizeLinear 输入顺序：(x, y_scale, y_zero_point[, axis])。
    覆盖上界 = (qmax - zp) * scale，int8 激活 qmax=127。
    """
    import onnx
    from onnx import numpy_helper
    m = onnx.load(qdq_path)
    ini = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type != 'QuantizeLinear' or n.input[0] not in MONITOR:
            continue
        sc = float(np.ravel(ini[n.input[1]]).astype(np.float64)[0])
        zp = float(np.ravel(ini[n.input[2]]).astype(np.float64)[0])
        out[n.input[0]] = {'scale': sc, 'zp': zp,
                           'covered_hi': (127.0 - zp) * sc}
    return out


def report_stage(tag, scales, per_point=None):
    print('--- %s ---' % tag, flush=True)
    rows = {}
    for p in MONITOR:
        s = scales.get(p)
        if not s:
            print('  %-16s (no Q node)' % p, flush=True)
            continue
        line = '%-16s scale=%.6g zp=%.1f covered_hi=%.3f' % (
            p, s['scale'], s['zp'], s['covered_hi'])
        if per_point and p in per_point:
            line += '  real_max=%.3f  ratio=%.2fx' % (
                per_point[p]['max'], s['covered_hi'] / per_point[p]['max'])
        print('  ' + line, flush=True)
        rows[p] = s
    return rows


def verdict(per_stats, scales_minmax):
    """判决 H1 / H2 / H3。全部依据 float 侧实测与 ORT 校准结果的比值。"""
    real = per_stats['per_point'].get(VERDICT_KEY, {}).get('max')
    cov = scales_minmax.get(VERDICT_KEY, {}).get('covered_hi')
    if not real or not cov:
        return {'key': VERDICT_KEY, 'verdict': 'INCONCLUSIVE（监控点缺失）'}
    ratio = cov / real
    if ratio < 2.0:
        v = ('H3 环境差异：ORT 校准与真实幅度自洽（ratio=%.2fx），'
             '此前的 46x 在本环境不可复现' % ratio)
    elif real > 100.0:
        v = ('H1 校准集含离群图：float 侧实测 max=%.1f 已可复现，'
             '指名图像见 top_outliers' % real)
    else:
        v = ('H2 量化器在本图上失效：float 侧正常（max=%.1f），'
             '但 ORT 校准给出 covered_hi=%.1f（ratio=%.1fx）' % (real, cov, ratio))
    return {'key': VERDICT_KEY, 'real_max': real, 'covered_hi': cov,
            'ratio': ratio, 'verdict': v}


def load_report(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_report(path, rep):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print('FORENSICS REPORT -> %s' % path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--float-onnx', required=True)
    ap.add_argument('--calib-dir', default='calib')
    ap.add_argument('--num-calib', type=int, default=300)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--outlier-k', type=float, default=OUTLIER_K)
    ap.add_argument('--stages', default='1,2,3',
                    help='逗号分隔阶段。必须分进程跑：entropy 校准在本图'
                         '（191 个大张量）内存峰值极高，同进程连跑会被 OOM kill')
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(',') if s.strip()}
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    report_path = os.path.join(args.outdir, 'ptq_forensics.json')
    rep = load_report(report_path)

    files = calib_files(args.calib_dir, args.num_calib)
    assert files, 'calib dir empty: %s' % args.calib_dir
    print('FORENSICS stages=%s float=%s calib=%d' % (
        sorted(stages), args.float_onnx, len(files)), flush=True)

    import gc

    # ---- 第1段：float 侧逐图 max（判决点，与量化无关）+ MinMax ----
    if '1' in stages:
        per = stage1_per_image(args.float_onnx, files, args.workdir)
        print('PERIMAGE n=%d median=%.2f p90=%.2f p99=%.2f max=%.2f' % (
            per['n_images'], per['mag_median'], per['mag_p90'],
            per['mag_p99'], per['mag_max']), flush=True)
        for o in per['top_outliers'][:5]:
            print('  TOP %-22s mag=%9.2f  dominant=%s' % (
                o['file'], o['mag'], o['dominant']), flush=True)
        rep['stage1_per_image'] = per
        q_minmax = os.path.join(args.workdir, 'qdq_minmax.onnx')
        quantize(args.float_onnx, q_minmax, files, 'minmax')
        rep['stage1_minmax'] = report_stage(
            'STAGE1 MinMax (全量)', dump_act_scales(q_minmax), per['per_point'])
        rep['args'] = vars(args)
        gc.collect()
        save_report(report_path, rep)

    # ---- 第2段：entropy + 同一批图（对照） ----
    if '2' in stages:
        pp = rep.get('stage1_per_image', {}).get('per_point')
        q_ent = os.path.join(args.workdir, 'qdq_entropy.onnx')
        quantize(args.float_onnx, q_ent, files, 'entropy')
        rep['stage2_entropy'] = report_stage(
            'STAGE2 Entropy (同批)', dump_act_scales(q_ent), pp)
        gc.collect()
        save_report(report_path, rep)

    # ---- 第3段：清洗后图集 + MinMax（验证清洗效果） ----
    if '3' in stages:
        per = rep.get('stage1_per_image')
        assert per and 'per_file' in per, 'stage3 依赖 stage1 的逐图结果'
        pp = per['per_point']
        thresh = per['mag_median'] * args.outlier_k
        drop = {r['file'] for r in per['per_file'] if r['mag'] > thresh}
        keep = [f for f in files if os.path.basename(f) not in drop]
        dropped = [{'file': r['file'], 'mag': r['mag']}
                   for r in per['per_file'] if r['mag'] > thresh]
        print('CLEAN: threshold=%.2f dropped=%d kept=%d' % (
            thresh, len(dropped), len(keep)), flush=True)
        for d in sorted(dropped, key=lambda r: -r['mag'])[:5]:
            print('  DROP %-22s mag=%.2f' % (d['file'], d['mag']), flush=True)
        s_clean = {}
        if dropped:
            q_clean = os.path.join(args.workdir, 'qdq_clean.onnx')
            quantize(args.float_onnx, q_clean, keep, 'minmax')
            s_clean = report_stage('STAGE3 MinMax (清洗后)',
                                   dump_act_scales(q_clean), pp)
            gc.collect()
        rep['stage3_clean'] = {'threshold': thresh, 'dropped': dropped,
                               'n_kept': len(keep), 'scales': s_clean}
        save_report(report_path, rep)

    # ---- 判决（任一阶段跑完即可给结论） ----
    per = rep.get('stage1_per_image')
    s_mm = rep.get('stage1_minmax')
    if per and s_mm:
        rep['verdict'] = verdict(per, s_mm)
        print('VERDICT %s' % rep['verdict']['verdict'], flush=True)
        save_report(report_path, rep)


if __name__ == '__main__':
    main()