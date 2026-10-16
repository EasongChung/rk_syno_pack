#!/usr/bin/env python3
"""ORT QDQ 预量化：绕开 MLIR PTQ，用 onnxruntime 成熟量化器做 INT8。

背景：TFLiteConverter MLIR PTQ 在本图上系统性崩坏（退化标量传染，
transplant/rebias/depad 全修仍红；-oiqt 则直接产出 float，无用）。
本路线：ORT quantize_static（QDQ，逐通道权重，INT8 激活）-> onnx2tf
仅做格式翻译（QDQ->TFLite 量化算子；onnx2tf 在此角色已证精确 cos=1.0）。
MLIR PTQ 全程不参与。

管线：
  repaired.onnx --ORT-QDQ--> qdq.onnx --validate-vs-float--> --onnx2tf--> INT8.tflite
输入：校准集 NHWC float32 [-1,1]（与服务端预处理一致）。

用法：
  python ort_quant.py --onnx work/concept_split.onnx --calib-dir calib \\
      --num-calib 300 --eval-dir eval --eval-num 50 \\
      --workdir work --outdir out
产物：out/concept_ortq.tflite（+ 中间 qdq.onnx）。
"""
import argparse
import glob
import os
import subprocess
import sys

NET = 395
MEAN = 127.5
SCALE = 0.00784313725


class Reader:
    """校准数据读取器。

    默认按 calib_dir 选取图片；传入 files 则由调用方决定用哪一批
    （校准集清洗场景，见 clean_calib_files）。
    流式读取，不把全量图像常驻内存 —— 300 张 395x395x3 float32 约 560MB，
    与 ORT 自身占用叠加后在 CI runner 上是内存峰值的主要来源之一。
    """

    def __init__(self, calib_dir, num, input_name, files=None):
        if files is None:
            files = sorted(glob.glob(os.path.join(calib_dir, '*')))
            files = [f for f in files
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
        assert len(files) >= min(num, 50), 'too few calib images'
        self.input_name = input_name
        self.files = files
        self.i = 0
        print('ort calib: %d images for %s' % (len(files), input_name),
              flush=True)

    def _load(self, path):
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert('RGB').resize((NET, NET), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        return ((arr - MEAN) * SCALE).reshape(1, NET, NET, 3)

    def get_next(self):
        # 注意：必须直接返回 dict/None，不能 yield（否则变成 generator 函数，
        # onnxruntime 内部对返回值调 .keys() 会 AttributeError）。
        if self.i >= len(self.files):
            return None
        d = {self.input_name: self._load(self.files[self.i])}
        self.i += 1
        return d

    def rewind(self):
        self.i = 0

    def __len__(self):
        return len(self.files)


def clean_calib_files(float_onnx, calib_dir, num_calib, k, workdir):
    """剔除校准集离群图，返回清洗后的文件列表。

    背景（CI run 38862205495 实测，300 张 picsum 图）：
        per-image 激活幅度 median=22.32 / p90=32.60 / p99=741.95 / max=1633
        p90/median=1.46x 而 p99/median=33x —— 断点干净，是少数几张脏图，
        不是连续长尾。MinMax 忠实把这些极值写进 scale，导致典型信号
        只占 int8 量程的 1/36（有效级数约 7 级），整网退化。
    清洗阈值 k×median：正常图（中位 22.32）与离群图（最低 675）之间有
    10 倍以上空隙，k=3 不存在误伤边界。
    """
    import ptq_forensics as pf
    os.makedirs(workdir, exist_ok=True)
    files = pf.calib_files(calib_dir, num_calib)
    per = pf.stage1_per_image(float_onnx, files, workdir)
    thresh = per['mag_median'] * k
    drop = {r['file'] for r in per['per_file'] if r['mag'] > thresh}
    keep = [f for f in files if os.path.basename(f) not in drop]
    dropped = sorted([r for r in per['per_file'] if r['mag'] > thresh],
                     key=lambda r: -r['mag'])
    print('CLEAN-CALIB: median=%.2f thresh=%.2f dropped=%d kept=%d/%d' % (
        per['mag_median'], thresh, len(dropped), len(keep), len(files)),
        flush=True)
    for d in dropped[:8]:
        print('  DROP %-22s mag=%.2f' % (d['file'], d['mag']), flush=True)
    assert len(keep) >= 50, 'cleaned calib too small: %d' % len(keep)
    return keep


def stage_ort_quant(onnx_path, qdq_path, calib_dir, num_calib,
                    calib_method='minmax', clean_calib=False, clean_k=3.0,
                    workdir='work'):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    import onnxruntime as ort
    method = {'minmax': CalibrationMethod.MinMax,
              'entropy': CalibrationMethod.Entropy,
              'percentile': CalibrationMethod.Percentile}[calib_method]
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    del sess
    files = None
    if clean_calib:
        files = clean_calib_files(onnx_path, calib_dir, num_calib, clean_k,
                                  workdir)
    quantize_static(
        model_input=onnx_path,
        model_output=qdq_path,
        calibration_data_reader=Reader(calib_dir, num_calib, iname,
                                       files=files),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=[],
        calibrate_method=method,
    )
    assert os.path.exists(qdq_path), 'ort quant produced nothing'
    print('ORT QDQ OK [%s] -> %s' % (calib_method, qdq_path), flush=True)


def stage_validate_qdq(ref_onnx, qdq_onnx, eval_dir, num):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import glob as gb
    import io
    files = sorted(gb.glob(os.path.join(eval_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    assert files, 'eval empty'
    sref = ort.InferenceSession(ref_onnx, providers=['CPUExecutionProvider'])
    sq = ort.InferenceSession(qdq_onnx, providers=['CPUExecutionProvider'])
    rn, qn = sref.get_inputs()[0].name, sq.get_inputs()[0].name
    coss, a1 = [], 0
    for f in files:
        img = Image.open(f).convert('RGB').resize((NET, NET), Image.BILINEAR)
        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        t = ((arr - MEAN) * SCALE).reshape(1, NET, NET, 3).astype(np.float32)
        rv = sref.run(None, {rn: t})[0].flatten()
        qv = sq.run(None, {qn: t})[0].flatten()
        denom = (np.linalg.norm(rv) * np.linalg.norm(qv)) + 1e-9
        c = float(np.dot(rv, qv) / denom)
        coss.append(c)
        r1 = int(np.argmax(rv))
        if r1 == int(np.argmax(qv)):
            a1 += 1
    cm, t1 = float(np.mean(coss)), a1 / len(files)
    print('QDQ-vs-FLOAT(onnx) n=%d cos=%.4f top1=%.4f' % (len(files), cm, t1),
          flush=True)
    assert cm >= 0.95 and t1 >= 0.85, 'ORT QDQ itself broken?!'
    print('QDQ VALIDATED', flush=True)


def stage_qdq_to_tflite(qdq_path, out_path):
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'run_onnx2tf.py'),
         '-i', qdq_path, '-o', out_path + '.qd',
         '--non_verbose', '-kt', 'inputs_0'],
        capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    print(r.stderr[-1500:], file=sys.stderr, flush=True)
    import glob as gb
    outs = gb.glob(os.path.join(out_path + '.qd', '*.tflite'))
    assert outs, 'qdq->tflite produced nothing'
    import shutil
    # 防 -oiqt 式假阳性：落盘前先验确为量化模型（量化张量数>0 且输入为整型）
    cand = sorted(outs, key=os.path.getsize, reverse=True)[0]
    import tflite
    _raw = open(cand, 'rb').read()
    _m = tflite.Model.GetRootAsModel(_raw, 0)
    _g = _m.Subgraphs(0)
    _nt = _g.TensorsLength()
    _nq = sum(1 for _i in range(_nt)
              if (_q := _g.Tensors(_i).Quantization()) is not None
              and _q.ScaleLength() > 0)
    _tin = _g.Tensors(_g.Inputs(0))
    # dtype 直方图：och 要求出件前同时报告量化张量数/总张量数、dtype 分布、
    # 文件大小。三者一致才能排除 -oiqt 式 float 假绿。
    from collections import Counter
    _dt = Counter(_g.Tensors(_i).Type() for _i in range(_nt))
    _names = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8',
              4: 'INT64', 5: 'STRING', 6: 'BOOL', 7: 'INT16', 9: 'INT8'}
    _hist = dict((_names.get(_k, 'type%d' % _k), _v)
                 for _k, _v in sorted(_dt.items()))
    print('qdq-tflite check: in_type=%d quantized=%d/%d (%.1f%%)' % (
        _tin.Type(), _nq, _nt, 100.0 * _nq / max(_nt, 1)), flush=True)
    print('  dtype histogram: %s' % _hist, flush=True)
    print('  file size: %d bytes' % os.path.getsize(cand), flush=True)
    # 真 INT8 必须：整型输入 + 存在整型张量 + 文件明显小于 float 版(12MB)
    assert _nq > 0 and _tin.Type() in (3, 9), \
        'qdq->tflite is NOT quantized (float false positive)!'
    assert _dt.get(9, 0) + _dt.get(3, 0) > 0, \
        'no integer tensors at all (float false positive)!'
    shutil.copy(cand, out_path)
    print('WROTE %s (ort-qdq via onnx2tf)' % out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='')
    ap.add_argument('--graph-file', default='work/graph_path.txt')
    ap.add_argument('--calib-dir', default='calib')
    ap.add_argument('--num-calib', type=int, default=300)
    ap.add_argument('--eval-dir', default='eval')
    ap.add_argument('--eval-num', type=int, default=50)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--calib-method', default='minmax',
                    choices=['minmax', 'entropy', 'percentile'])
    ap.add_argument('--clean-calib', action='store_true',
                    help='量化前剔除校准集离群图（见 clean_calib_files）')
    ap.add_argument('--clean-k', type=float, default=3.0,
                    help='离群判定阈值系数：mag > k * median（默认 3.0）')
    ap.add_argument('--tag', default='',
                    help="产物后缀：''=concept_ortq.tflite；'pre'=concept_ortq_pre.tflite 等")
    args = ap.parse_args()
    onnx_path = args.onnx or open(args.graph_file).read().strip()
    assert os.path.exists(onnx_path), onnx_path
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    suf = ('_' + args.tag) if args.tag else ''
    qdq_path = os.path.join(args.workdir, 'concept_ort_qdq%s.onnx' % suf)
    tflite_name = 'concept_ortq%s.tflite' % suf
    stage_ort_quant(onnx_path, qdq_path, args.calib_dir, args.num_calib,
                    args.calib_method, clean_calib=args.clean_calib,
                    clean_k=args.clean_k, workdir=args.workdir)
    stage_validate_qdq(onnx_path, qdq_path, args.eval_dir, args.eval_num)
    stage_qdq_to_tflite(qdq_path, os.path.join(args.outdir, tflite_name))
    print('ORTQ DONE [%s]' % (args.tag or 'default'), flush=True)


if __name__ == '__main__':
    main()