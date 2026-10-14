#!/usr/bin/env python3
"""张量剖面：跑 N 张图，记录每个张量反量化后的 (min,max,mean)。

用途：float 参考 vs fixed 逐张量比对，按拓扑序找 FIRST 分歧点
（移植是否生效、下个毒点在哪，一测便知）。

用法：
  python dump_regions.py --model out/concept_int8_fixed.tflite --calib-dir calib \\
      --num 3 --out profile_fixed.json [--uint8-input]
  float 模型用 --float-input（归一化输入）；量化模型 uint8 用原始像素。
"""
import argparse
import glob
import io
import json
import os

import numpy as np


def load_images(calib_dir, num):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(calib_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num]
    assert files, 'calib empty'
    out = []
    for f in files:
        img = Image.open(f).convert('RGB').resize((395, 395), Image.BILINEAR)
        out.append((os.path.basename(f),
                    np.frombuffer(img.tobytes(), dtype=np.uint8)))
    return out


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_to_int8 import make_interpreter
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--num', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--float-input', action='store_true')
    args = ap.parse_args()
    it = make_interpreter(args.model)['interp']
    ide = it.get_input_details()[0]
    infos = it.get_tensor_details()
    prof = {'model': args.model, 'tensors': []}
    for fname, raw in load_images(args.calib_dir, args.num):
        if args.float_input or ide['dtype'] == np.float32:
            t = (((raw.astype(np.float32) - 127.5) * 0.00784313725)
                 .reshape(1, 395, 395, 3))
        else:
            t = raw.reshape(1, 395, 395, 3)
        it.set_tensor(ide['index'], t.astype(ide['dtype']))
        it.invoke()
        row = {'file': fname, 'vals': {}}
        for info in infos:
            idx = info['index']
            try:
                v = it.get_tensor(idx).astype(np.float64)
            except Exception:
                continue
            raw_stats = [float(v.min()), float(v.max()), float(v.mean())]
            deq_stats = None
            q = info.get('quantization_parameters', None)
            if q is not None and q.get('scales') is not None and \
                    len(q['scales']) == 1 and \
                    str(info['dtype']) != "<class 'numpy.float32'>":
                sc = float(q['scales'][0])
                zp = float(q['zero_points'][0]) if len(q['zero_points']) else 0.0
                d = (v - zp) * sc
                deq_stats = [float(d.min()), float(d.max()), float(d.mean())]
            row['vals'][str(idx)] = {'raw': raw_stats, 'deq': deq_stats}
        prof['tensors'].append(row)
    # 附张量名表
    prof['names'] = {str(i['index']): i['name'] for i in infos}
    json.dump(prof, open(args.out, 'w'))
    print('PROFILE %s tensors=%d images=%d -> %s' % (
        args.model, len(infos), args.num, args.out), flush=True)


if __name__ == '__main__':
    main()
