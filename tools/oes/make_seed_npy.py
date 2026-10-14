#!/usr/bin/env python3
"""预置 onnx2tf 1.x 的测试图种子文件（免下载）.

onnx2tf 1.x 对一切 4D/batch<=20/last-dim=3 的模型，在转换后强制跑逐 OP
精度复核，测试图来自 download_test_image_data()：优先读 CWD 下的
calibration_image_sample_data_20x128x128x3_float32.npy，不存在才去 GitHub
下载（CI 里下载到的内容非法，直接崩）。本脚本用本次任务的真实校准图
生成同名同格式种子（20,128,128,3 float32，0~1），复核照常执行，
数据源从"网上下载"换成"本次校准集"，日志与报告中明示。

用法：
  python tools/oes/make_seed_npy.py --calib-dir calib --out \
      calibration_image_sample_data_20x128x128x3_float32.npy
"""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=20)
    args = ap.parse_args()
    import glob
    files = sorted(glob.glob(os.path.join(args.calib_dir, '*')))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) >= args.count, 'need %d images, have %d' % (
        args.count, len(files))
    arr = np.zeros((args.count, 128, 128, 3), dtype=np.float32)
    for i, f in enumerate(files[:args.count]):
        img = Image.open(f).convert('RGB').resize((128, 128), Image.BILINEAR)
        arr[i] = np.asarray(img, dtype=np.float32) / 255.0
    np.save(args.out, arr)
    print('SEED WROTE %s shape=%s range=[%.3f,%.3f]' % (
        args.out, arr.shape, arr.min(), arr.max()), flush=True)


if __name__ == '__main__':
    main()
