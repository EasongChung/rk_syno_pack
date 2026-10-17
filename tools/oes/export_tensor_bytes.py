#!/usr/bin/env python3
"""导出一张评测图的原始张量字节，供 och 在设备上验正确性。

## ⚠️ 两个基准，不要混（用户 2026-10-17 更正）

> 「CPU 跑的话直接使用原模型就可以，没必要用量化版，只有 NPU 才需要
>   转量化模型」

所以：

| 路径 | 模型 | 输入 | 输出 |
|---|---|---|---|
| **CPU（交付版）** | `tools/oes/models/model_float16.tflite`（原模型） | **FLOAT32** 1x395x395x3 = 1872300 B | **FLOAT32** 1x335 = 1340 B |
| NPU（在研） | `out/concept_int8_*.tflite`（量化版） | INT8 468075 B | INT8 335 B |

**本脚本对两者都能导出**，文件名带 dtype 后缀（`_f32` / `_i8`）以免混淆。
CPU 正确性验证应使用 **`_f32`** 那一对（原模型）；`_i8` 只服务 NPU 线。

## 为什么必须由本脚本导出
预处理链在 regress.py 里已有一份且经过验证。**重新实现一遍**，
正是最可能出现「两边都以为自己对齐了」的环节。故本脚本直接
import regress.preprocess_input —— **单一真相源**。

用法：
  # CPU 基准（原模型，交付用）
  python export_tensor_bytes.py \\
      --model tools/oes/models/model_float16.tflite \\
      --image eval/oes-eval-0000.jpg --outdir out/tensors

  # NPU 基准（量化模型）
  python export_tensor_bytes.py \\
      --model out/concept_int8_fixed.tflite \\
      --image eval/oes-eval-0000.jpg --outdir out/tensors
"""
import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--image', default=None,
                    help='默认取 eval/ 下第一张 oes-eval-*.jpg')
    ap.add_argument('--outdir', default='out/tensors')
    args = ap.parse_args()

    import numpy as np
    import regress as R

    img = args.image
    if img is None:
        cands = sorted(glob.glob('eval/oes-eval-*.jpg'))
        assert cands, 'no eval/oes-eval-*.jpg found'
        img = cands[0]
    assert os.path.exists(img), 'image not found: %s' % img
    stem = os.path.splitext(os.path.basename(img))[0]

    it = R.load_interp(args.model)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    print('model  : %s' % args.model)
    print('image  : %s' % img)
    print('input  : dtype=%s shape=%s quant=%s'
          % (inp['dtype'].__name__, list(inp['shape']),
             inp.get('quantization')))
    print('output : dtype=%s shape=%s quant=%s'
          % (out['dtype'].__name__, list(out['shape']),
             out.get('quantization')))

    b = open(img, 'rb').read()
    # 与 regress 同一份预处理函数（单一真相源）
    t = R.preprocess_input(inp, b)

    # dtype 决定文件名后缀，避免把 int8 字节和 float32 字节混为一谈
    # （用户更正：CPU 用原模型 = FLOAT32 I/O；量化模型只服务 NPU = INT8）
    import numpy as _np
    if t.dtype == _np.int8:
        sfx = 'i8'
    elif t.dtype == _np.uint8:
        sfx = 'u8'
    else:
        sfx = 'f32'
    in_bytes = t.tobytes()

    it.set_tensor(inp['index'], t)
    it.invoke()
    raw = it.get_tensor(out['index'])
    out_bytes = raw.tobytes()

    os.makedirs(args.outdir, exist_ok=True)
    num = stem.split('-')[-1]
    ip = os.path.join(args.outdir, 'input_%s_%s.bin' % (num, sfx))
    op = os.path.join(args.outdir, 'output_%s_%s.bin' % (num, sfx))
    with open(ip, 'wb') as f:
        f.write(in_bytes)
    with open(op, 'wb') as f:
        f.write(out_bytes)

    exp_in = int(_np.prod(inp['shape'])) * t.dtype.itemsize
    exp_out = int(_np.prod(out['shape'])) * raw.dtype.itemsize
    print()
    print('=== exported (%s path) ===' % sfx)
    print('%-34s %9d B (expect %d)  sha256=%s'
          % (ip, len(in_bytes), exp_in, sha256_of(ip)))
    print('%-34s %9d B (expect %d)  sha256=%s'
          % (op, len(out_bytes), exp_out, sha256_of(op)))
    print()
    print('input  stats: min=%s max=%s mean=%.4f'
          % (t.min(), t.max(), float(t.mean())))
    print('output stats: min=%s max=%s mean=%.4f'
          % (raw.min(), raw.max(), float(raw.mean())))

    if raw.dtype == _np.int8:
        sat_lo = int((raw == -128).sum())
        sat_hi = int((raw == 127).sum())
        print('output saturation: -128 x %d (%.1f%%), +127 x %d (%.1f%%)'
              % (sat_lo, 100.0 * sat_lo / raw.size,
                 sat_hi, 100.0 * sat_hi / raw.size))
        if sat_lo + sat_hi > 0.5 * raw.size:
            print('WARN: >50%% saturated -> preprocessing may still be off '
                  '(och saw this with synthetic input)')

    assert len(in_bytes) == exp_in, 'input byte count mismatch'
    assert len(out_bytes) == exp_out, 'output byte count mismatch'
    print('SIZES OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
