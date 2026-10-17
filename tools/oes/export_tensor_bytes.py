#!/usr/bin/env python3
"""导出一张评测图的原始张量字节，供 och 在设备上验 CPU 正确性。

## 背景（och 第18轮）
用户裁决 CPU 版先交付。要交付就必须证明**设备上对真实图片输出正确**，
而 regress 跑在 CI（x86）上，**没有设备侧证据**。
och 用合成输入（ramp）无法验正确性 —— 实测 CPU 输出几乎全饱和
（mean=-127.68），据此完全无法判断真实图片上是否正确。

## 交付物（och 指定）
| 文件 | 内容 | 大小 |
|---|---|---|
| input_NNNN.bin | 该图**预处理后**的输入张量原始 INT8 字节 | 1x395x395x3 = 468075 B |
| output_NNNN_cpu.bin | 参考输出张量原始 INT8 字节 | 1x335 = 335 B |
外加两个文件的 SHA256。

## 为什么必须由本脚本导出
预处理链在 regress.py 里已有一份且经过验证。**重新实现一遍**，
正是最可能出现「两边都以为自己对齐了」的环节。故本脚本直接
import regress.preprocess_input —— **单一真相源**。

## 用哪个模型
默认用 `out/concept_int8_fixed.tflite`（legacy 量化器 + PADV2 修复），
因为它是**唯一 cos 可用的产物（0.8879）**。
注意 `concept_int8_tf.tflite`（MLIR）cos=0.0021 **本身崩坏**，
不能用它做参考 —— 若设备上跑的是它，两边输出必然对不上，
那是模型本身的问题，不是本对齐流程的问题。

用法：
  python export_tensor_bytes.py --model out/concept_int8_fixed.tflite \\
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
    assert t.dtype == np.int8, 'expected int8 input, got %s' % t.dtype
    in_bytes = t.tobytes()

    it.set_tensor(inp['index'], t)
    it.invoke()
    raw = it.get_tensor(out['index'])
    out_bytes = raw.tobytes()

    os.makedirs(args.outdir, exist_ok=True)
    ip = os.path.join(args.outdir, 'input_%s.bin' % stem.split('-')[-1])
    op = os.path.join(args.outdir, 'output_%s_cpu.bin'
                      % stem.split('-')[-1])
    with open(ip, 'wb') as f:
        f.write(in_bytes)
    with open(op, 'wb') as f:
        f.write(out_bytes)

    exp_in = int(np.prod(inp['shape']))
    exp_out = int(np.prod(out['shape']))
    print()
    print('=== exported ===')
    print('%-34s %8d B (expect %d)  sha256=%s'
          % (ip, len(in_bytes), exp_in, sha256_of(ip)))
    print('%-34s %8d B (expect %d)  sha256=%s'
          % (op, len(out_bytes), exp_out, sha256_of(op)))
    print()
    print('input  stats: min=%d max=%d mean=%.3f  uniq=%d'
          % (t.min(), t.max(), float(t.mean()),
             len(np.unique(t))))
    print('output stats: min=%d max=%d mean=%.3f  uniq=%d'
          % (raw.min(), raw.max(), float(raw.mean()),
             len(np.unique(raw))))

    # 饱和检查：och 说他的合成输入导致 CPU 输出几乎全饱和
    # （mean=-127.68, max=-55）—— 用真实图片必须不出现这种情况，
    # 否则说明预处理链路仍有问题。
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
