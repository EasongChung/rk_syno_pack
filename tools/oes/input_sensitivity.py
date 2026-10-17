#!/usr/bin/env python3
"""输入敏感性门（och 第19轮新增的**阻塞性**验收判据）。

## 为什么必须有这道门
`concept_int8_tf.tflite` 通过了所有既有检查（真 INT8、dtype 直方图干净、
量化张量 94.1%、无 float32 假绿），**但它对任何输入都输出同一个常量**：

    in_zero    out_hash=0xefaf94f71a4b9f87
    in_all128  out_hash=0xefaf94f71a4b9f87
    in_all127  out_hash=0xefaf94f71a4b9f87
    ramp       out_hash=0xefaf94f71a4b9f87

⇒ **dtype 检查不足以判定模型可用。** 必须直接验证输出随输入变化。

## 根因（本脚本不修，仅检测）
TF MinMax 校准器对**从未被观测到**的张量保留初始值 min=FLT_MAX、
max=FLT_MAX；随后 min 被钳到 0，于是
    scale = (FLT_MAX - 0) / 255 = 1.3344405750530544e+36
int8 卷积 requant 乘数 = in_scale * w_scale / out_scale ≈ 0
⇒ 输出恒等于 zero_point ⇒ 全网络传播常量 ⇒ 与输入无关。

## 判据
喂 N 种明显不同的输入，输出哈希必须**两两不同**。
若出现相同 ⇒ exit 5（阻塞），并打印涉及的输入组合。

用法：
  python input_sensitivity.py --model foo.tflite
  退出码 0 = 通过；5 = 不通过（阻塞）
"""
import argparse
import hashlib
import sys


def fnv1a64(b):
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ref', default='tools/oes/models/model_float16.tflite',
                    help='float 参照模型；pass 空字符串则跳过')
    ap.add_argument('--eval-dir', default='eval')
    ap.add_argument('--n', type=int, default=50,
                    help='与 float 参照比对的图数')
    ap.add_argument('--min-cos', type=float, default=0.95)
    ap.add_argument('--min-top1', type=float, default=0.90)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    import numpy as np
    import regress as R

    it = R.load_interp(args.model)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    shape = [int(x) for x in inp['shape']]
    dt = inp['dtype']
    n = int(np.prod(shape))

    # 构造 N 种"明显不同"的输入。用该输入张量真实的 dtype 域，
    # 避免把非法值塞进去导致 set_tensor 报错（那会掩盖真正的常量输出问题）。
    if dt == np.int8:
        variants = [('all_0', np.zeros(shape, np.int8)),
                    ('all_-128', np.full(shape, -128, np.int8)),
                    ('all_+127', np.full(shape, 127, np.int8)),
                    ('ramp', (np.arange(n) % 256 - 128)
                     .astype(np.int8).reshape(shape))]
    elif dt == np.uint8:
        variants = [('all_0', np.zeros(shape, np.uint8)),
                    ('all_255', np.full(shape, 255, np.uint8)),
                    ('ramp', (np.arange(n) % 256).astype(np.uint8)
                     .reshape(shape))]
    else:
        variants = [('all_0', np.zeros(shape, np.float32)),
                    ('all_1', np.ones(shape, np.float32)),
                    ('all_-1', np.full(shape, -1.0, np.float32)),
                    ('ramp', (np.arange(n) % 512 / 256.0 - 1.0)
                     .astype(np.float32).reshape(shape))]

    print('=== input-sensitivity gate: %s ===' % args.model)
    print('  input dtype=%s shape=%s  variants=%d'
          % (dt.__name__, shape, len(variants)))

    hashes = {}
    raw_out = {}
    for name, arr in variants:
        it.set_tensor(inp['index'], arr)
        it.invoke()
        o = it.get_tensor(out['index'])
        h = fnv1a64(o.tobytes())
        hashes[name] = h
        raw_out[name] = (o.min(), o.max(), float(np.mean(o.astype(np.float64))))
        print('  %-10s out_hash=0x%016x  min=%s max=%s mean=%.4f'
              % (name, h, o.min(), o.max(), raw_out[name][2]))

    uniq = len(set(hashes.values()))
    print('  distinct hashes: %d / %d' % (uniq, len(hashes)))
    if uniq <= 1:
        print('  => VERDICT: CONSTANT OUTPUT - model ignores its input.')
        print('     BLOCKING: must not be delivered or deployed.')
        print('     (root cause seen here: tensors whose scale is')
        print('      FLT_MAX/255 = 1.3344405750530544e+36, produced by')
        print('      TF MinMax when a tensor is NEVER OBSERVED during')
        print('      calibration -> requant multiplier ~= 0 -> output')
        print('      collapses to zero_point -> constant everywhere.)')
        return 5
    if uniq < len(hashes):
        print('  => VERDICT: PARTIAL - some inputs collide (uniq=%d of %d)'
              % (uniq, len(hashes)))
        if args.verbose:
            seen = {}
            for k, v in hashes.items():
                seen.setdefault(v, []).append(k)
            for v, ks in seen.items():
                if len(ks) > 1:
                    print('     collide 0x%016x: %s' % (v, ks))
        return 5
    print('  => VERDICT: PASS - output responds to input.')
    return _ref_check(args, it)


def _ref_check(args, it):
    """与 float 参照逐图比对（och 第20轮要求：这才是**充分**判据）。

    为什么不能只靠「饱和率」判死：
    本模型输出层 scale=0.00390625 / zp=-128，而 float 输出 max 仅 0.3157、
    mean 0.0017 —— **绝大多数真值远小于半个 scale**，量化后必然舍入到
    -128。所以「输出饱和率高」本身**不能**证明模型坏了；真正的失败信号是
    top1 不一致 + 同图 cos 低。饱和率只作为诊断打印，不作硬阈值。
    """
    import glob
    if not args.ref:
        print('  (no --ref given; float-reference check skipped)')
        return 0

    files = sorted(glob.glob(os.path.join(args.eval_dir, '*')))
    files = [f for f in files
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:args.n]
    if not files:
        print('  !! no eval images under %s' % args.eval_dir)
        return 5

    import regress as R
    rit = R.load_interp(args.ref)
    cos_list, agree = [], 0
    print()
    print('  --- vs float reference (%d imgs) ---' % len(files))
    for fp in files:
        b = open(fp, 'rb').read()
        rv = R.run_once(rit, b)
        qv = R.run_once(it, b)
        cos_list.append(R.cos(rv, qv))
        if R.topk(rv, 1)[0] == R.topk(qv, 1)[0]:
            agree += 1
    m = sum(cos_list) / len(cos_list)
    t1 = float(agree) / len(files)
    print('  cos_mean=%.4f  top1=%.4f  n=%d' % (m, t1, len(files)))
    if m < args.min_cos or t1 < args.min_top1:
        print('  => VERDICT: FAIL vs float reference '
              '(need cos>=%.2f top1>=%.2f)' % (args.min_cos, args.min_top1))
        return 5
    print('  => VERDICT: PASS vs float reference')
    return 0


if __name__ == '__main__':
    sys.exit(main())
