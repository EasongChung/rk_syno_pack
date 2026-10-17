#!/usr/bin/env python3
"""产物级验证：conv/dwconv 权重是 per-tensor 还是 per-channel。

## 为什么必须从产物验证，而不是看代码
`control_allq` 的教训：代码里写了什么 ≠ 产物里实际是什么
（我设置了 scale 但 zero_point 没进 flatbuffer，只看代码看不出来）。
所以本脚本直接读 flatbuffer 的量化参数数组长度。

## 判据（teflon 25.0.7 的 tensor_quantization_supported）
    return affine->scale->size == 1 && affine->zero_point->size == 1;
per-channel 权重（scale->size == out_channels > 1）会导致
Conv2d / DepthwiseConv2d 的判定失败 -> 该算子留 CPU。

用法：
  python check_pc.py --model foo.tflite [--expect per-tensor|per-channel]

退出码：0 = 符合 --expect；5 = 不符合（用于 CI 硬门）。
"""
import argparse
import collections
import sys

CONV, DWCONV, FC = 3, 4, 9
OPN = {CONV: 'CONV_2D', DWCONV: 'DEPTHWISE_CONV_2D', FC: 'FULLY_CONNECTED'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--expect', choices=['per-tensor', 'per-channel'],
                    default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    import tflite as T
    raw = open(args.model, 'rb').read()
    mo = T.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    codes = {}
    for i in range(mo.OperatorCodesLength()):
        oc = mo.OperatorCodes(i)
        bc = oc.BuiltinCode()
        codes[i] = bc if bc < 127 else oc.DeprecatedBuiltinCode()

    print('=== per-tensor/per-channel audit: %s ===' % args.model)
    print('  size=%d  tensors=%d  ops=%d'
          % (len(raw), g.TensorsLength(), g.OperatorsLength()))

    stats = collections.Counter()
    details = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        c = codes.get(op.OpcodeIndex())
        if c not in (CONV, DWCONV, FC):
            continue
        if op.InputsLength() < 2:
            continue
        w = op.Inputs(1)
        tt = g.Tensors(w)
        q = tt.Quantization()
        ns = 0 if q is None else q.ScaleLength()
        nz = 0 if q is None else q.ZeroPointLength()
        shape = [tt.Shape(j) for j in range(tt.ShapeLength())]
        kind = OPN[c]
        if ns == 0:
            stats['%s:NO-QUANT' % kind] += 1
        elif ns == 1:
            stats['%s:per-tensor' % kind] += 1
        else:
            stats['%s:per-channel' % kind] += 1
        details.append((i, kind, w, shape, ns, nz))

    for k in sorted(stats):
        print('  %-28s %d' % (k, stats[k]))

    if args.verbose:
        print('  --- detail (first 8) ---')
        for i, kind, w, shape, ns, nz in details[:8]:
            print('    op#%-4d %-18s t%-4d shape=%s scale_len=%d zp_len=%d'
                  % (i, kind, w, shape, ns, nz))

    n_pc = sum(v for k, v in stats.items() if k.endswith('per-channel'))
    n_pt = sum(v for k, v in stats.items() if k.endswith('per-tensor'))
    n_no = sum(v for k, v in stats.items() if k.endswith('NO-QUANT'))
    total = n_pc + n_pt + n_no
    print('  TOTAL conv/dwconv/fc weights=%d  per-tensor=%d  '
          'per-channel=%d  unquantized=%d' % (total, n_pt, n_pc, n_no))

    # teflon 判据：四个张量必须 scale->size==1 && zp->size==1
    print('  teflon 25.0.7 接管判据满足的权重: %d / %d' % (n_pt, total))
    print('  => 预计被 delegate 接管的 conv/dwconv 数: %d' % n_pt)

    if args.expect:
        ok = ((args.expect == 'per-tensor' and n_pc == 0 and n_pt > 0)
              or (args.expect == 'per-channel' and n_pc > 0))
        print('  EXPECT=%s -> %s' % (args.expect, 'PASS' if ok else 'FAIL'))
        return 0 if ok else 5
    return 0


if __name__ == '__main__':
    sys.exit(main())