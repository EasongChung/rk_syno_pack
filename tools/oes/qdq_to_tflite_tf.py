#!/usr/bin/env python3
"""Route A：QDQ ONNX -> TFLite，不过 onnx2tf（och 止损线第 2 步）。

## 为什么换路
三条 onnx2tf 路线全部试完（CI runs 38891479973 / 38898318514 / 38899791397
/ 38900946439），结论一致：`output_integer_quantized_tflite=True` 根本不
产出 `_full_integer_quant.tflite`。CI 实测产物目录只有三个文件：

    concept_ort_qdq_clean_float32.tflite            (24.1MB, 量化张量 0/336)
    concept_ort_qdq_clean_float16.tflite
    concept_ort_qdq_clean_dynamic_range_quant.tflite

三次 flatbuffer_export 分别对应 float32 / float16 / dynamic_range，唯独
没有全整数量化那一档 —— 它在 onnx2tf 自己的 INT8 校准阶段失败后被跳过。
源码级原因：onnx2tf 把输入转成 NCHW (1,395,3,395)，其校准分支要求
`model_input.shape[-1] in [3,4]`；用 keep_nwc_... 修正后不再报该错，
但 INT8 导出仍然缺失（CI run 38900946439 实测）。

## 本路线做法
直接用 tf.lite.TFLiteConverter 消费 float ONNX + 我们已经验证达标的
ORT 量化参数：
  1. ORT QDQ 已产出 cos=0.9569 / top1=0.8800（清洗校准集 + exclude-head 6）
  2. 用 TFLiteConverter.from_concrete_functions / from_saved_model 不适用，
     故用 from_keras 或直接把 ONNX 权重灌进 tf.Module 再 converter.convert()
  3. representative_dataset 用同一份清洗后校准集
  4. target_spec = TFLITE_BUILTINS_INT8, inference_input/output_type = int8

关键点：不试图"搬运"ORT 的 QDQ scale（那正是 onnx2tf 失败的地方），
而是让 TFLiteConverter 用我们的清洗校准集**自己**做一次 INT8 校准。
清洗是根因修复（covered_hi 812 -> 35），所以这里重算出的 scale 应当同样健康。

用法：
  python qdq_to_tflite_tf.py --float-onnx work/concept_split.onnx \
      --calib-dir calib --calib-num 100 --out out/concept_int8_tf.tflite
"""
import argparse
import glob
import os
import sys
import sys as _sys

import numpy as np

def fix_pad_onnx(src, dst):
    """把 3 输入 Pad（constant_value=0）改写为 2 输入 Pad。

    ## 为什么必须改
    两种 TFLite 量化器都在 Pad 上崩，但根因同一个：

      MLIR   : constant_values=0 -> 0/0 退化出 scale=1.334441e+36，
               沿图传染到 relu/Add -> PadV2 -> MaxPool2d（CI run 38940758759
               实测 6 个退化张量）。
      Legacy : constant_values 张量保持 FLOAT32，而 input 被量化成 INT8，
               TFLite 内核直接拒绝 prepare：
               "pad.cc:230 op_context.input->type !=
                op_context.constant_values->type (INT8 != FLOAT32)
                Node number 4 (PADV2) failed to prepare"。

    ONNX 规范里 Pad 的 constant_value 默认为 0，所以 2 输入 Pad 与 3 输入
    Pad（constant_value=0）语义完全等价；2 输入 Pad 经 onnx2tf 会生成
    2 输入 TF Pad -> TFLite 的 PAD 算子（而非 PADV2），既没有
    constant_values 类型冲突，也不给 MLIR 留退化常量。
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(src)
    g = m.graph
    used = set()
    for n in g.node:
        for i in n.input:
            used.add(i)
    remap = {}
    changed = 0
    for n in g.node:
        if n.op_type != 'Pad' or len(n.input) != 3:
            continue
        cv = n.input[2]
        # constant_value 必须存在且为 0
        val = None
        for init in g.initializer:
            if init.name == cv:
                val = numpy_helper.to_array(init)
                break
        if val is None or np.any(np.asarray(val) != 0):
            continue                      # 非 0 不能删，保留 3 输入
        n.input.remove(cv)
        if cv not in used:
            remap[cv] = True
        changed += 1
    if not changed:
        print('fix_pad: no 3-input Pad with constant_value=0 found',
              flush=True)
        return 0
    keep = [i for i in g.initializer
            if not (i.name in remap and i.name not in used)]
    del g.initializer[:]
    g.initializer.extend(keep)
    onnx.save(m, dst)
    print('fix_pad: rewrote %d Pad node(s) 3-input -> 2-input (%s)'
          % (changed, os.path.basename(dst)), flush=True)
    _sys.stdout.flush()
    return changed


TFL_NAME = {0: 'FLOAT32', 1: 'FLOAT16', 2: 'INT32', 3: 'UINT8', 4: 'INT64',
            6: 'BOOL', 7: 'INT16', 9: 'INT8'}


def build_calib_npy(calib_dir, num, out_dir, net=395):
    """生成 (N,395,395,3) float32 校准张量。

    预处理必须与服务端、与 ort_quant 逐位一致：
    resize(395, BILINEAR) + (v-127.5)/127.5。
    """
    from PIL import Image
    files = sorted(f for f in os.listdir(calib_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))[:num]
    assert files, 'calib dir empty: %s' % calib_dir
    arr = np.zeros((len(files), net, net, 3), dtype=np.float32)
    for i, f in enumerate(files):
        img = Image.open(os.path.join(calib_dir, f)) \
            .convert('RGB').resize((net, net), Image.BILINEAR)
        v = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
        arr[i] = ((v - 127.5) * 0.00784313725).reshape(net, net, 3)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'calib_tf_%d.npy' % len(files))
    np.save(p, arr)
    return p


def report(cand, tag):
    import collections
    import tflite
    raw = open(cand, 'rb').read()
    mo = tflite.Model.GetRootAsModel(raw, 0)
    g = mo.Subgraphs(0)
    nt = g.TensorsLength()
    nq = sum(1 for i in range(nt)
             if (q := g.Tensors(i).Quantization()) is not None
             and q.ScaleLength() > 0)
    tin = g.Tensors(g.Inputs(0))
    dt = collections.Counter(TFL_NAME.get(g.Tensors(i).Type(), '?')
                             for i in range(nt))
    codes = {}
    for i in range(mo.OperatorCodesLength()):
        oc = mo.OperatorCodes(i)
        codes[i] = oc.BuiltinCode() if oc.BuiltinCode() < 127 \
            else oc.DeprecatedBuiltinCode()
    ops = collections.Counter()
    for i in range(g.OperatorsLength()):
        ops[codes.get(g.Operators(i).OpcodeIndex())] += 1
    print('%s: in_type=%s quantized=%d/%d (%.1f%%) size=%d' % (
        tag, TFL_NAME.get(tin.Type(), tin.Type()), nq, nt,
        100.0 * nq / max(nt, 1), len(raw)))
    print('  dtype: %s' % dict(sorted(dt.items())))
    print('  ops:   %s' % dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]))

    # 退化 scale 检测（不依赖任何对齐，纯阈值判定）。
    # 背景：MLIR PTQ 时期发现 Pad 的 0/0 constant_values 会产出
    # scale=1.334441e+36 的退化张量并沿图传染。och 第7轮确认：该退化
    # 在 delegate 侧不会被自动拒绝（scale>0 成立，0/1.33e36 不溢出），
    # 风险在 etnaviv 转定点格式时溢出，故必须在导出侧处理。
    # 本地实测种子：idx=133 pseudo_qconst112 -> 139 relu/Add ->
    # 140 PadV2 -> 141 MaxPool2d，全部 1.334441e+36。
    degen = []
    for i in range(nt):
        t = g.Tensors(i)
        q = t.Quantization()
        if q is None or q.ScaleLength() == 0:
            continue
        for k in range(q.ScaleLength()):
            s = q.Scale(k)
            if s > 1e6 or s <= 0:
                nm = (t.Name().decode('utf-8', 'replace')
                      if t.Name() else '')
                degen.append((i, k, s, nm, t.Type()))
                break
    print('  DEGENERATE scale: %d tensor(s)' % len(degen), flush=True)
    for i, k, s, nm, ty in degen[:12]:
        print('    idx=%-4d scale=%.6e type=%s %s'
              % (i, s, TFL_NAME.get(ty, ty), nm[:52]), flush=True)

    # 未量化张量（och 第5节：fill_tensor 无空指针检查，20 个未量化
    # INT32 张量若被 fill_tensor 处理会段错误；需真机确认）
    nq_unquant = sum(1 for i in range(nt)
                     if g.Tensors(i).Type() in (3, 9)
                     and (g.Tensors(i).Quantization() is None
                          or g.Tensors(i).Quantization().ScaleLength() == 0))
    print('  int8/uint8 tensors WITHOUT quantization: %d '
          '(och: fill_tensor 无空指针检查，需真机确认)' % nq_unquant,
          flush=True)

    ok = nq > 0 and tin.Type() in (3, 9) and not degen
    print('  -> %s' % ('INT8 OK' if ok else
                      ('FLOAT (FAILED)' if nq == 0 else
                       'INT8 BUT DEGENERATE SCALE')))
    return ok, len(degen), nq_unquant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--float-onnx', required=True,
                    help='float ONNX（concept_split.onnx）。注意：TFLiteConverter '
                         '不能直接吃 QDQ ONNX，故用 float 图 + 自建校准集。')
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--calib-num', type=int, default=100)
    ap.add_argument('--workdir', default='work')
    ap.add_argument('--out', required=True)
    ap.add_argument('--input-name', default='inputs_0')
    ap.add_argument('--clean-k', type=float, default=3.0,
                    help='校准集离群剔除阈值（mag > k * median）')
    ap.add_argument('--int16', action='store_true')
    ap.add_argument('--per-tensor-weights', action='store_true',
                    help='把 conv/dwconv/FC 权重从 per-channel 改成 '
                         'per-tensor。动机：teflon 的 '
                         'tensor_quantization_supported() 要求 '
                         'scale->size==1 && zero_point->size==1，'
                         'per-channel 权重导致 Conv/DWConv 全部留 CPU。'
                         '本开关只改这一个变量，其余保持不变。'
                         '注意 per-tensor 通常精度更低，如实报数。')
    ap.add_argument('--old-quantizer', action='store_true',
                    help='退回旧版 TFLite 量化器（experimental_new_quantizer'
                         '=False）。用于验证退化 scale 是否 MLIR 特有：'
                         'MLIR 的 pseudo_qconst 标量退化 '
                         '(scale=1.334441e+36) 在旧量化器下可能不存在。')
    args = ap.parse_args()

    print('=== TA0: args parsed, onnx exists=%s ==='
          % os.path.exists(args.float_onnx), flush=True)
    _sys.stdout.flush()

    try:
        import tensorflow as tf
        print('=== TA1: tensorflow %s ===' % tf.__version__, flush=True)
        _sys.stdout.flush()
    except Exception as e:
        print('=== TA1 FAILED: %s ===' % e, flush=True)
        raise

    print('=== TA2: cleaning calib set (mandatory) ===', flush=True)
    _sys.stdout.flush()
    # 必须清洗：本路线是 TFLiteConverter 独立重新校准（不搬运 ORT 的
    # QDQ scale），若喂未清洗的 calib 目录，那张 mag=541.54 的脏图
    # （oes-calib-0014.jpg，正好在排序前 100 张内）会再次撑爆 scale。
    # CI run 38911268432 实测：未清洗 -> regress cos=0.009 / top1=0.005。
    import shutil as _sh
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ort_quant as _oq
    qdq_or_float = args.float_onnx
    keep = _oq.clean_calib_files(qdq_or_float, args.calib_dir,
                                 args.calib_num * 4, args.clean_k,
                                 os.path.join(args.workdir, 'calib'))
    tmpdir = os.path.join(args.workdir, 'calib_clean')
    if os.path.isdir(tmpdir):
        _sh.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)
    for f in keep[:args.calib_num]:
        _sh.copy(f, os.path.join(tmpdir, os.path.basename(f)))
    print('cleaned calib: %d images' % len(os.listdir(tmpdir)), flush=True)
    _sys.stdout.flush()

    npy = build_calib_npy(tmpdir, args.calib_num,
                         os.path.join(args.workdir, 'tf_calib'))
    calib = np.load(npy)
    print('=== TA2b: calib %s shape=%s range=[%.3f,%.3f] ===' % (
        npy, calib.shape, calib.min(), calib.max()), flush=True)
    _sys.stdout.flush()

    # --- ONNX -> tf.Module（保持 NHWC） ---
    # 用 onnx2tf 只做"ONNX -> Keras/SavedModel"这一步仍会踩它的坑，
    # 故这里改用最朴素可靠的方式：直接构造 tf.function 并调用 converter。
    # 路径：onnxruntime 读 float 图的中间结果不可行（ORT 拒取中间名），
    # 因此用 tf.lite.TFLiteConverter.from_concrete_functions 需要 concrete
    # function —— 由 tf.Module + tf.function 包装 float ONNX 无法直接得到。
    # 最终方案：用 onnx2tf 只产出 SavedModel（其已验证 cos=1.0 的强项），
    # 再由 TFLiteConverter 做 INT8 量化。
    print('=== TA3: producing SavedModel from float ONNX via onnx2tf ===',
          flush=True)
    _sys.stdout.flush()

    import sys as _s2
    _s2.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as _np
    _orig = _np.load

    def _pl(a, **k):
        k.setdefault('allow_pickle', True)
        return _orig(a, **k)

    _np.load = _pl
    try:
        import importlib
        _cf = importlib.import_module('onnx2tf.utils.common_functions')
        if hasattr(_cf, 'download_test_image_data'):
            _od = _cf.download_test_image_data

            def _safe(*a, **k):
                try:
                    return _od(*a, **k)
                except Exception:
                    r = _np.random.default_rng(0)
                    return (r.random((395, 395, 3)) * 255).astype(_np.uint8)

            _cf.download_test_image_data = _safe
    except Exception:
        pass

    from onnx2tf import convert
    # onnx2tf 1.29.24 无 argparse 层（main() 直接调 convert），
    # 故走 API 直调，参数用真实签名（已 AST 校验）。
    sm_dir = os.path.join(args.workdir, 'sm_for_tf')
    os.makedirs(sm_dir, exist_ok=True)

    # 先修 Pad（3 输入 -> 2 输入）。见 fix_pad_onnx docstring：
    # 两种量化器都在 Pad 上崩，MLIR 退化 / Legacy 类型冲突，根因同一。
    pad_fixed = os.path.join(args.workdir, 'concept_split_padsafe.onnx')
    print('=== TA3-pre: fixing Pad (3-input -> 2-input) ===', flush=True)
    try:
        fix_pad_onnx(args.float_onnx, pad_fixed)
        src_onnx = pad_fixed
    except Exception as e:
        print('fix_pad failed (%s); use original' % str(e)[:120], flush=True)
        src_onnx = args.float_onnx
    _sys.stdout.flush()

    print('=== TA3a: onnx2tf.convert -> SavedModel (NHWC kept) ===',
          flush=True)
    _sys.stdout.flush()
    try:
        convert(input_onnx_file_path=src_onnx,
                output_folder_path=sm_dir,
                output_signaturedefs=False,
                non_verbose=True,
                not_use_onnxsim=False,
                keep_nwc_or_nhwc_or_ndhwc_input_names=[args.input_name])
    except BaseException as e:                                # noqa: BLE001
        print('onnx2tf raised %s: %s' % (type(e).__name__, str(e)[:200]),
              flush=True)
        _sys.stdout.flush()

    # 不假设目录名（onnx2tf 依 output_signaturedefs / -osd 等参数会在
    # float32/ 与 saved_model.pb 同级之间变化），递归找 saved_model.pb。
    sm = ''
    hits = glob.glob(os.path.join(sm_dir, '**', 'saved_model.pb'),
                     recursive=True)
    if hits:
        sm = os.path.dirname(hits[0])
        print('=== TA3b: saved_model.pb at %s ===' % sm, flush=True)
    else:
        print('=== TA3b: NO saved_model.pb under %s ===' % sm_dir,
              flush=True)
        for root, dirs, files in os.walk(sm_dir):
            for f in files:
                p = os.path.join(root, f)
                print('    %s (%d bytes)' % (
                    os.path.relpath(p, sm_dir), os.path.getsize(p)),
                    flush=True)
        return 5
    _sys.stdout.flush()

    print('=== TA4: SavedModel ready, converting to INT8 TFLite ===',
          flush=True)
    _sys.stdout.flush()

    def rep_gen():
        for i in range(calib.shape[0]):
            yield [calib[i:i + 1]]

    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
        print('converter created from %s' % sm, flush=True)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.old_quantizer:
            # 换量化器实现（不是调参）：验证 1.334441e+36 退化是否为 MLIR 特有。
            conv.experimental_new_quantizer = False
            print('using LEGACY TFLite quantizer '
                  '(experimental_new_quantizer=False)', flush=True)
        conv.representative_dataset = rep_gen
        if getattr(args, 'per_tensor_weights', False):
            # 不凭印象断言开关名字：先列出该 TF 版本里所有相关属性，
            # 再按优先级尝试设置，并把实际生效的那个名字打出来。
            # （och 明确要求：以 CI 上的真实行为为准。）
            cands = [a for a in dir(conv)
                     if not a.startswith('__')
                     and ('per_channel' in a.lower()
                          or 'per_axis' in a.lower()
                          or 'disable' in a.lower())]
            print('PER-TENSOR: candidate attrs=%s' % cands, flush=True)
            applied = None
            for name in ('_experimental_disable_per_channel',
                         '_experimental_disable_per_channel_quantization',
                         'experimental_disable_per_channel'):
                if hasattr(conv, name):
                    try:
                        setattr(conv, name, True)
                        if getattr(conv, name) is True:
                            applied = name
                            break
                    except Exception as e:                        # noqa: BLE001
                        print('PER-TENSOR: set %s raised %s'
                              % (name, str(e)[:80]), flush=True)
            if applied is None:
                print('PER-TENSOR: !! NO WORKING SWITCH FOUND in this TF '
                      'version. The variant will still be produced, but if '
                      'weights stay per-channel that is a REAL finding, not '
                      'a silent failure.', flush=True)
            else:
                print('PER-TENSOR: applied %s=True' % applied, flush=True)
        if args.int16:
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT16]
            conv.inference_input_type = tf.int16
            conv.inference_output_type = tf.int16
        else:
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        blob = conv.convert()
    except BaseException as e:                                # noqa: BLE001
        import traceback
        print('TA4 FAILED: %s: %s' % (type(e).__name__, str(e)[:300]),
              flush=True)
        traceback.print_exc()
        _sys.stdout.flush()
        return 6

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'wb').write(blob)
    print('=== TA5: wrote %s (%d bytes) ===' % (
        args.out, os.path.getsize(args.out)), flush=True)
    ok, ndegen, nq_unq = report(args.out, 'tf-converter')
    return 0 if ok else 4


if __name__ == '__main__':
    sys.exit(main())