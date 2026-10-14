#!/usr/bin/env python3
"""onnx2tf 包装运行器。

1. numpy>=1.24 下 np.load 默认 allow_pickle=False 会炸掉 onnx2tf 内部的
   .npy 加载 → 默认改回 True，让真实错误透出来。
2. onnx2tf 转换后的精度复核要从网上下载测试图（download_test_image_data）；
   该下载在 CI 里不可用（内容非 pickle）。打补丁：先试原函数，
   失败则用本地合成图代替 —— 复核本身（ONNX vs TF 同输入比对）照常执行，
   只是输入源从"下载"换成"本地合成"，会在日志中明示。
   （若复核逻辑要求特定形状，报错会说明，届时再按报错调整。）

用法：python tools/oes/run_onnx2tf.py [onnx2tf 原生参数...]
"""
import importlib
import runpy
import sys

import numpy as _np

_orig_load = _np.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_load(*args, **kwargs)


_np.load = _patched_load


def _patch_test_image_download():
    try:
        mod = importlib.import_module('onnx2tf.utils.common_functions')
    except Exception as e:
        print('no common_functions module: %s' % e, flush=True)
        return
    if not hasattr(mod, 'download_test_image_data'):
        print('no download_test_image_data attr, nothing to patch',
              flush=True)
        return
    _orig = mod.download_test_image_data

    def _safe(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except Exception as e:
            print('download_test_image_data failed (%s); '
                  'using local synthetic image for the check' % e, flush=True)
            rng = _np.random.default_rng(0)
            return (rng.random((395, 395, 3)) * 255).astype(_np.uint8)

    mod.download_test_image_data = _safe
    print('patched download_test_image_data (fallback=synthetic)', flush=True)


_patch_test_image_download()

sys.argv = ['onnx2tf'] + sys.argv[1:]
runpy.run_module('onnx2tf', run_name='__main__')
