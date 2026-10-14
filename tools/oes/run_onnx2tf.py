#!/usr/bin/env python3
"""onnx2tf 包装运行器：补 numpy>=1.24 的 np.load(pickle) 自爆。

背景：onnx2tf 的错误上报/精度复核路径调用 np.load 下载的 .npy，
在 numpy>=1.24 默认 allow_pickle=False 下直接崩，把真正的转换错误盖住。
本包装器先把 allow_pickle 默认改回 True，再同进程跑 onnx2tf，
让真实错误透出来（并让复核有机会跑完）。

用法：python tools/oes/run_onnx2tf.py [onnx2tf 原生参数...]
"""
import runpy
import sys

import numpy as _np

_orig_load = _np.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault('allow_pickle', True)
    return _orig_load(*args, **kwargs)


_np.load = _patched_load

sys.argv = ['onnx2tf'] + sys.argv[1:]
runpy.run_module('onnx2tf', run_name='__main__')
