#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码逐字提取锚点，生成/校验 build-recipe/patch-etnaviv-perchannel-quant.py。

## 为什么要有这个工具

补丁器里的 OLD_GATE / OLD_USE 是**逐字锚点**。只要上游 Mesa 动了那两段
（哪怕只是改注释、换缩进），补丁就会失配 —— 补丁器会 fail-closed 报错，
但那时已经浪费了一整轮 CI。这个工具让「重新生成锚点」变成一条命令。

## 它同时是一个**独立复核器**

    python3 tools/gen-perchannel-quant-patch.py --verify <etnaviv_ml.c>

`--verify` 不写任何文件，只回答一个问题：
**已入库的补丁器里的锚点，是否与这份真实源码逐字一致？**
这正是手抄锚点最容易出错、也最难在 review 里看出来的地方。

## 用法

    # 校验（只读，不写文件；CI 与本地都能跑）
    python3 tools/gen-perchannel-quant-patch.py --verify <etnaviv_ml.c>

    # 重新生成（会覆盖补丁器）
    python3 tools/gen-perchannel-quant-patch.py --src <etnaviv_ml.c> --out <补丁器路径>
"""

import argparse
import os
import sys

DEFAULT_SRC = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
               "src/gallium/drivers/etnaviv/etnaviv_ml.c")
DEFAULT_OUT = ("/root/workspace/oes-teflon/build-recipe/"
               "patch-etnaviv-perchannel-quant.py")

FN_GATE = "tensor_quantization_supported(struct pipe_tensor *tensor)"
FN_USE = "etna_ml_operation_supported(struct pipe_ml_device *pdevice"

STRUCT_GATE = (
    ("return tensor->scales == NULL && tensor->zero_points == NULL;", 1),
    ("Per-axis quantization not supported", 1),
)
STRUCT_USE = (
    ("tensor_quantization_supported(input_tensor)", 1),
    ("tensor_quantization_supported(weight_tensor)", 1),
    ("tensor_quantization_supported(bias_tensor)", 1),
    ("tensor_quantization_supported(output_tensor)", 1),
)


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def body_of(src, fn):
    at = src.find(fn)
    if at < 0:
        die("源码里找不到函数 " + fn)
    i = src.index("{", at)
    d = 0
    j = i
    while j < len(src):
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return src[at:j + 1]
        j += 1
    die("函数 " + fn + " 的花括号不配平（源码可能被截断）")


def extract_gate(src):
    """逐字取出 tensor_quantization_supported 的整个函数（含尾随换行）。

    ★ 连函数上面那行 `static bool` 一起取：入库的 OLD_GATE 就是从 `static bool`
      开始的，若这里从函数名开始抽，--verify 会拿「去头的真源码」去比
      「带头的入库锚点」，永远报不一致（这个坑踩过一次）。
    """
    b = body_of(src, FN_GATE)
    n = src.count(b)
    if n != 1:
        die("gate 函数体在文件里出现 %d 处（期望 1）" % n)
    at = src.find(b)
    head = ""
    prev = src.rfind("\n", 0, at - 1) + 1          # 函数名所在行的行首
    if prev > 0 and src[prev:at] == "static bool\n":
        head = "static bool\n"
    elif at >= len("static bool\n") and src[at - len("static bool\n"):at] == "static bool\n":
        head = "static bool\n"
    full = head + b + "\n"
    if src.count(full) != 1:
        die("含 static bool 的 gate 锚点在文件里出现 %d 处（期望 1）"
            % src.count(full))
    return full


def extract_use(src):
    """逐字取出四连调用（input → weight → bias → output）。

    从 `tensor_quantization_supported(input_tensor)` 起，向后吃掉连续三行
    `&& tensor_quantization_supported(<tensor>) &&`，保证四行一起被锚住 ——
    只锚一行会被上游重排打穿。
    """
    at = src.find("tensor_quantization_supported(input_tensor)")
    if at < 0:
        die("函数里找不到 tensor_quantization_supported(input_tensor)")
    lines = src[at:].split("\n")
    if len(lines) < 4:
        die("四连调用不足 4 行，源码结构可能已变")
    block = lines[0:4]
    for want in ("weight_tensor", "bias_tensor", "output_tensor"):
        if not any(want in l for l in block[1:4]):
            die("四连调用第 2-4 行里没有 " + want + " —— 源码结构可能已变")
    old = "\n".join(block)
    # 全文唯一性：锚点必须逐字唯一
    n = src.count(old)
    if n != 1:
        die("四连调用锚点在文件里出现 %d 处（期望 1）" % n)
    # 结构计数：限定在 etna_ml_operation_supported 的函数体内
    ub = body_of(src, FN_USE)
    for tok, want in STRUCT_USE:
        got = ub.count(tok)
        if got != want:
            die("etna_ml_operation_supported 内 `%s` 实得 %d（期望 %d）"
                % (tok, got, want))
    return old


def check_structs(src):
    gb = body_of(src, FN_GATE)
    for tok, want in STRUCT_GATE:
        got = gb.count(tok)
        if got != want:
            die("tensor_quantization_supported 内 `%s` 实得 %d（期望 %d）"
                % (tok, got, want))


def load_committed(out_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pcq_committed", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(src_path, out_path):
    src = open(src_path, encoding="utf-8").read().replace("\r\n", "\n")
    check_structs(src)
    gate = extract_gate(src)
    use = extract_use(src)
    P = load_committed(out_path)

    ok = True
    for name, real, committed in (("OLD_GATE", gate, P.OLD_GATE),
                                  ("OLD_USE", use, P.OLD_USE)):
        if real == committed:
            print("  %-10s 与真实源码逐字一致 ✅  (%d 字符)" % (name, len(real)))
        else:
            ok = False
            print("  %-10s ★不一致★" % name)
            print("    真实源码 : %r" % real[:220])
            print("    已入库   : %r" % committed[:220])
    print()
    if ok:
        print("VERIFY: PASS —— 已入库补丁器的锚点与真实源码完全一致")
        return 0
    print("VERIFY: FAIL —— 锚点与真实源码不符，请用 --src/--out 重新生成")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true",
                    help="只校验已入库补丁器的锚点，不写文件")
    args = ap.parse_args()

    if args.verify:
        return verify(args.src, args.out)

    src = open(args.src, encoding="utf-8").read().replace("\r\n", "\n")
    check_structs(src)
    gate = extract_gate(src)
    use = extract_use(src)
    print("--- OLD_GATE 逐字预览 ---")
    print(gate)
    print("--- OLD_USE 逐字预览 ---")
    print(use)
    print()
    print("把上面两段填进补丁器的 OLD_GATE / OLD_USE；"
          "或直接用 --verify 先确认现有锚点是否仍然命中。")


if __name__ == "__main__":
    sys.exit(main())
