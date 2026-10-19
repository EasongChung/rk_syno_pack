#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**真实 Mesa 源码**上复核 per-channel 量化门补丁是否正确落地。

## 为什么需要这个（而不是只靠 selftest）

`tools/selftest-perchannel-quant.py` 跑在**合成 mock** 上，它证的是
「补丁逻辑对不对」。本脚本跑在 build-teflon.sh 打完补丁后的**真实源码**上，
证的是「锚点在真源码上仍然命中、替换结果没有走形」。

两者互补，不可互相替代：
- 合成 mock 里锚点被我逐字喂进去 ⇒ 必然命中，验不了「上游是否改过这段」
- 真源码里锚点可能已被 Mesa 升级改写 ⇒ 这才是真实风险

## 硬判项（任一不满足 ⇒ exit 1，构建中止）

1. 新函数 `tensor_quantization_supported_per_axis` 恰好 1 处
2. 编译期标记 `och-patch-perchannel-quant` 恰好 1 处
3. weight / bias 各恰好 1 处改用宽松判定
4. ★ input / output **仍然**各恰好 1 处用严格判定
   （放行它们会读到未设置的标量 scale ⇒ 静默算错，这是本补丁唯一会静默出错的改法）
5. 旧 gate 函数体仍在（纯追加式，不应被删改）
6. 括号配平（`_body` 依赖它）

## 负控制（--selftest）

按纪律「门必须有负控制」：脚本会构造 3 个「看起来打了补丁、实际坏了」的源码，
要求本检查器**全部拒绝**。若负控制没被拒绝，说明这个检查器形同虚设。

用法：
    python3 tools/check-perchannel-quant.py <etnaviv_ml.c>
    python3 tools/check-perchannel-quant.py --selftest
"""

import argparse
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHER = os.path.join(os.path.dirname(HERE), "build-recipe",
                       "patch-etnaviv-perchannel-quant.py")

FN_NEW = "tensor_quantization_supported_per_axis"
MARK = "och-patch-perchannel-quant"
RETURN_LINE = "return tensor->scales == NULL && tensor->zero_points == NULL;"

FAILS = []


def report(label, ok, detail="", sink=None):
    """sink=None 时只打印不记账（负控制内部用）；否则追加到 sink。"""
    print("  %-58s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + detail))
    if not ok:
        if sink is None:
            FAILS.append(label)
        else:
            sink.append(label)


def check(text, label="真源码"):
    """返回 problems 列表（空 = 通过）。

    ★ 本函数**只把传入文本的问题**记进局部 bad，不污染全局 FAILS ——
      否则 selftest 的负控制（本来就该 FAIL）会被误判成自测失败。
    """
    bad = []

    def need(cond, msg):
        if not cond:
            bad.append(msg)

    def rep(lbl, ok, detail=""):
        report(lbl, ok, detail, sink=bad)

    if text.count("{") != text.count("}"):
        need(False, "括号不配平 { %d / } %d" % (text.count("{"), text.count("}")))

    n_fn = text.count(FN_NEW + "(struct pipe_tensor *tensor)")
    need(n_fn == 1, "新函数定义出现 %d 处（期望 1）" % n_fn)

    n_mark = text.count(MARK)
    need(n_mark == 1, "编译期标记出现 %d 处（期望 1）" % n_mark)

    for t, want in (("weight_tensor", 1), ("bias_tensor", 1),
                    ("input_tensor", 1), ("output_tensor", 1)):
        loose = text.count("%s(%s)" % (FN_NEW, t))
        strict = text.count("tensor_quantization_supported(%s)" % t)
        if t in ("weight_tensor", "bias_tensor"):
            need(loose == want, "%s 走宽松判定 %d 处（期望 %d）" % (t, loose, want))
        else:
            need(strict == want,
                 "★ %s 必须仍走**严格**判定，实得 %d 处（期望 %d）——"
                 "放行它会读到未设置的标量 scale，静默算错" % (t, strict, want))
            need(loose == 0, "★ %s 被误改成宽松判定（%d 处）" % (t, loose))

    need(text.count(RETURN_LINE) == 1,
         "旧 gate 的 return 语句出现 %d 处（期望 1，纯追加式不应改它）"
         % text.count(RETURN_LINE))

    print("=== %s ===" % label)
    rep("括号配平", "括号不配平" not in bad,
        "; ".join(x for x in bad if "括号" in x))
    rep("新函数 tensor_quantization_supported_per_axis 恰好 1 处",
        n_fn == 1, "实得 %d" % n_fn)
    rep("编译期标记恰好 1 处", n_mark == 1, "实得 %d" % n_mark)
    rep("weight 走宽松判定恰好 1 处",
        text.count("%s(weight_tensor)" % FN_NEW) == 1)
    rep("bias 走宽松判定恰好 1 处",
        text.count("%s(bias_tensor)" % FN_NEW) == 1)
    rep("★ input 仍走严格判定恰好 1 处",
        text.count("tensor_quantization_supported(input_tensor)") == 1
        and text.count("%s(input_tensor)" % FN_NEW) == 0,
        "严格 %d / 宽松 %d"
        % (text.count("tensor_quantization_supported(input_tensor)"),
           text.count("%s(input_tensor)" % FN_NEW)))
    rep("★ output 仍走严格判定恰好 1 处",
        text.count("tensor_quantization_supported(output_tensor)") == 1
        and text.count("%s(output_tensor)" % FN_NEW) == 0,
        "严格 %d / 宽松 %d"
        % (text.count("tensor_quantization_supported(output_tensor)"),
           text.count("%s(output_tensor)" % FN_NEW)))
    rep("旧 gate 的 return 仍在恰好 1 处（纯追加）",
        text.count(RETURN_LINE) == 1, "实得 %d" % text.count(RETURN_LINE))
    return bad


def load_patcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pcq", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_patched_real():
    """用补丁器把 NEW 文本贴到一个最小可编译形状的骨架上，得到「打好补丁」的真源码形状。"""
    P = load_patcher()
    src = P.OLD_GATE + "\nbool\n" + P.FN_USE_INPUT + \
        "\n                              const struct pipe_ml_operation *operation)\n{\n" + \
        P.CALL_OLD + "\n   return true;\n}\n"
    return src, P


def selftest():
    """负控制：三个「看起来打了补丁、实际坏了」的源码，本检查器必须全部拒绝。"""
    import contextlib
    import io
    src, P = build_patched_real()

    print("=== 负控制自测（4 例必须全部被拒）===")
    good = src.replace(P.OLD_GATE, P.NEW_GATE, 1).replace(P.OLD_USE, P.NEW_USE, 1)
    report("正控制 打好补丁的真源码形状（必须 PASS）", not check(good, "正控制"))

    # neg1：input 被误改成宽松判定（本补丁唯一会静默算错的改法）
    neg1 = good.replace("tensor_quantization_supported(input_tensor)",
                        "%s(input_tensor)" % FN_NEW, 1)
    report("neg1 input 被误改成宽松判定（必须 FAIL）", bool(check(neg1, "neg1")))

    # neg2：**半打补丁** —— 新函数贴进去了，但四连调用没改
    #   （最真实的失败形态：patcher 的「替换」那一步没跑，只有「追加」跑了。
    #    后果是 per-channel 仍然不被接管 ⇒ 接管数仍是 0，但 strings 门照样能过）
    neg2 = src.replace(P.OLD_GATE, P.NEW_GATE, 1)
    report("neg2 半打补丁：新函数在、调用点没改（必须 FAIL）",
           bool(check(neg2, "neg2")))

    # neg3：新函数被复制一份（重复定义，链接期会报重定义）
    neg3 = good.replace("static bool\n%s(struct pipe_tensor *tensor)\n{\n   /* per-tensor" % FN_NEW,
                        "static bool\n%s(struct pipe_tensor *tensor);\n\nstatic bool\n%s(struct pipe_tensor *tensor)\n{\n   /* per-tensor"
                        % (FN_NEW, FN_NEW), 1)
    report("neg3 新函数被复制一份（必须 FAIL）", bool(check(neg3, "neg3")))

    # neg4：编译期标记被条件编译掉（补丁实际未生效，但源码里看着有）
    neg4 = good.replace(MARK, "perchannel_marker_removed", 1)
    report("neg4 编译期标记被删（必须 FAIL）", bool(check(neg4, "neg4")))

    print()
    if FAILS:
        print("SELFTEST RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("SELFTEST RESULT: ALL PASS（正控制通过 + 4 个负控制全被拒）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.path:
        ap.error("需要 <etnaviv_ml.c> 路径，或用 --selftest")

    with open(args.path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")

    bad = check(text, os.path.basename(args.path))
    print()
    if bad:
        print("VERDICT: FAIL —— %s" % "; ".join(bad))
        return 1
    print("VERDICT: PASS —— per-channel 门补丁在真实源码上正确落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
