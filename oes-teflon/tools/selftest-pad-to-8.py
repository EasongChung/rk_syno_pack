#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-pad-to-8.py —— pad-to-8.py 的自测

门守什么：pad-to-8 是**零驱动改动路线**的主刀。它的正确性直接决定
「244 通道 1×1 conv 挂死」能否用「补到 248 ⇒ m=0 ⇒ 该机制不可能触发」来化解。
刀法错了 ⇒ ① 模型语义被改坏（数值错）② 补零没补对（m 仍 > 0）③ 半改的模型。

测试矩阵

  正控制 3：
    1) 对真实样本（sweep1x1-c0244.tflite）应用 ⇒ m 归零；**合法通道 0..243 的
       权重与偏置逐字节不变**（这是最重要的一条：手术只能动补出来的通道）；
    2) 补出来的通道：权重 == 权重 zero_point、偏置 == 0、量化 scale/zp 数组已延长；
    3) 回归：补出来的模型再扫一次 ⇒ 幻影通道数 m == 0，且分区算术与已实测不挂的
       248 行一致（kpc0=31 sb=7 d=5 rem=1）。

  负控制 4：
    1) 真实模型（含 Sandglass 交错块）必须**拒绝** ⇒ 不产出半改的模型；
    2) 已补齐的模型再补一次 ⇒ 拒绝（幂等，不做无意义改写）；
    3) 截断的输入文件 ⇒ 必须失败，不能产出坏模型；
    4) 图输出通道数被改且未加 --allow-output-pad ⇒ 拒绝。

  反作弊：补出来的权重值与「从输出模型的量化参数独立读出的 zero_point」比对，
          不用工具内部状态。

用法:
    python3 selftest-pad-to-8.py [工具路径]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_TOOL = os.path.join(HERE, "pad-to-8.py")
SAMPLE = os.path.join(REPO, "model-surgery", "diag", "sweep", "sweep1x1-c0244.tflite")
REAL_MODEL = os.path.join(REPO, "artifact-int8", "concept_int8_pertensor.tflite")


def run(tool, *args):
    return subprocess.run([sys.executable, tool] + list(args),
                          capture_output=True, text=True)


def fail(msg, detail=""):
    sys.stderr.write("FAIL: " + msg + "\n")
    if detail:
        sys.stderr.write("      " + detail.strip().replace("\n", "\n      ") + "\n")
    sys.exit(1)


def load(path):
    sys.path.insert(0, HERE)
    import tflite_fold_rank5_to_4d as fold
    return fold.read_model(path)


def weight_bytes(mdl, ti):
    t = mdl["tensors"][ti]
    return mdl["buffers"][t["buffer"]]["data"]


def bias_bytes(mdl, ti):
    t = mdl["tensors"][ti]
    return mdl["buffers"][t["buffer"]]["data"]


def main():
    tool = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOOL
    if not os.path.isfile(tool):
        fail("找不到工具: " + tool)
    if not os.path.isfile(SAMPLE):
        fail("找不到真实样本: " + SAMPLE)
    tmp = tempfile.mkdtemp(prefix="pad8-selftest-")

    # ---------- 正控制 1/2/3 ----------
    out = os.path.join(tmp, "pad8.tflite")
    r = run(tool, SAMPLE, out, "--allow-output-pad")
    if r.returncode != 0:
        fail("正控制 1 失败：对真实样本应用应成功", r.stderr)
    src = load(SAMPLE)
    dst = load(out)
    # 合成模型里 conv 是 op#2，权重 t#6、偏置 t#7
    sw = weight_bytes(src, 6)
    sb = bias_bytes(src, 7)
    dw = weight_bytes(dst, 6)
    db = bias_bytes(dst, 7)
    n_old = 244
    # 形状按**实际张量**读，不写死：补齐只动输出轴，输入轴（图的输入张量）不变
    wshape_old = list(src["tensors"][6]["shape"])      # [244,1,1,244]
    wshape_new = list(dst["tensors"][6]["shape"])      # [248,1,1,244]
    n_new = wshape_new[0]
    in_axis = wshape_new[3]
    in_axis_old = wshape_old[3]
    if wshape_old[0] != n_old:
        fail("样本结构异常：原权重形状 %s" % wshape_old)
    if len(sw) != n_old * in_axis_old:
        fail("样本结构异常：原权重 %d B，形状 %s" % (len(sw), wshape_old))
    if len(dw) != n_new * in_axis:
        fail("正控制 1 失败：补后权重 %d B，期望 %d B（形状 %s）"
             % (len(dw), n_new * in_axis, wshape_new))
    # 输入轴也可能被补（内部张量）⇒ 合法输入通道那一段仍必须逐字节未变
    if len(db) != n_new * 4:
        fail("正控制 1 失败：补后偏置 %d B，期望 %d B" % (len(db), n_new * 4))

    # (a) 合法通道 × 合法输入通道的权重逐字节不变
    #     （输入轴也可能被补 ⇒ 只比对前 in_axis_old 个字节那一段）
    for oc in range(n_old):
        row_old = sw[oc * in_axis_old:(oc + 1) * in_axis_old]
        row_new = dw[oc * in_axis:(oc + 1) * in_axis][:in_axis_old]
        if row_old != row_new:
            fail("正控制 1 失败：合法通道 %d 的权重被改动了 ⇒ 手术越界" % oc)
    # (b) 合法通道的偏置逐字节不变
    if sb != db[:len(sb)]:
        fail("正控制 1 失败：合法通道的偏置前缀被改动")
    print("  [ok] 正控制 1：合法通道 0..%d 的权重/偏置逐字节未变" % (n_old - 1))

    # (c) 反作弊：补出来的权重 == 从**输出模型**的量化参数独立读出的 zp
    q = dst["tensors"][6]["quant"]
    zps = q.get("zp") or [0]
    wzp = zps[min(n_new - 1, len(zps) - 1)]
    for oc in range(n_old, n_new):
        row = dw[oc * in_axis:(oc + 1) * in_axis]
        if any(v != (wzp & 0xFF) for v in row):
            fail("正控制 2 失败：补出的通道 %d 的权重不是 zp(%d)" % (oc, wzp))
    # 补出来的偏置 == 0（int32 小端）
    for oc in range(n_old, n_new):
        v = int.from_bytes(db[oc * 4:oc * 4 + 4], "little", signed=True)
        if v != 0:
            fail("正控制 2 失败：补出的偏置 %d != 0" % oc)
    # per-axis 量化的 scale/zp 数组长度必须 = 输出通道数
    if len(q.get("scale") or []) not in (0, n_new):
        fail("正控制 2 失败：权重 scale 数组长度 %d != %d" % (len(q["scale"]), n_new))
    # ★ 偏置的 per-axis 量化数组同样必须延长（实测漏过这一条 ⇒ 设备上解释器直接拒绝加载：
    #   "num_scales must be 1 ... or 248 ... but got 244. Tensor 7 = bias"）
    bq = dst["tensors"][7].get("quant") or {}
    if len(bq.get("scale") or []) not in (0, 1, n_new):
        fail("正控制 2 失败：偏置 scale 数组长度 %d 既不是 1 也不是 %d"
             % (len(bq["scale"]), n_new))
    print("  [ok] 正控制 2：补出的通道 权重==zp(%d)、偏置==0、量化数组已延长" % wzp)

    # (d) 回归：m 必须归零
    scan = os.path.join(REPO, "verify", "scan-phantom-risk.py")
    r = subprocess.run([sys.executable, scan, out, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        fail("正控制 3 失败：扫描补后模型失败", r.stderr)
    import json
    rows = json.loads(r.stdout)
    if not rows or rows[0]["m"] != 0:
        fail("正控制 3 失败：补后 m=%s（应为 0）" % (rows[0]["m"] if rows else "?"))
    if (rows[0]["kpc0"], rows[0]["sb"], rows[0]["d"], rows[0]["rem"]) != (31, 7, 5, 1):
        fail("正控制 3 失败：补后分区 %s ≠ 已实测不挂的 248 行 (31,7,5,1)"
             % str((rows[0]["kpc0"], rows[0]["sb"], rows[0]["d"], rows[0]["rem"])))
    print("  [ok] 正控制 3：补后 m=0，且分区 (31,7,5,1) 与已实测不挂的 248 行一致")

    # ---------- 负控制 ----------
    if os.path.isfile(REAL_MODEL):
        # ★ 正控制 4：真实模型必须被**拒绝**，且理由必须是「畸形 conv」
        #   （权重 in 维 != 输入通道数）—— 这独立复核了「2 个结构畸形 CONV_2D」这条发现。
        out2 = os.path.join(tmp, "real.tflite")
        r = run(tool, REAL_MODEL, out2, "--allow-output-pad")
        if r.returncode == 0:
            fail("正控制 4 失败：真实模型含 2 个结构畸形 CONV_2D，应被拒绝")
        if "权重 in 维" not in r.stderr:
            fail("正控制 4 失败：拒绝理由不是「权重 in 维 != 输入通道数」", r.stderr)
        if os.path.exists(out2):
            fail("正控制 4 失败：被拒绝时不应留下输出文件")
        print("  [ok] 正控制 4：真实模型被拒，理由=畸形 conv（权重 in 维 != 输入通道数）")

        # 负控制 1：把 SE 块的 RESHAPE 常量改坏 ⇒ 必须被 pre-flight 拒绝
        broken = os.path.join(tmp, "broken-se.tflite")
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import tflite_fold_rank5_to_4d as fold
        mdl = fold.read_model(REAL_MODEL)
        t5 = mdl["tensors"][5]
        buf = mdl["buffers"][t5["buffer"]]
        d = bytearray(buf["data"])
        d[8:12] = (121).to_bytes(4, "little", signed=True)   # [1,2,122,50,50] → 121
        buf["data"] = bytes(d)
        with open(broken, "wb") as f:
            f.write(fold.build_model(mdl))
        out2b = os.path.join(tmp, "broken-se-out.tflite")
        r = run(tool, broken, out2b, "--allow-output-pad")
        if r.returncode == 0:
            fail("负控制 1 失效：SE 块常量被改坏的模型应被 pre-flight 拒绝")
        if "元素数不自洽" not in r.stderr:
            fail("负控制 1 失效：拒绝理由不是「元素数不自洽」", r.stderr)
        print("  [ok] 负控制 1：SE 块常量被改坏（122→121）被 pre-flight 拒绝")

    out3 = os.path.join(tmp, "again.tflite")
    r = run(tool, out, out3, "--allow-output-pad")
    if r.returncode == 0:
        fail("负控制 2 失效：已补齐的模型再补一次应拒绝（幂等）")
    print("  [ok] 负控制 2：已补齐的模型再补一次被拒绝")

    trunc = os.path.join(tmp, "trunc.tflite")
    with open(SAMPLE, "rb") as f:
        head = f.read(4096)
    with open(trunc, "wb") as f:
        f.write(head)
    out4 = os.path.join(tmp, "trunc-out.tflite")
    r = run(tool, trunc, out4, "--allow-output-pad")
    if r.returncode == 0:
        fail("负控制 3 失效：截断输入应失败")
    print("  [ok] 负控制 3：截断输入被拒绝")

    out5 = os.path.join(tmp, "noallow.tflite")
    r = run(tool, SAMPLE, out5)
    if r.returncode == 0:
        fail("负控制 4 失效：改图输出通道数且未加 --allow-output-pad，应拒绝")
    print("  [ok] 负控制 4：图输出被改且未放行时被拒绝")

    print("  [ok] selftest-pad-to-8 全部通过（正控制 4 + 负控制 4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
