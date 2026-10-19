# -*- coding: utf-8 -*-
"""本地端到端验证：把新补丁打到真源码副本上，检查结构/幂等/两种 diag 模式。

用法：
  python3 tools/selftest-perchannel-layout.py --src-root <etnaviv 目录>
默认取 CI 上的 third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv。
"""
import io
import os
import shutil
import argparse
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

# ⚠️ 默认值取的是 **och 开发机**的路径，不是 CI 的，也不是本机的。
#    CI 里 build-teflon.sh 显式传了 --src-root，所以当前不致命；
#    但它属于「换台机器就炸」的那一类陷阱（och 报告第二节第 3 条）。
#    唯一可靠的用法是**显式传 --src-root**。
DEFAULT_SRC_ROOT = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
                     "src/gallium/drivers/etnaviv")

# ★ 路径必须按 __file__ 推导 —— 写死本地绝对路径会让 CI 上子进程直接失败，
#   而失败会伪装成「补丁打不上」，极具误导性（run 39228292468 就这样红的）。
GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "gen-perchannel-layout-patch.py")
PATCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "build-recipe", "patch-etnaviv-perchannel-layout.py")
for _p, _what in ((GEN, "生成器"), (PATCH, "补丁器")):
    if not os.path.isfile(_p):
        sys.stderr.write("ERROR: 找不到%s %s\n" % (_what, _p))
        sys.exit(2)
SRC = None

_ap = argparse.ArgumentParser()
_ap.add_argument("--src-root", default=DEFAULT_SRC_ROOT,
                help="directory holding etnaviv_ml_nn.c / etnaviv_ml_nn_v7.c")
_args = _ap.parse_args()
SRC = _args.src_root
if not os.path.isdir(SRC):
    sys.stderr.write("ERROR: --src-root is not a directory: " + SRC + "\n")
    sys.exit(1)

fails = []


def ck(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def fresh():
    d = tempfile.mkdtemp(prefix="pcl-")
    for f in ("etnaviv_ml_nn.c", "etnaviv_ml_nn_v7.c"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(d, f))
    return d


def run(args):
    return subprocess.run([sys.executable] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


print("=== 1. --verify（已入库锚点 vs 真源码）===")
r = run([GEN, "--verify", "--src", os.path.join(SRC, "etnaviv_ml_nn.c"),
         "--src-v7", os.path.join(SRC, "etnaviv_ml_nn_v7.c"), "--out", PATCH])
print(r.stdout.strip())
print(r.stderr.strip())
ck(r.returncode == 0, "锚点与真实源码逐字一致")

print()
print("=== 2. 应用 --diag 1 到真源码副本 ===")
d1 = fresh()
r = run([PATCH, os.path.join(d1, "etnaviv_ml_nn.c"),
         "--v7", os.path.join(d1, "etnaviv_ml_nn_v7.c"), "--diag", "1"])
print(r.stdout.strip() or r.stderr.strip())
ck(r.returncode == 0, "补丁应用成功")

nn1 = io.open(os.path.join(d1, "etnaviv_ml_nn.c"), encoding="utf-8").read()
v71 = io.open(os.path.join(d1, "etnaviv_ml_nn_v7.c"), encoding="utf-8").read()

for tok, cnt, why in (
        ("#define OCH_PCL_DIAG 1", 1, "nn.c 定义了诊断开关"),
        ("och patch: per-channel layout contract", 3, "nn.c 带补丁标记（前奏+两处函数注释）"),
        ("uint8_t (*weights)[real_in]", 1, "calc_quantization 读侧 real_in"),
        ("uint8_t (*map_in)[real_in]", 1, "requantize_weights 读侧 real_in"),
        ("uint8_t (*map_out)[out_dim]", 1, "requantize_weights 写侧 out_dim"),
        ("unsigned src_ic = depthwise ? oc : ic;", 1, "depthwise 对角线取权重"),
        ("nn_core_version >= 8 && poperation->conv.depthwise", 0, "v7 条件已去掉"),
        ("[och-pcl] CQ-OUT", 1, "calc_quantization 出口诊断"),
        ("[och-pcl] RW-OUT", 1, "requantize_weights 出口诊断"),
        ("new_size = oc_max *", 1, "new_size 按 oc_max 计算"),
        ("(unsigned)operation->weight_tensor_size", 1, "打印字段已显式转换")):
    ck(nn1.count(tok) == cnt, "%s（实得 %d，期望 %d）" % (why, nn1.count(tok), cnt))

for tok, cnt, why in (("[och-pcl] SEL", 2, "v7 两处选路都注入了诊断"),
                      ("write_core_sequential(subgraph, NULL", 1, "试算段保留"),
                      ("write_core_sequential(subgraph, map", 1, "正式段保留")):
    ck(v71.count(tok) == cnt, "%s（实得 %d，期望 %d）" % (why, v71.count(tok), cnt))

print()
print("=== 3. 幂等：再打一次必须被拒 ===")
r = run([PATCH, os.path.join(d1, "etnaviv_ml_nn.c"),
         "--v7", os.path.join(d1, "etnaviv_ml_nn_v7.c"), "--diag", "1"])
ck(r.returncode == 1 and "拒绝重复应用" in (r.stderr + r.stdout),
   "第二次应用被拒绝（返回 %d）" % r.returncode)

print()
print("=== 4. 失配 fail-closed：改坏一个锚点后必须报错 ===")
d2 = fresh()
p = os.path.join(d2, "etnaviv_ml_nn.c")
t = io.open(p, encoding="utf-8").read()
t = t.replace("input_channels = 1;", "input_channels = 2;", 1)
io.open(p, "w", encoding="utf-8").write(t)
r = run([PATCH, p, "--v7", os.path.join(d2, "etnaviv_ml_nn_v7.c"), "--diag", "1"])
ck(r.returncode == 1, "结构变了就打不上（返回 %d）" % r.returncode)
msg = (r.stderr + r.stdout).strip().splitlines()
print("       " + (msg[-1] if msg else ""))

print()
print("=== 5. --diag 0 必须产出无诊断代码 ===")
d3 = fresh()
r = run([PATCH, os.path.join(d3, "etnaviv_ml_nn.c"),
         "--v7", os.path.join(d3, "etnaviv_ml_nn_v7.c"), "--diag", "0"])
print(r.stdout.strip() or r.stderr.strip())
ck(r.returncode == 0, "无诊断模式应用成功")
nn0 = io.open(os.path.join(d3, "etnaviv_ml_nn.c"), encoding="utf-8").read()
v70 = io.open(os.path.join(d3, "etnaviv_ml_nn_v7.c"), encoding="utf-8").read()
ck("OCH_PCL_DIAG" not in nn0, "nn.c 无诊断残留")
ck("och-pcl" not in nn0, "nn.c 无诊断打印")
ck("OCH_PCL_DIAG" not in v70 and "och-pcl" not in v70, "v7 无诊断残留")
ck("uint8_t (*map_out)[out_dim]" in nn0, "功能改动仍在（out_dim 布局）")

for d in (d1, d2, d3):
    shutil.rmtree(d, ignore_errors=True)

print()
print("SELFTEST: " + ("PASS" if not fails else "FAIL (%d)" % len(fails)))
sys.exit(0 if not fails else 1)