#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等 oca 的 art13 模型落盘，落盘即自动跑静态门 a/b/a2/c **与 5 构建崩溃矩阵**。

背景：/root/clawtemp 是 oca 工作区的**实时只读视图**，所以不必去问 oca、
也不占它算力，直接等文件出现即可。

判据（静态，纯本地、不编译）：
  a  退化张量 = 0       → teflon_partition_sim.py 的「退化张量 ...: N」
  a2 量化张量数 = 317   → 与移植前一致，防止「凭空给张量塞量化参数」的写坏
  b  无空形状张量       → 报告里的 shape=[] 条目
  c  conv/dwconv per-tensor → 全量报告的 per-tensor/per-channel 统计

数值门 d/e/f 与崩溃矩阵必须在**设备**上跑（会写设备临时目录，跑完自动清理）：
  → verify/device_gate.sh        （数值门 CPU 通路）
  → verify/crash_matrix.sh       （接管数 + 5 构建崩溃签名）

纪律：不猜、不抢跑。矩阵**不 invoke**，只测「接管能否成立」。
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

ROOT = "/root/clawtemp/o2tf-src"
PROJECT = "/root/workspace/oes-teflon"
VERIFY = os.path.join(PROJECT, "verify")
REPORTS = os.path.join(VERIFY, "reports")
SKIP_DIRS = {".git", "@eaDir", "__pycache__"}


def scan():
    found = {}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith(".tflite"):
                p = os.path.join(dp, f)
                try:
                    found[p] = os.stat(p).st_size
                except OSError:
                    pass
    return found


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gate(model):
    name = os.path.basename(model)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, name + ".sim.txt")
    with open(out, "w") as fh:
        subprocess.run([sys.executable, "teflon_partition_sim.py", model],
                       cwd=VERIFY, stdout=fh, stderr=subprocess.STDOUT)
    text = open(out, encoding="utf-8", errors="replace").read()
    h = hashlib.sha256(open(model, "rb").read()).hexdigest()
    size = os.path.getsize(model)

    print("=" * 78)
    print(f"候选模型: {model}")
    print(f"  大小   : {size} B")
    print(f"  sha256 : {h}")
    print(f"  报告   : {out}")
    print("=" * 78)

    fails = []
    # 门 a：退化张量数
    deg = None
    for line in text.splitlines():
        if "退化张量 (|scale|>1e10 或 scale<=0 等):" in line:
            deg = int(line.rsplit(":", 1)[1].strip())
    ok_a = deg == 0
    print(f"  门 a 退化张量数        : {deg}  -> {'PASS' if ok_a else 'FAIL'}")
    if not ok_a:
        fails.append("门 a 退化张量 != 0")

    # 门 a2（移植不变性）：有仿射量化参数的张量数必须与移植前一致。
    # oca 曾把「无量化参数」当成「退化」给 20 个 INT32 常量硬塞 scale=1.0/zp=0，
    # 该数从 317 变成 337 —— 那是模型损坏，不是修复。
    quant = None
    for line in text.splitlines():
        if "有仿射量化参数" in line:
            try:
                quant = int(line.rsplit(":", 1)[1].strip())
            except ValueError:
                pass
    # 参照：concept_int8_pertensor.tflite 自身（移植前的源模型）
    REF_Q = 317
    ok_a2 = (quant == REF_Q)
    print(f"  门 a2 量化张量数      : {quant}（移植前应为 {REF_Q}）"
          f"  -> {'PASS' if ok_a2 else 'FAIL'}"
          f"{'' if ok_a2 else '  ← 模型可能被写坏（凭空给张量塞量化参数）'}")
    if not ok_a2:
        fails.append(f"门 a2 量化张量数 {quant} != {REF_Q}")

    # 门 b：空形状张量
    empty = [ln.strip() for ln in text.splitlines() if "shape=[]" in ln]
    ok_b = not empty
    print(f"  门 b 空形状张量        : {len(empty)} 条 -> {'PASS' if ok_b else 'FAIL'}")
    for ln in empty[:6]:
        print(f"      {ln}")
    if not ok_b:
        fails.append("门 b 存在空形状张量")

    # 门 c：per-tensor / per-channel 统计
    for line in text.splitlines():
        if "per-tensor" in line and "conv" in line.lower():
            print(f"  门 c 统计: {line.strip()}")

    print("-" * 78)
    print(f"静态门小结: {'全部通过' if not fails else '未过 -> ' + '; '.join(fails)}")
    print("  ---- 数值门 d/e/f 需上设备 tf_load_check --invoke，本脚本不涉及 ----")
    return deg, empty


def run_matrix(model):
    """设备侧 5 构建崩溃矩阵（不 invoke；脚本自己核对 sha256 并清理临时目录）。"""
    script = os.path.join(VERIFY, "crash_matrix.sh")
    if not os.path.isfile(script):
        log(f"跳过矩阵：找不到 {script}")
        return
    log("开始设备侧崩溃矩阵（上传模型 → 5 构建逐个跑 PrepareDelegate → 清理）")
    p = subprocess.run(["sh", script, model], cwd=PROJECT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout.splitlines():
        print("    " + line, flush=True)
    log(f"矩阵结束，退出码 {p.returncode}（0=有构建未崩 / 5=全崩）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--no-matrix", action="store_true",
                    help="只跑静态门，不跑设备崩溃矩阵")
    args = ap.parse_args()

    base = scan()
    log(f"基线快照: {len(base)} 个 .tflite 已在 {ROOT} 下（这些不算新产出）")

    deadline = time.time() + args.timeout
    prev = {}
    while time.time() < deadline:
        time.sleep(args.poll)
        now = scan()
        new = {p: s for p, s in now.items() if p not in base}
        for p, s in new.items():
            stable = prev.get(p) == s and s > 0
            if stable:
                log(f"新模型落盘且大小稳定: {p} ({s} B)")
                deg, empty = gate(p)
                if not args.no_matrix:
                    run_matrix(p)
                return 0
        prev = new
        if new and int(time.time()) % 300 < args.poll:
            log(f"… 已见 {len(new)} 个新 .tflite，等大小稳定")
    log(f"超时 {args.timeout}s：{ROOT} 下仍未出现新的 .tflite")
    return 3


if __name__ == "__main__":
    sys.exit(main())
