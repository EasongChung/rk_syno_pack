#!/bin/sh
# 交付版产物的**设备回归**：三个模型 × (NPU / CPU) 两组。
#
# 用法：
#   tools/device-delivery-regression.sh <设备上的 libteflon.so> [证据目录名]
#
# ★ 交付语义：NPU 那一组**不设任何环境变量** —— 修复必须默认生效。
#   （要单独关掉某个修复：TEFLON_WDOMAIN=0 / TEFLON_ODETRANS=0 / TEFLON_PCDOMAIN=0）
#
# 判据（按模型分列，不同模型的「好」不一样）：
#
#   模型                     基准                       期望
#   ─────────────────────    ──────────────────────    ──────────────────────────────
#   c248 合成（per-tensor）  精确整数数学               100% 逐字节一致
#                            TFLite CPU                 ~76.5%，差值单侧 −1
#   官方 MobileNetV1        TFLite CPU                 ~99.7%，差值 ≤ ±1
#   c248-perchannel 手术版  TFLite CPU                 ~78%，差值 ≤ ±1，**不得饱和**
#
#   「精确整数数学」由 verify/exact-math-prediction.py 算（报告 43 的基准）。

set -u

SO_DEV="${1:?用法: $0 <设备上的 libteflon.so> [证据子目录名]}"
TAG="${2:-delivery-regression-$(date +%Y%m%d-%H%M)}"
CTL=/tmp/opencode/ssh-ctl
KEY=/root/.config/opencode/.secrets/ssh/id_ed25519
DEV=root@192.168.123.184
WORK=/volume1/OES/och-ms
LOCAL_EV="$(dirname "$0")/../verify/evidence/$TAG"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$LOCAL_EV"

if ! ssh -S "$CTL" -O check "$DEV" >/dev/null 2>&1; then
    echo "==> 建立复用连接"
    rm -f "$CTL"
    ssh -M -S "$CTL" -o ControlPersist=1800 \
        -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -o ConnectTimeout=10 -fN "$DEV" || exit 1
fi
SSH="ssh -S $CTL -T -o BatchMode=yes -o LogLevel=ERROR $DEV"

echo "==> 产物核对"
$SSH "md5sum '$SO_DEV'; ls -l '$SO_DEV' | awk '{print \$5\" B\"}'"

# 交付语义：NPU 组**不设任何环境变量**
echo "==> 跑三模型 × (NPU 无环境变量 / CPU)"
$SSH "cd $WORK
mkdir -p nndump/probe
run() {
  tag=\$1; model=\$2; mode=\$3
  L=nndump/probe/dr-\$tag.log
  if [ \"\$mode\" = cpu ]; then
    timeout 180 env LD_LIBRARY_PATH=nndump:. ./tf_load_check \"\$model\" --no-delegate \\
      --invoke 1 --output-dump /tmp/dr-\$tag.bin > \"\$L\" 2>&1
  else
    timeout 120 env ETNA_MESA_DEBUG=ml_msgs LD_LIBRARY_PATH=nndump:. \\
      ./tf_load_check \"\$model\" '$SO_DEV' --invoke 1 --output-dump /tmp/dr-\$tag.bin > \"\$L\" 2>&1
  fi
  echo \"  rc=\$? [\$tag]\"
  grep -a -m1 'INVOKE n=' \"\$L\" | cut -c1-160
  cp /tmp/dr-\$tag.bin nndump/probe/ 2>/dev/null
}
run c248   nndump/sweep/sweep1x1-c0248-nondg.tflite   npu
run c248c  nndump/sweep/sweep1x1-c0248-nondg.tflite   cpu
run mb     nndump/sweep/mobilenet_v1_0.25_128_quant.tflite npu
run mbc    nndump/sweep/mobilenet_v1_0.25_128_quant.tflite cpu
run pc     nndump/sweep/c248-perchannel.tflite        npu
run pcc    nndump/sweep/c248-perchannel.tflite        cpu
sync && echo '  sync done'"

echo "==> 拉回证据到 $LOCAL_EV"
for t in c248 c248c mb mbc pc pcc; do
    $SSH "cat $WORK/nndump/probe/dr-$t.bin" > "$LOCAL_EV/dr-$t.bin" 2>/dev/null
    $SSH "cat $WORK/nndump/probe/dr-$t.log" > "$LOCAL_EV/dr-$t.log" 2>/dev/null
done

M248="$HERE/../verify/evidence/roundmode-20261021/sweep1x1-c0248-nondg.tflite"
MPC="$HERE/../verify/evidence/perchannel-20261021/c248-perchannel.tflite"

echo
echo "==================== 1) c248 合成（per-tensor int8）===================="
python3 "$HERE/../verify/exact-math-prediction.py" "$M248" \
        "$LOCAL_EV/dr-c248.bin" "$LOCAL_EV/dr-c248c.bin"

echo
echo "==================== 2) 官方 MobileNetV1 ===================="
python3 - "$LOCAL_EV/dr-mb.bin" "$LOCAL_EV/dr-mbc.bin" <<'PY'
import collections, sys
i8 = lambda b: [x - 256 if x > 127 else x for x in b]
npu = i8(open(sys.argv[1], "rb").read())
cpu = i8(open(sys.argv[2], "rb").read())
h = collections.Counter(a - b for a, b in zip(npu, cpu))
gt = sum(1 for a, b in zip(npu, cpu) if a > b)
print("  NPU vs CPU : %d/%d = %.2f%%  差值分布 %s  (gt=%d lt=%d)"
      % (h[0], len(cpu), 100.0 * h[0] / len(cpu), sorted(h.items())[:4], gt,
         sum(1 for a, b in zip(npu, cpu) if a < b)))
print("  NPU 取值集合大小: %d（若为 1 说明饱和了 ⇒ 修复没生效）" % len(set(npu)))
PY

echo
echo "==================== 3) c248-perchannel 手术版 ===================="
python3 - "$LOCAL_EV/dr-pc.bin" "$LOCAL_EV/dr-pcc.bin" <<'PY'
import collections, sys
i8 = lambda b: [x - 256 if x > 127 else x for x in b]
npu = i8(open(sys.argv[1], "rb").read())
cpu = i8(open(sys.argv[2], "rb").read())
h = collections.Counter(a - b for a, b in zip(npu, cpu))
gt = sum(1 for a, b in zip(npu, cpu) if a > b)
print("  NPU vs CPU : %d/%d = %.2f%%  差值分布 %s  (gt=%d lt=%d)"
      % (h[0], len(cpu), 100.0 * h[0] / len(cpu), sorted(h.items())[:4], gt,
         sum(1 for a, b in zip(npu, cpu) if a < b)))
print("  NPU 取值集合大小: %d（若为 1 说明饱和了 ⇒ per-channel 域修复没生效）" % len(set(npu)))
PY
