#!/bin/sh
# per-channel 域修复的上机验证（报告 46 §7）。
#
# 用法：
#   tools/device-perchannel-probe.sh <设备上的 .so 路径> <设备上的模型路径> [证据目录名]
#
# 跑三组：
#   fix      TEFLON_PCDOMAIN=1   → 期望 = 离线 B 预测（逐字节 100%）
#   upstream 不设 PCDOMAIN       → 期望 = 饱和输出（与修复前逐字节相同 ⇒ 同产物 A/B 有效）
#   cpu      --no-delegate       → 权威 CPU 基准
#
# 判据（verify/perchannel-prediction.py 给出）：
#   fix  == B 预测          （100.00%）
#   fix  vs CPU             （约 78%，其余 ≤1 LSB）
#   upstream == 修复前结果   （100.00%）

set -u

SO_DEV="${1:?用法: $0 <设备上的 .so> <设备上的模型> [证据子目录名]}"
MODEL_DEV="${2:?用法: $0 <设备上的 .so> <设备上的模型> [证据子目录名]}"
TAG="${3:-perchannel-fix-$(date +%Y%m%d-%H%M)}"
CTL=/tmp/opencode/ssh-ctl
KEY=/root/.config/opencode/.secrets/ssh/id_ed25519
DEV=root@192.168.123.184
LOCAL_EV="$(dirname "$0")/../verify/evidence/$TAG"
WORK_DEV=/volume1/OES/och-ms

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
$SSH "md5sum '$SO_DEV'"

echo "==> 跑三组（每组 90 s 硬超时）"
$SSH "cd $WORK_DEV
mkdir -p nndump/probe
run() {
  tag=\$1; shift
  L=nndump/probe/pf-\$tag.log
  timeout 90 env ETNA_MESA_DEBUG=ml_msgs \"\$@\" LD_LIBRARY_PATH=nndump:. \
    ./tf_load_check '$MODEL_DEV' '$SO_DEV' --invoke 1 --output-dump /tmp/pf-\$tag.bin > \"\$L\" 2>&1
  echo \"  rc=\$? [\$tag]\"
  grep -a -m1 'INVOKE n=' \"\$L\" | cut -c1-160
  grep -a -m1 'per-channel scheme' \"\$L\" | cut -c1-60
  cp /tmp/pf-\$tag.bin nndump/probe/ 2>/dev/null
}
run fix TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1 TEFLON_PCDOMAIN=1
run upstream TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1
timeout 120 env LD_LIBRARY_PATH=nndump:. ./tf_load_check '$MODEL_DEV' --no-delegate \
  --invoke 1 --output-dump /tmp/pf-cpu.bin > nndump/probe/pf-cpu.log 2>&1
echo \"  rc=\$? [cpu]\"
grep -a -m1 'INVOKE n=' nndump/probe/pf-cpu.log | cut -c1-160
cp /tmp/pf-cpu.bin nndump/probe/ && sync && echo '  sync done'"

echo "==> 拉回证据到 $LOCAL_EV"
for t in fix upstream cpu; do
    $SSH "cat $WORK_DEV/nndump/probe/pf-$t.bin" > "$LOCAL_EV/pf-$t.bin" 2>/dev/null
    $SSH "cat $WORK_DEV/nndump/probe/pf-$t.log" > "$LOCAL_EV/pf-$t.log" 2>/dev/null
done
ls -l "$LOCAL_EV" | head -8

echo "==> 离线判决"
MODEL_LOCAL="$(dirname "$0")/../verify/evidence/perchannel-20261021/c248-perchannel.tflite"
python3 "$(dirname "$0")/../verify/perchannel-prediction.py" \
    "$MODEL_LOCAL" "$LOCAL_EV/pf-fix.bin" "$LOCAL_EV/pf-cpu.bin"
