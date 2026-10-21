#!/bin/sh
# 舍入模式探针：在设备上跑 5 组（默认 + ROUNDMODE=0/1/2/3），拉回输出并比对 CPU。
#
# 用法：
#   tools/device-roundmode-probe.sh <设备上的 .so 路径> [evidence 子目录名]
#
# 前提：
#   - 产物必须含 TEFLON_ROUNDMODE 门控（构建时 TEFLON_ROUNDMODE_PATCH=1）
#   - 设备侧模型：/volume1/OES/och-ms/nndump/sweep/sweep1x1-c0248-nondg.tflite
#   - 设备侧工作目录：/volume1/OES/och-ms
#
# 设计要点（每条都踩过坑）：
#   - 走 ControlMaster 复用连接，避免连续新建 ssh 被节流（255）
#   - 每组 timeout 60（设备不可达时作业不会赖着不走）
#   - 输出先写 /tmp 再 cp 到持久卷 + sync（硬断电会丢未落盘文件）
#   - 判据是「解析后的数值」，不是「有没有打印」：每组都记录 rc 与 hash

set -u

SO_DEV="${1:?用法: $0 <设备上的 .so 路径> [evidence 子目录名]}"
TAG="${2:-roundmode-$(date +%Y%m%d-%H%M)}"
CTL=/tmp/opencode/ssh-ctl
KEY=/root/.config/opencode/.secrets/ssh/id_ed25519
DEV=root@192.168.123.184
LOCAL_EV="$(dirname "$0")/../verify/evidence/$TAG"
MODEL_DEV=/volume1/OES/och-ms/nndump/sweep/sweep1x1-c0248-nondg.tflite
WORK_DEV=/volume1/OES/och-ms

mkdir -p "$LOCAL_EV"

# ---- 0. 复用连接 ----
if ! ssh -S "$CTL" -O check "$DEV" >/dev/null 2>&1; then
    echo "==> 建立复用连接"
    rm -f "$CTL"
    ssh -M -S "$CTL" -o ControlPersist=1800 \
        -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -o ConnectTimeout=10 -fN "$DEV" || exit 1
fi

SSH="ssh -S $CTL -T -o BatchMode=yes -o LogLevel=ERROR $DEV"

echo "==> 产物核对（设备侧 md5）"
$SSH "md5sum '$SO_DEV' | cut -d' ' -f1; wc -c < '$SO_DEV'"

# ---- 1. 五组：默认 + 0/1/2/3 ----
echo "==> 跑 5 组（每组 60 s 硬超时）"
$SSH "cd $WORK_DEV
mkdir -p nndump/probe
run() {
  tag=\$1; shift
  L=nndump/probe/rm-\$tag.log
  timeout 60 env ETNA_MESA_DEBUG=ml_msgs \"\$@\" LD_LIBRARY_PATH=nndump:. \
    ./tf_load_check $MODEL_DEV '$SO_DEV' --invoke 1 --output-dump /tmp/rm-\$tag.bin > \"\$L\" 2>&1
  echo \"  rc=\$? [\$tag]\"
  grep -a -m1 'INVOKE n=' \"\$L\" | cut -c1-160
  grep -a -m1 'och-roundmode' \"\$L\" | cut -c1-60
  cp /tmp/rm-\$tag.bin nndump/probe/ 2>/dev/null
}
run default TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1
run rm0 TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1 TEFLON_ROUNDMODE=0
run rm1 TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1 TEFLON_ROUNDMODE=1
run rm2 TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1 TEFLON_ROUNDMODE=2
run rm3 TEFLON_WDOMAIN=1 TEFLON_ODETRANS=1 TEFLON_ROUNDMODE=3
sync; echo '  sync done'
ls -l nndump/probe/rm-*.bin | awk '{print \"  \"\$5\" B  \"\$9}'"

# ---- 2. 拉回输出与日志 ----
echo "==> 拉回证据到 $LOCAL_EV"
for t in default rm0 rm1 rm2 rm3; do
    $SSH "cat $WORK_DEV/nndump/probe/rm-$t.bin" > "$LOCAL_EV/rm-$t.bin" 2>/dev/null
    $SSH "cat $WORK_DEV/nndump/probe/rm-$t.log" > "$LOCAL_EV/rm-$t.log" 2>/dev/null
done
ls -l "$LOCAL_EV" | head -12

# ---- 3. 比对 CPU 基准 ----
echo "==> vs CPU 基准（cpu-248.bin）"
python3 - "$LOCAL_EV" "$(dirname "$0")/../verify/evidence/coef-probe-20261021/cpu-248.bin" <<'PY'
import os
import re
import sys
import collections

ev, cpu_path = sys.argv[1], sys.argv[2]
i8 = lambda b: [x - 256 if x > 127 else x for x in b]
cpu = i8(open(cpu_path, "rb").read())

def log_hash(t):
    p = os.path.join(ev, "rm-%s.log" % t)
    try:
        m = re.search(r"out_hash=(0x[0-9a-f]+)", open(p, encoding="utf-8",
                                                    errors="replace").read())
    except OSError:
        return "?"
    return m.group(1) if m else "?"

print("  %-8s %-20s %-8s %-9s %s" % ("组", "hash", "一致率", "npu>cpu", "差值分布"))
for t in ("default", "rm0", "rm1", "rm2", "rm3"):
    p = os.path.join(ev, "rm-%s.bin" % t)
    if not os.path.exists(p) or os.path.getsize(p) != len(cpu):
        print("  %-8s （缺失或大小不符）" % t)
        continue
    d = i8(open(p, "rb").read())
    h = collections.Counter(a - b for a, b in zip(d, cpu))
    gt = sum(1 for a, b in zip(d, cpu) if a > b)
    lt = sum(1 for a, b in zip(d, cpu) if a < b)
    print("  %-8s %-20s %6.2f%%  gt=%-7d %s (lt=%d)"
          % (t, log_hash(t), 100.0 * h[0] / len(cpu), gt,
             sorted(h.items())[:4], lt))
PY
