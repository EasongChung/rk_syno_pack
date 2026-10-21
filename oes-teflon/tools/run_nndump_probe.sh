#!/bin/sh
# run_nndump_probe.sh —— 用「NN core 描述符 dump」产物做一次取证比对
#
# 目的（承接报告 28）：
#   挂死最小单元已缩到**单个 1×1 CONV_2D（244→244 通道）**；
#   对照 122→122 的同型 conv **不挂**。
#   驱动判挂死的依据是「前端 DMA 指针不再前进」⇒ NPU 连作业都没开始推进
#   ⇒ 极可能是**下发给硬件的描述符本身不合法**。
#   本脚本用带 [och-nn] dump 的产物，把两个作业的描述符**逐字节**打出来，
#   离线 diff 即可锁定是哪个字段越界/错位。
#
# ⚠️ 风险分级（务必读）：
#   本脚本**只跑 1×1 conv 的段**（#9 前 3 算子 / #12 前 4 算子 / #4 全段），
#   这三个在真机上实测均为**可恢复挂死**（驱动复位、机器存活）。
#   ★ **绝不**跑含 depthwise 的段（#9 前 5 算子 / #12 前 6 算子）：
#     实测 244 通道 depthwise 会**整机冻死**（需物理断电）。
#   脚本里对模型名做白名单校验，含 "s09p05" / "s12p06" 一律拒绝。
#
# 用法:
#   tools/run_nndump_probe.sh <libteflon.so 路径> [设备上的诊断模型目录]
#
#   ★ KPAT 单变量对照（同一份 .so，运行期切 pattern 强制值）：
#       tools/run_nndump_probe.sh <so>              # 基线（不设 KPAT ⇒ 原分档逻辑）
#       TEFLON_KPAT=6 tools/run_nndump_probe.sh <so> # 强制 pattern 0x6（122 通道 conv 挂 ⇒ H 成立）
#       TEFLON_KPAT=3 tools/run_nndump_probe.sh <so> # 强制 pattern 0x3f（244 通道 conv 不挂 ⇒ H 成立）
#     产物文件名带 -kpat6 / -kpat3 后缀，互不覆盖。
#
# 产物（落在本机）:
#   verify/evidence/npu-hang/nndump-<模型名>.log   每个模型的 run 日志
#   verify/evidence/npu-hang/nndump-summary.md     两个作业的描述符 diff 摘要

set -u
LIB="${1:-}"
DIAGDIR="${2:-/volume1/OES/och-ms/model-surgery/diag}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT="$HERE/verify/evidence/npu-hang"
SSH="ssh -T -i /root/.config/opencode/.secrets/ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@192.168.123.184"

[ -n "$LIB" ] || { echo "用法: $0 <libteflon.so 路径> [诊断模型目录]"; exit 2; }

# ★ 安全白名单：只允许这四个（挂死的 1×1 conv 两个 + 正常对照两个）
ALLOW="bis-s09p03 bis-s12p04 iso-seg09 iso-seg04"
for BAD in bis-s09p05 bis-s12p06; do
  case " $ALLOW " in
    *" $BAD "*) echo "FATAL: 白名单里出现了已知会冻死整机的 $BAD —— 拒绝执行"; exit 3;;
  esac
done

mkdir -p "$OUT"
RECOVER_BEFORE=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null)
BOOT_BEFORE=$($SSH 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null)
echo "实验前: recover=$RECOVER_BEFORE boot_id=$BOOT_BEFORE"

for M in $ALLOW; do
  echo "=== $M ==="
  # KPAT 环境变量透传到设备端（同一份 .so，切运行期开关做单变量对照）
  KPENV=""
  [ -n "${TEFLON_KPAT:-}" ] && KPENV="TEFLON_KPAT=$TEFLON_KPAT"
  SUF=""
  [ -n "$KPENV" ] && SUF="-kpat$TEFLON_KPAT"
  $SSH "cd /volume1/OES/och-ms && TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs $KPENV LD_LIBRARY_PATH=. \
        ./tf_load_check $DIAGDIR/$M.tflite $LIB --invoke 1 --output-dump $M.bin 2>&1; \
        echo \"---REMOTE_EXIT=\$?---\"" > "$OUT/nndump-$M$SUF.log" 2>&1
  grep -aoE "\[och-nn\].*" "$OUT/nndump-$M$SUF.log" | head -3 | cut -c1-200 | sed 's/^/  /'
  grep -aE "^INVOKE n=|REMOTE_EXIT" "$OUT/nndump-$M$SUF.log" | head -2 | cut -c1-160 | sed 's/^/  /'
done

RECOVER_AFTER=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null)
BOOT_AFTER=$($SSH 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null)
echo "实验后: recover=$RECOVER_AFTER (新增 $((RECOVER_AFTER-RECOVER_BEFORE)))  boot_id=$BOOT_AFTER"
if [ "$BOOT_BEFORE" != "$BOOT_AFTER" ]; then
  echo "⚠️ 设备重启过（boot_id 变了）—— 说明有作业触发了整机复位"
fi

# ---- 离线 diff：把两个作业的 raw 描述符字节对齐比较 ----
python3 "$HERE/tools/diff_nn_desc.py" "$OUT" 2>&1 | tee "$OUT/nndump-summary.md"
echo
echo "产物在 $OUT/（nndump-*.log + nndump-summary.md）"
