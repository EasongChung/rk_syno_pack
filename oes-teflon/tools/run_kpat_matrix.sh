#!/bin/sh
# run_kpat_matrix.sh —— 一条命令跑完「kernel pattern 0x6」的三组单变量对照并给判决
#
# 判据（报告 30 的假设 H）：
#   H：描述符声明 kernel_pattern = 0x6（0x7e/0x0）这一档，会让 A311D 的 VIPNano-QI
#      NN core 无法推进作业 ⇒ 挂死（驱动判「前端 DMA 指针不再前进」）。
#
# 三组对照（**同一份 .so**，只切运行期 TEFLON_KPAT，其余一律不动）：
#
#   组 1  基线（不设 KPAT）      122 通道 conv（iso-seg04）  期望【不挂】
#                                244 通道 conv（bis-s09p03） 期望【挂】
#   组 2  TEFLON_KPAT=6          122 通道 conv              期望【挂】 ⇒ H 成立
#   组 3  TEFLON_KPAT=3          244 通道 conv              期望【不挂】⇒ H 成立
#
# ⚠️ 安全：只跑 1×1 conv 的段（真机实测为**可恢复挂死**：驱动复位、机器存活）。
#   **绝不**跑含 depthwise 的段（244 通道 depthwise 会**整机冻死**，需物理断电）。
#   本脚本对 bis-s09p05 / bis-s12p06 做硬拒绝。
#
# 用法:
#   tools/run_kpat_matrix.sh <设备上的 kpat 版 libteflon.so 路径> [诊断模型目录]
#
# 产物: verify/evidence/npu-hang/nndump-*.log（带 -kpat6 / -kpat3 后缀）
#        verify/evidence/npu-hang/kpat-verdict.md（判决表）

set -u
LIB="${1:-}"
DIAGDIR="${2:-/volume1/OES/och-ms/model-surgery/diag}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
PROBE="$HERE/tools/run_nndump_probe.sh"
SSH="ssh -T -i /root/.config/opencode/.secrets/ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@192.168.123.184"

[ -n "$LIB" ] || { echo "用法: $0 <kpat 版 libteflon.so 路径> [诊断模型目录]"; exit 2; }
[ -x "$PROBE" ] || { echo "ERROR: 找不到 $PROBE"; exit 2; }

for BAD in bis-s09p05 bis-s12p06; do
  case "$LIB$DIAGDIR" in
    *"$BAD"*) echo "FATAL: 命中已知会冻死整机的 $BAD —— 拒绝执行"; exit 3;;
  esac
done

OUT="$HERE/verify/evidence/npu-hang"
mkdir -p "$OUT"

RECOVER_BEFORE=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null)
BOOT_BEFORE=$($SSH 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null)
echo "实验前: recover=$RECOVER_BEFORE boot_id=$BOOT_BEFORE"

echo
echo "########## 组 1：基线（不设 KPAT ⇒ 原分档逻辑）##########"
"$PROBE" "$LIB" "$DIAGDIR"

echo
echo "########## 组 2：TEFLON_KPAT=6（强制 pattern 0x6/0x7e/0x0）##########"
TEFLON_KPAT=6 "$PROBE" "$LIB" "$DIAGDIR"

echo
echo "########## 组 3：TEFLON_KPAT=3（强制 pattern 0x3f/0xfffffffe/0xffffffff）##########"
TEFLON_KPAT=3 "$PROBE" "$LIB" "$DIAGDIR"

RECOVER_AFTER=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null)
BOOT_AFTER=$($SSH 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null)
echo
echo "实验后: recover=$RECOVER_AFTER (新增 $((RECOVER_AFTER-RECOVER_BEFORE)))  boot_id=$BOOT_AFTER"
[ "$BOOT_BEFORE" != "$BOOT_AFTER" ] && echo "⚠️ 设备重启过（boot_id 变了）—— 有作业触发了整机复位"

python3 "$HERE/tools/kpat_verdict.py" "$OUT" 2>&1 | tee "$OUT/kpat-verdict.md"
