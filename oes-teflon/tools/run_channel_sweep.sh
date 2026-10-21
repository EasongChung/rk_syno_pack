#!/bin/sh
# 通道数扫描：只含 1×1 conv 的合成模型（不含 depthwise ⇒ 不会整机冻死）
#
# 用法：  sh tools/run_channel_sweep.sh <模型目录> <通道列表...>
#   例：  sh tools/run_channel_sweep.sh sweep2 242 246 248
#
# 判据（每道门都要自证）：
#   · 接管：日志里 3 个算子（RESHAPE/TRANSPOSE/CONV）必须全部 supported
#           ⇒ 否则「不挂」是假绿（回退 CPU 不算）
#   · 挂死：内核日志 "recover hung GPU" 计数增量 > 0
#   · 跑起：REMOTE_EXIT=0 且日志含 [och-nn] 描述符 dump
set -eu

DEV=192.168.123.184
SSH="ssh -T -i /root/.config/opencode/.secrets/ssh/id_ed25519 -o BatchMode=yes \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=8 root@${DEV}"
DIR=${1:?用法: run_channel_sweep.sh <模型目录> <通道...>}
shift
# 目录可写相对名（sweep2）或绝对路径；绝对路径原样用，否则补前缀。
# 2026-10-20 02:42 事故：传绝对路径又补前缀 ⇒ 路径重复 ⇒ 全点 "Could not open"。
case "$DIR" in
  /*) MODEL_DIR=$DIR ;;
  *)  MODEL_DIR=/volume1/OES/och-ms/nndump/$DIR ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
EV=${EVIDENCE_DIR:-$HERE/../verify/evidence/npu-hang/diag}
mkdir -p "$EV"

printf "%-6s %-10s %-6s %-8s %-8s %-6s %-10s %-10s\n" \
  C 结果 复位 接管 superblk kpc img_cache coef_cache

for C in "$@"; do
  # 设备上的模型名是 4 位补零（sweep1x1-c0248.tflite）。这里少补零 ⇒
  # tf_load_check 直接 "Could not open" ⇒ 全点假失败（2026-10-20 02:04 事故）。
  M=$(printf 'sweep1x1-c%04d' "$C")
  L="$EV/$M.log"
  # 注意：远端 grep -c 无匹配时打印 0 但退出码 1，若用 `|| echo 0` 会得到两行
  # "0\n0" ⇒ $(( )) 报 bad number。故只取最后一行、不追加。
  RB=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null | tr -d '\r' | tail -1 || true); RB=${RB:-0}
  timeout 600 $SSH "cd /volume1/OES/och-ms && TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs \
    LD_LIBRARY_PATH=. ./tf_load_check $MODEL_DIR/$M.tflite \
    /volume1/OES/och-ms/nndump/libteflon-kpat.so --invoke 1 --output-dump $M.bin 2>&1; \
    echo \"---REMOTE_EXIT=\$?---\"" > "$L" 2>&1 || true
  RA=$($SSH 'grep -ac "recover hung GPU" /var/log/kern.log' 2>/dev/null | tr -d '\r' | tail -1 || true); RA=${RA:-0}

  SUP=$(grep -ac ' supported ' "$L" 2>/dev/null | head -1); SUP=${SUP:-0}
  UNS=$(grep -ac ' unsupported ' "$L" 2>/dev/null | head -1); UNS=${UNS:-0}
  EX=$(grep -ao 'REMOTE_EXIT=[0-9-]*' "$L" 2>/dev/null | head -1 | cut -d= -f2); EX=${EX:-none}
  D=$(grep -ao '\[och-nn\][^|]*' "$L" 2>/dev/null | head -1); D=${D:-}
  HANG=$((RA - RB))

  if [ "${SUP:-0}" -lt 3 ]; then R="★未接管"; elif [ "${HANG:-0}" -gt 0 ]; then R="★挂"; else R="不挂"; fi
  [ "${EX:-1}" != "0" ] && R="$R/exit${EX:-?}"

  printf "%-6s %-10s %-6s %-8s %-8s %-6s %-10s %-10s\n" "$C" "$R" "$HANG" "$SUP/$UNS" \
    "$(echo "$D" | grep -oE 'superblocks=[0-9]+' | cut -d= -f2)" \
    "$(echo "$D" | grep -oE ' kpc=[0-9]+' | cut -d= -f2)" \
    "$(echo "$D" | grep -oE 'img_cache=[0-9]+' | cut -d= -f2)" \
    "$(echo "$D" | grep -oE 'coef_cache=[0-9]+' | cut -d= -f2)"
done
