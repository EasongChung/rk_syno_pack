#!/bin/sh
# ============================================================================
# ssh-mux-upload.sh —— 单连接复用 + 小块的上传器（针对本设备实测的两种故障）
#
# 实测故障（2026-10-20，本机 → 192.168.123.184）：
#   ① 连接节流：连续新建 ~6 条连接后，新连接被整体拒绝（rc=255，立即返回）
#   ② 窗口停住：单条流推过 ~96 KB 就停住（TCP 窗口不增长），远端只留半截文件
#
# 对策：
#   - 用 ControlMaster **只建一条** TCP 连接，所有块走复用通道（治 ①）
#   - 块大小 32 KB，保证每块能在单个窗口内送完（治 ②）
#   - 每块**独立成文件** + 同连接内回读字节数校验，坏块只重传该块
#   - 主连接死了就自动重建（最多 3 次）
#
# 用法：
#   sh ssh-mux-upload.sh <本地文件> <远端目录> <远端文件名> [块大小=32768]
# 环境：
#   CTLSOCK  控制套接字路径（默认 /tmp/opencode/xfer/mux）
#
# 退出码：0 成功（md5 一致）/ 1 重试耗尽 / 2 参数或环境错误 / 3 md5 不一致
# ============================================================================
set -u

SRC=${1:-}
RDIR=${2:-}
RNAME=${3:-}
CHUNK=${4:-32768}
CTLSOCK=${CTLSOCK:-/tmp/opencode/xfer/mux}
TMPCH=${TMPCH:-/tmp/opencode/xfer/_chunk.bin}

[ -f "$SRC" ] || { echo "ERROR: 本地文件不存在: $SRC"; exit 2; }
[ -n "$RDIR" ] && [ -n "$RNAME" ] || { echo "用法: $0 <本地文件> <远端目录> <远端文件名> [块大小]"; exit 2; }
mkdir -p "$(dirname "$CTLSOCK")" "$(dirname "$TMPCH")"

HOST=root@192.168.123.184
BASEOPT="-i /root/.config/opencode/.secrets/ssh/id_ed25519 \
 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=8 \
 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

TOTAL=$(stat -c %s "$SRC")
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
LOCAL_MD5=$(md5sum "$SRC" | cut -d' ' -f1)

# 复用通道；主连接不在就建一条
# 超时放进函数内部：timeout 不能执行 shell 函数
mux() { timeout 90 ssh -S "$CTLSOCK" $BASEOPT $HOST "$@"; }
ensure_master() {
    if [ -S "$CTLSOCK" ] && timeout 15 ssh -S "$CTLSOCK" -O check $BASEOPT $HOST >/dev/null 2>&1; then
        return 0
    fi
    rm -f "$CTLSOCK"
    timeout 30 ssh -M -S "$CTLSOCK" -o ControlPersist=900 -fN $BASEOPT $HOST 2>/dev/null
    sleep 1
    [ -S "$CTLSOCK" ]
}

echo "文件: $SRC  ($TOTAL B, md5=$LOCAL_MD5)"
echo "分块: $N × $CHUNK B  →  $RDIR/$RNAME"

ensure_master || { echo "ERROR: 主连接建立失败"; exit 2; }
mux "mkdir -p '$RDIR/.chunks' && rm -f '$RDIR/.chunks'/c_*" || { echo "ERROR: 远端准备失败"; exit 2; }

i=0
while [ "$i" -lt "$N" ]; do
    idx=$(printf '%04d' "$i")
    dd if="$SRC" bs="$CHUNK" skip="$i" count=1 2>/dev/null > "$TMPCH"
    want=$(stat -c %s "$TMPCH")

    try=1; ok=0; got=""
    while [ "$try" -le 8 ]; do
        ensure_master || { echo "ERROR: 主连接重建失败（块 $idx）"; exit 1; }
        got=$(cat "$TMPCH" | mux \
                "cat > '$RDIR/.chunks/c_$idx'; stat -c %s '$RDIR/.chunks/c_$idx'" 2>/dev/null | tail -1)
        if [ "$got" = "$want" ]; then ok=1; break; fi
        try=$((try + 1))
        sleep 2
    done
    [ "$ok" -eq 1 ] || { echo "ERROR: 块 $idx 重试耗尽（期望 $want，得到 '${got:-空}'）"; exit 1; }

    i=$((i + 1))
    [ $((i % 10)) -eq 0 ] || [ "$i" -eq "$N" ] && echo "  进度 $i/$N"
done

echo "合并 + 校验 ..."
REMOTE_MD5=$(mux "cd '$RDIR/.chunks' && cat \$(ls c_* | sort) > '$RDIR/$RNAME' && md5sum '$RDIR/$RNAME' | cut -d' ' -f1" 2>/dev/null | tail -1)
if [ "$REMOTE_MD5" != "$LOCAL_MD5" ]; then
    echo "ERROR: md5 不一致 —— 本地 $LOCAL_MD5 / 远端 ${REMOTE_MD5:-空}"
    exit 3
fi
mux "rm -rf '$RDIR/.chunks'" >/dev/null 2>&1
echo "OK: 上传完成且 md5 一致（$RNAME, $TOTAL B）"
