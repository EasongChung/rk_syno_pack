#!/bin/sh
# ============================================================================
# ssh-chunk-upload.sh —— 分块 + 逐块校验重试的 SSH 上传（该设备链路随机 255）
#
# 为什么需要它：2026-10-20 实测，本机 → 192.168.123.184 的 ssh 通道**随机**失败
#   （rc=255，远端 0 字节），与块大小无关：8 KB 失败过、32 KB 成功过。
#   34 MB 的 .so 因此无法一次传完。
#
# 做法：
#   1) 每块单独上传到独立文件 c_NNNN（坏块只重传该块，不污染已成功的块）
#   2) 每次上传在**同一个连接**里回报远端字节数 ⇒ 一次往返即可判定成败
#   3) 全部到齐后按序 cat 合并，再用 md5 与本地比对
#
# 用法：
#   sh ssh-chunk-upload.sh <本地文件> <远端目录> <远端文件名> [块大小=32768]
#
# 退出码：0 成功（md5 一致）/ 1 某块重试耗尽 / 2 参数或环境错误 / 3 md5 不一致
# ============================================================================
set -u

SRC=${1:-}
RDIR=${2:-}
RNAME=${3:-}
CHUNK=${4:-32768}

[ -f "$SRC" ] || { echo "ERROR: 本地文件不存在: $SRC"; exit 2; }
[ -n "$RDIR" ] && [ -n "$RNAME" ] || { echo "用法: $0 <本地文件> <远端目录> <远端文件名> [块大小]"; exit 2; }

# ★ 该设备 sshd 有连接节流：连续新建十几条连接后会整体拒绝（255）。
#   实测「5 条连发全成、第 6 条起全灭」⇒ 必须复用同一条连接。
#   先建主连接：  ssh -M -S $CTL -o ControlPersist=600 -fN <opts> root@host
#   然后本脚本带 CTL=... 运行，所有命令都走复用通道。
CTL=${CTL:-}
MUX=""
[ -n "$CTL" ] && [ -S "$CTL" ] && MUX="-S $CTL -o ControlMaster=no"
SSHOPT="-i /root/.config/opencode/.secrets/ssh/id_ed25519 \
 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10 -o ServerAliveInterval=10 $MUX"
HOST=root@192.168.123.184

TOTAL=$(stat -c %s "$SRC")
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
LOCAL_MD5=$(md5sum "$SRC" | cut -d' ' -f1)

echo "文件: $SRC  ($TOTAL B, md5=$LOCAL_MD5)"
echo "分块: $N 块 × $CHUNK B  →  $RDIR/$RNAME"

# 远端准备
timeout 60 ssh -T $SSHOPT $HOST "mkdir -p '$RDIR/.chunks' && rm -f '$RDIR/.chunks'/c_*" || {
    echo "ERROR: 远端目录准备失败"; exit 2; }

i=0
while [ "$i" -lt "$N" ]; do
    idx=$(printf '%04d' "$i")
    dd if="$SRC" bs="$CHUNK" skip="$i" count=1 2>/dev/null > /tmp/opencode/xfer/_chunk.bin
    want=$(stat -c %s /tmp/opencode/xfer/_chunk.bin)

    try=1
    ok=0
    while [ "$try" -le 6 ]; do
        got=$(cat /tmp/opencode/xfer/_chunk.bin | \
              timeout 90 ssh -T $SSHOPT $HOST \
                "cat > '$RDIR/.chunks/c_$idx'; stat -c %s '$RDIR/.chunks/c_$idx'" 2>/dev/null | tail -1)
        if [ "$got" = "$want" ]; then ok=1; break; fi
        try=$((try + 1))
        sleep 1
    done
    if [ "$ok" -ne 1 ]; then
        echo "ERROR: 块 $idx 重试耗尽（期望 $want B，最后得到 '${got:-空}'）"
        exit 1
    fi
    i=$((i + 1))
    # 每 10 块报一次进度，避免刷屏
    if [ $((i % 10)) -eq 0 ] || [ "$i" -eq "$N" ]; then
        echo "  进度: $i/$N 块"
    fi
done

echo "合并 + 校验 ..."
REMOTE_MD5=$(timeout 180 ssh -T $SSHOPT $HOST \
  "cd '$RDIR/.chunks' && cat \$(ls c_* | sort) > '$RDIR/$RNAME' && md5sum '$RDIR/$RNAME' | cut -d' ' -f1" 2>/dev/null | tail -1)

if [ "$REMOTE_MD5" != "$LOCAL_MD5" ]; then
    echo "ERROR: md5 不一致 —— 本地 $LOCAL_MD5 / 远端 ${REMOTE_MD5:-空}"
    exit 3
fi

timeout 60 ssh -T $SSHOPT $HOST "rm -rf '$RDIR/.chunks'" >/dev/null 2>&1
echo "OK: 上传完成且 md5 一致（$RNAME, $TOTAL B）"
