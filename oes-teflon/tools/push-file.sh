#!/bin/sh
# push-file.sh —— 通过 ssh 把本机文件可靠地推到设备
#
# 背景：该设备的 sshd 对大块 stdin 推送**会卡住**（实测 8 KB 稳定、70 KB 挂死），
#       且单次推送偶发超时（约 1/9 概率）⇒ 必须**分片 + 逐片校验 + 重试**。
#       ★ 不能只 append：丢一片就整文件错位 ⇒ 先把每片写成 partNNN 并核大小，
#         全部到位后在设备端按序 cat 拼接。
#
# 用法：tools/push-file.sh <本机文件> <设备路径> [片大小=8192] [重试=3]

set -u
LOCAL="${1:?用法: push-file.sh <local> <remote> [chunk] [retries]}"
REMOTE="${2:?}"
CHUNK="${3:-8192}"
RETRIES="${4:-3}"
SSH="ssh -T -i /root/.config/opencode/.secrets/ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@192.168.123.184"

[ -f "$LOCAL" ] || { echo "ERROR: 本机文件不存在: $LOCAL" >&2; exit 2; }

TMP=$(mktemp -d /tmp/opencode/push.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT
split -b "$CHUNK" -a 3 "$LOCAL" "$TMP/part."

total=$(ls "$TMP" | wc -l)
echo "push: $LOCAL -> $REMOTE  ($(wc -c < "$LOCAL") B, $total 片 × $CHUNK B)"
echo "[1/3] 清掉旧片"
timeout 30 $SSH "rm -f $REMOTE.part.*"; rc=$?
[ $rc -eq 0 ] || { echo "ERROR: 清旧片失败 rc=$rc" >&2; exit 3; }

echo "[2/3] 逐片上传（每片核大小，失败重试 $RETRIES 次）"
fail=0
for f in "$TMP"/part.*; do
    name=$(basename "$f")
    want=$(wc -c < "$f")
    ok=0
    i=1
    while [ $i -le "$RETRIES" ]; do
        timeout 40 $SSH "cat > $REMOTE.$name" < "$f"
        got=$(timeout 25 $SSH "stat -c %s $REMOTE.$name 2>/dev/null || echo -1")
        got=$(echo "$got" | tr -d '\r\n ')
        if [ "$got" = "$want" ]; then ok=1; break; fi
        i=$((i + 1))
    done
    if [ $ok -ne 1 ]; then
        echo "  ✗ $name 失败（期望 $want，实得 $got）"
        fail=$((fail + 1))
    fi
done
[ $fail -eq 0 ] || { echo "ERROR: $fail 片上传失败" >&2; exit 4; }

echo "[3/3] 设备端拼接 + 校验"
timeout 40 $SSH "cd \$(dirname $REMOTE) && cat \$(basename $REMOTE).part.* > \$(basename $REMOTE) && rm -f $(basename $REMOTE).part.* && ls -l \$(basename $REMOTE) && cksum \$(basename $REMOTE)" > "$TMP/remote.sum"
rc=$?
cat "$TMP/remote.sum"
[ $rc -eq 0 ] || { echo "ERROR: 拼接失败 rc=$rc" >&2; exit 5; }
echo "本机: $(cksum < "$LOCAL")"
echo "（两行 cksum 的 CRC 与字节数必须一致）"
