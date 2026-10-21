#!/bin/sh
# 通过 SSH 向设备可靠传输文件（应对 SSH 通道 ~20% 静默数据损坏）
#
# 背景：OES 重装后 SSH 通道出现间歇性数据损坏（实测 16KB 分块 10 次中 2 次损坏），
#       而 SMB/HTTP 需要认证。本脚本用「分块 + 每块 SHA256 校验 + 失败重传」
#       把不可靠通道变成可靠通道。
#
# 用法：./push-reliable.sh <本地文件> [远端路径]
# 依赖：ssh/scp 可用；设备需已装公钥（见 docs/08-network-diagnosis.md）

set -u

SRC="$1"
DST="${2:-/tmp/$(basename "$1")}"
SSH_HOST="root@192.168.123.183"
SSHOPT="-o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts \
        -o BatchMode=yes -o ConnectTimeout=10 -i /root/.ssh/id_ed25519"
BLOCK=8192
MAX_RETRY=12

[ -f "$SRC" ] || { echo "用法: $0 <本地文件> [远端路径]" >&2; exit 1; }

WANT=$(sha256sum "$SRC" | awk '{print $1}')
SIZE=$(wc -c < "$SRC")
N=$(( (SIZE + BLOCK - 1) / BLOCK ))
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "=== 传输 $SRC → $DST ==="
echo "    大小 $SIZE B，分 $N 块（${BLOCK}B/块），每块校验"

REMOTE_TMP="/tmp/.oca_push.$$"
ssh $SSHOPT "$SSH_HOST" "rm -f $REMOTE_TMP; touch $REMOTE_TMP" >/dev/null 2>&1

i=0
while [ "$i" -lt "$N" ]; do
	dd if="$SRC" bs=$BLOCK skip=$i count=1 of="$TMP/chunk" 2>/dev/null
	want=$(sha256sum "$TMP/chunk" | awk '{print $1}')

	got=""
	attempt=0
	while [ "$attempt" -lt "$MAX_RETRY" ]; do
		attempt=$((attempt + 1))
		ssh $SSHOPT "$SSH_HOST" "rm -f $REMOTE_TMP.c" >/dev/null 2>&1
		ssh $SSHOPT "$SSH_HOST" 'cat > '"$REMOTE_TMP"'.c' < "$TMP/chunk" >/dev/null 2>&1
		got=$(ssh $SSHOPT "$SSH_HOST" "sha256sum $REMOTE_TMP.c 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n')
		[ "$got" = "$want" ] && break
		sleep 1
	done

	if [ "$got" != "$want" ]; then
		echo "  块 $i 在 $MAX_RETRY 次重试后仍失败，放弃" >&2
		ssh $SSHOPT "$SSH_HOST" "rm -f $REMOTE_TMP $REMOTE_TMP.c" >/dev/null 2>&1
		exit 1
	fi

	ssh $SSHOPT "$SSH_HOST" "cat $REMOTE_TMP.c >> $REMOTE_TMP; rm -f $REMOTE_TMP.c" >/dev/null 2>&1
	i=$((i + 1))
	[ $((i % 10)) -eq 0 ] && echo "  ...$i/$N"
done

ssh $SSHOPT "$SSH_HOST" "mv $REMOTE_TMP $DST && chmod 755 $DST" >/dev/null 2>&1

FINAL=$(ssh $SSHOPT "$SSH_HOST" "sha256sum $DST 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n')
echo
if [ "$FINAL" = "$WANT" ]; then
	echo "✓ 传输完成且校验通过"
	echo "  $(basename "$DST")  sha256=$FINAL"
	exit 0
else
	echo "✗ 最终校验失败：$FINAL != $WANT" >&2
	exit 1
fi