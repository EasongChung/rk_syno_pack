#!/bin/sh
# push_to_device.sh —— 分块并行推送大文件到 A311D 设备
#
# 为什么需要这个：
#   设备（DSM）sshd 对**单条连接**的上行带宽限在 ~2–4 KB/s（实测：100 KB 单路要 30 s，
#   且以 32768 B 为块、每 ~15 s 放一块）。scp 不可用。实测 **并行 N 路可线性提速**
#   （4 路 100 KB / 10 s）。故：本地切块 → 并行推 → 设备端按序 cat 合并 → 双端 sha256 校验。
#
# 用法: tools/push_to_device.sh <本地文件> <设备绝对路径> [并行路数，默认 8]
# 依赖: ssh（非交互、无密码、BatchMode）；不改动设备上任何既有文件（只写目标路径）
set -u

SRC="$1"
DST="$2"
JOBS="${3:-4}"
SSH="ssh -T -i /root/.config/opencode/.secrets/ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@192.168.123.184"
# 块大小：实测单连接上行 ~0.8 KB/s ⇒ 256 KB/块需要 ~320 s，会撞上单块超时（180 s）
# ⇒ 每块必须 **小于** 单块超时 × 单连接速率。默认 64 KB（≈80 s/块，留足余量）。
CHUNK=${CHUNK_BYTES:-65536}
WORK="/tmp/opencode/push_$$"
RDIR="/tmp/_push_$$"

[ -f "$SRC" ] || { echo "错误: 本地文件不存在: $SRC"; exit 1; }
mkdir -p "$WORK"

TOTAL=$(wc -c < "$SRC")
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "推送: $SRC -> $DST"
echo "  大小 $TOTAL B，切成 $NCHUNK 块（每块 $CHUNK B），并行 $JOBS 路"
LSHA=$(sha256sum "$SRC" | cut -d' ' -f1)

# 设备端先清干净临时目录
# ★ 同样不能只看返回码（远端无 stdout ⇒ DSM sshd 回 255）⇒ 改用「目录是否真的建出来」判定
$SSH "rm -rf $RDIR; mkdir -p $RDIR" >/dev/null 2>&1 || true
$SSH "test -d $RDIR" >/dev/null 2>&1 || { echo "错误: 设备端临时目录建不出来: $RDIR"; exit 1; }

i=0
while [ "$i" -lt "$NCHUNK" ]; do
  dd if="$SRC" of="$WORK/c$(printf %03d "$i").bin" bs="$CHUNK" skip="$i" count=1 2>/dev/null
  i=$((i+1))
done

# 并行推（DSM 的 sshd 对并发连接数有限制，实测 8 路会被 reset，故默认 4 路 + 重试）
push_chunk() {
  _i="$1"
  _f="$WORK/c$(printf %03d "$_i").bin"
  _r="$RDIR/c$(printf %03d "$_i").bin"
  _want=$(wc -c < "$_f")
  for _a in 1 2 3; do
    # ★ 必须给单块加超时：实测某条连接会**停在半途不再前进**（10 分钟 0 字节），
    #   没有超时就会永远挂住；超时后重推（远端文件会被覆盖，不追加）。
    # ★★ **不能只看 ssh 的返回码**：DSM 的 sshd 在**远端命令不产生任何 stdout** 时
    #   返回 255（实测 `cat > f` ⇒ 255；`cat > f; echo X` ⇒ 0；`tee f >/dev/null` ⇒ 0），
    #   而数据其实早已落盘 ⇒ 早期版本把 255 当失败 ⇒ 每块重传最多 6 次
    #   ⇒ 433 KB 的文件最多重传 2.6 MB ⇒ **自己把上行打成重传风暴** ⇒ 越推越慢。
    #   ⇒ 判据改为：**核对远端文件的实际大小**，返回码只当线索。
    timeout 180 $SSH "cat > $_r" < "$_f" 2>/dev/null || true
    _got=$($SSH "stat -c %s $_r 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r')
    if [ "$_got" = "$_want" ]; then
      return 0
    fi
    echo "    块 $(printf %03d "$_i") 第 $_a 次推送未完成（远端 $_got != $_want），重试"
    sleep 2
  done
  return 1
}

i=0
FAILED=""
PIDS=""
# ★★ 限流必须**自己数 PID**：`$(jobs -p | wc -l)` 在命令替换的**子 shell** 里执行，
#    看不到父 shell 的后台作业 ⇒ 计数恒为 0 ⇒ 限流从未生效 ⇒ 27 块**全在并发推**
#    ⇒ 把 DSM 的 sshd 顶爆（连接被 reset）⇒ 每块都失败 ⇒ 重传风暴。
#    （实测：DSM sshd 基本只吃 1 条并发会话 ⇒ JOBS=1 才稳）
n_alive() {
  _c=0
  for _p in $PIDS; do
    kill -0 "$_p" 2>/dev/null && _c=$((_c+1))
  done
  echo "$_c"
}
while [ "$i" -lt "$NCHUNK" ]; do
  ( push_chunk "$i" || echo "$i" >> "$WORK/failed.list" ) &
  PIDS="$PIDS $!"
  i=$((i+1))
  # 维持最多 $JOBS 个并发（自己数，不用 jobs）
  while [ "$(n_alive)" -ge "$JOBS" ]; do sleep 0.2; done
done
wait

# 逐块核对大小，不对的重推
i=0
while [ "$i" -lt "$NCHUNK" ]; do
  WANT=$(wc -c < "$WORK/c$(printf %03d "$i").bin")
  GOT=$($SSH "stat -c %s $RDIR/c$(printf %03d "$i").bin 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r')
  if [ "$GOT" != "$WANT" ]; then
    echo "  块 $(printf %03d "$i") 大小不符（$GOT != $WANT），重推"
    push_chunk "$i"
  fi
  i=$((i+1))
done

# 设备端按序合并 + 校验
echo "  合并并校验..."
$SSH "cat $RDIR/c*.bin > '$DST' && sync && sha256sum '$DST' && rm -rf $RDIR"
RSHA=$($SSH "sha256sum '$DST'" 2>/dev/null | cut -d' ' -f1)
rm -rf "$WORK"

# ★ 必须确认落地大小：DSM 的 /volume1 若在硬断电时丢页缓存，文件会变成 0 字节
#   （实测 2026-10-19 23:11 推送的 4 个诊断模型在硬断电后全部变 0 字节）。
#   ⇒ 合并后 sync，再核对大小；不一致直接失败，不要"看起来成功了"。
RSIZE=$($SSH "stat -c %s '$DST' 2>/dev/null" 2>/dev/null | tr -d '\r')
LSIZE=$(wc -c < "$SRC")
echo "  本地 sha256: $LSHA"
echo "  设备 sha256: $RSHA"
echo "  落地大小: ${RSIZE:-?} / 本地 ${LSIZE}"
if [ "$RSIZE" != "$LSIZE" ]; then
  echo "  ❌ 设备端大小与本地不一致（可能被截断或未落盘）"
  exit 1
fi
if [ "$LSHA" = "$RSHA" ]; then
  echo "  ✅ 一致"
  exit 0
else
  echo "  ❌ 不一致"
  exit 1
fi
