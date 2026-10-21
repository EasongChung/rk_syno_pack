#!/bin/sh
# 独立复核 oca 的 check-tee-pipefail.py：正控制 + 负控制 + 三组反作弊变体 + 计数自指
#
# 用法：tools/review_check_tee_pipefail.sh <门脚本路径> <workflow 路径>
#
# 为什么要有这个脚本
# ------------------
# 这道门的存在意义就是「防止 `cmd | tee` 吞掉退出码造成假绿」。
# 一道**专门防假绿**的门，如果自己能被文字游戏骗过，那它守住的只是心理安慰。
# 所以验收它不能只跑正/负控制，必须主动**攻击它**。
#
# 期望结果：
#   [1] 正控制              真 workflow                             ⇒ EXIT=0，且输出必须含「证据:」
#   [2] 负控制·真实样本      去掉 pipefail                           ⇒ EXIT=1
#   [3] 反作弊 A            去掉 pipefail + 注释含「pipefail」二字     ⇒ EXIT=1
#   [4] 反作弊 B            去掉 pipefail + 注释伪装成 set -euxo pipefail ⇒ EXIT=1
#   [5] 反作弊 C            去掉 pipefail + echo 里喊 set -o pipefail  ⇒ EXIT=1
#   [6] 计数自指            门报的 tee 块数 == 真实 | tee 行数          ⇒ 相等
#
# ★ 两处刻意的写法，都是为了避开本仓库反复出现的坑：
#   1) 退出码直接取自 python，**不经任何管道** —— 管道会吞掉退出码（取到 sed/tee 的恒 0）。
#      本脚本初版用 `python3 ... | sed` 后取 `$?`，差点报出「门在 FAIL 时也返回 0」的假结论。
#   2) 样本放 src/ 子目录，被测副本放 cases/ 子目录 —— 否则清理动作会删掉样本本身
#      （本脚本初版用 `rm -f "$TMP"/*.yml` 把刚 sed 出来的样本删了，导致负控制**假绿**）。
set -u

GATE=${1:?用法: $0 <门脚本路径> <workflow 路径>}
WF=${2:?用法: $0 <门脚本路径> <workflow 路径>}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src" "$TMP/cases"

[ -f "$GATE" ] || { echo "ERROR: 找不到门脚本 $GATE"; exit 2; }
[ -f "$WF" ]   || { echo "ERROR: 找不到 workflow $WF"; exit 2; }

fails=0
POS_OUT="$TMP/pos.out"

# run_case <标签> <期望退出码> <样本路径>
run_case() {
	label="$1"; expect="$2"; src="$3"
	rm -f "$TMP/cases"/*.yml
	if ! cp "$src" "$TMP/cases/case.yml" 2>"$TMP/cp.err"; then
		printf '  [FAIL] %-24s 样本不存在：%s\n' "$label" "$(cat "$TMP/cp.err")"
		fails=$((fails + 1)); echo; return
	fi
	python3 "$GATE" --dir "$TMP/cases" > "$TMP/out.txt" 2>&1
	rc=$?
	if [ "$rc" -eq "$expect" ]; then
		printf '  [OK  ] %-24s EXIT=%s（期望 %s）\n' "$label" "$rc" "$expect"
	else
		printf '  [FAIL] %-24s EXIT=%s（期望 %s）\n' "$label" "$rc" "$expect"
		fails=$((fails + 1))
	fi
	sed 's/^/          /' "$TMP/out.txt"
	echo
}

# run_case 的变体：把正控制的输出留档，供 [7] 检查「PASS 必须带证据」
run_case_keep() {
	run_case "$@"
	cp "$TMP/out.txt" "$POS_OUT" 2>/dev/null
}

# 把真 workflow 里的 `set -euxo pipefail` 降级为 `set -eux`，
# 并在其后追加一行 $1（用于构造各种欺骗样本）
make_fooled() {
	_payload="$1"
	python3 - "$WF" "$TMP/src/wf-fooled.yml" "$_payload" <<'PY'
import sys
wf, out, payload = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(wf, encoding='utf-8').read().split('\n')
res = []
for l in lines:
    if 'set -euxo pipefail' in l:
        res.append(l.replace('set -euxo pipefail', 'set -eux'))
        if payload:
            res.append('          ' + payload)
    else:
        res.append(l)
open(out, 'w', encoding='utf-8').write('\n'.join(res))
PY
}

# 只统计「真正的」| tee：排除整行注释，也排除 `- name:` 声明行
real_tee_count() {
	grep -nE '\|\s*tee' "$1" 2>/dev/null \
		| grep -vE '^[0-9]+:[[:space:]]*#' \
		| grep -vE '^[0-9]+:[[:space:]]*-[[:space:]]*name:' \
		| wc -l | tr -d ' '
}

echo '=== 复核 check-tee-pipefail.py ==='
echo

echo '--- [1] 正控制：真 workflow（应 PASS，且必须带证据）---'
run_case_keep '正控制' 0 "$WF"

echo '--- [2] 负控制·真实样本：去掉 pipefail ---'
sed 's/set -euxo pipefail/set -eux/' "$WF" > "$TMP/src/wf-bad.yml"
run_case '负控制' 1 "$TMP/src/wf-bad.yml"

echo '--- [3] 反作弊 A：注释里含「pipefail」二字 ---'
make_fooled '# 本步未开 pipefail（复核用注释）'
run_case '反作弊A' 1 "$TMP/src/wf-fooled.yml"

echo '--- [4] 反作弊 B：注释伪装成 set -euxo pipefail ---'
make_fooled '# set -euxo pipefail 其实没开（伪装注释）'
run_case '反作弊B' 1 "$TMP/src/wf-fooled.yml"

echo '--- [5] 反作弊 C：echo 里喊 set -o pipefail，实际没开 ---'
make_fooled 'echo "set -o pipefail 已开启"'
run_case '反作弊C' 1 "$TMP/src/wf-fooled.yml"

echo '--- [6] 计数自指检查 ---'
real=$(real_tee_count "$WF")
rm -f "$TMP/cases"/*.yml
cp "$WF" "$TMP/cases/case.yml"
python3 "$GATE" --dir "$TMP/cases" > "$TMP/out6.txt" 2>&1
reported=$(grep -oE '共扫 [0-9]+ 个' "$TMP/out6.txt" | grep -oE '[0-9]+' | head -1)
printf '  真正的 | tee 行数（已排注释与 name 行） = %s\n' "$real"
printf '  门报告的 tee 块数                        = %s\n' "${reported:-未知}"
if [ "${reported:-0}" = "$real" ]; then
	echo '  [OK  ] 计数一致'
else
	echo '  [FAIL] 计数不一致 ⇒ 门很可能把注释行 / 自身步骤名数了进去（自指）'
	fails=$((fails + 1))
fi
echo

echo '--- [7] 纪律检查：PASS 必须给出「它匹配到了哪段文本」---'
if grep -q '证据:' "$POS_OUT" 2>/dev/null; then
	echo '  [OK  ] PASS 带证据行'
	grep -o '证据:.*' "$POS_OUT" | head -1 | sed 's/^/          /'
else
	echo '  [FAIL] PASS 未给出匹配到的文本（违反「一条 PASS 必须给出它匹配到了哪段文本」）'
	fails=$((fails + 1))
fi
echo

if [ "$fails" -eq 0 ]; then
	echo '复核结果: 全部通过'
	exit 0
fi
echo "复核结果: $fails 项不符预期 —— 该门存在缺陷，详见上方输出"
exit 1
