#!/bin/sh
# ============================================================================
# probe-cross-ini.sh —— 本地校验 build-teflon.sh 生成的 cross.ini 是否正确
#
# 为什么需要它（真实踩坑）
# ----------------------
# 曾在 `cat > cross.ini <<EOF … EOF` **内部**写了 DBG_ARGS 的 if/else/fi，
# 于是 shell 的 `else` / `fi` 被当成 cross.ini 的内容写进文件，
# meson 直接报：
#
#     Malformed machine file: Source contains parsing errors:
#       [line 28]: 'else\n'
#
# 而 ninja 那步的失败又被管道 / `||` 掩盖，整个构建**看起来还是 success**，
# 结果是「拿着一个 cross.ini 损坏的产物上机，得到零信息」。
#
# 这个脚本把「渲染 cross.ini」这一步单独拎出来本地跑一遍，
# 不需要编译、不需要网络，几秒就能判定对错。
#
# 用法：sh build-recipe/probe-cross-ini.sh
# 退出码：0 = 渲染正确；2 = 渲染有问题（并打印问题清单）
# ============================================================================
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC="$SCRIPT_DIR/build-teflon.sh"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT INT TERM

[ -f "$SRC" ] || { echo "ERROR: 找不到 $SRC"; exit 2; }

# 从 build-teflon.sh 里抠出 heredoc 那一段（从 cat > ... <<EOF 到独占一行的 EOF）
sed -n '/^cat > "\$BUILD_DIR\/cross.ini" <<EOF$/,/^EOF$/{/^cat > /d; /^EOF$/d; p;}' "$SRC" \
	> "$OUT/block.tmpl"
if [ ! -s "$OUT/block.tmpl" ]; then
	echo "ERROR: 没能在 $SRC 里定位到 cross.ini 的 heredoc 段（脚本结构变了？）"
	exit 2
fi

fail=0
note() { printf '  %s\n' "$*"; }

# 用假变量渲染一次；DBG_ARGS 由调用方注入，模拟开关两种状态。
# 注意必须用 eval + 真正的 heredoc：`cat 文件` **不做变量展开**
#（真脚本是靠 heredoc 在 cat 时展开 $SR 等），直接 cat 模板会看到未展开的字面量。
render() {
	( # 子 shell：避免污染本进程
		BUILD_DIR="$OUT"
		CC_BIN=aarch64-linux-gnu-gcc
		CXX_BIN=aarch64-linux-gnu-g++
		AR_BIN=aarch64-linux-gnu-ar
		STRIP_BIN=aarch64-linux-gnu-strip
		SR=/fake/sysroot
		SR_INC_MULTI=""
		SR_INC_LIBDRM=""
		SR_INC="$SR/usr/include"
		SR_LIB="$SR/usr/lib"
		EXTRA_LINK=""
		DBG_ARGS="$1"
		eval "cat <<OCH_EOF
$(cat "$OUT/block.tmpl")
OCH_EOF"
	)
}

check() { # check <期望含-g3:0|1>
	want_g3="$2"
	TEFLON_DEBUG_INFO="$1"
	if [ "$TEFLON_DEBUG_INFO" = "1" ]; then
		export DBG_ARGS="-g3"
	else
		DBG_ARGS=""
	fi
	render "$DBG_ARGS" > "$OUT/cross.ini"

	printf '\n--- TEFLON_DEBUG_INFO=%s ---\n' "$TEFLON_DEBUG_INFO"
	if grep -nE '^\s*(else|fi|then|done)\s*$|^DBG_ARGS=' "$OUT/cross.ini"; then
		note "FAIL: cross.ini 里出现了 shell 控制流/赋值语句 —— 说明它们被写进了 heredoc"
		fail=1
	else
		note "OK: cross.ini 内无 shell 控制流残留"
	fi
	if grep -q '\[binaries\]' "$OUT/cross.ini" && grep -q '\[host_machine\]' "$OUT/cross.ini" \
		&& grep -q 'c_args' "$OUT/cross.ini"; then
		note "OK: cross.ini 三个关键段都在"
	else
		note "FAIL: cross.ini 结构不完整"
		fail=1
	fi
	if grep -q -- "'-g3'" "$OUT/cross.ini"; then
		got=1
	else
		got=0
	fi
	if [ "$got" = "$want_g3" ]; then
		note "OK: -g3 期望=$want_g3 实际=$got"
	else
		note "FAIL: -g3 期望=$want_g3 实际=$got"
		fail=1
	fi
}

echo "=== cross.ini 渲染探针（不需要编译/网络）==="
check 0 0
check 1 1

printf '\n=== 渲染样例（DEBUG_INFO=1 的 c_args 行）===\n'
render "-g3" | grep '^c_args' || true

if [ "$fail" -eq 0 ]; then
	echo
	echo "CROSS-INI PROBE PASSED"
	exit 0
fi
echo
echo "CROSS-INI PROBE FAILED"
exit 2