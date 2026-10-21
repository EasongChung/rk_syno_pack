#!/bin/sh
# 本轮上机验证：布局修法产物 b8681800（calc_quantization + requantize_weights + v7 写回 + 诊断）
#
# 目的（严格按四级判据，逐条独立取证）：
#   ① 编译/分区不崩      —— 上一轮 14e099a7 + per-channel 模型是 EXIT=139 SIGSEGV
#   ② 接管数             —— 期望 supported=153，卷积族 54（37 CONV + 17 DWCONV）
#   ③ 推理执行           —— 另用 --invoke 单独跑（本脚本只跑前两项）
#
# 零侵入：不替换任何已部署文件，不触碰在跑的 npu_server。
# 变量控制：同目录、同 tf_load_check、同一命令，只换 delegate / 模型。
#
# ★ 可证伪预测（不符即证伪 och 报告 20 的改动 1）：
#     [och-pcl] CQ-OUT dw=1 ... visited=2196      （若为 535824 ⇒ 我的覆盖分析是错的）
#     [och-pcl] RW-OUT dw=1 ... new_size=2196
#     [och-pcl] SEL    dw=1 input_channels=1 -> write_core_sequential
set -u

BASE=/volume1/OES/teflon-perchannel-test
DIR=$BASE/art-b8681800
PC_MODEL=/volume1/OES/artifact-model/concept_int8_tf.tflite
PT_MODEL=/volume1/OES/photos-npu/model.tflite

cd "$DIR" || exit 1
cp -f "$BASE/tf_load_check" . 2>/dev/null
chmod 755 tf_load_check 2>/dev/null

echo "=============================================="
echo " 布局修法产物上机验证"
echo " delegate: $DIR/libteflon.so"
echo "   sha256: $(sha256sum "$DIR/libteflon.so" | cut -c1-16)..."
echo "   字节数: $(stat -c %s "$DIR/libteflon.so")"
echo "=============================================="
echo

run() {
	name="$1"; model="$2"; dele="$3"; shift 3
	echo "=============== $name ==============="
	echo "  model: $model  ($(stat -c %s "$model") 字节)"
	echo "  dele : $dele  sha=$(sha256sum "$dele" | cut -c1-12)"
	TEFLON_DEBUG=verbose LD_LIBRARY_PATH="$DIR" timeout 600 \
		./tf_load_check "$model" "$dele" "$@" > "$name.log" 2>&1
	rc=$?
	echo "  EXIT=$rc   日志行数=$(wc -l < "$name.log")"

	if ! grep -qE 'compiling graph| supported' "$name.log" 2>/dev/null; then
		echo "  [警告] 日志里没有 support 表 —— 提前崩溃或 delegate 未加载"
		echo "  日志尾部:"
		tail -12 "$name.log" | sed 's/^/    /'
		echo
		return
	fi

	echo "  --- supported 分布 ---"
	grep ' supported' "$name.log" | awk '{print $2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
	echo "  --- unsupported 分布 ---"
	grep 'unsupported' "$name.log" | awk '{print $2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
	printf "  --- 合计 ---\n    查询节点=%s  supported=%s  unsupported=%s\n" \
		"$(grep -cE ' (un)?supported ' "$name.log")" \
		"$(grep -c ' supported' "$name.log")" \
		"$(grep -c 'unsupported' "$name.log")"
	echo "  --- 卷积族（CONV/DWCONV 各自 supported 与 unsupported）---"
	printf "    CONV    supported=%s  unsupported=%s\n" \
		"$(grep -cE '  CONV ' "$name.log" 2>/dev/null || echo 0)" \
		"$(grep -cE '  CONV .*unsupported|unsupported.*  CONV ' "$name.log" 2>/dev/null || echo 0)"
	printf "    DWCONV  supported=%s  unsupported=%s\n" \
		"$(grep -cE '  DWCONV ' "$name.log" 2>/dev/null || echo 0)" \
		"$(grep -cE '  DWCONV .*unsupported|unsupported.*  DWCONV ' "$name.log" 2>/dev/null || echo 0)"
	echo
}

echo "=== G：per-channel 模型 + b8681800（主判据）==="
run G_perchannel_b8681800 "$PC_MODEL" "$DIR/libteflon.so"

echo "=== H：per-tensor 模型 + b8681800（对照：无 scales ⇒ 不应进 requantize）==="
run H_pertensor_b8681800 "$PT_MODEL" "$DIR/libteflon.so"

echo "=============== och-pcl 诊断（可证伪预测）==============="
for f in G_perchannel_b8681800 H_pertensor_b8681800; do
	[ -f "$f.log" ] || continue
	n=$(grep -c 'och-pcl' "$f.log" 2>/dev/null || echo 0)
	echo "  $f : och-pcl 行数 = $n"
done
echo
echo "  --- CQ-OUT（覆盖字节数；预测 depthwise 应为 2196，不是 535824）---"
grep -oE '\[och-pcl\] CQ-OUT[^\\]*' G_perchannel_b8681800.log 2>/dev/null | head -8 | sed 's/^/    /'
echo "  --- RW-OUT（new_size）---"
grep -oE '\[och-pcl\] RW-OUT[^\\]*' G_perchannel_b8681800.log 2>/dev/null | head -8 | sed 's/^/    /'
echo "  --- SEL（选路：预测 depthwise 应是 write_core_sequential）---"
grep -oE '\[och-pcl\] SEL [^\\]*' G_perchannel_b8681800.log 2>/dev/null | head -8 | sed 's/^/    /'
echo
echo "=============== 结论对照 ==============="
printf "  %-30s %-8s %-8s %-10s %-10s\n" 样本 EXIT supported unsupported och-pcl行数
for f in G_perchannel_b8681800 H_pertensor_b8681800; do
	[ -f "$f.log" ] || continue
	sup=$(grep -c ' supported' "$f.log" 2>/dev/null || echo 0)
	uns=$(grep -c 'unsupported' "$f.log" 2>/dev/null || echo 0)
	pcl=$(grep -c 'och-pcl' "$f.log" 2>/dev/null || echo 0)
	printf "  %-30s %-8s %-8s %-10s %-10s\n" "$f" "见上" "$sup" "$uns" "$pcl"
done
echo
echo "注：本脚本只回答 ①（崩不崩）与 ②（接管数）。③ 推理执行须另跑 --invoke。"
