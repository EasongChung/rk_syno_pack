#!/bin/sh
# ③ 推理执行 + ④ 逐字节比对的**准备**（本脚本只取证，不下结论）
#
#   I：per-channel + NPU 推理（--invoke）
#   J：per-channel + CPU 基准（--no-delegate）  ← ④ 的比对基准
#
# 已知独立阻塞（报告 09）：per-tensor 路线在 NPU 跑 2 轮后崩在
#   etna_ml_subgraph_read_outputs+0x38c 的 pipe_buffer_map_range assert。
#   本轮 per-channel 会撞上同一堵墙 —— 预期会看到同样的塌方，那是**已知问题**，不是新问题。
set -u

BASE=/volume1/OES/teflon-perchannel-test
DIR=$BASE/art-b8681800
MODEL=/volume1/OES/artifact-model/concept_int8_tf.tflite

cd "$DIR" || exit 1
cp -f "$BASE/tf_load_check" . 2>/dev/null
chmod 755 tf_load_check 2>/dev/null

run() {
	name="$1"; shift
	echo "=============== $name ==============="
	echo "  参数: $*"
	TEFLON_DEBUG=verbose LD_LIBRARY_PATH="$DIR" timeout 600 \
		./tf_load_check "$MODEL" "$DIR/libteflon.so" "$@" > "$name.log" 2>&1
	rc=$?
	echo "  EXIT=$rc   日志行数=$(wc -l < "$name.log")"
	echo "  --- 关键结果行 ---"
	grep -iE 'out_hash|out_bytes|median|out_min|out_max|out_mean|invoked graph|took .* ms|Assertion|assert|SIGSEGV|Aborted|core dumped|pipe_buffer_map_range|read_outputs' \
		"$name.log" 2>/dev/null | head -14 | sed 's/^/    /'
	echo "  --- 尾部 8 行 ---"
	tail -8 "$name.log" | sed 's/^/    /'
	echo
}

echo "=== I：per-channel + NPU 推理（--invoke 1）==="
run I_perchannel_npu_invoke --invoke 1

echo "=== J：per-channel + CPU 基准（--no-delegate）==="
run J_perchannel_cpu --no-delegate

echo "=============== ④ 比对材料 ==============="
hi=$(grep -oE 'out_hash=0x[0-9a-f]+' I_perchannel_npu_invoke.log 2>/dev/null | head -1)
hj=$(grep -oE 'out_hash=0x[0-9a-f]+' J_perchannel_cpu.log 2>/dev/null | head -1)
echo "  NPU out_hash : ${hi:-（无 —— 没拿到）}"
echo "  CPU out_hash : ${hj:-（无 —— 没拿到）}"
if [ -n "$hi" ] && [ -n "$hj" ]; then
	[ "$hi" = "$hj" ] && echo "  ⇒ 逐字节一致：**是**" || echo "  ⇒ 逐字节一致：**否**"
else
	echo "  ⇒ 无法比对（有一侧没拿到 out_hash）"
fi
