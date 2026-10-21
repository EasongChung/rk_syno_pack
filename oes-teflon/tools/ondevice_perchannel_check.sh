#!/bin/sh
# per-channel 补丁上机验证：双对照
#
# 对照 A：per-channel 模型 + teflon-2625-dual/libteflon.so
#         目的 = 复现报告 02 的已知基线（supported=99 / CONV 39 un / DWCONV 17 un）
#         ★ 这一步是「方法自检」：基线复现不出来，后面的结果就不可解读
# 对照 B：per-channel 模型 + 本轮 perchannel 产物 (14e099a7)
#         目的 = 看卷积族 CONV/DWCONV 是否从 unsupported 变 supported
#
# 变量控制：同一目录、同一 tf_load_check、同一模型、同一命令，**只换 delegate 一个变量**。
# 零侵入：不替换任何已部署文件，不触碰正在运行的 npu_server。
set -u

DIR=/volume1/OES/teflon-perchannel-test
MODEL=/volume1/OES/artifact-model/concept_int8_tf.tflite
cd "$DIR" || exit 1

cp -f /volume1/OES/teflon-v7fix/tf_load_check . 2>/dev/null
cp -f /volume1/OES/teflon-v7fix/libdrm.so.2 . 2>/dev/null
chmod 755 tf_load_check libdrm.so.2 2>/dev/null

echo "模型: $MODEL  ($(stat -c %s "$MODEL") 字节)"
echo

run() {
    name="$1"; dele="$2"
    echo "=============== $name ==============="
    echo "delegate: $dele  ($(stat -c %s "$dele" 2>/dev/null) 字节)"
    TEFLON_DEBUG=verbose LD_LIBRARY_PATH=. timeout 420 \
        ./tf_load_check "$MODEL" "$dele" > "$name.log" 2>&1
    rc=$?
    echo "EXIT=$rc   日志行数=$(wc -l < "$name.log")"
    if ! grep -q 'compiling graph\| supported' "$name.log" 2>/dev/null; then
        echo "  [警告] 日志里没有任何 support 表 —— 可能是提前崩溃或 delegate 未加载"
        echo "  日志尾部:"; tail -6 "$name.log" | sed 's/^/    /'
        echo
        return
    fi
    echo "--- supported 分布 ---"
    grep ' supported' "$name.log" | awk '{print $2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
    echo "--- unsupported 分布 ---"
    grep 'unsupported' "$name.log" | awk '{print $2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
    echo "--- 合计 ---"
    printf "    查询节点=%s  supported=%s  unsupported=%s\n" \
        "$(grep -cE ' (un)?supported ' "$name.log")" \
        "$(grep -c ' supported' "$name.log")" \
        "$(grep -c 'unsupported' "$name.log")"
    grep 'compiling graph' "$name.log" | awk '{s+=$6} END {printf "    接管节点数(日志求和) = %d\n", s+0}'
    echo
}

run A_baseline_teflon2625dual /volume1/OES/teflon-2625-dual/libteflon.so
run B_perchannel_14e099a7     "$DIR/libteflon.so"

echo "=============== 结论对照 ==============="
for f in A_baseline_teflon2625dual B_perchannel_14e099a7; do
    [ -f "$f.log" ] || continue
    conv=$(grep -cE '^ *[0-9]+ CONV ' "$f.log" 2>/dev/null)
    dw=$(grep -cE '^ *[0-9]+ DWCONV ' "$f.log" 2>/dev/null)
    sup=$(grep -c ' supported' "$f.log" 2>/dev/null)
    uns=$(grep -c 'unsupported' "$f.log" 2>/dev/null)
    printf "  %-26s supported=%-4s unsupported=%-4s  CONV行=%-4s DWCONV行=%s\n" \
        "$f" "$sup" "$uns" "$conv" "$dw"
done
echo
echo "注：CONV/DWCONV 行数含 supported 与 unsupported 两类，需配合上方分布表读。"
