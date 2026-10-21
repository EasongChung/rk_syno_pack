#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 Mesa 源码里**逐字**提取锚点，生成 patch-etnaviv-write-core-diag.py。

为什么要程序化生成：
1. 手敲模板时缩进空格数极易错（已踩过一次：15 vs 18 空格 → 锚点匹配 0 处）；
2. 模板里同时含 Python 的 `%` 格式化与 C 的 `%u` 占位，手写极易互相打架
   （已踩过一次：`%%u` 与命名占位 `{s}` 混用 → 生成出字面量 `{ob}`）。

所以这里用 repr() 把逐字原文嵌进补丁脚本，并且**完全不做 `%` 格式化**。
"""

SRC = "/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
OUT = "/root/workspace/oes-teflon/build-recipe/patch-etnaviv-write-core-diag.py"

src = open(SRC, encoding="utf-8").read()
key = "wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);"
i = src.index(key)
anchor_start = src.rindex("for (unsigned y = 0; y < stride; y++) {", 0, i)
anchor_start = src.rindex("\n", 0, anchor_start) + 1
OLD = src[anchor_start:i + len(key) + 1]
assert OLD.count(key) == 1, "锚点不唯一"
assert "for (unsigned y = 0; y < stride; y++) {" in OLD

# 插入内容：单引号拼接、无任何格式化，%u 就是 C 的占位符。
PRINT = (
    '/* och patch: write_core_interleaved out_channel diagnostics */\n'
    '                        fprintf(stderr,\n'
    '                                "[och-wc] core=%u kernel=%u z=%u sb=%u/%u "\n'
    '                                "kpc=%u/%u cores=%u out_ch=%u out_channels=%u "\n'
    '                                "w=%u h=%u wt_size=%u%s\\n",\n'
    '                                core, kernel, z, superblock, superblocks,\n'
    '                                kernels_in_superblock, kernels_per_core, cores_used,\n'
    '                                out_channel, output_channels,\n'
    '                                operation->weight_width, operation->weight_height,\n'
    '                                operation->weight_tensor_size,\n'
    '                                (out_channel >= output_channels) ? "  *** OOB ***" : "");\n'
)

# 自检：格式符个数必须等于参数个数（差一个就会读到垃圾 vararg）
_fmts = PRINT.count('%u') + PRINT.count('%s')
_tail = PRINT[PRINT.index('core, kernel'):]
_args = _tail.count(',') + 1
if _fmts != _args:
    raise SystemExit('生成器自检失败：格式符 %d 个 != 参数 %d 个' % (_fmts, _args))
print('生成器自检：格式符 %d == 参数 %d' % (_fmts, _args))

# 在 wb_stream_write 那一行之前插入
wb_off = OLD.index(key)
line_off = OLD.rindex("\n", 0, wb_off) + 1
NEW = OLD[:line_off] + PRINT + OLD[line_off:]

TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断补丁：write_core_interleaved() 写出权重前打印循环变量，out_channel 越界时报警。

## 背景（addr2line 硬证据，不再靠反推）

CI 里 `addr2line -e libteflon.so 0x256a60` 直接给出：

    write_core_interleaved()   etnaviv_ml_nn_v7.c:345（打 FIX_V7 补丁后的行号）

崩溃行正是 `wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);`。
而 `weights_map` 只是把权重缓冲区**强转**成四维视图：

    uint8_t (*weights_map)[input_channels][weight_width][weight_height] = (void *)input;

`out_channel` 一旦超过 `output_channels`，`weights_map[out_channel]` 就指向缓冲区之外
→ 越界读 → SIGSEGV；越界多少取决于堆布局，所以表现为**非确定性**。

**待证实/证伪的就是这一点** —— 所以本补丁只把数字打出来，不改任何行为。

## 本补丁做什么

在写出权重前打印：core / kernel / z / superblock、superblocks、kernels_in_superblock、
kernels_per_core、cores_used、out_channel、output_channels、weight_width/height、
weight_tensor_size；并在 `out_channel >= output_channels` 时追加 `*** OOB ***`。

**刻意不做**：不改行为、不钳界、不跳过循环。
打印紧挨崩溃行发生，而 `tf_load_check` 已 `setvbuf(_IONBF)`（日志无缓冲），
所以即使下一秒仍然 SIGSEGV，**这一行已经落进日志** ——
我们要的是「机制被证实」，不是「让它跑过去」。

## 用法

    python3 patch-etnaviv-write-core-diag.py <etnaviv_ml_nn_v7.c 路径>

由 `build-teflon.sh` 的 `TEFLON_WCDIAG=1` 启用，默认关。
"""

import argparse
import sys

MARKER = "och patch: write_core_interleaved out_channel diagnostics"

# ↓↓↓ 逐字锚点（由 tools/gen-wcdiag-patch.py 从真实源码提取，勿手改）↓↓↓
OLD = __OLD__
NEW = __NEW__
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        sys.stderr.write("ERROR: " + args.path + " 已包含本补丁（标记 \"" + MARKER +
                         "\"），拒绝重复应用。\n")
        return 1

    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            "ERROR: 在 " + args.path + " 中匹配到 " + str(n) + " 处锚点，期望恰好 1 处。\n"
            "       Mesa 可能已改动此处 —— 请人工核对后重新生成补丁"
            "（python3 tools/gen-wcdiag-patch.py），不要跳过校验\n"
            "       （会静默产出未打补丁的 .so，诊断结论无效）。\n")
        return 1

    if "weights_map[out_channel][z][x][y]" not in src:
        sys.stderr.write("ERROR: 源文件里找不到 weights_map[out_channel][z][x][y]，"
                         "结构与预期不符\n")
        return 1

    src = src.replace(OLD, NEW, 1)

    if MARKER not in src or "[och-wc]" not in src:
        sys.stderr.write("ERROR: 打完补丁后自检失败\n")
        return 1

    with open(args.path, "w", encoding="utf-8") as f:
        f.write(src)

    print("  [och-wc] 已注入 out_channel 诊断: " + args.path)
    print("  [och-wc] 日志形如 [och-wc] ... out_ch=N out_channels=M *** OOB ***")
    print("  [och-wc] 不改行为：崩溃行仍会照旧触发，但这一行已在日志里")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

body = TEMPLATE.replace("__OLD__", repr(OLD)).replace("__NEW__", repr(NEW))
open(OUT, "w", encoding="utf-8").write(body)
print("已生成:", OUT)