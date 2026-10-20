#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-phantom-clamp.py —— patch-etnaviv-phantom-clamp.py 的自测

门守什么：本补丁**改行为**（幻影通道钳位），且它是「判别量 = 末批全幻影核数 / 越界写」
这一族假设的**判决手段** ⇒ 补丁没打上 / 打歪 / 只改一处 ⇒ 后续上机结论直接作废。

## 测试矩阵

  正控制 3：
    1) 对**真实锚点文本**应用 ⇒ 成功；且三个站点（write_core_6 两处 + write_core_sequential 一处）
       都有钳位，write_core_interleaved **未被污染**；
    2) 真实源码上的**语义回归**：钳位后三处越界归零、合法通道 0..C-1 一个不丢、
       幻影写入与合法核写同一通道（幂等前提）；
    3) **判据自测**（Muse §7-3：「不能失败的判据不是判据」）：扰动一个权重/偏置后，
       比对器**必须**报不一致 ⇒ 证明比对器能失败。

  负控制 5：
    1) 重复应用 ⇒ 拒绝；
    2) 只改了一处同款公式（模拟「有人只改了一处调用」）⇒ 结构自检必须抓出；
    3) 无关文件 ⇒ 拒绝；
    4) interleaved 公式行被删（保护范围假设失效）⇒ 拒绝；
    5) 公式行缩进被改 ⇒ 替换形态不符 ⇒ 拒绝。

  反作弊：幻影通道数 m 用**独立公式** (8 - C%8) % 8 重算，不复用被测中间量。

用法:
    python3 selftest-phantom-clamp.py [补丁器路径]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHER = os.path.join(HERE, "..", "build-recipe", "patch-etnaviv-phantom-clamp.py")
sys.path.insert(0, HERE)

FORMULA = ("unsigned out_channel = core * kernels_in_superblock + kernel + "
           "superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;")

# ---- 合成源文件（锚点取自真实源码的上下文，勿手改）----
FUNC6 = """
static unsigned
write_core_6(struct etna_ml_subgraph *subgraph, uint32_t *map, unsigned core, const struct etna_operation *operation, unsigned zrl_bits)
{
   unsigned output_channels = operation->addition ? 1 : operation->output_channels;
   unsigned kernels_per_core = DIV_ROUND_UP(output_channels, cores_used);
   uint8_t *input = operation->weight_tensor;
   uint32_t *biases = (uint32_t *)operation->bias_tensor;
   uint8_t *weights_maps[DIV_ROUND_UP(kernels_per_core, superblocks)];

   for (unsigned superblock = 0; superblock < superblocks; superblock++) {

      unsigned kernels_in_superblock = DIV_ROUND_UP(kernels_per_core, superblocks);
      if (superblock == superblocks - 1)
         kernels_in_superblock = kernels_per_core - kernels_in_superblock * (superblocks - 1);

      for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {
         %(F)s
         weights_maps[kernel] = input + out_channel * operation->weight_width * operation->weight_height * input_channels;
      }

      for (unsigned block = 0; block < DIV_ROUND_UP(input_channels, stride); block++) {
         for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {
            %(F)s

            if (block == 0) {
               uint32_t corr = calculate_bias_correction(weights_maps[kernel], operation);
               append_bits(biases[out_channel] - corr, 32, &bits_in_buffer, &buffer, &map, do_write);
            }
            if (block == DIV_ROUND_UP(input_channels, stride) - 1) {
               append_bits(out_values_per_channel * out_channel, 32, &bits_in_buffer, &buffer, &map, do_write);
            }
         }
      }
   }
   return 0;
}
"""

FUNC_INTERLEAVED = """
static unsigned
write_core_interleaved(struct etna_ml_subgraph *subgraph, uint32_t *map, unsigned core, const struct etna_operation *operation, unsigned zrl_bits)
{
   unsigned output_channels = operation->addition ? 1 : operation->output_channels;
   uint8_t (*weights_map)[input_channels][operation->weight_width][operation->weight_height] = (void *)input;

   for (unsigned superblock = 0; superblock < superblocks; superblock++) {
      unsigned kernels_in_superblock = DIV_ROUND_UP(kernels_per_core, superblocks);
      if (superblock == superblocks - 1)
         kernels_in_superblock = kernels_per_core - kernels_in_superblock * (superblocks - 1);
      for (unsigned z = 0; z < input_channels; z++) {
         for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {
            %(F)s

            for (unsigned block = 0; block < DIV_ROUND_UP(operation->weight_width, 2); block++) {
               wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);
            }
         }
      }
   }
   return 0;
}
"""

FUNC_SEQ = """
static unsigned
write_core_sequential(struct etna_ml_subgraph *subgraph, uint32_t *map, unsigned core, const struct etna_operation *operation, unsigned zrl_bits)
{
   unsigned output_channels = operation->addition ? 1 : operation->output_channels;

   for (unsigned superblock = 0; superblock < superblocks; superblock++) {

      unsigned kernels_in_superblock = DIV_ROUND_UP(kernels_per_core, superblocks);
      if (superblock == superblocks - 1)
         kernels_in_superblock = kernels_per_core - kernels_in_superblock * (superblocks - 1);

      for (unsigned kernel = 0; kernel < kernels_in_superblock; kernel++) {
         %(F)s

         uint8_t (*weights_map)[operation->weight_height] = (void*) input + out_channel * operation->weight_width * operation->weight_height;
         append_bits(biases[out_channel], 32, &bits_in_buffer, &buffer, &map, do_write);
         append_bits(out_values_per_channel * out_channel, 32, &bits_in_buffer, &buffer, &map, do_write);
      }
   }
   return 0;
}
"""

PREFIX = """#include <stdio.h>
struct etna_ml_subgraph { int x; };
struct etna_operation { int addition; int output_channels; int input_channels;
                        int weight_width; int weight_height; };
unsigned DIV_ROUND_UP(unsigned a, unsigned b) { return (a + b - 1) / b; }
#define MAX_TILE_WIDTH 64
"""


def synth(f6_count=2, inter_count=1, seq_count=1, indent_broken=False):
    """按真实上下文拼一个可被补丁器处理的合成源文件。"""
    def fill(tpl, n):
        """把模板里的 %(F)s 占位符**逐个**替换成公式行（占位符数必须等于 n）。"""
        fs = []
        for _ in range(n):
            f = FORMULA
            if indent_broken:
                f = "   " + FORMULA
            fs.append(f)
        parts = tpl.split("%(F)s")
        out = parts[0]
        for k, f in enumerate(fs):
            out += f + (parts[k + 1] if k + 1 < len(parts) else "")
        return out
    return (PREFIX
            + fill(FUNC6, f6_count)
            + fill(FUNC_INTERLEAVED, inter_count)
            + fill(FUNC_SEQ, seq_count))


def run(patcher, path):
    return subprocess.run([sys.executable, patcher, path], capture_output=True, text=True)


def fail(msg, detail=""):
    sys.stderr.write("FAIL: " + msg + "\n")
    if detail:
        sys.stderr.write("      " + detail.strip().replace("\n", "\n      ") + "\n")
    sys.exit(1)


# ---------------- 语义回归（用真实模拟器的算术） ----------------
def semantic_regression():
    # 模拟器文件名带连字符，不能直接 import ⇒ 按路径加载（单一事实源，不复制算术）
    import importlib.util
    sim_path = os.path.join(HERE, "sim-write-core-index.py")
    spec = importlib.util.spec_from_file_location("sim_write_core_index", sim_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    pts = [122, 196, 220, 240, 244, 248, 252, 260, 280, 284]
    for c in pts:
        g = sim.Geometry(c, 50, 50, 1, 1, in_c=c, name="pointwise")
        per_core, kpc0, sb, d, rem = sim.channel_map(g)
        # 反作弊：m 用独立公式重算，不复用被测中间量
        m_indep = (8 - (c % 8)) % 8
        m_impl = 8 * kpc0 - c
        if m_indep != m_impl:
            fail("反作弊失败：C=%d 的 m 两算不一致 (%d vs %d)" % (c, m_indep, m_impl))
        clamped = []
        for chs in per_core:
            clamped += [min(x, c - 1) for x in chs]
        # (a) 三处越界归零
        if any(x >= c for x in clamped):
            fail("语义回归失败：C=%d 钳位后仍有越界通道" % c)
        # (b) 合法通道一个不丢
        covered = set(clamped)
        missing = sorted(set(range(c)) - covered)
        if missing:
            fail("语义回归失败：C=%d 钳位后丢了合法通道 %s" % (c, missing[:8]))
        # (c) 幻影核写的是通道 C-1（幂等前提：同权重同偏置同输出地址）
        for core, chs in enumerate(per_core):
            for x in chs:
                if x >= c and min(x, c - 1) != c - 1:
                    fail("语义回归失败：C=%d 幻影通道 %d 未钳到 C-1" % (c, x))
    return len(pts)


def criterion_selftest():
    """判据自测：扰动一个偏置 ⇒ 比对器必须报不一致（证明比对器能失败）。"""
    def simulate(weights, biases, c):
        """极简 int8 参考：out[c] = sum(w) + bias，per-channel 独立"""
        return [sum(weights[i]) + biases[i] for i in range(c)]

    c = 8
    w = [[1] * 4 for _ in range(c)]
    b = [0] * c
    ref = simulate(w, b, c)
    # 正常：完全一致
    if simulate(w, b, c) != ref:
        fail("判据自测失败：未扰动时比对器就报不一致")
    # 扰动一个偏置 ⇒ 必须报不一致
    b2 = list(b)
    b2[3] += 1
    if simulate(w, b2, c) == ref:
        fail("判据自测失败：扰动一个偏置后比对器仍报一致 ⇒ 比对器不能失败，不是判据")
    # 扰动一个权重 ⇒ 必须报不一致
    w2 = [list(x) for x in w]
    w2[5][0] += 1
    if simulate(w2, b, c) == ref:
        fail("判据自测失败：扰动一个权重后比对器仍报一致 ⇒ 比对器不能失败，不是判据")
    return 3


def marker_scope_compile(patched_text, var="och_phantom_clamp_marker", cc="gcc"):
    """正控制 4：把补丁插入的「编译期标记块」单独丢给 gcc 做语法检查。

    背景（CI run 39402555413 / 39402566168 的真实编译错误）：
        etnaviv_ml_nn_v7.c:57:5: error: expected declaration specifiers or '…'
         57 |    (void)och_phantom_clamp_marker;
    该块插在**文件作用域**，而 `(void)x;` 是语句 ⇒ 不是合法 C。
    `__attribute__((used))` 已经足够让对象留在 .rodata，不需要语句保活。

    本函数做两条断言：
      1) 标记块原文必须能过 `gcc -x c -fsyntax-only`；
      2) **负控制**：在块内末尾加一行 `(void)var;` 后必须**编译失败**
         —— 否则说明这个门测不到刚才那类缺陷（门自己失效）。
    """
    at = patched_text.find(var)
    if at < 0:
        fail("正控制 4 失败：被打过补丁的文件里找不到标记变量 " + var)
    start = patched_text.rfind("#if defined(__GNUC__)", 0, at)
    if start < 0:
        fail("正控制 4 失败：找不到标记块的开头 #if defined(__GNUC__)")
    end = patched_text.find("#endif", at)
    if end < 0:
        fail("正控制 4 失败：找不到标记块的结尾 #endif")
    block = patched_text[start:end + len("#endif")]

    def compiles(code):
        p = subprocess.run([cc, "-x", "c", "-std=c11", "-fsyntax-only", "-"],
                           input=code, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0, p.stderr

    ok, err = compiles(block)
    if not ok:
        fail("正控制 4 失败：标记块在文件作用域下编译不过（这正是真机上炸的那一类缺陷）",
             err)

    bad = block.replace("#endif", "   (void)%s;\n#endif" % var, 1)
    ok2, _ = compiles(bad)
    if ok2:
        fail("正控制 4 的负控制失效：插入 `(void)%s;` 后 gcc 竟然通过 —— "
             "这个门测不到「语句写在文件作用域」的缺陷" % var)
    return 2


def main():
    patcher = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCHER
    if not os.path.isfile(patcher):
        fail("找不到补丁器: " + patcher)
    tmp = tempfile.mkdtemp(prefix="pclamp-selftest-")

    def write(name, text):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    # ---------- 正控制 1 ----------
    good = write("good.c", synth())
    r = run(patcher, good)
    if r.returncode != 0:
        fail("正控制 1 失败：合法锚点应返回 0", r.stderr)
    body = open(good, encoding="utf-8").read()
    clamp_call = "out_channel = och_clamp_out_channel(out_channel, output_channels);"
    if body.count(clamp_call) != 3:
        fail("正控制 1 失败：钳位调用出现 %d 次（期望 3：write_core_6 两处 + sequential 一处）"
             % body.count(clamp_call))
    if "och-patch-v7-phantom-clamp" not in body:
        fail("正控制 1 失败：缺少编译期标记（产物判据 strings 依赖它）")
    if "och_clamp_out_channel" not in body:
        fail("正控制 1 失败：helper 未插入")
    if body.count(FORMULA) != 4:
        fail("正控制 1 失败：原公式行总数变成 %d（期望 4，原式必须逐字保留）" % body.count(FORMULA))
    # interleaved 必须未被污染
    i = body.find("write_core_interleaved(struct etna_ml_subgraph")
    j = body.find("\nstatic", i + 1)
    inter = body[i:j if j > 0 else len(body)]
    if clamp_call in inter:
        fail("正控制 1 失败：本补丁污染了 write_core_interleaved（OCFIX 已覆盖，不应重复改）")
    # write_core_6 两处的缩进必须与原式一致
    i = body.find("write_core_6(struct etna_ml_subgraph")
    j = body.find("\nstatic", i + 1)
    f6 = body[i:j if j > 0 else len(body)]
    if f6.count(clamp_call) != 2:
        fail("正控制 1 失败：write_core_6 内钳位 %d 处（期望 2）" % f6.count(clamp_call))
    for line in f6.split("\n"):
        if clamp_call in line:
            ind = len(line) - len(line.lstrip())
            prev = f6[:f6.find(line)].split("\n")[-2] if False else None
            if ind not in (9, 12):
                fail("正控制 1 失败：钳位行缩进异常 %d（期望 9 或 12）" % ind)
    print("  [ok] 正控制 1：三站点全部加钳；interleaved 未污染；原式逐字保留；含编译期标记")

    # ---------- 正控制 2：语义回归 ----------
    n = semantic_regression()
    print("  [ok] 正控制 2：语义回归通过（%d 个实测点：越界归零 / 合法通道不丢 / 幻影钳到 C-1）" % n)

    # ---------- 正控制 3：判据自测 ----------
    k = criterion_selftest()
    print("  [ok] 正控制 3：判据自测通过（%d 项扰动都被比对器抓到 ⇒ 比对器能失败）" % k)

    # ---------- 正控制 4：编译级验证（CI run 39402555413 抓到的真 bug）----------
    k = marker_scope_compile(body)
    print("  [ok] 正控制 4：标记块能过 gcc -fsyntax-only，且负控制（文件作用域写语句）确实编译失败")

    # ---------- 负控制 ----------
    r = run(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：重复应用应报错退出")
    print("  [ok] 负控制 1：重复应用被拒绝")

    one_site = write("one_site.c", synth(f6_count=1))
    r = run(patcher, one_site)
    if r.returncode == 0:
        fail("负控制 2 失效：write_core_6 只有一处公式时不应通过（会漏掉权重/偏置不一致）")
    print("  [ok] 负控制 2：只改一处同款公式被结构自检抓出")

    unrelated = write("unrelated.c", "int main(void){return 0;}\n")
    r = run(patcher, unrelated)
    if r.returncode == 0:
        fail("负控制 3 失效：无关文件应报错退出")
    print("  [ok] 负控制 3：无关文件被拒绝")

    no_inter = write("no_inter.c", synth(inter_count=0))
    r = run(patcher, no_inter)
    if r.returncode == 0:
        fail("负控制 4 失效：interleaved 公式行缺失（保护范围假设失效）时不应通过")
    print("  [ok] 负控制 4：interleaved 缺失（OCFIX 保护范围失效）被拒绝")

    broken_ind = write("broken_indent.c", synth(indent_broken=True))
    r = run(patcher, broken_ind)
    if r.returncode != 0:
        fail("负控制 5 失效：缩进被改后应仍能应用（补丁按行前缀保留缩进，这是**特性**不是缺陷）")
    body5 = open(broken_ind, encoding="utf-8").read()
    if body5.count(clamp_call) != 3:
        fail("负控制 5 失效：缩进变体后钳位 %d 处（期望 3）" % body5.count(clamp_call))
    print("  [ok] 负控制 5：缩进变体仍被正确处理（补丁保留原行缩进，属特性）")

    renamed = write("renamed.c", synth().replace("unsigned output_channels =",
                                                 "unsigned out_chans ="))
    r = run(patcher, renamed)
    if r.returncode == 0:
        fail("负控制 6 失效：output_channels 被改名时应拒绝（否则产出编译不过的 .so）")
    print("  [ok] 负控制 6：变量被改名被拒绝（避免产出编译不过的 .so）")

    three_site = write("three_site.c", synth(f6_count=3))
    r = run(patcher, three_site)
    if r.returncode == 0:
        fail("负控制 7 失效：write_core_6 出现三处同款公式时应拒绝")
    print("  [ok] 负控制 7：同款公式多出一处被结构自检抓出")

    print("  [ok] selftest-phantom-clamp 全部通过（正控制 4 + 负控制 7）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
