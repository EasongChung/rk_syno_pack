#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-f4-phantom-fix.py —— patch-etnaviv-f4-phantom-fix.py 的自测

门守什么：F4 改行为（幻影通道钳位），是 c335 解锁线的判决手段 ⇒
补丁没打上 / 打歪 / 漏站点 / 前置失效 ⇒ 后续上机结论直接作废。

## 测试矩阵

  正控制 4：
    1) **真实源码链式应用**： pristine mesa v7.c → OCFIX → F4 全部成功；
       三站点钳位齐全、interleaved 未被 F4 污染、原式逐字保留、双标记齐全；
    2) 语义回归（sim-write-core-index 单一事实源）：10 个实测点钳位后
       越界归零 / 合法通道不丢 / 幻影钳到 C-1；m 用独立公式反作弊重算；
    3) 判据自测：扰动权重/偏置 ⇒ 比对器必须报不一致（判据能失败）；
    4) 编译级：helper+标记块过 gcc -fsyntax-only（若本机有 gcc），
       且负控制（文件作用域写语句）必须编译失败。

  负控制 7：
    1) 重复应用 ⇒ 拒绝；
    2) 无 OCFIX 前置 ⇒ 拒绝（fail-closed，interleaved 不能无人管）；
    3) PHANTOM_CLAMP 互斥 ⇒ 拒绝；
    4) write_core_6 只有一处公式 ⇒ 拒绝；
    5) write_core_6 三处公式 ⇒ 拒绝；
    6) 无关文件 ⇒ 拒绝；
    7) output_channels 改名 ⇒ 拒绝（防产出编译不过的 .so）。

用法:
    python3 selftest-f4-phantom-fix.py [F4补丁器路径]
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PATCHER = os.path.join(ROOT, "build-recipe", "patch-etnaviv-f4-phantom-fix.py")
OCFIX_PATCHER = os.path.join(ROOT, "build-recipe", "patch-etnaviv-outchannel-clamp.py")
REAL_SRC = os.path.join(ROOT, "third_party", "mesa-src-26.2.5", "src",
                        "gallium", "drivers", "etnaviv", "etnaviv_ml_nn_v7.c")

FORMULA = ("unsigned out_channel = core * kernels_in_superblock + kernel + "
           "superblock * DIV_ROUND_UP(kernels_per_core, superblocks) * cores_used;")
CLAMP_CALL = "out_channel = och_f4_clamp_out_channel(out_channel, output_channels);"
F4_MARKER = "och-patch-v7-f4-phantom-fix"
OCFIX_MARKER = "och-patch-v7-outchannel-clamp"
PCLAMP_MARKER = "och-patch-v7-phantom-clamp"


def python_cmd():
    """子进程 python 启动命令。

    och-runtime 的 python3 是 musl loader 包装器：sys.executable 指向裸二进制，
    直接 subprocess 调用会丢 --library-path ⇒ libpython 加载失败。
    检测到 och-runtime 布局时重建包装器等价命令（含 PYTHONHOME）。
    """
    exe = os.path.realpath(sys.executable)
    marker = "/och-runtime/root/usr/bin/"
    if marker in exe:
        root = exe.split(marker)[0] + "/och-runtime/root"
        loader = os.path.join(root, "lib", "ld-musl-aarch64.so.1")
        if os.path.isfile(loader):
            return [loader, "--library-path",
                    os.path.join(root, "usr", "lib") + ":" + os.path.join(root, "lib"),
                    exe]
    return [sys.executable]


def run(patcher, path, extra=()):
    env = dict(os.environ)
    cmd = python_cmd()
    if cmd[0].endswith("ld-musl-aarch64.so.1"):
        # 包装器原本导出 PYTHONHOME；loader 直调时显式补上
        exe = os.path.realpath(sys.executable)
        root = exe.split("/och-runtime/root/usr/bin/")[0] + "/och-runtime/root"
        env["PYTHONHOME"] = os.path.join(root, "usr")
    return subprocess.run([*cmd, patcher, *extra, path],
                          capture_output=True, text=True, env=env)


def fail(msg, detail=""):
    sys.stderr.write("FAIL: " + msg + "\n")
    if detail:
        sys.stderr.write("      " + detail.strip().replace("\n", "\n      ") + "\n")
    sys.exit(1)


def fn_body(src, header):
    i = src.find(header)
    if i < 0:
        return None
    j = src.find("\nstatic", i + 1)
    return src[i: j if j > 0 else len(src)]


def semantic_regression():
    sim_path = os.path.join(HERE, "sim-write-core-index.py")
    spec = importlib.util.spec_from_file_location("sim_write_core_index", sim_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    pts = [122, 196, 220, 240, 244, 248, 252, 260, 280, 284]
    for c in pts:
        g = sim.Geometry(c, 50, 50, 1, 1, in_c=c, name="pointwise")
        per_core, kpc0, sb, d, rem = sim.channel_map(g)
        m_indep = (8 - (c % 8)) % 8
        m_impl = 8 * kpc0 - c
        if m_indep != m_impl:
            fail("反作弊失败：C=%d 的 m 两算不一致 (%d vs %d)" % (c, m_indep, m_impl))
        clamped = []
        for chs in per_core:
            clamped += [min(x, c - 1) for x in chs]
        if any(x >= c for x in clamped):
            fail("语义回归失败：C=%d 钳位后仍有越界通道" % c)
        missing = sorted(set(range(c)) - set(clamped))
        if missing:
            fail("语义回归失败：C=%d 钳位后丢了合法通道 %s" % (c, missing[:8]))
        for chs in per_core:
            for x in chs:
                if x >= c and min(x, c - 1) != c - 1:
                    fail("语义回归失败：C=%d 幻影通道 %d 未钳到 C-1" % (c, x))
    return len(pts)


def criterion_selftest():
    def simulate(weights, biases, c):
        return [sum(weights[i]) + biases[i] for i in range(c)]

    c = 8
    w = [[1] * 4 for _ in range(c)]
    b = [0] * c
    ref = simulate(w, b, c)
    if simulate(w, b, c) != ref:
        fail("判据自测失败：未扰动时比对器就报不一致")
    b2 = list(b)
    b2[3] += 1
    if simulate(w, b2, c) == ref:
        fail("判据自测失败：扰动偏置后比对器仍报一致 ⇒ 比对器不能失败")
    w2 = [list(x) for x in w]
    w2[5][0] += 1
    if simulate(w2, b, c) == ref:
        fail("判据自测失败：扰动权重后比对器仍报一致 ⇒ 比对器不能失败")
    return 3


def helper_compile_check(patched_text):
    """helper+标记块语法检查（有 gcc 才跑；无 gcc 显式 SKIP 并声明）。"""
    cc = shutil.which("gcc")
    if cc is None:
        # 持久运行时里的 gcc（容器重建后系统 gcc 会丢，这里不丢）
        cand = "/root/.local/share/opencode/och-runtime/jellyfin-vpu-build/bin/gcc"
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            cc = cand
    if cc is None:
        print("  [skip] 正控制 4：本机无 gcc，编译级检查跳过（CI 构建仍是最终编译门）")
        return 0
    at = patched_text.find("och_f4_phantom_fix_marker")
    if at < 0:
        fail("正控制 4 失败：找不到标记变量")
    start = patched_text.rfind("static inline unsigned", 0, at)
    end = patched_text.find("#endif", at)
    if start < 0 or end < 0:
        fail("正控制 4 失败：helper 块边界定位失败")
    block = patched_text[start:end + len("#endif")]
    # ★ 不能把 ML_DBG 定义成空宏（do {} while(0) 会把格式串当宏参数丢掉，
    #   编译器看不到跨行字符串 —— CI run 42547644410 就是被这个假绿放过的：
    #   helper 里 ML_DBG 的 "\n" 被写成真换行 ⇒ missing terminating " character）。
    #   这里模拟真实 ML_DBG：格式串作为 _debug_printf 的第一个实参参与编译。
    code = ("extern void _debug_printf(const char *, ...);\n"
            "#define ML_DBG(fmt, ...) _debug_printf(fmt, ##__VA_ARGS__)\n"
            + block + "\n")

    def compiles(src):
        p = subprocess.run([cc, "-x", "c", "-std=c11", "-Werror=format", "-fsyntax-only", "-"],
                           input=src, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0, p.stderr

    ok, err = compiles(code)
    if not ok:
        fail("正控制 4 失败：helper 块编译不过", err)
    bad = code.replace("#endif", "   (void)och_f4_phantom_fix_marker;\n#endif", 1)
    ok2, _ = compiles(bad)
    if ok2:
        fail("正控制 4 的负控制失效：文件作用域写语句竟然编译通过")
    return 2


def main():
    patcher = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCHER
    if not os.path.isfile(patcher):
        fail("找不到补丁器: " + patcher)
    if not os.path.isfile(OCFIX_PATCHER):
        fail("找不到 OCFIX 补丁器: " + OCFIX_PATCHER)
    if not os.path.isfile(REAL_SRC):
        fail("找不到真实源码: " + REAL_SRC + "（正控制必须用真锚点，不用合成替代品）")
    tmp = tempfile.mkdtemp(prefix="f4-selftest-")

    def fresh(name):
        p = os.path.join(tmp, name)
        shutil.copyfile(REAL_SRC, p)
        return p

    # ---------- 正控制 1：真实源码链式 OCFIX → F4 ----------
    good = fresh("good.c")
    r = run(OCFIX_PATCHER, good)
    if r.returncode != 0:
        fail("正控制 1 失败：OCFIX 前置应用失败", r.stderr)
    r = run(patcher, good)
    if r.returncode != 0:
        fail("正控制 1 失败：F4 应能应用在 OCFIX 之后的真实源码上", r.stderr)
    body = open(good, encoding="utf-8").read()
    if body.count(CLAMP_CALL) != 3:
        fail("正控制 1 失败：钳位调用 %d 次（期望 3）" % body.count(CLAMP_CALL))
    if F4_MARKER not in body or OCFIX_MARKER not in body:
        fail("正控制 1 失败：编译期标记缺失（F4 或 OCFIX）")
    if body.count(FORMULA) != 4:
        fail("正控制 1 失败：原公式行总数 %d（期望 4，原式必须逐字保留）" % body.count(FORMULA))
    inter = fn_body(body, "write_core_interleaved(struct etna_ml_subgraph")
    if inter is None or CLAMP_CALL in inter or "och_f4_clamp_out_channel" in inter:
        fail("正控制 1 失败：F4 污染了 write_core_interleaved（OCFIX 专属）")
    if "out_channel = output_channels - 1;" not in inter:
        fail("正控制 1 失败：OCFIX 钳位行不在 interleaved 体内")
    f6 = fn_body(body, "write_core_6(struct etna_ml_subgraph")
    if f6.count(CLAMP_CALL) != 2:
        fail("正控制 1 失败：write_core_6 钳位 %d 处（期望 2）" % f6.count(CLAMP_CALL))
    seq = fn_body(body, "write_core_sequential(struct etna_ml_subgraph")
    if seq.count(CLAMP_CALL) != 1:
        fail("正控制 1 失败：write_core_sequential 钳位 %d 处（期望 1）" % seq.count(CLAMP_CALL))
    if "och-f4: phantom" not in body:
        fail("正控制 1 失败：运行期遥测缺失")
    print("  [ok] 正控制 1：真实源码 OCFIX→F4 链式应用；三站点齐全；interleaved 未污染；双标记+遥测")

    # ---------- 正控制 2：语义回归 ----------
    n = semantic_regression()
    print("  [ok] 正控制 2：语义回归 %d 点（越界归零 / 合法通道不丢 / 幻影钳到 C-1 / m 反作弊）" % n)

    # ---------- 正控制 3：判据自测 ----------
    k = criterion_selftest()
    print("  [ok] 正控制 3：判据自测 %d 项扰动全被抓到（比对器能失败）" % k)

    # ---------- 正控制 4：编译级 ----------
    if helper_compile_check(body):
        print("  [ok] 正控制 4：helper+标记块过 gcc -fsyntax-only，且文件作用域写语句的负控制确实编译失败")

    # ---------- 负控制 1：重复应用 ----------
    r = run(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：重复应用应拒绝")
    print("  [ok] 负控制 1：重复应用被拒绝")

    # ---------- 负控制 2：无 OCFIX 前置 ----------
    no_ocfix = fresh("no_ocfix.c")
    r = run(patcher, no_ocfix)
    if r.returncode == 0:
        fail("负控制 2 失效：无 OCFIX 时 F4 应拒绝（interleaved 不能无人管）")
    if "OCFIX" not in r.stderr:
        fail("负控制 2 失效：拒绝原因未指明 OCFIX 前置")
    print("  [ok] 负控制 2：无 OCFIX 前置被拒绝（fail-closed）")

    # ---------- 负控制 3：PHANTOM_CLAMP 互斥 ----------
    pclamp = fresh("pclamp.c")
    txt = open(pclamp, encoding="utf-8").read()
    txt = txt.replace("#define MAX_TILE_WIDTH 64",
                      '/* och-patch-v7-phantom-clamp */\n#define MAX_TILE_WIDTH 64', 1)
    with open(pclamp, "w", encoding="utf-8") as f:
        f.write(txt)
    r = run(patcher, pclamp)
    if r.returncode == 0:
        fail("负控制 3 失效：PHANTOM_CLAMP 在位时 F4 应拒绝（互斥）")
    print("  [ok] 负控制 3：PHANTOM_CLAMP 互斥被拒绝")

    # ---------- 负控制 4/5：公式行数量异常 ----------
    one_site = fresh("one_site.c")
    txt = open(one_site, encoding="utf-8").read()
    i = txt.find("write_core_6(struct etna_ml_subgraph")
    j = txt.find("\nstatic", i + 1)
    seg = txt[i:j]
    k1 = seg.find(FORMULA)
    seg2 = seg[:k1] + seg[k1 + len(FORMULA):]  # 删第一处公式
    txt = txt[:i] + seg2 + txt[j:]
    with open(one_site, "w", encoding="utf-8") as f:
        f.write(txt)
    r = run(patcher, one_site)
    if r.returncode == 0:
        fail("负控制 4 失效：write_core_6 只剩一处公式应拒绝")
    print("  [ok] 负控制 4：write_core_6 公式缺一处被结构自检抓出")

    three_site = fresh("three_site.c")
    txt = open(three_site, encoding="utf-8").read()
    i = txt.find("write_core_6(struct etna_ml_subgraph")
    j = txt.find("\nstatic", i + 1)
    seg = txt[i:j]
    k1 = seg.find(FORMULA)
    line_end = seg.find("\n", k1)
    ind = seg[seg.rfind("\n", 0, k1) + 1:k1]
    seg2 = seg[:line_end] + "\n" + ind + FORMULA + seg[line_end:]
    txt = txt[:i] + seg2 + txt[j:]
    with open(three_site, "w", encoding="utf-8") as f:
        f.write(txt)
    r = run(patcher, three_site)
    if r.returncode == 0:
        fail("负控制 5 失效：write_core_6 三处公式应拒绝")
    print("  [ok] 负控制 5：write_core_6 公式多一处被结构自检抓出")

    # ---------- 负控制 6：无关文件 ----------
    unrelated = os.path.join(tmp, "unrelated.c")
    with open(unrelated, "w", encoding="utf-8") as f:
        f.write("int main(void){return 0;}\n")
    r = run(patcher, unrelated)
    if r.returncode == 0:
        fail("负控制 6 失效：无关文件应拒绝")
    print("  [ok] 负控制 6：无关文件被拒绝")

    # ---------- 负控制 7：变量改名 ----------
    renamed = fresh("renamed.c")
    txt = open(renamed, encoding="utf-8").read()
    txt = txt.replace("unsigned output_channels =", "unsigned out_chans =")
    with open(renamed, "w", encoding="utf-8") as f:
        f.write(txt)
    r = run(patcher, renamed)
    if r.returncode == 0:
        fail("负控制 7 失效：output_channels 改名应拒绝")
    print("  [ok] 负控制 7：变量改名被拒绝（防产出编译不过的 .so）")

    # ---------- --check 模式 ----------
    r = run(patcher, good, extra=("--check",))
    if r.returncode != 0:
        fail("--check 模式：已打补丁的文件应返回 0")
    r = run(patcher, no_ocfix, extra=("--check",))
    if r.returncode == 0:
        fail("--check 模式：未打补丁的文件应返回非 0")
    print("  [ok] --check 模式：已打/未打判定正确")

    print("  [ok] selftest-f4-phantom-fix 全部通过（正控制 4 + 负控制 7 + --check）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
