#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-phantom-gate.py —— patch-etnaviv-phantom-gate.py（含 F4 运行期开关）自测

门守什么：phantom-gate 是 c335 的安全兜底，F4 判决实验要靠它的运行期开关
（TEFLON_PHANTOM_GATE=0）放行 122/244 形状。开关语义打歪 ⇒ 要么判决实验
假阳性（门没关），要么交付件漏放行（门失效挂死照旧）。

## 测试矩阵

  正控制 3：
    1) 真实源码新装：锚点/标记/getenv/双日志/stdlib 齐全；
    2) 运行期语义 harness（gcc 编译+运行守卫块）：
       unset+122 拒 / =1+122 拒 / =0+122 接（带关门通告）/ unset+120 接；
    3) 旧版升级路径：git HEAD 旧补丁器先打 ⇒ 新补丁器原地升级成功。

  负控制 5：
    1) 新版重复应用 ⇒ 拒绝；
    2) 无关文件 ⇒ 拒绝；
    3) harness 判据自测：把 %8 改 %4 ⇒ unset+122 必须变「接」（harness 能抓语义变化）；
    4) harness 判据自测：把 env 判断改成恒门开 ⇒ =0+122 必须变「拒」；
    5) FSGUARD 锚点缺失 ⇒ 拒绝。

用法:
    python3 tools/selftest-phantom-gate.py [补丁器路径]
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PATCHER = os.path.join(ROOT, "build-recipe", "patch-etnaviv-phantom-gate.py")
REAL_ML = os.path.join(ROOT, "third_party", "mesa-src-26.2.5", "src",
                       "gallium", "drivers", "etnaviv", "etnaviv_ml.c")
GIT = "/root/.local/share/opencode/och-runtime/bin/git"


def fail(msg, detail=""):
    sys.stderr.write("FAIL: " + msg + "\n")
    if detail:
        sys.stderr.write("      " + detail.strip().replace("\n", "\n      ") + "\n")
    sys.exit(1)


def python_cmd():
    exe = os.path.realpath(sys.executable)
    marker = "/och-runtime/root/usr/bin/"
    if marker in exe:
        root = exe.split(marker)[0] + "/och-runtime/root"
        loader = os.path.join(root, "lib", "ld-musl-aarch64.so.1")
        if os.path.isfile(loader):
            return [loader, "--library-path",
                    os.path.join(root, "usr", "lib") + ":" + os.path.join(root, "lib"), exe]
    return [sys.executable]


def run_patcher(patcher, path):
    env = dict(os.environ)
    cmd = python_cmd()
    if cmd[0].endswith("ld-musl-aarch64.so.1"):
        exe = os.path.realpath(sys.executable)
        root = exe.split("/och-runtime/root/usr/bin/")[0] + "/och-runtime/root"
        env["PYTHONHOME"] = os.path.join(root, "usr")
    return subprocess.run([*cmd, patcher, path], capture_output=True, text=True, env=env)


def find_gcc():
    cc = shutil.which("gcc")
    if cc:
        return cc
    cand = "/root/.local/share/opencode/och-runtime/jellyfin-vpu-build/bin/gcc"
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None


def extract_guard(patched_text):
    """从打过补丁的源码里抽出运行期开关守卫块（含外层花括号）。"""
    head = "         {\n            /* och patch: 幻影门运行期开关"
    at = patched_text.find(head)
    if at < 0:
        fail("extract_guard：找不到守卫块头部")
    end = patched_text.find("\n#if defined(__GNUC__)", at)
    if end < 0:
        fail("extract_guard：找不到守卫块尾部")
    return patched_text[at:end]


HARNESS_TMPL = """#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
struct pipe_tensor { unsigned dims[4]; };
struct op_t { struct pipe_tensor *output_tensors[1]; };
#define ML_DBG(...) printf(__VA_ARGS__)
static int supported(struct op_t *operation)
{
%(GUARD)s
   return 1;
}
int main(int argc, char **argv)
{
   unsigned c = (unsigned)atoi(argv[1]);
   struct pipe_tensor t = {{0, 0, 0, c}};
   struct op_t op = { {&t} };
   int r = supported(&op);
   printf("RESULT=%%d\\n", r);
   return 0;
}
"""


def harness_run(cc, tmp, guard, env_val, out_c):
    """编译并运行 harness；返回 (result, stdout)。env_val=None ⇒ unset。"""
    src = HARNESS_TMPL % {"GUARD": guard}
    cpath = os.path.join(tmp, "harness.c")
    exe = os.path.join(tmp, "harness")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(src)
    p = subprocess.run([cc, "-x", "c", "-std=c11", "-o", exe, cpath],
                       capture_output=True, text=True)
    if p.returncode != 0:
        fail("harness 编译失败", p.stderr)
    env = dict(os.environ)
    env.pop("TEFLON_PHANTOM_GATE", None)
    if env_val is not None:
        env["TEFLON_PHANTOM_GATE"] = env_val
    r = subprocess.run([exe, str(out_c)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        fail("harness 运行失败", r.stderr)
    m = re.search(r"RESULT=(\d)", r.stdout)
    if not m:
        fail("harness 输出无 RESULT", r.stdout)
    return int(m.group(1)), r.stdout


def main():
    patcher = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCHER
    if not os.path.isfile(patcher):
        fail("找不到补丁器: " + patcher)
    if not os.path.isfile(REAL_ML):
        fail("找不到真实源码: " + REAL_ML)
    tmp = tempfile.mkdtemp(prefix="pgate-selftest-")

    def fresh(name):
        p = os.path.join(tmp, name)
        shutil.copyfile(REAL_ML, p)
        return p

    # ---------- 正控制 1：真实源码新装 ----------
    good = fresh("good.c")
    r = run_patcher(patcher, good)
    if r.returncode != 0:
        fail("正控制 1 失败：真实源码新装应成功", r.stderr)
    body = open(good, encoding="utf-8").read()
    for tok in ("och-patch-v7-phantom-gate", 'getenv("TEFLON_PHANTOM_GATE")',
                "och-pgate: 幻影门运行期关闭", "幻影通道形状", "return false;",
                "#include <stdlib.h>"):
        if tok not in body:
            fail("正控制 1 失败：缺少 " + tok)
    if body.count('getenv("TEFLON_PHANTOM_GATE")') != 1:
        fail("正控制 1 失败：getenv 出现 %d 次（期望 1）"
             % body.count('getenv("TEFLON_PHANTOM_GATE")'))
    print("  [ok] 正控制 1：真实源码新装，标记/开关/双日志/stdlib 齐全")

    # ---------- 正控制 2：运行期语义 harness ----------
    cc = find_gcc()
    if cc is None:
        fail("正控制 2 需要 gcc（系统或 och-runtime），本机找不到 —— 不得静默跳过")
    guard = extract_guard(body)
    cases = [
        (None, 122, 0, "幻影通道形状"),     # unset+122 ⇒ 拒 + 拒绝日志
        ("1", 122, 0, "幻影通道形状"),      # =1+122 ⇒ 拒
        ("0", 122, 1, "幻影门运行期关闭"),  # =0+122 ⇒ 接 + 关门通告
        (None, 120, 1, None),               # unset+120 ⇒ 接（8 倍数放行）
        ("0", 244, 1, "幻影门运行期关闭"),  # =0+244 ⇒ 接（取证形态）
    ]
    for env_val, c, want, log in cases:
        got, out = harness_run(cc, tmp, guard, env_val, c)
        if got != want:
            fail("正控制 2 失败：env=%r out_c=%d ⇒ RESULT=%d（期望 %d）"
                 % (env_val, c, got, want), out)
        if log and log not in out:
            fail("正控制 2 失败：env=%r out_c=%d 缺少日志「%s」" % (env_val, c, log), out)
        if log is None and ("幻影" in out):
            fail("正控制 2 失败：放行 case 不该有幻影日志", out)
    print("  [ok] 正控制 2：运行期语义 5 case 全对（unset/1 拒、0 接、双日志不静默、8 倍数放行）")

    # ---------- 正控制 3：旧版升级路径 ----------
    legacy_patcher = os.path.join(tmp, "legacy-pgate.py")
    g = subprocess.run([GIT, "-C", ROOT, "show",
                        "HEAD:build-recipe/patch-etnaviv-phantom-gate.py"],
                       capture_output=True, text=True)
    if g.returncode != 0:
        fail("正控制 3 失败：取不到 git HEAD 旧版补丁器", g.stderr)
    if 'getenv("TEFLON_PHANTOM_GATE")' in g.stdout:
        fail("正控制 3 setup：HEAD 已是新版，升级路径无法用 HEAD 旧件测试（需人工指定旧件）")
    with open(legacy_patcher, "w", encoding="utf-8") as f:
        f.write(g.stdout)
    up = fresh("upgrade.c")
    r = run_patcher(legacy_patcher, up)
    if r.returncode != 0:
        fail("正控制 3 失败：旧版补丁器应用失败", r.stderr)
    r = run_patcher(patcher, up)
    if r.returncode != 0 or "旧版已升级" not in r.stdout:
        fail("正控制 3 失败：新版应原地升级旧版", r.stderr + r.stdout)
    ub = open(up, encoding="utf-8").read()
    if ub.count('getenv("TEFLON_PHANTOM_GATE")') != 1:
        fail("正控制 3 失败：升级后 getenv 不唯一")
    if "return false;" not in ub or "och-patch-v7-phantom-gate" not in ub:
        fail("正控制 3 失败：升级丢了拒绝出口或编译期标记")
    print("  [ok] 正控制 3：旧版（git HEAD）原地升级成功，开关/出口/标记齐全")

    # ---------- 负控制 1：新版重复应用 ----------
    r = run_patcher(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：新版重复应用应拒绝")
    print("  [ok] 负控制 1：新版重复应用被拒绝")

    # ---------- 负控制 2：无关文件 ----------
    unrelated = os.path.join(tmp, "unrelated.c")
    with open(unrelated, "w", encoding="utf-8") as f:
        f.write("int main(void){return 0;}\n")
    r = run_patcher(patcher, unrelated)
    if r.returncode == 0:
        fail("负控制 2 失效：无关文件应拒绝")
    print("  [ok] 负控制 2：无关文件被拒绝")

    # ---------- 负控制 3：判据自测（%8→%4 必须改变行为） ----------
    bad_guard = guard.replace("och_out_c % 8 != 0", "och_out_c % 4 != 0")
    # 分离点选 124：124%8=4 ⇒ 原版拒；124%4=0 ⇒ 篡改版放行。
    got0, _ = harness_run(cc, tmp, guard, None, 124)
    got1, _ = harness_run(cc, tmp, bad_guard, None, 124)
    if not (got0 == 0 and got1 == 1):
        fail("负控制 3 失效：判据扰动（%%8→%%4）未改变 124 的判定（原版 %d / 篡改 %d）"
             % (got0, got1))
    print("  [ok] 负控制 3：判据自测 —— %%8→%%4 使 124 由拒变接（harness 能抓语义变化）")

    # ---------- 负控制 4：判据自测（env 判断恒门开 ⇒ =0 必须失效） ----------
    bad_guard2 = guard.replace(
        "int och_pgate_on = !(och_pgate_env && och_pgate_env[0] == '0' && och_pgate_env[1] == '\\0');",
        "int och_pgate_on = 1; (void)och_pgate_env;")
    if bad_guard2 == guard:
        fail("负控制 4 setup：env 判断行替换失败（锚点漂移）")
    got, out = harness_run(cc, tmp, bad_guard2, "0", 122)
    if got != 0:
        fail("负控制 4 失效：恒门开篡改后 =0+122 应被拒（开关失效必须可检测）")
    print("  [ok] 负控制 4：判据自测 —— 恒门开篡改使 =0 失效（开关语义可检测）")

    # ---------- 负控制 5：FSGUARD 锚点缺失 ----------
    nofsg = fresh("nofsg.c")
    txt = open(nofsg, encoding="utf-8").read()
    anchor = "         struct pipe_tensor *output_tensor = operation->output_tensors[0];\n"
    if anchor not in txt:
        fail("负控制 5 setup：真实源码里找不到 FSGUARD 锚点")
    txt = txt.replace(anchor, "", 1)
    with open(nofsg, "w", encoding="utf-8") as f:
        f.write(txt)
    r = run_patcher(patcher, nofsg)
    if r.returncode == 0:
        fail("负控制 5 失效：FSGUARD 锚点缺失应拒绝")
    print("  [ok] 负控制 5：FSGUARD 锚点缺失被拒绝（共存前提守住）")

    print("  [ok] selftest-phantom-gate 全部通过（正控制 3 + 负控制 5）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
