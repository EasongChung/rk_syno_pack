# -*- coding: utf-8 -*-
"""所有 workflow 的 `run:` 块做 `bash -n` 语法检查。

## 为什么需要（run 39209044465 第 23 步红的根因）

我在那个 `for pat in …` 列表末尾留了续行反斜杠，而被替换掉的那段里
带着 `; do` ⇒ **`for` 循环没有终止符** ⇒ shell 语法错误，
`set -e` 直接退出，日志里**一行 `OK:` 都没有**，只看到脚本回显。

它完全可以在本地抓到：`bash -n` 是零成本的静态检查。
⇒ 这道检查并入本地入口（§5），并且**每次都跑全部 workflow**，不留死角。

依赖：`bash`（Git Bash / WSL / Linux runner 都有）。
若本机没有 bash，脚本**明确跳过并说明**，不静默当通过。
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import yaml

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WF_DIR = os.path.join(REPO, ".github", "workflows")


def self_test(bash):
    """★ 先自证 `bash -n` 可信，再采信它的结论。

    实测：本机的 `bash` 是 **WSL 的**，且 `/etc/fstab` 挂载失败
    （`wsl: Processing /etc/fstab with mount -a failed`）。
    那种情况下 `bash -n` 的返回值**不可信** ——
    门若照单全收，就又是一道「给出虚假保证」的门（§2.1）。

    ⇒ 先喂一段**已知错误**的脚本；抓不住 ⇒ 明确报「不可信」并 exit 2，
      **绝不**输出 PASS。
    """
    bad = ("for pat in 'a' \\\n"
           "           'b' \\\n"
           "\n"
           "  echo \"$pat\"\n"
           "done\n")
    good = "for pat in 'a' 'b'; do\n  echo \"$pat\"\ndone\n"
    rb = subprocess.run([bash, "-n"], input=bad, text=True,
                        capture_output=True, encoding="utf-8", errors="replace")
    rg = subprocess.run([bash, "-n"], input=good, text=True,
                        capture_output=True, encoding="utf-8", errors="replace")
    ok = (rb.returncode != 0) and (rg.returncode == 0)
    if not ok:
        print("★ bash -n **不可信**，本工具不出结论（这不是通过）：")
        print("   已知错误版本 rc=%d（应为非 0）" % rb.returncode)
        print("   已知正确版本 rc=%d（应为 0）" % rg.returncode)
        for ln in ((rb.stderr or "") + (rg.stderr or "")).splitlines()[:4]:
            print("   bash: %s" % ln[:140])
        print()
        print("   本机 bash 可能来自未正确挂载的 WSL。换 Git Bash / WSL 修好后重跑，")
        print("   或在 CI（ubuntu-latest）上跑 —— 那里 bash 一定是可靠的。")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="检查全部 workflow（默认也全部）")
    args = ap.parse_args()

    bash = shutil.which("bash")
    if not bash:
        print("SKIP: 本机没有 bash —— 这道检查**未运行**，不是通过。")
        print("      CI（ubuntu-latest）一定有 bash；本地请用 Git Bash 或 WSL。")
        return 2

    if not self_test(bash):
        return 2

    bad = []
    n = 0
    for f in sorted(glob.glob(os.path.join(WF_DIR, "*.yml"))):
        name = os.path.basename(f)
        try:
            d = yaml.safe_load(open(f, encoding="utf-8"))
        except Exception as exc:                                  # noqa: BLE001
            bad.append((name, "?", "YAML 解析失败: %r" % (exc,)))
            continue
        for jn, job in (d.get("jobs") or {}).items():
            for st in (job.get("steps") or []):
                run = st.get("run") or ""
                if not run.strip():
                    continue
                n += 1
                r = subprocess.run([bash, "-n"], input=run, text=True,
                                   capture_output=True, encoding="utf-8",
                                   errors="replace")
                if r.returncode != 0:
                    err = (r.stderr or "").strip().splitlines()
                    bad.append((name, st.get("name", "?"),
                                err[0][:160] if err else "?"))

    print("=== bash -n：检查 %d 个 run 块，语法错 %d 个 ===" % (n, len(bad)))
    for nm, sn, why in bad:
        print("  %-52s %s" % (nm, sn))
        print("      %s" % why)
    print()
    if bad:
        print("RESULT: FAIL —— 有 run 块是**语法错误**的，CI 上必然失败。")
        return 1
    print("RESULT: ALL PASS（%d 个 run 块语法正确）" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
