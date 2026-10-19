#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-nn-desc-dump.py —— patch-etnaviv-nn-desc-dump.py 的自测（CI Preflight 用）

## 为什么要这道门

项目铁律：**补丁必须有「正控制 + 至少一个负控制 + 一次回归」**，且
**「检查器自己会错」** —— 门触发时的错误信息退化成 traceback 是本项目吃过的两次亏
（见 build-teflon-fix-rofix-perchannel.yml 里 Preflight 段的注释）。

本补丁把 Mesa 的 `create_nn_config()` 收尾插入一段打印。风险有三类，本门逐个封死：

| 风险 | 负控制 | 期望 |
|---|---|---|
| 重复应用（幂等） | 同一份文件打两次 | 第二次**必须报错退出**（非 0） |
| 锚点被上游改动 | 把锚点里的一行改掉 | **必须报错退出**（不许模糊匹配） |
| 源文件结构不符 | 给一个完全无关的文件 | **必须报错退出** |

正控制：对**真实锚点文本**的合成文件应用 ⇒ 成功，且产物含标记与 `[och-nn]`。

## 用法

    python3 selftest-nn-desc-dump.py [补丁器路径]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHER = os.path.join(HERE, "..", "build-recipe", "patch-etnaviv-nn-desc-dump.py")

# 与补丁器里的 OLD 锚点**逐字一致**（合成文件必须复现真实源码的上下文）
ANCHOR = """   map->per_channel_post_mul = 0x0;

   etna_bo_cpu_fini(bo);

   return bo;
}
"""

PREFIX = """#include <stdio.h>
struct etna_nn_params { unsigned int a : 3; };
struct etna_operation { int type; };
"""


def run_patcher(patcher, path):
    return subprocess.run([sys.executable, patcher, path],
                          capture_output=True, text=True)


def fail(msg, detail=""):
    sys.stderr.write("FAIL: %s\n" % msg)
    if detail:
        sys.stderr.write("      %s\n" % detail.strip().replace("\n", "\n      "))
    sys.exit(1)


def main():
    patcher = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCHER
    if not os.path.isfile(patcher):
        fail("找不到补丁器: %s" % patcher)

    tmp = tempfile.mkdtemp(prefix="nndesc-selftest-")
    good = os.path.join(tmp, "good.c")
    with open(good, "w", encoding="utf-8") as f:
        f.write(PREFIX + ANCHOR)

    # ── 正控制 ───────────────────────────────────────────────
    r = run_patcher(patcher, good)
    if r.returncode != 0:
        fail("正控制失败：补丁器对合法锚点应返回 0", r.stderr)
    body = open(good, encoding="utf-8").read()
    if "och patch: NN core descriptor dump" not in body:
        fail("正控制失败：打完补丁看不到标记")
    if "[och-nn]" not in body:
        fail("正控制失败：打完补丁看不到输出前缀 [och-nn]")
    if "sizeof(*map)" not in body:
        fail("正控制失败：补丁里没有原始字节 dump")
    if body.count("och patch: NN core descriptor dump") != 1:
        fail("正控制失败：标记应恰好出现 1 次")
    print("  [ok] 正控制：对合法锚点应用成功")

    # ── 负控制 1：重复应用必须拒绝 ───────────────────────────
    r = run_patcher(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：对已打补丁的文件再次应用应报错退出")
    if "拒绝重复应用" not in r.stderr:
        fail("负控制 1 失效：错误信息不对（应说明「拒绝重复应用」）", r.stderr)
    print("  [ok] 负控制 1：重复应用被拒绝")

    # ── 负控制 2：锚点被改动必须拒绝 ─────────────────────────
    bad2 = os.path.join(tmp, "bad2.c")
    with open(bad2, "w", encoding="utf-8") as f:
        f.write(PREFIX + ANCHOR.replace("map->per_channel_post_mul = 0x0;",
                                        "map->per_channel_post_mul = 0x1;"))
    r = run_patcher(patcher, bad2)
    if r.returncode == 0:
        fail("负控制 2 失效：锚点被改动后应报错退出（不许模糊匹配）")
    if "匹配到 0 处锚点" not in r.stderr:
        fail("负控制 2 失效：错误信息不对（应报「匹配到 0 处锚点」）", r.stderr)
    print("  [ok] 负控制 2：锚点被改动被拒绝")

    # ── 负控制 3：结构完全不符必须拒绝 ───────────────────────
    bad3 = os.path.join(tmp, "bad3.c")
    with open(bad3, "w", encoding="utf-8") as f:
        f.write("int main(void){return 0;}\n")
    r = run_patcher(patcher, bad3)
    if r.returncode == 0:
        fail("负控制 3 失效：无关文件应报错退出")
    print("  [ok] 负控制 3：无关文件被拒绝")

    # ── 负控制 4：锚点出现两次必须拒绝（不许改第一处就蒙对）──
    bad4 = os.path.join(tmp, "bad4.c")
    with open(bad4, "w", encoding="utf-8") as f:
        f.write(PREFIX + ANCHOR + ANCHOR)
    r = run_patcher(patcher, bad4)
    if r.returncode == 0:
        fail("负控制 4 失效：锚点多处出现时必须报错退出")
    print("  [ok] 负控制 4：锚点多处出现被拒绝")

    print("  [ok] selftest-nn-desc-dump 全部通过（正控制 1 + 负控制 4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
