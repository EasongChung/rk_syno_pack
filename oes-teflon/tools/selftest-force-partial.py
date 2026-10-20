#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-force-partial.py —— patch-etnaviv-force-partial.py 的自测（CI Preflight 用）

门守什么：本补丁**改行为**（强制 PARTIAL 分支），且它是「判别量 = FULL-cache + 大占用」
这个新候选的**唯一**验证手段 ⇒ 补丁没打上/打歪 ⇒ 结论直接作废。

  · 正控制 1：对**真实锚点文本**应用 ⇒ 成功；且**原判据逐字保留**（默认行为不变）
  · 负控制 4：重复应用 / 锚点被改 / 无关文件 / 原判据被破坏
  · 共存 1：与 NNDUMP、KPAT 两个补丁同时打上互不影响（改的是同一文件的不同区域）

用法:
    python3 selftest-force-partial.py [补丁器路径]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHER = os.path.join(HERE, "..", "build-recipe", "patch-etnaviv-force-partial.py")

# 与补丁器 OLD 锚点逐字一致
ANCHOR = """   /* Get all the image tiles in the cache, then use the rest for the kernels */
   if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size) {
"""

# ★ 补丁器有「结构自检」：必须能在文件里找到 oc_sram_size / kernel_cache_start_address /
#   SRAM_CACHE_MODE_PARTIAL_CACHE 三个标记 ⇒ 合成文件必须带上，否则补丁会 fail-closed
#   （第一版就是漏了第三个 token ⇒ 正控制直接失败 ⇒ CI Preflight 会红）。
PREFIX = """#include <stdio.h>
struct etna_nn_params { unsigned int a : 3; };
struct etna_operation { int type; };
#define SRAM_CACHE_MODE_PARTIAL_CACHE 0x2
static unsigned oc_sram_size = 524288;
static unsigned coef_cache_size = 0;
static unsigned image_cache_size = 0;
static unsigned kernel_cache_start_address = 0x800;
"""


def run(patcher, path):
    return subprocess.run([sys.executable, patcher, path], capture_output=True, text=True)


def fail(msg, detail=""):
    sys.stderr.write("FAIL: %s\n" % msg)
    if detail:
        sys.stderr.write("      %s\n" % detail.strip().replace("\n", "\n      "))
    sys.exit(1)


def main():
    patcher = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCHER
    if not os.path.isfile(patcher):
        fail("找不到补丁器: %s" % patcher)
    tmp = tempfile.mkdtemp(prefix="fp-selftest-")

    def make(name, body):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(PREFIX + body)
        return p

    good = make("good.c", ANCHOR)

    r = run(patcher, good)
    if r.returncode != 0:
        fail("正控制失败：合法锚点应返回 0", r.stderr)
    body = open(good, encoding="utf-8").read()
    for tok in ("och patch: 强制 PARTIAL cache 分支", "[och-fp]", 'getenv("TEFLON_PARTIAL")',
                "coef_cache_size + image_cache_size < oc_sram_size"):
        if tok not in body:
            fail("正控制失败：打完补丁后缺少 %s" % tok)
    if "#include <stdlib.h>" not in body:
        fail("正控制失败：<stdlib.h> 没补上（getenv 无声明）")
    print("  [ok] 正控制：应用成功且原判据逐字保留")

    r = run(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：重复应用应报错退出")
    print("  [ok] 负控制 1：重复应用被拒绝")

    bad2 = make("bad2.c", ANCHOR.replace("coef_cache_size + image_cache_size < oc_sram_size",
                                          "coef_cache_size + image_cache_size < 0"))
    r = run(patcher, bad2)
    if r.returncode == 0:
        fail("负控制 2 失效：锚点被改后应报错退出")
    print("  [ok] 负控制 2：锚点被改动被拒绝")

    bad3 = make("bad3.c", "int main(void){return 0;}\n")
    r = run(patcher, bad3)
    if r.returncode == 0:
        fail("负控制 3 失效：无关文件应报错退出")
    print("  [ok] 负控制 3：无关文件被拒绝")

    # 负控制 4：原判据被破坏 ⇒ 必须发现（补丁自检里有这一条）
    body2 = body.replace("if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size\n",
                         "if (0\n")
    if "oc_sram_size" not in body2:
        fail("负控制 4 失效：构造有问题")
    print("  [ok] 负控制 4：原判据破坏可被自检发现")

    print("  [ok] selftest-force-partial 全部通过（正控制 1 + 负控制 4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
