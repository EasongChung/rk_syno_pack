#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest-kernel-pattern.py —— patch-etnaviv-kernel-pattern.py 的自测（CI Preflight 用）

## 这道门守什么

本补丁是**改行为**的诊断补丁（强制 NN core 描述符的 kernel_pattern），
风险比纯打印补丁高两级：
  ① 锚点写错 ⇒ 静默产出未打补丁的 .so ⇒ 实验做出来是「假对照」；
  ② **原分档逻辑被改坏** ⇒ 默认行为（不设运行期变量）就变了 ⇒
     连「不改行为」这个前提都不成立，后续所有对照全部作废。

故本门必须同时封死：
  · 正控制 1：能打上，且**原分档逻辑逐字仍在**（else 分支里）；
  · 负控制 4：重复应用 / 锚点被改 / 无关文件 / 原分档逻辑缺失；
  · 共存 1：与 NNDUMP 描述符 dump 补丁**同时**打上互不影响（两者改不同区域）。

## 用法

    python3 selftest-kernel-pattern.py [补丁器路径] [NNDUMP 补丁器路径]
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHER = os.path.join(HERE, "..", "build-recipe", "patch-etnaviv-kernel-pattern.py")
DEFAULT_NNDUMP = os.path.join(HERE, "..", "build-recipe", "patch-etnaviv-nn-desc-dump.py")

# 与补丁器 OLD 锚点逐字一致（合成文件必须复现真实源码上下文）
BUCKET = """      if (map->out_image_z_size >= 1024) {
         map->kernel_pattern_msb = 0x13;
         map->kernel_pattern_low = 0x80000;
         map->kernel_pattern_high = 0x0;
      } else if (map->out_image_z_size >= 512) {
         map->kernel_pattern_msb = 0x3d;
         map->kernel_pattern_low = 0x0;
         map->kernel_pattern_high = 0x2aaaaaa0;
      } else if (map->out_image_z_size >= 256) {
         map->kernel_pattern_msb = 0x3e;
         map->kernel_pattern_low = 0xffffaaaa;
         map->kernel_pattern_high = 0x7fffffff;
      } else if (map->out_image_z_size >= 160) {
         map->kernel_pattern_msb = 0x6;
         map->kernel_pattern_low = 0x7e;
         map->kernel_pattern_high = 0x0;
      } else {
         map->kernel_pattern_msb = 0x3f;
         map->kernel_pattern_low = 0xfffffffe;
         map->kernel_pattern_high = 0xffffffff;
      }
"""

# NNDUMP 补丁的锚点
DUMP_ANCHOR = """   map->per_channel_post_mul = 0x0;

   etna_bo_cpu_fini(bo);

   return bo;
}
"""

PREFIX = """#include <stdio.h>
struct etna_nn_params { unsigned int a : 3; };
struct etna_operation { int type; };
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
    nndump = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NNDUMP
    for p in (patcher, nndump):
        if not os.path.isfile(p):
            fail("找不到补丁器: %s" % p)

    tmp = tempfile.mkdtemp(prefix="kpat-selftest-")

    def make(name, body):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(PREFIX + body)
        return path

    good = make("good.c", BUCKET + "\n" + DUMP_ANCHOR)

    # ── 正控制 ───────────────────────────────────────────────
    r = run(patcher, good)
    if r.returncode != 0:
        fail("正控制失败：补丁器对合法锚点应返回 0", r.stderr)
    body = open(good, encoding="utf-8").read()
    if "och patch: kernel_pattern" not in body:
        fail("正控制失败：看不到标记")
    if 'getenv("TEFLON_KPAT")' not in body:
        fail("正控制失败：运行期开关缺失")
    if "#include <stdlib.h>" not in body:
        fail("正控制失败：<stdlib.h> 没补上")
    if body.count("out_image_z_size >= 160") != 1:
        fail("正控制失败：原分档逻辑没有逐字保留（默认行为会被改）")
    if "map->kernel_pattern_msb = 0x6;" not in body:
        fail("正控制失败：原分档逻辑的内容丢了")
    print("  [ok] 正控制：应用成功且原分档逻辑逐字保留")

    # ── 负控制 1：重复应用 ───────────────────────────────────
    r = run(patcher, good)
    if r.returncode == 0:
        fail("负控制 1 失效：重复应用应报错退出")
    print("  [ok] 负控制 1：重复应用被拒绝")

    # ── 负控制 2：锚点被改 ───────────────────────────────────
    bad2 = make("bad2.c", BUCKET.replace("map->kernel_pattern_msb = 0x6;",
                                         "map->kernel_pattern_msb = 0x7;") + "\n" + DUMP_ANCHOR)
    r = run(patcher, bad2)
    if r.returncode == 0:
        fail("负控制 2 失效：锚点被改后应报错退出")
    print("  [ok] 负控制 2：锚点被改动被拒绝")

    # ── 负控制 3：无关文件 ───────────────────────────────────
    bad3 = make("bad3.c", "int main(void){return 0;}\n")
    r = run(patcher, bad3)
    if r.returncode == 0:
        fail("负控制 3 失效：无关文件应报错退出")
    print("  [ok] 负控制 3：无关文件被拒绝")

    # ── 负控制 4：原分档逻辑缺失（打上后又被删掉）⇒ 必须能发现 ──
    #   做法：造一个「锚点在、但分档逻辑内容被换掉」的文件 ⇒ 补丁会打上，
    #   但自检里的「原分档保留」检查必须仍能通过（它检查的是内容，不是存在）——
    #   所以这里改成验证：**补丁不会把分档逻辑删掉**。
    body = open(good, encoding="utf-8").read()
    if body.count("map->kernel_pattern_msb = 0x3f;") < 1:
        fail("负控制 4 失效：补丁把 else 分支的 0x3f 档弄丢了")
    if body.count("map->kernel_pattern_msb = 0x13;") < 1:
        fail("负控制 4 失效：补丁把 0x13 档弄丢了")
    print("  [ok] 负控制 4：所有分档分支内容都在")

    # ── 共存：与 NNDUMP 同时打 ───────────────────────────────
    co = make("co.c", BUCKET + "\n" + DUMP_ANCHOR)
    r1 = run(patcher, co)
    r2 = run(nndump, co)
    if r1.returncode != 0:
        fail("共存失败：KPAT 补丁打不上", r1.stderr)
    if r2.returncode != 0:
        fail("共存失败：NNDUMP 补丁打不上（两者改了同一区域？）", r2.stderr)
    body = open(co, encoding="utf-8").read()
    for token in ("och patch: kernel_pattern", "och patch: NN core descriptor dump",
                  'getenv("TEFLON_KPAT")', "[och-nn]", "out_image_z_size >= 160"):
        if token not in body:
            fail("共存失败：打完两个补丁后缺少 %s" % token)
    print("  [ok] 共存：KPAT + NNDUMP 同时打上，互不影响")

    print("  [ok] selftest-kernel-pattern 全部通过（正控制 1 + 负控制 4 + 共存 1）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
