#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付版的门控语义**行为级**自检助手。

## 为什么要它

2026-10-21 起，三个修复补丁的门控语义从「设了才开」改成
**「默认开，`=0` 才关」**（交付版要求修复默认生效，产品侧不设任何环境变量）：

| 环境变量 | 修复 | 打印 |
|---|---|---|
| 不设 | **开** | 关（交付版不刷屏） |
| `=0` | 关 | 关 |
| `=1` / 其它 | **开** | **开**（调试用） |

原先的自测只验「门控字符串在不在文件里」，**验不出这三种状态的行为**。
本助手把门控片段从补丁产物里抽出来，用宿主 gcc 真编译、真运行，
逐一断言三种状态的 (修复, 打印) 组合。

## 用法

    from gate_semantics_check import check_gate

    ok, msg = check_gate(patched_text, "TEFLON_WDOMAIN", "och_wd")
    # 期望：unset -> (1,0)   "0" -> (0,0)   "1" -> (1,1)

## 设计要点

- **抽片段而不是重写**：从**打完补丁的真实文本**里抽，避免「测的代码和编译的代码
  不是同一份」这种最常见的自欺
- 片段里的变量名可参数化（不同补丁的变量名不同）
- 编译失败 ⇒ 直接判失败并回显编译器输出（不静默跳过）
"""

import os
import re
import subprocess
import tempfile


def _extract(patched_text, var_prefix):
    """抽出门控片段（含变量声明 + 两个判定）。"""
    m = re.search(
        r"(const char \*%s = getenv\(\"[A-Z_]+\"\);\s*\n"
        r"\s*int %s_on = .*?;\s*\n"
        r"\s*int %s_log = .*?;)" % (var_prefix, var_prefix, var_prefix),
        patched_text, re.S)
    return m.group(1) if m else None


def check_gate(patched_text, env_name, var_prefix):
    """返回 (ok, 详情)。抽取门控片段并真编译真跑，断言三种状态的语义。"""
    frag = _extract(patched_text, var_prefix)
    if frag is None:
        return False, "抽不出门控片段（%s）" % env_name

    harness = """
#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv) {
   if (argc > 1)
      setenv("%(env)s", argv[1], 1);
   else
      unsetenv("%(env)s");
%(frag)s
   printf("%%d %%d\\n", %(prefix)s_on, %(prefix)s_log);
   return 0;
}
""" % {"env": env_name, "frag": frag, "prefix": var_prefix}

    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        cc = os.environ.get("CC", "cc")
        r = subprocess.run([cc, "-std=gnu99", "-Wall", "-Wextra", "-fsyntax-only", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "门控片段编译失败: %s" % (r.stderr or "")[:300]
        r = subprocess.run([cc, "-std=gnu99", "-o", path + ".bin", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "门控片段链接失败: %s" % (r.stderr or "")[:300]

        def run(args):
            rr = subprocess.run([path + ".bin"] + args, capture_output=True, text=True)
            return [int(x) for x in rr.stdout.split()]

        unset = run([])
        zero = run(["0"])
        one = run(["1"])
    finally:
        for suf in ("", ".bin"):
            try:
                os.unlink(path + suf)
            except OSError:
                pass

    expect = {"unset": (1, 0), "0": (0, 0), "1": (1, 1)}
    got = {"unset": tuple(unset), "0": tuple(zero), "1": tuple(one)}
    bad = [k for k in expect if got[k] != expect[k]]
    if bad:
        return False, "门控语义错: %s（期望 %s，实得 %s）" % (
            ", ".join(bad), expect, got)
    return True, "门控语义正确: unset=%s 0=%s 1=%s" % (got["unset"], got["0"], got["1"])


def check_gate_simple(patched_text, env_name, var_prefix):
    """简化版：只有「开/关」两态（没有打印分支的补丁用）。

    期望：unset -> 1（开）   "0" -> 0（关）   "1" -> 1（开）
    """
    # ★ 用字符串拼接而不是 % 格式化：模式里含 %(.*?) 会被 % 解析成格式说明符
    pat = (r"(const char \*" + re.escape(var_prefix) + r" = getenv\(\"" +
           re.escape(env_name) + r"\"\);)\s*\n" + r"\s*if \((.*?)\) \{")
    m = re.search(pat, patched_text, re.S)
    if not m:
        return False, "抽不出门控片段（%s，简化版）" % env_name
    decl, cond = m.group(1), m.group(2)

    harness = """
#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv) {
   if (argc > 1)
      setenv("%(env)s", argv[1], 1);
   else
      unsetenv("%(env)s");
%(decl)s
   printf("%%d\\n", (%(cond)s) ? 1 : 0);
   return 0;
}
""" % {"env": env_name, "decl": decl, "cond": cond}

    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        cc = os.environ.get("CC", "cc")
        r = subprocess.run([cc, "-std=gnu99", "-Wall", "-Wextra", "-o", path + ".bin", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "门控片段编译失败: %s" % (r.stderr or "")[:300]

        def run(args):
            rr = subprocess.run([path + ".bin"] + args, capture_output=True, text=True)
            return int(rr.stdout.strip() or -1)

        unset, zero, one = run([]), run(["0"]), run(["1"])
    finally:
        for suf in ("", ".bin"):
            try:
                os.unlink(path + suf)
            except OSError:
                pass

    expect = {"unset": 1, "0": 0, "1": 1}
    got = {"unset": unset, "0": zero, "1": one}
    bad = [k for k in expect if got[k] != expect[k]]
    if bad:
        return False, "门控语义错: %s（期望 %s，实得 %s）" % (", ".join(bad), expect, got)
    return True, "门控语义正确: unset=%s 0=%s 1=%s" % (got["unset"], got["0"], got["1"])
