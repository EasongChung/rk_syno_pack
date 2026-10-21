#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 etnaviv ML 驱动源码补上被间接包含掩盖的头文件依赖。

## 为什么需要

`etnaviv_ml.c` 只有 `#include <stdio.h>`，没有 `<stdlib.h>`；
`etnaviv_ml_nn.c` 两个都没有。我们后续的补丁（权重域 +128、图输出收口、
舍入探针）首次在这两个文件里引入 `getenv()` / `atoi()` / `fprintf()`。

当前能编能跑，只是因为 mesa 的头文件链恰好把 `stdlib.h` / `stdio.h` 间接带进来了。
这属于**脆弱依赖**：上游改一次 include、或编译选项一变，就会退化成
`implicit declaration`（返回值被当 int，指针被截断）。

## 做法

只加 include，不改任何逻辑。幂等：已经存在就跳过，不重复插入。

  etnaviv_ml.c    在 `#include <stdio.h>` 之后补 `#include <stdlib.h>`
  etnaviv_ml_nn.c 在首个 `#include "pipe/p_state.h"` 之前补 `<stdio.h>` `<stdlib.h>`

## 自证

  python3 build-recipe/patch-etnaviv-includes.py --selftest

正控制：两个真实源码副本各自被改动、且各新增恰好 1 个 `<stdlib.h>`。
负控制：重复应用不产生重复行；认不出的文件必须报错（不静默跳过）。
"""

import io
import os
import shutil
import sys
import tempfile

MARKER = "och patch: explicit std headers"

# ml.c：紧跟已有的 stdio.h 之后
ANCHOR_ML = "#include <stdio.h>\n"
NEW_ML = ("#include <stdio.h>\n"
          "#include <stdlib.h> /* %s: getenv()/atoi() 的显式声明 */\n" % MARKER)

# nn.c：首个 pipe/p_state.h 之前（标记只出现一次，便于幂等判定）
ANCHOR_NN = '#include "pipe/p_state.h"\n'
NEW_NN = ("#include <stdio.h>   /* fprintf() */\n"
          "#include <stdlib.h>  /* getenv()/atoi() — %s */\n"
          '#include "pipe/p_state.h"\n' % MARKER)


def apply(path):
    """返回 0 = 已应用或已满足；1 = 失败。"""
    src = io.open(path, encoding="utf-8").read()

    if MARKER in src:
        print("  [och-inc] 已有显式 include，跳过: %s" % path)
        return 0

    # 已经是「有 stdlib.h」的文件也不必再动（幂等，不依赖标记）
    if "#include <stdlib.h>" in src:
        print("  [och-inc] 已存在 <stdlib.h>，跳过: %s" % path)
        return 0

    if src.count(ANCHOR_ML) == 1 and "#include <stdlib.h>" not in src:
        # ml.c 分支
        io.open(path, "w", encoding="utf-8").write(
            src.replace(ANCHOR_ML, NEW_ML, 1))
        print("  [och-inc] 已补 <stdlib.h>: %s" % path)
        return 0

    if src.count(ANCHOR_NN) == 1:
        io.open(path, "w", encoding="utf-8").write(
            src.replace(ANCHOR_NN, NEW_NN, 1))
        print("  [och-inc] 已补 <stdio.h>/<stdlib.h>: %s" % path)
        return 0

    sys.stderr.write(
        "ERROR: %s 里既没有唯一的 `#include <stdio.h>`，也没有唯一的\n"
        '       `#include "pipe/p_state.h"` —— 认不出这是哪个文件。\n'
        "       源码可能已升级，拒绝盲目插入。\n" % path)
    return 1


def selftest(mesa_src):
    ok = True
    tdir = os.path.join(mesa_src, "src/gallium/drivers/etnaviv")
    files = [os.path.join(tdir, "etnaviv_ml.c"),
             os.path.join(tdir, "etnaviv_ml_nn.c")]
    for f in files:
        if not os.path.isfile(f):
            print("  [SKIP] 找不到真实源码 %s" % f)
            return True

    tmpdir = tempfile.mkdtemp(prefix="och-inc-")
    try:
        # ---- 正控制 1/2：两个文件各自被改动 ----
        for real in files:
            dst = os.path.join(tmpdir, os.path.basename(real))
            shutil.copyfile(real, dst)
            raw = io.open(real, encoding="utf-8").read()

            if apply(dst) != 0:
                print("  [BAD] 正控制：应用到 %s 失败" % os.path.basename(real))
                ok = False
                continue
            out = io.open(dst, encoding="utf-8").read()

            for cond, desc in [
                (MARKER in out,
                 "标记已写入: %s" % os.path.basename(real)),
                (out.count("#include <stdlib.h>") == 1,
                 "新增恰好 1 个 <stdlib.h>: %s" % os.path.basename(real)),
                (out.count(MARKER) == 1,
                 "标记只出现一次: %s" % os.path.basename(real)),
                # 除插入的两三行外，原文件所有行都还在（不改逻辑）
                (len(out.splitlines()) == len(raw.splitlines()) + (1 if "etnaviv_ml.c" == os.path.basename(real) else 2),
                 "行数只增加插入的 include 行: %s" % os.path.basename(real)),
            ]:
                print("  [%s] 正控制：%s" % ("ok" if cond else "BAD", desc))
                ok = ok and cond

            # ---- 负控制 1：重复应用不产生重复行 ----
            before = out.count("#include <stdlib.h>")
            apply(dst)
            after = io.open(dst, encoding="utf-8").read().count("#include <stdlib.h>")
            cond = (before == after == 1)
            print("  [%s] 负控制 1：重复应用不重复插入: %s"
                  % ("ok" if cond else "BAD", os.path.basename(real)))
            ok = ok and cond

        # ---- 负控制 2：认不出的文件必须报错 ----
        bad = os.path.join(tmpdir, "unrelated.c")
        io.open(bad, "w", encoding="utf-8").write("int main(void) { return 0; }\n")
        cond = (apply(bad) == 1)
        print("  [%s] 负控制 2：认不出的文件报错（非静默跳过）"
              % ("ok" if cond else "BAD"))
        ok = ok and cond

        # ---- 负控制 3：n.c 的改动不得触碰函数体（只改文件头） ----
        nn = os.path.join(tmpdir, "etnaviv_ml_nn.c")
        if os.path.isfile(nn):
            out = io.open(nn, encoding="utf-8").read()
            head, _, body = out.partition('#include "etnaviv_ml.h"')
            cond = (MARKER in head and MARKER not in body)
            print("  [%s] 负控制 3：插入只发生在文件头 include 区"
                  % ("ok" if cond else "BAD"))
            ok = ok and cond
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    mesa = os.environ.get("OCH_MESA_SRC",
                          os.path.join(here, "..", "third_party", "mesa-src-26.2.5"))
    if len(argv) == 2 and argv[1] == "--selftest":
        ok = selftest(mesa)
        print("自测%s" % ("通过" if ok else "失败"))
        return 0 if ok else 1
    if len(argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for p in argv[1:]:
        rc |= apply(p)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
