#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen-force-partial-patch.py —— 从**真实 Mesa 源码**逐字生成 force-partial 补丁的锚点

项目规矩（见 build-recipe/patch-etnaviv-*.py 的注释）：锚点必须由生成器从真实源码提取，
**不许手打**（手打过一次，缩进 3/5/6 空格全试过，全部 count==0 ⇒ 门 fail-closed 报错）。

用法:
    python3 gen-force-partial-patch.py <etnaviv_ml_nn.c> <补丁器输出路径>

产出：把生成好的 OLD/NEW 锚点写进补丁器（替换其中的锚点常量块）。
"""
import re
import sys

# 目标：create_nn_config() 里 FULL/PARTIAL 的判据那一行
PAT = re.compile(r'^(\s*)if \(map->kernel_cache_start_address \+ coef_cache_size \+ image_cache_size < oc_sram_size\) \{$')
COMMENT = "Get all the image tiles in the cache, then use the rest for the kernels"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src_path, patcher_path = sys.argv[1], sys.argv[2]
    src = open(src_path, encoding="utf-8").read().split("\n")

    # 找到注释行与紧随其后的判据行
    ci = None
    for i, l in enumerate(src):
        if COMMENT in l:
            ci = i
            break
    if ci is None:
        print("ERROR: 找不到注释 %r" % COMMENT)
        return 1
    mi = None
    for i in range(ci + 1, min(ci + 6, len(src))):
        m = PAT.match(src[i])
        if m:
            mi = i
            indent = m.group(1)
            break
    if mi is None:
        print("ERROR: 注释之后没找到 FULL/PARTIAL 判据行")
        return 1

    old = "\n".join(src[ci:mi + 1]) + "\n"
    new = (
        src[ci] + "\n"
        + indent + "/* och patch: 强制 PARTIAL cache 分支（运行期开关，默认不变） */\n"
        + indent + "{\n"
        + indent + "   const char *fp = getenv(\"TEFLON_PARTIAL\");\n"
        + indent + "   int force_partial = (fp && fp[0] == '1');\n"
        + indent + "   unsigned long long total_needed =\n"
        + indent + "      (unsigned long long)map->kernel_cache_start_address + coef_cache_size + image_cache_size;\n"
        + indent + "   if (force_partial)\n"
        + indent + "      fprintf(stderr,\n"
        + indent + "              \"[och-fp] 强制 PARTIAL: total_needed=%llu oc_sram=%u coef=%u image=%u\\n\",\n"
        + indent + "              total_needed, oc_sram_size, coef_cache_size, image_cache_size);\n"
        + indent + "}\n"
        + indent + "if (map->kernel_cache_start_address + coef_cache_size + image_cache_size < oc_sram_size\n"
        + indent + "    && !(getenv(\"TEFLON_PARTIAL\") && getenv(\"TEFLON_PARTIAL\")[0] == '1')) {\n"
    )

    # 替换补丁器里的 OLD/NEW 块
    p = open(patcher_path, encoding="utf-8").read()
    p = re.sub(r'OLD = """.*?"""\n', 'OLD = ' + repr(old).replace("'", '"', 0) + '\n', p, count=1, flags=re.S) \
        if False else p
    # 用更稳的方式：按 OLD = """ 到 """ 的边界替换
    m_old = re.search(r'OLD = """.*?"""\n', p, re.S)
    m_new = re.search(r'NEW = """.*?"""\n', p, re.S)
    if not m_old or not m_new:
        print("ERROR: 补丁器里找不到 OLD/NEW 锚点块")
        return 1
    p = p[:m_old.start()] + 'OLD = """' + old + '"""\n' + p[m_old.end():]
    m_new = re.search(r'NEW = """.*?"""\n', p, re.S)
    p = p[:m_new.start()] + 'NEW = """' + new + '"""\n' + p[m_new.end():]
    open(patcher_path, "w", encoding="utf-8").write(p)

    print("  已把逐字锚点写入 %s" % patcher_path)
    print("  锚点起始行: %d（缩进 %d 空格）" % (ci + 1, len(indent)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
