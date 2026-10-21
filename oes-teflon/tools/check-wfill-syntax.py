#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从打过补丁的真实源码里抽出「写端填充探针」代码块，套桩编译，做语法/类型检查。

桩只覆盖该代码块用到的符号（util_dynarray_*、pipe_buffer_*、etna_bo_gpu_va、
etna_buffer_resource、ML_DBG、PIPE_MAP_WRITE），足够抓出：
  少分号、括号不配平、格式串与参数个数不符、类型不匹配（-Werror=format）。
不覆盖：真实头文件里的宏展开差异（那由 CI 承担）。
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIXTURE = os.path.join(ROOT, "build-recipe", "fixtures", "etnaviv_ml.ref.c")
PATCH = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-diag.py")
ROFIX = os.path.join(ROOT, "build-recipe", "patch-etnaviv-readout-offset-fix.py")

STUB = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct pipe_resource { unsigned size; } pipe_resource;
typedef struct pipe_transfer pipe_transfer;
typedef struct etna_bo etna_bo;
typedef struct etna_buffer_resource { etna_bo *bo; } etna_buffer_resource;

struct etna_ml_tensor {
   pipe_resource *resource;
   unsigned offset;
   unsigned size;
};

/* util_dynarray 桩：签名与 Mesa 一致（第二个参数是元素类型） */
#define util_dynarray_num_elements(arr, type) ((arr)->n)
#define util_dynarray_element(arr, type, idx) ((type *)((arr)->data + (idx)))

struct etna_ml_subgraph {
   struct { struct etna_ml_tensor **data; unsigned n; } tensors;
};

#define PIPE_MAP_READ  (1u << 1)
#define PIPE_MAP_WRITE (1u << 2)
#define PIPE_MAP_UNSYNCHRONIZED (1u << 9)
#define ML_DBG(...) printf(__VA_ARGS__)

static pipe_transfer *pipe_buffer_map_range(void *ctx, pipe_resource *res,
                                            unsigned offset, unsigned length,
                                            unsigned usage, pipe_transfer **out)
{
   (void)ctx; (void)offset; (void)length; (void)usage;
   *out = (pipe_transfer *)res;
   return *out;
}
static unsigned pipe_buffer_size(pipe_resource *res) { return res->size; }
static void pipe_buffer_unmap(void *ctx, pipe_transfer *tr) { (void)ctx; (void)tr; }
static etna_buffer_resource *etna_buffer_resource_of(pipe_resource *res) { (void)res; return NULL; }
static uint32_t etna_bo_gpu_va(etna_bo *bo) { (void)bo; return 0; }
#define etna_buffer_resource(res) etna_buffer_resource_of(res)

/* —— 下面是补丁原样注入的代码块 —— */
int probe(void *context, struct etna_ml_subgraph *subgraph,
          unsigned outputs_count, unsigned output_idxs[], void *outputs[],
          unsigned operation_count)
{
__BLOCK__
   return 0;
}
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=PATCH, help="补丁路径（负控制用：指向被改坏的副本）")
    ap.add_argument("--fixture", default=FIXTURE)
    args = ap.parse_args()

    if subprocess.run(["sh", "-c", "command -v gcc >/dev/null 2>&1"]).returncode != 0:
        sys.stderr.write("SKIP: 没有 gcc，无法做语法检查（不算通过）\n")
        return 2

    tmp = tempfile.mkdtemp(prefix="wfill-syntax-")
    src = os.path.join(tmp, "etnaviv_ml.c")
    with open(args.fixture, encoding="utf-8") as f:
        text = f.read()
    with open(src, "w", encoding="utf-8") as f:
        f.write(text)

    if subprocess.run([sys.executable, ROFIX, src], capture_output=True).returncode != 0:
        sys.stderr.write("ERROR: ROFIX 应用失败\n")
        return 2
    r = subprocess.run([sys.executable, args.patch, src], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("ERROR: RODIAG 应用失败\n" + r.stderr)
        return 2

    with open(src, encoding="utf-8") as f:
        patched = f.read()

    at = patched.find("och patch: 写端填充探针")
    if at < 0:
        sys.stderr.write("ERROR: 打完后找不到探针代码块\n")
        return 2
    # 从注释开头回溯到块首的 `   {`，截到 `operation_count > 0) {` 之前
    end = patched.find("   if (operation_count > 0) {", at)
    if end < 0:
        sys.stderr.write("ERROR: 找不到探针块尾部锚点\n")
        return 2
    blk = patched[at:end]
    blk = blk[blk.find("{\n"):]          # 丢掉注释，保留 `{ ... }`
    blk = blk.rstrip()
    if not blk.endswith("}"):
        sys.stderr.write("ERROR: 探针块不以 } 结尾\n")
        return 2

    harness = os.path.join(tmp, "h.c")
    with open(harness, "w", encoding="utf-8") as f:
        f.write(STUB.replace("__BLOCK__", blk))

    cmd = ["gcc", "-std=gnu11", "-Wall", "-Wextra", "-Werror=format", "-Wno-unused-parameter",
           "-fsyntax-only", harness]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("ERROR: 探针代码块编译失败\n" + r.stderr[:2000])
        return 1
    print("  [ok] 写端探针代码块通过 gcc 语法/格式检查（%d 行）" % blk.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
