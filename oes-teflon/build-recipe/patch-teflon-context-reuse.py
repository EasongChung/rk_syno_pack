#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 teflon 胶水层加「pipe_context 跨分区复用」（消除每分区的 create+destroy 开销）。

## 为什么需要（报告 60 的性能判决）

真机实测（2026-10-22，concept 模型整图，218 op）：

| 口径 | 耗时 |
|---|---|
| CPU（不挂 delegate） | **293.15 ms** |
| NPU（挂 delegate，生产路径口径） | **698.89 ms** ⇒ **慢 2.38×** |
| 其中：NN 作业 GPU 时间合计 | 197 ms（32 个作业，调试口径实测） |
| 其中：CPU 侧算子（187 个） | ≈193 ms（由 CPU 整图耗时反推） |
| **差额 ≈ 300 ms** | **= 委托税（delegate 分区固定开销）** |

## 税的来源（本补丁要消除的那一条）

`src/gallium/frontends/teflon/tfl_device.c` 的 `partition_invoke()`：

```c
struct pipe_context *context =
   delegate->screen->context_create(delegate->screen, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
...
context->destroy(context);
```

⇒ **每个被接管的算子（= 每个 delegate 分区）每次推理都新建 + 销毁一个 pipe_context**。
`etna_context_create()`（`etnaviv_context.c:730`）会做：
`u_upload_create_default()`（**默认上传器 = 一块 staging 缓冲**）+ `etna_cmd_stream_new()`
+ 两个哈希表 ⇒ **每次都是一份完整的 GPU 侧资源**。

⇒ 我们自己在 `build-recipe/patch-teflon-conv-only.py` 的文档里已经量过这个固定开销：
**单分区 6–31 ms（均值 57 ms，最大 1005 ms）**。
⇒ 14 个分区 × ~20 ms ≈ **280 ms** —— 与上面反推的 ≈300 ms 吻合。

## 本补丁做三件事（都是**行为等价**，不改数值）

1. `struct teflon_delegate` 增加 `struct pipe_context *compute_ctx;`
2. `partition_invoke()`：**首次使用时创建，之后所有分区复用**同一个 context
   ⇒ 既省掉 create/destroy，也顺带让所有分区共用一条命令流（flush 次数 14 → 1）
3. `tflite_plugin_destroy_delegate()`：销毁时 `context->destroy(context)` ⇒ **不泄漏**

## 为什么不写成「编译期开关」

这个补丁**不改变任何数值结果**（同样的算子、同样的顺序、同样的系数），
只改资源生命周期 ⇒ 做成**恒生效**的修复，而不是开关：
开关会让「哪个产物更快」这件事依赖环境变量，而交付件必须**不设任何环境变量**就对。

## 风险与边界

* 复用 ⇒ 所有分区共用一条命令流 ⇒ 必须确认 `etna_context_force_flush` 在
  `ml_subgraph_invoke` 内部已有 flush（`etnaviv_ml.c` 的 `close_batch` + `flush`），
  否则跨分区的依赖会读到未完成的数据 ⇒ **本补丁在真机验证时会用「逐字节比对」判定**。
* 一个 context 跨多个 `ml_subgraph_invoke` ⇒ 其 `pending_resources` 会累积
  （在同一个 interpreter 生命周期内）⇒ 内存占用上升，但**有界**（原来是无界的重复分配）。

## 适用性

针对 mesa-26.2.5 的 `tfl_device.c`（`struct teflon_delegate` / `partition_invoke` /
`tflite_plugin_destroy_delegate`）。若上游结构不同，脚本**断言失败并中止**（不静默跳过）。

## 用法

    python3 patch-teflon-context-reuse.py <tfl_device.c 路径> [--selftest]
"""
import argparse
import os
import shutil
import sys
import tempfile

MARK = "och patch: ctx-reuse"

# ── [1] delegate 结构体：加一个缓存的 context ───────────────────────────
OLD_STRUCT = """\
struct teflon_delegate {
   TfLiteDelegate base;
   struct pipe_loader_device *dev;
   struct pipe_screen *screen;
   struct pipe_ml_device *ml_dev;
   struct pipe_tensor *tensors;
   unsigned tensor_count;
};"""
NEW_STRUCT = """\
struct teflon_delegate {
   TfLiteDelegate base;
   struct pipe_loader_device *dev;
   struct pipe_screen *screen;
   struct pipe_ml_device *ml_dev;
   struct pipe_tensor *tensors;
   unsigned tensor_count;
   /* och patch: ctx-reuse —— 跨分区复用的 compute context（见 patch 文档） */
   struct pipe_context *compute_ctx;
};"""

# ── [2] partition_invoke：创建一次，之后复用 ───────────────────────────
OLD_CREATE = """\
   struct pipe_context *context = delegate->screen->context_create(delegate->screen, NULL, PIPE_CONTEXT_COMPUTE_ONLY);"""
NEW_CREATE = """\
   /* och patch: ctx-reuse —— 每个分区都 create+destroy 一个 pipe_context 代价很大
    * （etna_context_create 会建默认上传器 staging 缓冲 + 命令流 + 哈希表；
    *   本项目实测单分区固定开销 6–31 ms）⇒ 改为**整个 delegate 只建一次**、跨分区复用。
    *   数值行为不变：算子集合、顺序、系数都一致。 */
   struct pipe_context *context = delegate->compute_ctx;
   if (context == NULL) {
      context = delegate->screen->context_create(delegate->screen, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
      if (context == NULL) {
         fprintf(stderr, "teflon: context_create failed\\n");
         return kTfLiteError;
      }
      delegate->compute_ctx = context;
   }"""

# ── [3] partition_invoke 尾部：不再在这里 destroy（改成 delegate 销毁时统一做）──
OLD_DESTROY = """\
   context->destroy(context);
   context = NULL;

   return kTfLiteOk;
}"""
NEW_DESTROY = """\
   /* och patch: ctx-reuse —— context 由 delegate 持有，**这里不销毁**
    * （销毁统一放在 tflite_plugin_destroy_delegate）⇒ 跨分区复用。 */
   context = NULL;

   return kTfLiteOk;
}"""

# ── [4] delegate 销毁：统一释放缓存的 context ──────────────────────────
OLD_FREE = """\
   free(delegate->tensors);

   delegate->screen->destroy(delegate->screen);"""
NEW_FREE = """\
   free(delegate->tensors);

   /* och patch: ctx-reuse —— 释放复用的 compute context（不释放就是泄漏） */
   if (delegate->compute_ctx) {
      delegate->compute_ctx->destroy(delegate->compute_ctx);
      delegate->compute_ctx = NULL;
   }

   delegate->screen->destroy(delegate->screen);"""

EDITS = [
    ("delegate 结构体加 compute_ctx", OLD_STRUCT, NEW_STRUCT),
    ("partition_invoke 复用 context", OLD_CREATE, NEW_CREATE),
    ("partition_invoke 不再 destroy", OLD_DESTROY, NEW_DESTROY),
    ("delegate 销毁时释放 context", OLD_FREE, NEW_FREE),
]


def apply(text, path=""):
    if MARK in text:
        raise SystemExit("ERROR: %s 已包含本补丁，拒绝重复应用。" % path)
    for name, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("ERROR: 在 %s 中匹配到 %d 处锚点「%s」，期望恰好 1 处。"
                             "（Mesa 版本结构不同 ⇒ 拒绝打补丁，避免半改的产物）"
                             % (path, n, name))
        text = text.replace(old, new)
    return text


def selftest(pristine):
    """正控制 + 负控制 + 幂等性（项目纪律：检查器必须带正/负控制）。"""
    fails = []

    # 正控制：应用成功，且 4 处改动都在
    try:
        out = apply(open(pristine, encoding="utf-8").read(), pristine)
    except SystemExit as e:
        fails.append("正控制：应用失败 %s" % e)
        out = ""
    if out:
        if MARK not in out:
            fails.append("正控制：缺少标记")
        if "compute_ctx" not in out:
            fails.append("正控制：没有 compute_ctx")
        if out.count("context_create(") != 1:
            fails.append("正控制：context_create 次数 = %d（期望 1）"
                         % out.count("context_create("))
        if "context->destroy(context);" in out:
            fails.append("正控制：partition_invoke 里的 context->destroy(context) 还在")
        if "delegate->compute_ctx->destroy(delegate->compute_ctx);" not in out:
            fails.append("正控制：delegate 销毁处没有释放 compute_ctx")
        if out.count("context_create(") != 1:
            fails.append("正控制：context_create 次数 = %d（期望 1）" % out.count("context_create("))
        # 幂等：再打一次必须被拒绝
        try:
            apply(out, "二次应用")
            fails.append("负控制：二次应用未被拒绝")
        except SystemExit:
            pass
        # 负控制：结构不同的文件必须被拒绝（不能半改）
        broken = out.replace("   struct pipe_context *compute_ctx;\n", "")
        try:
            apply(broken, "结构被改")
            fails.append("负控制：结构不匹配时未被拒绝")
        except SystemExit:
            pass
        # 负控制：空文件
        try:
            apply("", "空文件")
            fails.append("负控制：空文件未被拒绝")
        except SystemExit:
            pass

    print("== 4 项检查，%d 项失败 ==" % len(fails))
    for f in fails:
        print("   " + f)
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--selftest", action="store_true",
                    help="对 pristine 源做正/负控制自测（不写文件）")
    args = ap.parse_args(argv[1:])

    if args.selftest:
        return selftest(args.path)

    if not os.path.exists(args.path):
        raise SystemExit("ERROR: 文件不存在：%s" % args.path)
    with open(args.path, encoding="utf-8") as f:
        text = f.read()
    new = apply(text, args.path)
    with open(args.path, "w", encoding="utf-8") as f:
        f.write(new)
    print("  [ctx-reuse] 已打补丁：pipe_context 跨分区复用（4 处改动）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
