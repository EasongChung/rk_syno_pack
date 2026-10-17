#!/usr/bin/env python3
"""给 teflon 加「只接受卷积」的运行时开关（TEFLON_CONV_ONLY=1）。

## 为什么需要

真机实测（2026-10-17，A311D + DSM 7.4.1）：

| 路径 | 耗时 | 正确性 |
|---|---|---|
| CPU（不挂 delegate） | **302 ms** | 基准 |
| mesa-25.0.7 + drm_version 补丁（19 算子 / 18 分区） | **1128 ms** | 输出错误 |
| mesa-26.2.5 + drm_version 补丁（99 算子 / 48 分区） | SIGBUS | — |

**慢的根因在 Mesa 的实现里**（`tfl_device.c` `partition_invoke`）：

    struct pipe_context *context =
       delegate->screen->context_create(delegate->screen, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
    ...  ml_subgraph_invoke / ml_subgraph_read_output  ...
    context->destroy(context);

**每次调用、每个分区都创建并销毁一次 pipe_context。**
teflon 自己的计时行实测：单分区固定开销 **6–31 ms**（均值 57 ms，最大 1005 ms）。
→ 18 个分区 × ~20 ms ≈ 360 ms，**光固定开销就超过 CPU 全部 302 ms**。

## 而当前被接管的算子恰好是「最不该上 NPU」的那批

26.2.5 上被接管的 99 个是：

    TRANSPOSE 47 / RESHAPE 32 / CONCAT 16 / PAD 3 / LOG 1
    CONV 0 / DWCONV 0

**全是布局/内存操作** —— 放 NPU 只带来数据搬运与分区固定开销，**不省计算**。

## 所以本补丁做一件事

让 delegate **只接受 Conv2d / DepthwiseConv2d**：

- 布局操作留 CPU（它们本来就便宜）
- 分区数大幅下降 → 固定开销随之下降
- 换来的是**真实计算**（卷积）上 NPU

## 为什么做成运行时开关而不是编译期分支

用**同一个 .so** 就能做 A/B：不设环境变量 = 原行为，设 `=1` = 只收卷积。
这样两次真机测量之间**唯一的变量就是这个开关**，
不会混入「不同构建」的其它差异。实验结束后把这个补丁去掉即可。

## 适用性

针对 `mesa-26.2.5` 的 `check_op_support()`（它把判定委托给
`ml_operation_supported`）。若 Mesa 改动此处，脚本**断言失败并中止构建**。
"""

import sys

# 在 check_op_support 的变量声明之后插入开关判断。
# 用「声明块 + 第一个 if」两行共同定位，避免误伤同名结构。
OLD = """\
   struct teflon_delegate *delegate = (struct teflon_delegate *)tf_delegate;
   struct pipe_ml_operation operation = {0};
   bool supported = false;

   if (!fill_operation(delegate, tf_context, node, registration, &operation))
      return false;
"""

NEW = """\
   struct teflon_delegate *delegate = (struct teflon_delegate *)tf_delegate;
   struct pipe_ml_operation operation = {0};
   bool supported = false;

   /* ---- och patch: TEFLON_CONV_ONLY -------------------------------------
    * 运行时开关「只接受卷积」。
    *
    * 真机实测：本 delegate 接管的 99 个算子全是 TRANSPOSE/RESHAPE/CONCAT/PAD
    * （布局操作，CONV/DWCONV 为 0），而每个分区每次 invoke 都要
    * context_create/destroy，固定开销 6–31 ms → 18 个分区就超过 CPU 全部耗时。
    *
    * 置 TEFLON_CONV_ONLY=1 后只接受 Conv2d / DepthwiseConv2d：
    * 布局操作留 CPU（本就便宜），分区数下降，固定开销随之下降，
    * 换来的是真实计算上 NPU。
    *
    * 「已读环境变量」缓存成 static，避免每次判定都 getenv（判定在
    * AllocateTensors 期间对每个节点调用一次）。
    * ------------------------------------------------------------------- */
   {
      static int conv_only = -1;
      if (conv_only < 0) {
         const char *e = getenv("TEFLON_CONV_ONLY");
         conv_only = (e != NULL && e[0] == '1') ? 1 : 0;
      }
      if (conv_only &&
          registration->builtin_code != kTfLiteBuiltinConv2d &&
          registration->builtin_code != kTfLiteBuiltinDepthwiseConv2d)
         return false;
   }

   if (!fill_operation(delegate, tf_context, node, registration, &operation))
      return false;
"""

MARKER = "och patch: TEFLON_CONV_ONLY"


def main(path: str, path2: str | None = None) -> int:
    # 目标文件因 Mesa 版本而异：26.2.5 起判定在 tfl_device.c；
    # 若拆分到别的文件，允许传第二个候选路径。
    candidates = [path] + ([path2] if path2 else [])
    chosen = None
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        if MARKER in src:
            sys.stderr.write(f"ERROR: {c} 已包含本补丁，拒绝重复应用。\n")
            return 1
        if src.count(OLD) == 1:
            chosen, chosen_src = c, src
            break

    if chosen is None:
        sys.stderr.write(
            f"ERROR: 在 {candidates} 中找不到 check_op_support 的目标片段。\n"
            "       Mesa 可能已改动此处 —— 请人工核对后再更新本补丁，\n"
            "       不要跳过校验直接构建（会静默产出未打补丁的 .so）。\n"
        )
        return 1

    src = chosen_src.replace(OLD, NEW)
    if MARKER not in src:
        sys.stderr.write("ERROR: 补丁内容未写入\n")
        return 1

    with open(chosen, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [och patch] 已修 {chosen}")
    print("  [och patch] TEFLON_CONV_ONLY=1 时只接受 Conv2d/DWConv2d（默认关闭，行为不变）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(f"用法: {sys.argv[0]} <tfl_device.c> [备选路径]\n")
        sys.exit(2)
    sys.exit(main(*sys.argv[1:3]))
