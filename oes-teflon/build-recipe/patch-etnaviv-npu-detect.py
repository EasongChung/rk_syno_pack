#!/usr/bin/env python3
"""修 Mesa etnaviv 的 NPU 识别：绕过 drm_version 1.4 的门槛判断。

## 为什么需要这个补丁

`libteflon.so` 在真机上段错误，崩溃栈：

    etna_ml_subgraph_create+0x44
    libteflon.so(+0x83c6c)                  <- teflon 的子图 kernel init
    tflite::Subgraph::AddNodeWithParameters
    ...

崩在单条指令上（已反汇编核实）：

    1f52a0: bl  <etna_ml_get_core_info>
    1f52a4: ldr w0, [x0]        <- 解引用返回值，崩在这里

`etnaviv_ml.c`：

    etna_ml_get_core_info(struct etna_context *context) {
       struct etna_screen *screen = context->screen;
       struct etna_core_info *info = etna_gpu_get_core_info(screen->npu);
       return &info->npu;             /* info == NULL 时不检查，取地址得野指针 */
    }

即 `screen->npu == NULL`。

## 根因链

1. `screen_create()`（winsys/etnaviv/drm/etnaviv_drm_winsys.c）只在
   `info->type == ETNA_CORE_NPU` 时才给 `npu` 赋值。
2. `info->type` 的**唯一** NPU 赋值点在 `etna_query_feature_db()`
   （src/etnaviv/hwdb/etna_hwdb.c）：`if (db->NNCoreCount) info->type = ETNA_CORE_NPU;`
3. 而 `etna_query_feature_db()` 只在 **`drm_version >= 1.4`** 时才被调用
   （src/etnaviv/drm/etnaviv_gpu.c `etna_gpu_new()`）：

       if (dev->drm_version >= ETNA_DRM_VERSION(1, 4)) {
          gpu->info.product_id  = get_param(dev, core, ETNAVIV_PARAM_GPU_PRODUCT_ID);
          gpu->info.customer_id = get_param(dev, core, ETNAVIV_PARAM_GPU_CUSTOMER_ID);
          gpu->info.eco_id      = get_param(dev, core, ETNAVIV_PARAM_GPU_ECO_ID);
          core_info_okay = etna_query_feature_db(&gpu->info);
       }
       if (!core_info_okay) {
          query_features_from_kernel(gpu);   /* <- 把 type 写死 ETNA_CORE_GPU */
          query_limits_from_kernel(gpu);
       }

4. 本设备内核（DSM 7.4.1 / A311D）报告：

       [drm] Initialized etnaviv 1.3.0 20151214

   `ETNA_DRM_VERSION(1,3) = 0x10003` < `ETNA_DRM_VERSION(1,4) = 0x10004`
   → **hwdb 查找被整个跳过** → `type` 恒为 `ETNA_CORE_GPU`
   → `screen->npu` 恒为 NULL → `etna_ml_get_core_info()` 空指针 → 段错误。

## 这不是内核的问题

该内核**实际支持**这三个 param（`drivers/gpu/drm/etnaviv/etnaviv_gpu.c`
L152/156/160 分别处理 PRODUCT_ID/CUSTOMER_ID/ECO_ID），
编号 `0x1c/0x1d/0x1e` 与 Mesa 的 `enum etna_param_id` **完全一致**；
hwdb 里也有完全匹配的条目（`amlogic/gc_feature_database.h`，
ChipID `0x8000` / ChipRevision `0x7120` / ProductID `0x45080009` /
CustomerID `0x88`，`NNCoreCount = 8`）。

**内核只是没把 DRM 版本号报成 1.4。** Mesa 拿版本号当「内核有没有这些 param」
的代理，**在这个内核上代理判断错了**。

## 修法（保守）

保留版本号判断作为快速路径；版本号不够时**实测一次** PRODUCT_ID：

- 读得到（非 0）→ 内核确实支持这三个 param → 照常走 hwdb ← 本设备
- 读不到（`get_param` 返回 0）→ 内核真不支持 → 维持原行为，回落 kernel 查询

即：**只在证据充分时才改变行为**，不会让旧内核走到不存在的 param 上。

## 适用性

`mesa-24.3.0` 与 `mesa-25.0.7` 的这段代码**逐字节完全一致**（已 diff 核实），
同一份改法同时适用两者。若将来 Mesa 改动此处，脚本会**断言失败并中止构建**，
而不是静默产出一个没打上补丁的 .so。
"""

import re
import sys

# 目标片段：etna_gpu_new() 里的 drm_version 门槛。
# 用「门槛 + 紧随其后的 product_id 读取」两行共同定位，避免将来误伤同名的其他判断。
OLD = (
    "\tif (dev->drm_version >= ETNA_DRM_VERSION(1, 4)) {\n"
    "\t\tgpu->info.product_id = get_param(dev, core, ETNAVIV_PARAM_GPU_PRODUCT_ID);\n"
)

NEW = """\
\t/* ---- och 补丁：本内核 DRM 版本报 1.3.0，但实际支持这三个 param ----------
\t * Mesa 用版本号作为「内核有没有 PRODUCT_ID/CUSTOMER_ID/ECO_ID」的代理，
\t * 本内核上该代理判断错误：版本号 1.3.0，param 0x1c/0x1d/0x1e 却可用。
\t * 后果：hwdb 查找被跳过 → info->type 恒为 ETNA_CORE_GPU
\t *       → screen->npu 恒为 NULL → teflon 的 etna_ml_get_core_info()
\t *       空指针解引用 → 真机段错误（已用崩溃栈 + 反汇编定位到单条指令）。
\t *
\t * 保守修法：版本号不够时**实测一次** PRODUCT_ID。
\t *   读得到 → 内核确实支持，照常走 hwdb（本设备走这条）
\t *   读不到 → 内核真不支持，维持原行为回落 kernel 查询
\t * get_param() 在失败时返回 0，所以这个探测不会误判。
\t * ------------------------------------------------------------------------ */
\tif (dev->drm_version < ETNA_DRM_VERSION(1, 4)) {
\t\tDEBUG_MSG(" och patch: drm_version < 1.4, probing PRODUCT_ID param\\n");
\t\tif (get_param(dev, core, ETNAVIV_PARAM_GPU_PRODUCT_ID) != 0) {
\t\t\tDEBUG_MSG(" och patch: kernel supports PRODUCT_ID despite old drm_version\\n");
\t\t\tdev->drm_version = ETNA_DRM_VERSION(1, 4);
\t\t}
\t}

"""


def main(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # 防重入：补丁已打过就必须拒绝。
    # 否则第二次运行会匹配到**补丁自己插入的**那个 `if`，把一个探测块
    # 嵌进另一个探测块里 —— 实测确实会重复命中（当时靠人工看 diff 才发现）。
    MARKER = "och patch: kernel supports PRODUCT_ID"
    if MARKER in src:
        sys.stderr.write(
            f"ERROR: {path} 已包含本补丁（找到标记 \"{MARKER}\"），拒绝重复应用。\n"
        )
        return 1

    # 只匹配「紧接着 product_id 读取」的那一处，避免将来误伤别的同名判断
    OLD = (
        "\tif (dev->drm_version >= ETNA_DRM_VERSION(1, 4)) {\n"
        "\t\tgpu->info.product_id = get_param(dev, core, ETNAVIV_PARAM_GPU_PRODUCT_ID);\n"
    )
    n = src.count(OLD)
    if n != 1:
        sys.stderr.write(
            f"ERROR: 在 {path} 中匹配到 {n} 处 drm_version 门槛（含紧随的 product_id 读取），\n"
            "       期望恰好 1 处。Mesa 可能已改动此处 —— 请人工核对后再更新本补丁，\n"
            "       不要跳过校验直接构建（会静默产出未打补丁的 .so）。\n"
        )
        return 1

    src = src.replace(OLD, NEW + OLD)

    # 自检：补丁必须真的落在 etna_gpu_new() 里，且 NPU 判定入口仍在
    if "core_info_okay = etna_query_feature_db(&gpu->info);" not in src:
        sys.stderr.write("ERROR: 打完补丁后找不到 etna_query_feature_db 调用，异常\n")
        return 1
    if "och patch: kernel supports PRODUCT_ID" not in src:
        sys.stderr.write("ERROR: 补丁内容未写入\n")
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"  [och patch] 已修 {path}")
    print("  [och patch] drm_version < 1.4 时将实测 PRODUCT_ID，能读到则照常走 hwdb")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(f"用法: {sys.argv[0]} <etnaviv_gpu.c 路径>\n")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
