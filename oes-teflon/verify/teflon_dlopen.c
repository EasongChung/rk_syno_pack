/*
 * teflon_dlopen.c —— libteflon.so 上机加载测试
 *
 * 用途：验证 libteflon.so 能否在设备上成功 dlopen 并枚举到 NPU。
 *       这是"静态检查都通过"到"真的能用"之间的最后一道关。
 *
 * 本文件需**交叉编译后**放到设备上运行（设备 DSM 无 gcc）：
 *   aarch64-linux-gnu-gcc -o teflon_dlopen teflon_dlopen.c -ldl
 *   # 或在 och 的构建机上：
 *   gcc --sysroot=<sr> -B <sr>/usr/lib/aarch64-linux-gnu \
 *       -o teflon_dlopen teflon_dlopen.c -ldl
 *
 * 用法：./teflon_dlopen /path/to/libteflon.so
 *
 * 退出码：
 *   0 = dlopen 成功、符号齐全、delegate 可创建
 *   1 = dlopen 失败
 *   2 = 符号缺失
 *   3 = delegate 创建失败
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* TFLite external delegate ABI（见 tfl_device.c L1631/L1682） */
typedef void *TfLiteDelegate;
typedef TfLiteDelegate *(*create_fn)(char **options_keys, char **options_values);
typedef void (*destroy_fn)(TfLiteDelegate *delegate);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "用法: %s <libteflon.so 路径>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    printf("=== teflon dlopen 测试 ===\n");
    printf("目标: %s\n\n", path);

    /* 1) dlopen */
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("[FAIL] dlopen 失败: %s\n", dlerror());
        printf("\n排查：\n");
        printf("  - libdrm.so.2 是否与本文件同目录？（DSM 系统级没有 libdrm）\n");
        printf("  - readelf -d %s | grep RUNPATH  是否含 $ORIGIN\n", path);
        printf("  - readelf -d %s | grep NEEDED    列出的库是否都存在\n", path);
        return 1;
    }
    printf("[ OK ] dlopen 成功\n");

    /* 2) 导出符号 */
    create_fn  create  = (create_fn)  dlsym(h, "tflite_plugin_create_delegate");
    destroy_fn destroy = (destroy_fn)dlsym(h, "tflite_plugin_destroy_delegate");

    if (!create) {
        printf("[FAIL] 缺少符号 tflite_plugin_create_delegate\n");
        printf("        说明这不是 teflon delegate 或构建不完整\n");
        return 2;
    }
    printf("[ OK ] tflite_plugin_create_delegate 存在\n");

    if (!destroy) {
        printf("[WARN] 缺少 tflite_plugin_destroy_delegate（可选，影响资源回收）\n");
    } else {
        printf("[ OK ] tflite_plugin_destroy_delegate 存在\n");
    }

    /* 3) 真正调用 create —— 这一步会触发 etnaviv 枚举与
     *    etna_ml_get_core_info() 读取 nn_core_count。
     *    若 nn_core_count < 1，etnaviv 会直接 abort()（不是返回错误），
     *    所以这里若进程异常终止，说明 NPU 没被正确识别。 */
    printf("\n调用 tflite_plugin_create_delegate()...\n");
    fflush(stdout);

    TfLiteDelegate *d = create(NULL, NULL);

    if (!d) {
        printf("[WARN] create_delegate 返回 NULL\n");
        printf("       可能原因：\n");
        printf("       - 未指定 TFLITE delegate 选项（正常应传 options）\n");
        printf("       - 无 /dev/dri/renderD129，或节点权限不足\n");
        printf("       - hwdb 里 nn_core_count = 0（NPU 不可用）\n");
        return 3;
    }

    printf("[ OK ] create_delegate 返回非 NULL (%p)\n", (void *)d);
    printf("       delegate 已成功枚举到 ML 设备（NPU 可用）\n");

    if (destroy)
        destroy(d);

    printf("\n结果: 全部通过 —— libteflon.so 可在设备上加载并枚举 NPU\n");
    return 0;
}