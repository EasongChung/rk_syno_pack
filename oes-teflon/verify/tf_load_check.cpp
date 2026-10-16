/* tf_load_check.cpp —— 上机验证 Teflon delegate 能否接管 .tflite 模型的节点
 *
 * ## 为什么需要这个程序
 *
 * 我们之前只验证过「delegate 能加载」：
 *   teflon_dlopen → dlopen libteflon.so → 找到 tflite_plugin_create_delegate
 *   → 调用它 → 返回非 NULL
 * 那只能证明 **库能加载、符号能解析、设备能被枚举**。
 *
 * 但 TFLite 源码 tfl_device.c 的 tflite_plugin_create_delegate()（L1687-1712）
 * 只做两件事：
 *   delegate->dev  = find_accel_device() / find_drm_device()
 *   delegate->base.Prepare = PrepareDelegate
 * 它**从不调用 PrepareDelegate**。
 *
 * PrepareDelegate 真正被回调是在 TFLite 解释器 AllocateTensors() 期间
 * （ReplaceNodeSubsetsWithDelegateKernels 那一步）。
 * 所以「能加载」与「能接管节点」是两件事，必须用解释器验证。
 *
 * ## 为什么用 C API 而不是 C++ API
 *
 * 前一版用 tflite::InterpreterOptions / tflite::TfLiteModel，
 * 在 TFLite 2.13 下编译失败：
 *     'TfLiteModel' is not a member of 'tflite'
 *     'class tflite::InterpreterOptions' has no member named 'AddDelegate'
 * C++ 封装层在不同版本间变动较大（2.13 与后续版本的
 * InterpreterOptions::AddDelegate 签名与可用性都不同）。
 *
 * C API（tensorflow/lite/c/c_api.h）是稳定 ABI，而且 **delegate 本身就是 C 接口**
 * （tflite_plugin_create_delegate 的签名就是 TfLiteDelegate*(*)(const char*)），
 * 用 C API 与它天然对齐。
 *
 * ## 本程序做什么 / 不做什么
 *
 * 做：
 *   1. 加载 .tflite 模型
 *   2. 建解释器，把 libteflon.so 的 external delegate 挂上去
 *   3. AllocateTensors()   ← 这一步触发 PrepareDelegate
 *   4. 打印张量统计与**分区数**
 *
 * 不做：
 *   - 不做正确的前处理/反量化
 *   - 不要求推理结果正确
 *   - 不 Invoke()
 *
 * 因为接管分区在 AllocateTensors 阶段就已完成，
 * 数值正确与否与「哪些节点被接管」是两件独立的事。
 *
 * ## 关键环境变量
 *
 *   TEFLON_DEBUG=2   打开 delegate 的逐节点日志（注意是 2 不是 1）
 *
 *   源码 tfl_device.c L31  TEFLON_DEBUG_VERBOSE = 1 << 1   → 值是 2
 *   源码 tfl_device.c L43  teflon_debug() 整个函数体被
 *                               & TEFLON_DEBUG_VERBOSE 门控
 *   设成 1 只置 bit0，1 & 2 == 0 → 一行日志都不打。
 *   而空日志极易被误读成「delegate 没接管任何节点」—— 方向正好反了。
 *
 * ## 退出码
 *
 *   0   成功，AllocateTensors 通过
 *   64  用法错误
 *   1   模型加载失败
 *   2   AllocateTensors 失败
 *   3   解释器创建失败
 */

#include <cstdio>
#include <cstdlib>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"

/* libteflon.so 导出的 external delegate 入口（Mesa teflon，ABI 见 tfl_device.c） */
extern "C" TfLiteDelegate* tflite_plugin_create_delegate(const char*);
extern "C" void tflite_plugin_destroy_delegate(TfLiteDelegate*);

/* tflite_type_name 在 teflon 内部是 file-static 短名版本（L1370），
 * 与公开枚举名不同。这里用公开枚举打印，便于阅读。 */
static const char* dtype_name(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "FLOAT32";
    case kTfLiteInt32:   return "INT32";
    case kTfLiteUInt8:   return "UINT8";
    case kTfLiteInt64:   return "INT64";
    case kTfLiteString:  return "STRING";
    case kTfLiteBool:    return "BOOL";
    case kTfLiteInt16:   return "INT16";
    case kTfLiteComplex64:  return "COMPLEX64";
    case kTfLiteInt8:    return "INT8";
    case kTfLiteFloat16: return "FLOAT16";
    case kTfLiteFloat64: return "FLOAT64";
    default:             return "OTHER";
  }
}

static void print_shape(const TfLiteTensor* t) {
  std::fprintf(stderr, "[");
  for (int d = 0; d < t->dims->size; d++) {
    std::fprintf(stderr, "%s%d", d ? "," : "", t->dims->data[d]);
  }
  std::fprintf(stderr, "]");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "用法: %s <model.tflite>\n", argv[0]);
    return 64;
  }
  const char* model_path = argv[1];

  std::fprintf(stderr, "=== tf_load_check ===\n");
  std::fprintf(stderr, "模型        : %s\n", model_path);
  std::fprintf(stderr, "TEFLON_DEBUG: %s\n",
               getenv("TEFLON_DEBUG") ? getenv("TEFLON_DEBUG") : "(未设置)");

  /* 1. 加载模型 */
  TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
  if (model == nullptr) {
    std::fprintf(stderr, "结果: FAIL - 模型加载失败\n");
    return 1;
  }
  std::fprintf(stderr, "模型        : flatbuffer 解析成功\n");

  /* 2. 建解释器并挂 delegate */
  TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
  if (opts == nullptr) {
    TfLiteModelDelete(model);
    std::fprintf(stderr, "结果: FAIL - 选项创建失败\n");
    return 3;
  }
  TfLiteInterpreterOptionsAddDelegate(opts, tflite_plugin_create_delegate,
                                     tflite_plugin_destroy_delegate);
  TfLiteInterpreterOptionsSetNumThreads(opts, 1);

  TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, opts);
  /* 解释器创建后会拷贝需要的选项，官方示例即在此处 Delete */
  TfLiteInterpreterOptionsDelete(opts);
  if (interp == nullptr) {
    TfLiteModelDelete(model);
    std::fprintf(stderr, "结果: FAIL - 解释器创建失败\n");
    return 3;
  }

  /* 3. AllocateTensors —— 触发 PrepareDelegate，分区在这一步完成 */
  if (TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
    std::fprintf(stderr, "结果: FAIL - AllocateTensors 失败\n");
    TfLiteInterpreterDelete(interp);
    TfLiteModelDelete(model);
    return 2;
  }
  std::fprintf(stderr, "AllocateTensors: OK（delegate 分区已完成）\n");

  /* 4. 分区数 —— 接管是否发生的直接证据
   *
   * AllocateTensors 之后：
   *   1  → 只有主图，delegate 一个节点都没接管
   *   >1 → delegate 切出了分区
   */
  const int n_partitions = TfLiteInterpreterExecutionPlanSize(interp);

  /* 5. dtype 统计 —— 判「真 int8 vs 假 int8」的关键指标 */
  const int n_tensors = TfLiteInterpreterGetTensorCount(interp);
  int n_int8 = 0, n_uint8 = 0, n_int32 = 0, n_float32 = 0, n_other = 0;
  int n_quantized = 0;   /* 带有效 scale 的量化张量 */
  int n_degenerate = 0;  /* scale 异常大，退化标量传染的征兆 */

  for (int i = 0; i < n_tensors; i++) {
    const TfLiteTensor* t = TfLiteInterpreterGetTensor(interp, i);
    if (t == nullptr) continue;
    switch (t->type) {
      case kTfLiteInt8:    n_int8++;    break;
      case kTfLiteUInt8:   n_uint8++;   break;
      case kTfLiteInt32:   n_int32++;   break;
      case kTfLiteFloat32: n_float32++; break;
      default:             n_other++;   break;
    }
    if ((t->type == kTfLiteInt8 || t->type == kTfLiteUInt8) &&
        t->params.scale > 0.0f) {
      n_quantized++;
      /* 正常 scale 在 1e-3 ~ 1e1 量级；
         1e10 以上说明是常量 0 张量导致的 0/0 退化传染。 */
      if (t->params.scale > 1.0e10f) n_degenerate++;
    }
  }

  /* 6. 输入/输出张量信息 */
  const int n_in = TfLiteInterpreterGetInputTensorCount(interp);
  const int n_out = TfLiteInterpreterGetOutputTensorCount(interp);
  std::fprintf(stderr, "输入张量数    : %d\n", n_in);
  for (int i = 0; i < n_in; i++) {
    const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp, i);
    if (t == nullptr) continue;
    std::fprintf(stderr, "  输入[%d] dtype=%s scale=%g zp=%lld shape=",
                 i, dtype_name(t->type), t->params.scale,
                 (long long)t->params.zero_point);
    print_shape(t);
    std::fprintf(stderr, "\n");
  }
  std::fprintf(stderr, "输出张量数    : %d\n", n_out);
  for (int i = 0; i < n_out; i++) {
    const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp, i);
    if (t == nullptr) continue;
    std::fprintf(stderr, "  输出[%d] dtype=%s scale=%g zp=%lld shape=",
                 i, dtype_name(t->type), t->params.scale,
                 (long long)t->params.zero_point);
    print_shape(t);
    std::fprintf(stderr, "\n");
  }

  /* 7. 机器可读摘要：一行 key=value，便于 grep/awk 判定 */
  std::fprintf(stderr,
               "SUMMARY partitions=%d tensors=%d int8=%d uint8=%d int32=%d "
               "float32=%d other=%d quantized_tensors=%d degenerate_scales=%d\n",
               n_partitions, n_tensors, n_int8, n_uint8, n_int32,
               n_float32, n_other, n_quantized, n_degenerate);

  if (n_degenerate > 0) {
    std::fprintf(stderr,
                 "警告: %d 个张量的 scale > 1e10 —— 疑似 0/0 退化传染\n",
                 n_degenerate);
  }

  /* 分区数 > 1 才说明 delegate 真的切出了分区。
   *
   * 注意：partitions == 1 **不一定**是 bug。
   * teflon 对每个节点独立判定（PrepareDelegate 里 if (supported) 才入委托），
   * 若一个节点都不支持，就不会产生分区。
   * 要与 TEFLON_DEBUG=2 的逐节点日志一起看 —— 空日志无法区分这两种情况，
   * 而设错 TEFLON_DEBUG 的值（=1）恰好就会得到空日志。 */
  const char* verdict = (n_partitions > 1) ? "DELEGATE_ACTIVE"
                                           : "DELEGATE_NO_PARTITION";
  std::fprintf(stderr, "结果: OK - %s (partitions=%d)\n", verdict, n_partitions);

  TfLiteInterpreterDelete(interp);
  TfLiteModelDelete(model);
  return 0;
}