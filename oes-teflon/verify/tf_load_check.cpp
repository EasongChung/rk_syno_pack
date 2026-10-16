/* tf_load_check.cpp —— 上机验证 Teflon delegate 能否接管 .tflite 模型的节点
 *
 * ## 为什么需要这个程序
 *
 * 我们之前只验证过「delegate 能加载」：
 *   teflite_dlopen → dlopen libteflon.so → 找到 tflite_plugin_create_delegate
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
 * ## 本程序做什么 / 不做什么
 *
 * 做：
 *   1. 加载 .tflite 模型
 *   2. 建解释器，把 libteflon.so 的 external delegate 挂上去
 *   3. AllocateTensors()   ← 这一步触发 PrepareDelegate
 *   4. 打印张量/算子统计与子图数
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
 *   3   delegate 未被挂载（AddDelegate 失败）
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/c/c_api_types.h"

/* libteflon.so 导出的 external delegate 入口（Mesa teflon，ABI 见 tfl_device.c） */
extern "C" TfLiteDelegate* tflite_plugin_create_delegate(const char*);
extern "C" void tflite_plugin_destroy_delegate(TfLiteDelegate*);

/* tflite_type_name 在 teflon 内部是 file-static 短名版本（L1370），
 * 与 TFLite 公开头里的枚举名不同。这里用公开枚举打印，便于阅读。 */
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
  std::unique_ptr<tflite::TfLiteModel> model =
      tflite::TfLiteModel::CreateFromFile(model_path);
  if (!model) {
    std::fprintf(stderr, "结果: FAIL - 模型加载失败\n");
    return 1;
  }
  std::fprintf(stderr, "模型        : flatbuffer 解析成功\n");

  /* 2. 建解释器并挂 delegate */
  tflite::InterpreterOptions options;
  options.AddDelegate(tflite_plugin_create_delegate,
                      tflite_plugin_destroy_delegate);
  options.SetNumThreads(1);

  std::unique_ptr<tflite::TfLiteInterpreter> interpreter;
  try {
    interpreter = std::make_unique<tflite::TfLiteInterpreter>(
        model.get(), options);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "结果: FAIL - 建解释器异常: %s\n", e.what());
    return 2;
  }

  const TfLiteModel* keep_alive = model.get();
  (void)keep_alive;

  /* 3. AllocateTensors —— 触发 PrepareDelegate，分区在这一步完成 */
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::fprintf(stderr, "结果: FAIL - AllocateTensors 失败\n");
    return 2;
  }
  std::fprintf(stderr, "AllocateTensors: OK（delegate 分区已完成）\n");

  /* 4. 统计 —— 以下均为机器可读，用于验收脚本判定 */
  const int n_sub = interpreter->subgraphs().size();
  int n_tensors = 0, n_ops = 0;

  /* 遍历所有子图累计算子数与张量数 */
  for (int sg = 0; sg < n_sub; sg++) {
    const auto* sub = interpreter->subgraph(sg);
    if (sub == nullptr) continue;
    n_tensors += sub->tensors().size();
    n_ops += static_cast<int>(sub->operators().size());
  }

  /* 输入/输出张量的 dtype 与量化参数 */
  for (int i = 0; i < interpreter->inputs().size(); i++) {
    const TfLiteTensor* t = interpreter->input_tensor(i);
    if (t == nullptr) continue;
    float scale = 0.0f;
    int64_t zp = 0;
    TfLiteQuantizationParams q;
    if (TfLiteTensorGetQuantizationParams(t, &q) == kTfLiteOk) {
      scale = q.scale;
      zp = q.zero_point;
    }
    std::fprintf(stderr,
                 "输入[%d]      : dtype=%s scale=%g zp=%lld shape=[",
                 i, dtype_name(t->type), scale, (long long)zp);
    for (size_t d = 0; d < t->dims->size; d++) {
      std::fprintf(stderr, "%s%d", d ? "," : "", t->dims->data[d]);
    }
    std::fprintf(stderr, "]\n");
  }

  for (int i = 0; i < interpreter->outputs().size(); i++) {
    const TfLiteTensor* t = interpreter->output_tensor(i);
    if (t == nullptr) continue;
    std::fprintf(stderr, "输出[%d]      : dtype=%s shape=[", i,
                 dtype_name(t->type));
    for (size_t d = 0; d < t->dims->size; d++) {
      std::fprintf(stderr, "%s%d", d ? "," : "", t->dims->data[d]);
    }
    std::fprintf(stderr, "]\n");
  }

  /* dtype 直方图：量化张量占比是判「真 int8 vs 假 int8」的关键指标 */
  int n_quantized = 0;
  int n_float = 0;
  for (int sg = 0; sg < n_sub; sg++) {
    const auto* sub = interpreter->subgraph(sg);
    if (sub == nullptr) continue;
    for (const auto* t : sub->tensors()) {
      if (t->type == kTfLiteInt8 || t->type == kTfLiteUInt8) {
        float s = 0.0f;
        int64_t z = 0;
        TfLiteQuantizationParams q;
        if (TfLiteTensorGetQuantizationParams(t, &q) == kTfLiteOk && q.scale > 0) {
          n_quantized++;
        }
      } else if (t->type == kTfLiteFloat32 || t->type == kTfLiteFloat16) {
        n_float++;
      }
    }
  }

  /* 机器可读摘要：一行 key=value，便于 grep/awk 判定 */
  std::fprintf(stderr,
               "SUMMARY subgraphs=%d tensors=%d ops=%d quantized_tensors=%d "
               "float_tensors=%d\n",
               n_sub, n_tensors, n_ops, n_quantized, n_float);

  /* 子图数 > 1 说明 delegate 成功切出了分区。
   * 若 subgraphs == 1，说明没有任何节点被接管 —— 这是需要重点排查的信号。 */
  const char* verdict = (n_sub > 1) ? "DELEGATE_ACTIVE"
                                     : "DELEGATE_INACTIVE(no partition)";
  std::fprintf(stderr, "结果: OK - %s\n", verdict);

  /* 注意：subgraphs==1 不一定是 bug。delegate 对每个节点独立判定
   * （PrepareDelegate 里 if (supported) 才入委托），若一个节点都不支持，
   * 就不会产生分区。这要与 TEFLON_DEBUG 的逐节点日志一起看。 */
  return 0;
}