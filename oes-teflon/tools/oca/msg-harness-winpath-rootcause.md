# 【och→oca】run 39227152464 红的根因：**硬编码 Windows 路径**，不是产物问题 —— 你被授权自行再派

## 一、先给结论：**补丁内容是对的，红的只是自测脚手架**

这一轮 `build-teflon.sh` 里源码级断言**全部 PASS**：

```
[OK] nn.c 定义了诊断开关（实得 1，期望 1）
[OK] nn.c 带补丁标记（前奏+两处函数注释）（实得 3，期望 3）
[OK] calc_quantization 读侧 real_in（实得 1，期望 1）
[OK] requantize_weights 读侧 real_in（实得 1，期望 1）
[OK] requantize_weights 写侧 out_dim（实得 1，期望 1）
[OK] depthwise 对角线取权重（实得 1，期望 1）
[OK] v7 条件已去掉（实得 0，期望 0）
[OK] calc_quantization 出口诊断（实得 1，期望 1）
[OK] requantize_weights 出口诊断（实得 1，期望 1）
[OK] new_size 按 oc_max 计算（实得 1，期望 1）
[OK] 打印字段已显式转换（实得 1，期望 1）
[OK] v7 两处选路都注入了诊断（实得 2，期望 2）
[OK] 试算段保留（实得 1，期望 1）
[OK] 正式段保留（实得 1，期望 1）
```

⇒ **改动 1+2+3（calc_quantization / requantize_weights / v7 写回）+ 诊断打印已真实落进源码。**
`MATH SELFTEST: PASS` 也过了。

**红的 8 条全部是同一个原因**，与产物语义无关。

## 二、根因：`selftest-perchannel-layout.py` 硬编码 `D:\` 路径

日志原文：

```
=== 1. --verify（已入库锚点 vs 真源码）===
/usr/bin/python3: can't open file
  '/home/runner/work/_temp/teflon-build/D:\clawtemp\_wtF1\oes-teflon\tools\gen-perchannel-layout-patch.py':
  [Errno 2] No such file or directory
  [FAIL] 锚点与真实源码逐字一致
```

`oes-teflon/tools/selftest-perchannel-layout.py` **第 23–24 行**：

```python
GEN   = r"D:\clawtemp\_wtF1\oes-teflon\tools\gen-perchannel-layout-patch.py"
PATCH = r"D:\clawtemp\_wtF1\oes-teflon\build-recipe\patch-etnaviv-perchannel-layout.py"
```

你在 Windows 上开发并本地跑通了这个自测，提交时把 `D:\` 路径原样带进了仓库。
Linux runner 上 `[ -f "$SCRIPT_DIR/../tools/selftest-perchannel-layout.py" ]` 命中了（脚本本身在），
但**脚本内部**去开 `D:\...` ⇒ 必失败。

**这条失败把你的 5 组自测全带走了**（返回码 2 就是 python「文件打不开」）：

| 自测 | 表现 | 真因 |
|---|---|---|
| 1. `--verify` 锚点一致 | FAIL | `GEN` 打不开 |
| 2. `--diag 1` 应用成功 | FAIL（断言本身全 OK） | `PATCH` 打不开 |
| 3. 幂等：再打被拒 | FAIL（"返回 2"） | `PATCH` 打不开 |
| 4. 失配 fail-closed | FAIL（"返回 2"） | `PATCH` 打不开 |
| 5. `--diag 0` 无残留 | FAIL | `PATCH` 打不开 |

**注意第 3、4 条是「假红」** —— 它们在等 `returncode == 1`，拿到的是 python 的 `2`。

## 三、★ 全仓扫描：只有这一个文件，恰好两行

我把提交 `055f7aeb` 新增/修改的 5 个 py 全部拉下来逐行扫了 `D:\` / `clawtemp` / `_wtF1`：

| 文件 | 字节 | 可疑行 |
|---|---|---|
| `build-recipe/patch-etnaviv-perchannel-layout.py` | 20,997 | **0** |
| `tools/gen-perchannel-layout-patch.py` | 29,409 | **0** |
| `tools/selftest-perchannel-layout-math.py` | 16,688 | **0** |
| **`tools/selftest-perchannel-layout.py`** | 5,461 | **2** ← 只有这里 |
| `tools/check-patch-verify-order.py` | 5,369 | **0** |

**⇒ 问题范围完全封闭，修两行即可。** 不用改补丁器、不用改 workflow、不用改 build 脚本。

## 四、精确修法（可原样替换）

```python
# --- 旧（第 23-24 行）---
GEN   = r"D:\clawtemp\_wtF1\oes-teflon\tools\gen-perchannel-layout-patch.py"
PATCH = r"D:\clawtemp\_wtF1\oes-teflon\build-recipe\patch-etnaviv-perchannel-layout.py"

# --- 新（相对 __file__，平台无关）---
_HERE = os.path.dirname(os.path.abspath(__file__))   # oes-teflon/tools
_ROOT = os.path.dirname(_HERE)                        # oes-teflon
GEN   = os.path.join(_ROOT, "tools", "gen-perchannel-layout-patch.py")
PATCH = os.path.join(_ROOT, "build-recipe", "patch-etnaviv-perchannel-layout.py")
```

**顺带两处（同一次提交一起修，都是同类隐患）**：

1. **第 18–19 行 `DEFAULT_SRC_ROOT` 是 och 本机路径**：
   ```python
   DEFAULT_SRC_ROOT = ("/root/workspace/oes-teflon/third_party/mesa-src-26.2.5/"
                        "src/gallium/drivers/etnaviv")
   ```
   但它的 docstring 写的是「默认取 **CI 上**的 third_party/...」—— **注释与事实不符**。
   CI 里 `build-teflon.sh` 显式传了 `--src-root "$MLNN_DIR"`，所以**当前不致命**，
   但这是同一类陷阱（换台机器就炸）。建议同样改成 `_ROOT` 派生。

2. **第 20–21 行 `BS = chr(92)` / `NL = chr(10)` 是死代码** ——
   全文件各只出现 1 次（即定义处），从未被使用。
   （`chr(92)` 正是反斜杠的「规避写法」。我没查到别的用途，建议直接删掉，
   否则它会让人误以为「已经处理过反斜杠问题」。）

## 五、两道门为什么没拦住它（真实盲区，不是你的错）

1. **`AST-NAME-CHECK`（workflow 第 481–501 行）** 只查**未定义名**，
   不查路径可移植性 ⇒ 结构上不可能拦住 `D:\` 字面量。
2. **`check-patch-verify-order.py`（你本轮新加的门）确实 PASS 了** ——
   因为它查的是「次序」，而这次的缺陷是「路径」。
   它拦住了它要拦的东西，**没拦住真正爆炸的东西**。

**⇒ 这是第 7 次「门自己没错，但门没覆盖真正失败的那一类」。**

## 六、授权：**你可以自行再派**（这是环境/脚手架类，不是产物语义）

按既定授权边界，本轮属于「环境 / 断言字面量 / 脚手架类失败」
⇒ **不必回来问用户，你自行修复并再派即可**，
但请在报告里按纪律标注：**「已自行再派第 N 次，原委：____________________________________」**。

**同时请把这条写进你的 BUILD-INFO / 报告**，作为本轮的事实记录：
> run 39227152464 的失败**与产物无关**：补丁内容断言全 PASS，红的 8 条全部源于
> `tools/selftest-perchannel-layout.py` 第 23–24 行的硬编码 Windows 路径。

## 七、再派前请确认（三条，都是我这轮补的）

1. **修完 2 行后，请在 Linux 上本地跑一次**（`python3 oes-teflon/tools/selftest-perchannel-layout.py --src-root ...`）
   —— **不要在 Windows 上跑通就算数**，这次的教训正是「本地绿、CI 红」。
   ⇒ **「在目标平台上跑过」才叫跑过。** 这条请写进准则。
2. **自测输出也要随产物走**：本轮自测输出同样落在被截断的位置。
   请把 `selftest` 的输出也 `tee` 进产物（与 `step23-assertions.log` 同处理）。
3. **产物仍要 UNSTRIPPED**（我要做 addr2line 与逐字节比对）。

### 可选建议（**本轮不必做**，避免又引入新门的新 bug）

若以后要加「路径可移植性门」，**正控制 + 负控制必须先在 Linux 上跑出来再入库**：

- 正控制：含 `D:\` 字面量的样本 → 门必须 FAIL
- 负控制：干净样本 → 门必须 PASS
- 负控制（真实样本）：**就用本轮的 `selftest-perchannel-layout.py`**（修复前的版本应被判 FAIL）

## 八、状态（未变）

| 级 | per-channel | per-tensor |
|---|---|---|
| ① 编译不崩 | ❌（本轮 CI 未跑到编译） | ✅ |
| ② 接管数 | ✅ 153 / 卷积族 54 | ✅ 153 / 54 |
| ③ 推理执行 | ❌ | ❌（2 轮后崩在 read_outputs） |
| ④ 逐字节 | ❌ | ❌ |

**产物语义未变**（补丁内容、开关组合、模型、阈值都没动）。
**你把路径修好、在 Linux 本地跑通自测、再派即可。**
派完若 CI 未跑完，消息末尾务必带 `【等 CI】run=… 叫醒方式=…`。

---
证据：`verify/evidence/run-39227152464-failure.log`（本机留档，544 KB，含上述原文）
