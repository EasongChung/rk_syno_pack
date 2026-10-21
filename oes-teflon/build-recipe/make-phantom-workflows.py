#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make-phantom-workflows.py —— 生成 B 段 CI 的两个 workflow（fix / diag）

## 为什么必须是两个产物（而不是一个）

`build-teflon.sh` 现在保证 **RODIAG 必须配 ROFIX**，且顺序固定为「先 ROFIX、再取证叠加」
（3j → 3j-2）。原因（2026-10-20 事故，报告 38）：**ROFIX=0 的取证产物**会把
`read_outputs` 的越界读重新引入，在真机上**整机硬挂**（无 Oops、需断电重启）。

于是 B 段仍是两个**同源**产物，但差别只在幻影补丁与取证：

| 产物 | 开关 | 用途 |
|---|---|---|
| `...-kpat-phantomfix` | ROFIX=1，CLAMP=1，GATE=1 | C3 / C4 上机载体（能正常回写，不崩） |
| `...-kpat-phantomdiag` | **ROFIX=1**，CLAMP=1，GATE=1，**RODIAG=1** | 取证仪器：打印 tsize / bsize / head 16 B（**叠加在 ROFIX 之上**） |

## 用法

    python3 build-recipe/make-phantom-workflows.py <基准 kpat.yml> <输出目录> [fix|diag|both]

## 产物判据

    strings libteflon.so | grep -E 'och-patch-v7-phantom-clamp|och-patch-v7-phantom-gate'
    # diag 额外：strings libteflon.so | grep -F '[och-ro]'
    # 两者都必须**没有** [och-fp]（FPARTIAL 不在 B 段内）
"""

import os
import sys


def common_rules(suffix):
    """通用变换。suffix=None 时**不改 workflow 名**（delivery 模式自己改名）。"""
    rules = []
    if suffix is not None:
        rules.append((
            "workflow 名",
            "name: build-teflon-fix-rofix-perchannel-kpat\n",
            "name: build-teflon-fix-rofix-perchannel-kpat-%s\n" % suffix,
        ))
    # ★ BUILD-INFO 台账：run 39407410761 / 39407423126 的产物里**没有**记录
    #   PHANTOM_CLAMP / GATE / RODIAG（只靠编译期标记证明）。开关台账应随产物走，
    #   否则「这个 .so 到底开了什么」只能靠人回忆。以后每轮必带。
    rules.append((
        "BUILD-INFO 台账补三个开关",
        'TEFLON_NO_STRIP=${TEFLON_NO_STRIP}"\n',
        'TEFLON_NO_STRIP=${TEFLON_NO_STRIP}'
        ' TEFLON_PHANTOM_CLAMP=${TEFLON_PHANTOM_CLAMP:-0}'
        ' TEFLON_PHANTOM_GATE=${TEFLON_PHANTOM_GATE:-0}'
        ' TEFLON_RODIAG=${TEFLON_RODIAG:-0}"\n',
    ))
    return rules

ENV_ANCHOR = "  TEFLON_KPAT: '1'\n"

ASSERT_ROFIX = (
    "                     '[3j/6] 打【修复】read_outputs 范围对齐 invoke + idx 边界检查' \\\n"
    "                     '[och-rofix] 已加 read_outputs 范围对齐' \\\n"
)

ASSERT_PERCHANNEL = "                     '[perchannel] 已放开 weight/bias 的 per-axis 量化接管'; do\n"

ASSERT_PERCHANNEL_MORE = (
    "                     '[perchannel] 已放开 weight/bias 的 per-axis 量化接管' \\\n"
    "                     '[3g2/6] 打【修复】幻影通道 out_channel 钳位' \\\n"
    "                     '[phantom-clamp] 已加幻影通道钳位' \\\n"
    "                     '[3h2/6] 打【修复】幻影通道形状门控' \\\n"
    "                     '[och-pgate] 已加幻影通道形状门控'; do\n"
)

NEG_RO = "                       '[och-wc]' '[och-ro]'; do\n"
NEG_NO_RO = "                       '[och-wc]'; do\n"

ARTIFACT = "          name: libteflon-fix-rofix-perchannel-kpat\n"

# 注意：YAML plain scalar 里不能出现「冒号+空格」——
# 实测 run 39401638871：步骤名写成 `B段: ...` ⇒ 整个 workflow 解析失败。
UPLOAD_FIX = (
    "      - name: Upload artifact (B-fix — FIX_V7+OCFIX+FSGUARD+ROFIX+PERCHANNEL+NNDUMP+KPAT"
    " + PHANTOM_CLAMP + PHANTOM_GATE)\n"
)
UPLOAD_DIAG = (
    "      - name: Upload artifact (B-diag — FIX_V7+OCFIX+FSGUARD+PERCHANNEL+NNDUMP+KPAT"
    " + PHANTOM_CLAMP + PHANTOM_GATE + RODIAG, ROFIX ON + superimposed)\n"
)
UPLOAD_BASE = (
    "      - name: Upload artifact (FIX_V7 + OCFIX + FSGUARD + ROFIX + PERCHANNEL + NNDUMP + KPAT,"
    " UNSTRIPPED evidence)\n"
)

ANCHOR_SUMMARY = '          if [ "$ok" -ne 1 ]; then\n'

POSITIVE_FIX = """
          # ★★ B 段正控制：两条幻影补丁的编译期标记**必须**都在产物里，
          #   否则「开关设了但补丁没打上」会被构建成功掩盖（本仓踩过同类）。
          for pat in 'och-patch-v7-phantom-clamp' \\
                     'och-patch-v7-phantom-gate'; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  OK(positive): $pat"
            else
              echo "  ERROR: 产物里缺少「$pat」—— 开关设了但补丁没进产物"
              ok=0
            fi
          done
          # ★ 反控制：FPARTIAL / RODIAG 都必须缺席（B-fix 不带）
          for pat in '[och-fp]' '[och-ro]'; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  ERROR: 产物里出现 $pat —— 开关误开，B-fix 不允许"
              ok=0
            else
              echo "  OK(negative): $pat absent"
            fi
          done
"""

POSITIVE_DIAG = """
          # ★★ B 段正控制：两条幻影补丁的编译期标记 + RODIAG 取证打印**必须**都在。
          for pat in 'och-patch-v7-phantom-clamp' \\
                     'och-patch-v7-phantom-gate' \\
                     '[och-ro]' \\
                     '[och-ro]   head=' \\
                     '[och-ro]   sample ' \\
                     '[och-wd]' \\
                     '[och-td]' \\
                     '[och-coef]' \\
                     '[och-wdom]' \\
                     'TEFLON_WDOMAIN' \\
                     '[och-odetrans]' \\
                     'TEFLON_ODETRANS' \\
                     '[och-roundmode]' \\
                     'TEFLON_ROUNDMODE' \\
                     '[och-pcdomain]' \\
                     'TEFLON_PCDOMAIN' \\
                     'TEFLON_WFILL'; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  OK(positive): $pat"
            else
              echo "  ERROR: 产物里缺少「$pat」—— 开关设了但补丁没进产物"
              ok=0
            fi
          done
          # ★ 反控制：FPARTIAL 必须缺席。
          #   ★ ROFIX 的标记则**必须存在**（2026-10-20 事故后：取证产物必须含 ROFIX，
          #     否则 read_outputs 的越界读重新出现，真机会整机硬挂 —— 报告 38）。
          for pat in '[och-fp]'; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  ERROR: 产物里出现 $pat —— 与 B-diag 的开关组合矛盾"
              ok=0
            else
              echo "  OK(negative): $pat absent"
            fi
          done
          if ! strings -a "$SO" | grep -qF '[och-rofix]'; then
            echo "  ERROR: 产物里缺少 [och-rofix] —— 取证产物必须含 ROFIX（报告 38）"
            ok=0
          fi
"""


# ══════════════════════════════════════════════════════════════════
# 交付版（delivery）：修复**默认开**、诊断/实验全关、strip 过的 .so
# ══════════════════════════════════════════════════════════════════
#
# 与 phantom* 变体的区别：那些是**取证**产物（诊断全开、带 DWARF、不 strip）；
# 交付版反过来 —— 诊断全关、strip，修复默认生效（产品侧不设任何环境变量）。
#
# 开关集合的依据（报告 42/43/44/46）：
#   开  FIX_V7   v7 calculate_bias_correction 的 depthwise/addition 修复
#   开  OCFIX    write_core 的 out_channel 钳位（防越界）
#   开  FSGUARD  不自洽卷积回退 CPU
#   开  ROFIX    read_outputs 范围对齐 invoke（RODIAG 也依赖它）
#   开  PERCHANNEL  per-axis 量化的接管门 + 布局契约 + 域修复
#   开  GATE     out_c%8!=0 的卷积不接管（幻影通道的**兜底**，零挂死风险）
#   开  WDOM/ODETRANS/PCDOMAIN  三个修复（**门控语义已改成默认开**）
#   关  NNDUMP/RODIAG/ALIASDIAG/CREATEDIAG/WCDIAG  诊断（体积 + 刷屏）
#   关  CLAMP    被 GATE 取代（且是那条高危路线）
#   关  KPAT/FPARTIAL/ROUNDMODE  实验/探针
#   关  R5GUARD  P0 未做，误命中率未知
#   开  STDINC   纯卫生（补显式头文件）

DELIVERY_ENV = """env:
  MESA_TAG: ${{ github.event.inputs.mesa_tag || 'mesa-26.2.5' }}
  TEFLON_FIX_V7: '1'
  TEFLON_OCFIX: '1'
  TEFLON_FSGUARD: '1'
  TEFLON_ROFIX: '1'
  TEFLON_PERCHANNEL: '1'
  TEFLON_PERCHANNEL_DIAG: '0'
  TEFLON_PHANTOM_GATE: '1'
  TEFLON_PHANTOM_CLAMP: '0'
  TEFLON_WDOM_PATCH: '1'
  TEFLON_ODETRANS_PATCH: '1'
  TEFLON_PCDOMAIN_PATCH: '1'
  TEFLON_STDINC_PATCH: '1'
  TEFLON_ROUNDMODE_PATCH: '0'
  TEFLON_NNDUMP: '0'
  TEFLON_KPAT: '0'
  TEFLON_FPARTIAL: '0'
  TEFLON_R5GUARD: '0'
  TEFLON_RODIAG: '0'
  TEFLON_ALIASDIAG: '0'
  TEFLON_CREATEDIAG: '0'
  TEFLON_DIAG: '0'
  TEFLON_WCDIAG: '0'
  TEFLON_DEBUG_INFO: '0'
  TEFLON_NO_STRIP: '0'
"""

DELIVERY_STRINGS = """          # ★ 交付版的额外门：三个修复的标记**必须在**，诊断/实验标记**必须不在**
          for pat in '[och-wdom]' '[och-odetrans]' '[och-pcdomain]'; do
            if ! strings -a "$SO" | grep -qF "$pat"; then
              echo "ERROR: 产物里缺少 $pat —— 该修复没进二进制，交付产物会带缺陷"
              exit 1
            fi
            echo "STRING-CHECK: $pat PRESENT (fix really in .so)"
          done
          for probe in '[och-nn]' '[och-fp]' '[och-pcl]' 'och-patch-v7-phantom-clamp'; do
            if strings -a "$SO" | grep -qF "$probe"; then
              echo "ERROR: 产物里出现 $probe —— 诊断/实验开关误开，这不是交付产物"
              exit 1
            fi
            echo "STRING-CHECK: $probe absent"
          done
"""


def build(base: str, mode: str) -> str:
    src = open(base, encoding="utf-8").read()
    out = src

    def sub(label, old, new, expect=1):
        nonlocal out
        n = out.count(old)
        if n != expect:
            raise SystemExit("ERROR: 「%s」命中 %d 次（期望 %d）—— 基准 workflow 已改动"
                             % (label, n, expect))
        out = out.replace(old, new)
        print("  [ok] %s" % label)

    for label, old, new in common_rules(None if mode == "delivery" else mode):
        sub(label, old, new)

    if mode == "fix":
        sub("env: + CLAMP/GATE（ROFIX 保持 1）",
            ENV_ANCHOR,
            ENV_ANCHOR +
            "  # ★ B-fix：两条幻影通道修复路线（ROFIX 保持 1 ⇒ 回写不崩）\n"
            "  TEFLON_PHANTOM_CLAMP: '1'\n"
            "  TEFLON_PHANTOM_GATE: '1'\n")
        sub("断言表 + 两条幻影补丁的应用证据", ASSERT_PERCHANNEL, ASSERT_PERCHANNEL_MORE)
        sub("上传步骤名", UPLOAD_BASE, UPLOAD_FIX)
        sub("产物名", ARTIFACT, "          name: libteflon-fix-rofix-perchannel-kpat-phantomfix\n")
        sub("正/反控制块", ANCHOR_SUMMARY, POSITIVE_FIX + "\n" + ANCHOR_SUMMARY)
    elif mode == "diag":
        sub("env: + CLAMP/GATE/RODIAG",
            ENV_ANCHOR,
            ENV_ANCHOR +
            "  # ★ B-diag：两条幻影补丁 + read_outputs 取证（★ ROFIX 必须为 1；取证叠加在 ROFIX 之上，\n"
            "  #   见 build-teflon.sh 3j-2 与报告 38：ROFIX=0 的取证产物会在真机上整机硬挂）\n"
            "  TEFLON_PHANTOM_CLAMP: '1'\n"
            "  TEFLON_PHANTOM_GATE: '1'\n"
            "  TEFLON_RODIAG: '1'\n"
            "  # ★ 权重 int8→uint8 域（+128）：编进产物，但**运行期**再由 TEFLON_WDOMAIN 门控，\n"
            "  #   便于同产物 A/B（不设 = 与上游逐字节一致 = 全零输出；设 = 应有实数输出）。\n"
            "  TEFLON_WDOM_PATCH: '1'\n"
            "  # ★ 图输出 NCHW→NHWC 收口（同样运行期 TEFLON_ODETRANS 门控）\n"
            "  TEFLON_ODETRANS_PATCH: '1'\n"
            "  # ★ 显式 std 头文件（纯卫生，无行为变化）\n"
            "  TEFLON_STDINC_PATCH: '1'\n"
            "  # ★ 舍入模式探针：编进产物，运行期由 TEFLON_ROUNDMODE=0..3 覆盖，\n"
            "  #   不设 = 与上游逐字节一致。用于实测哪个值等价于 TFLite 的四舍五入。\n"
            "  TEFLON_ROUNDMODE_PATCH: '1'\n"
            "  # ★ per-channel 权重折叠后的域修复（报告 46）：\n"
            "  #   signed 分支多减 0x80 ⇒ 非对角权重 0→128 ⇒ 输出饱和到 255。\n"
            "  #   运行期由 TEFLON_PCDOMAIN 门控，不设 = 与上游逐字节一致。\n"
            "  TEFLON_PCDOMAIN_PATCH: '1'\n")
        sub("断言表 + 两条幻影补丁的应用证据", ASSERT_PERCHANNEL, ASSERT_PERCHANNEL_MORE)
        sub("负控制表：RODIAG 由「必须缺席」改为「必须存在」", NEG_RO, NEG_NO_RO)
        sub("workflow 名（与产物名对齐，避免 run 列表里和旧 diag 混淆）",
            "name: build-teflon-fix-rofix-perchannel-kpat-diag\n",
            "name: build-teflon-fix-rofix-perchannel-kpat-phantomdiag\n")
        sub("上传步骤名", UPLOAD_BASE, UPLOAD_DIAG)
        sub("产物名", ARTIFACT, "          name: libteflon-fix-rofix-perchannel-kpat-phantomdiag\n")
        sub("正/反控制块", ANCHOR_SUMMARY, POSITIVE_DIAG + "\n" + ANCHOR_SUMMARY)
        # ★ Verify artifact 步（step 33）里还有两处与 ROFIX 强绑定的硬门，
        #   run 39406884649 实测：`ERROR: 产物里找不到 [och-rofix]`。
        sub("Verify artifact：RODIAG 反门 → 正门",
            "          if strings -a \"$SO\" | grep -qF '[och-ro]'; then\n"
            "            echo \"ERROR: 产物里出现 [och-ro] —— RODIAG 取证开关误开，这不是交付产物\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"STRING-CHECK: [och-ro] absent (RODIAG switch really off)\"\n",
            "          # ★ 本变体就是要 RODIAG ⇒ 必须存在，否则产物没有诊断价值\n"
            "          if ! strings -a \"$SO\" | grep -qF '[och-ro]'; then\n"
            "            echo \"ERROR: 产物里找不到 [och-ro] —— RODIAG 没打进去\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"STRING-CHECK: [och-ro] PRESENT (RODIAG really in .so)\"\n")
    elif mode == "delivery":
        # ── 交付版：修复默认开、诊断全关、strip ──
        # 1) 整个 env 块换掉（base 的 env 是取证配置：NNDUMP/KPAT/NO_STRIP=1）
        i_env = out.index("env:\n  MESA_TAG:")
        j_env = out.index("\njobs:", i_env)
        out = out[:i_env] + DELIVERY_ENV.rstrip("\n") + out[j_env:]

        sub("workflow 名",
            "name: build-teflon-fix-rofix-perchannel-kpat\n",
            "name: build-teflon-delivery\n")

        # 头部说明：base 写的是「交付候选轮 / 数值未验证，NOT DELIVERABLE」，
        # 交付版必须改口径（否则文档与产物自相矛盾）。
        sub("头部说明",
            "# 【交付候选轮 / 数值未验证，NOT DELIVERABLE】四处修复 + per-channel 量化门（UNSTRIPPED）",
            "# 【交付版】三个修复默认开（per-tensor 权重域 / 图输出收口 / per-channel 域）\n"
            "#   + 四处安全修复（FIX_V7 / OCFIX / FSGUARD / ROFIX）+ per-channel 门与布局契约\n"
            "#   + 幻影通道形状门控 GATE（out_c%8!=0 回落 CPU，零挂死风险）\n"
            "#   诊断与实验开关全关（NNDUMP / RODIAG / ALIASDIAG / CREATEDIAG / WCDIAG\n"
            "#   / KPAT / FORCE-PARTIAL / ROUNDMODE 探针 / R5GUARD），产物已 strip。\n"
            "#   ★ 修复的运行期门控语义：**默认开**，TEFLON_WDOMAIN=0 / TEFLON_ODETRANS=0 /\n"
            "#     TEFLON_PCDOMAIN=0 可单独关掉某一个（设 =1 则额外打印，供调试）。")

        sub("job 名",
            "name: build libteflon.so FIX_V7+OCFIX+FSGUARD+ROFIX+PERCHANNEL+NNDUMP+KPAT"
            " (UNSTRIPPED evidence, NOT deliverable)\n",
            "name: build libteflon.so DELIVERY"
            " (fixes default-on, diagnostics off, stripped)\n")

        # 2) 产物级字符串检查：加交付版专属的正/反门
        sub("产物级门：修复标记必须在 / 诊断标记必须不在",
            "            echo \"STRING-CHECK: $probe absent (diag switch really off)\"\n"
            "          done\n",
            "            echo \"STRING-CHECK: $probe absent (diag switch really off)\"\n"
            "          done\n" + DELIVERY_STRINGS)

        # 3) DWARF 硬门反过来：交付版是 strip 的
        sub("DWARF 硬门 → strip 硬门",
            "          if ! readelf -S \"$SO\" | grep -q '\\.debug_info'; then\n"
            "            echo \"ERROR: 产物缺 .debug_info —— NO_STRIP/DEBUG_INFO 未生效，无法 addr2line\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"DWARF: OK (.debug_info present)\"\n",
            "          if readelf -S \"$SO\" | grep -q '\\.debug_info'; then\n"
            "            echo \"ERROR: 交付产物里还有 .debug_info —— NO_STRIP 未生效\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"STRIP: OK (.debug_info absent)\"\n")

        # 4) 上传步骤名 + 产物名
        sub("上传步骤名",
            UPLOAD_BASE,
            "      - name: Upload artifact (DELIVERY — fixes default-on, diagnostics off,"
            " stripped)\n")
        sub("产物名", ARTIFACT, "          name: libteflon-delivery\n")

        # 5) BUILD-INFO 台账：把交付版的开关集记进产物
        sub("BUILD-INFO 台账补交付开关",
            'TEFLON_NO_STRIP=${TEFLON_NO_STRIP}'
            ' TEFLON_PHANTOM_CLAMP=${TEFLON_PHANTOM_CLAMP:-0}'
            ' TEFLON_PHANTOM_GATE=${TEFLON_PHANTOM_GATE:-0}'
            ' TEFLON_RODIAG=${TEFLON_RODIAG:-0}"\n',
            'TEFLON_NO_STRIP=${TEFLON_NO_STRIP}'
            ' TEFLON_WDOM_PATCH=${TEFLON_WDOM_PATCH:-0}'
            ' TEFLON_ODETRANS_PATCH=${TEFLON_ODETRANS_PATCH:-0}'
            ' TEFLON_PCDOMAIN_PATCH=${TEFLON_PCDOMAIN_PATCH:-0}'
            ' TEFLON_PHANTOM_GATE=${TEFLON_PHANTOM_GATE:-0}"\n')

    elif mode in ("clamp", "gate"):
        # ★★ 为什么必须分开：两个补丁都改「接管/写通道」这条链。
        #   GATE 对 out_c % 8 != 0 的卷积**直接 return false**（回退 CPU），
        #   会把 CLAMP 的效果完全遮住 ⇒ 同一产物里放两个，C3 永远看不到 CLAMP。
        #   run 39403704522 那个「两个都开」的产物只能用于 out_c % 8 == 0 的模型
        #   （此时两者都是 no-op），不能用来做 C3/C4 判决。
        is_clamp = (mode == "clamp")
        sub("env: + %s（ROFIX 保持 1）" % ("CLAMP" if is_clamp else "GATE"),
            ENV_ANCHOR,
            ENV_ANCHOR +
            "  # ★ B-%s：只带一条幻影路线（两条同产物会互相遮蔽，见 make-phantom-workflows.py）\n"
            % mode +
            ("  TEFLON_PHANTOM_CLAMP: '1'\n" if is_clamp else "  TEFLON_PHANTOM_GATE: '1'\n"))
        if is_clamp:
            sub("断言表 + CLAMP 应用证据", ASSERT_PERCHANNEL,
                "                     '[perchannel] 已放开 weight/bias 的 per-axis 量化接管' \\\n"
                "                     '[3g2/6] 打【修复】幻影通道 out_channel 钳位' \\\n"
                "                     '[phantom-clamp] 已加幻影通道钳位'; do\n")
        else:
            sub("断言表 + GATE 应用证据", ASSERT_PERCHANNEL,
                "                     '[perchannel] 已放开 weight/bias 的 per-axis 量化接管' \\\n"
                "                     '[3h2/6] 打【修复】幻影通道形状门控' \\\n"
                "                     '[och-pgate] 已加幻影通道形状门控'; do\n")
        sub("上传步骤名", UPLOAD_BASE,
            "      - name: Upload artifact (B-%s — FIX_V7+OCFIX+FSGUARD+ROFIX+PERCHANNEL+NNDUMP+KPAT"
            " + %s only)\n" % (mode, "PHANTOM_CLAMP" if is_clamp else "PHANTOM_GATE"))
        sub("产物名", ARTIFACT,
            "          name: libteflon-fix-rofix-perchannel-kpat-phantom%s\n" % mode)
        pos = (
            "          # ★★ 只带一条幻影路线：本产物必须**只有** %s 的标记，另一条必须缺席。\n"
            "          if strings -a \"$SO\" | grep -qF '%s'; then\n"
            "            echo \"  OK(positive): %s\"\n"
            "          else\n"
            "            echo \"  ERROR: 产物里缺少 %s —— 开关设了但补丁没进产物\"\n"
            "            ok=0\n"
            "          fi\n"
            "          for pat in '%s' '[och-fp]' '[och-ro]'; do\n"
            "            if strings -a \"$SO\" | grep -qF \"$pat\"; then\n"
            "              echo \"  ERROR: 产物里出现 $pat —— 与本变体的开关组合矛盾\"\n"
            "              ok=0\n"
            "            else\n"
            "              echo \"  OK(negative): $pat absent\"\n"
            "            fi\n"
            "          done\n"
        ) % (("CLAMP" if is_clamp else "GATE"),
             ("och-patch-v7-phantom-clamp" if is_clamp else "och-patch-v7-phantom-gate"),
             ("och-patch-v7-phantom-clamp" if is_clamp else "och-patch-v7-phantom-gate"),
             ("och-patch-v7-phantom-clamp" if is_clamp else "och-patch-v7-phantom-gate"),
             ("och-patch-v7-phantom-gate" if is_clamp else "och-patch-v7-phantom-clamp"))
        sub("正/反控制块", ANCHOR_SUMMARY, pos + "\n" + ANCHOR_SUMMARY)
    else:
        raise SystemExit("未知 mode: %s" % mode)

    # 兜底自检：YAML plain scalar 里不能有「: 」（本仓实测踩过，run 39401638871）
    for i, line in enumerate(out.splitlines(), 1):
        s = line.strip()
        if s.startswith("- name:") or s.startswith("name:"):
            body = s.split(":", 1)[1]
            if ": " in body:
                raise SystemExit("ERROR: 第 %d 行 name 里含「冒号+空格」，YAML 会解析失败: %s"
                                 % (i, s))
    return out


def main() -> int:
    if len(sys.argv) not in (4, 5):
        sys.stderr.write("用法: make-phantom-workflows.py <base.yml> <outdir> [fix|diag|both]\n")
        return 2
    base, outdir = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) == 4 else "both"
    modes = ["fix", "diag"] if which == "both" else [which]

    for mode in modes:
        print("== 生成 %s ==" % mode)
        text = build(base, mode)
        suffix = {"fix": "phantomfix", "diag": "phantomdiag",
                  "clamp": "phantomclamp", "gate": "phantomgate",
                  "delivery": "delivery"}[mode]
        path = os.path.join(outdir, "build-teflon-fix-rofix-perchannel-kpat-%s.yml" % suffix)
        os.makedirs(outdir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("  已写出: %s (%d B)" % (path, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
