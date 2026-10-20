#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make-phantom-workflows.py —— 生成 B 段 CI 的两个 workflow（fix / diag）

## 为什么必须是两个产物（而不是一个）

`build-teflon.sh` 里 **ROFIX 与 RODIAG 是硬互斥**（:798-799 直接 `exit 1`）：
两者都改 `etna_ml_subgraph_read_outputs()` 的同一处循环体，锚点会互相打架。

于是 B 段拆成两个**同源**产物，各自的用途写死在名字里：

| 产物 | 开关 | 用途 |
|---|---|---|
| `...-kpat-phantomfix` | ROFIX=1，CLAMP=1，GATE=1 | C3 / C4 上机载体（能正常回写，不崩） |
| `...-kpat-phantomdiag` | ROFIX=0，CLAMP=1，GATE=1，**RODIAG=1** | C2b 仪器：打印 tsize / bsize / head 16 B |

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
    return [
        (
            "workflow 名",
            "name: build-teflon-fix-rofix-perchannel-kpat\n",
            "name: build-teflon-fix-rofix-perchannel-kpat-%s\n" % suffix,
        ),
    ]


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
    " + PHANTOM_CLAMP + PHANTOM_GATE + RODIAG, ROFIX OFF)\n"
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
                     '[och-ro]   head='; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  OK(positive): $pat"
            else
              echo "  ERROR: 产物里缺少「$pat」—— 开关设了但补丁没进产物"
              ok=0
            fi
          done
          # ★ 反控制：FPARTIAL 必须缺席；ROFIX 的标记也必须缺席（本产物 ROFIX=0）
          for pat in '[och-fp]' '[och-rofix]'; do
            if strings -a "$SO" | grep -qF "$pat"; then
              echo "  ERROR: 产物里出现 $pat —— 与 B-diag 的开关组合矛盾"
              ok=0
            else
              echo "  OK(negative): $pat absent"
            fi
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

    for label, old, new in common_rules(mode):
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
        sub("env: ROFIX 关掉（与 RODIAG 互斥）", "  TEFLON_ROFIX: '1'\n", "  TEFLON_ROFIX: '0'\n")
        sub("env: + CLAMP/GATE/RODIAG",
            ENV_ANCHOR,
            ENV_ANCHOR +
            "  # ★ B-diag：两条幻影补丁 + read_outputs 取证（ROFIX 必须为 0，见 build-teflon.sh:798）\n"
            "  TEFLON_PHANTOM_CLAMP: '1'\n"
            "  TEFLON_PHANTOM_GATE: '1'\n"
            "  TEFLON_RODIAG: '1'\n")
        sub("断言表：去掉 ROFIX 两条（本产物 ROFIX=0）", ASSERT_ROFIX, "")
        # ★ 门 32「ROFIX anchors vs real source」直接检查**构建出来的** etnaviv_ml.c 里
        #   ROFIX 锚点是否落地；本变体 ROFIX=0 ⇒ 该门必然红（run 39403716188 实测）。
        #   用 if: 跳过，而不是删掉——保留原文便于对照。
        sub("ROFIX 源码锚点门：按 env 条件跳过",
            "      - name: Static check - ROFIX anchors vs real source (post-build)\n",
            "      - name: Static check - ROFIX anchors vs real source (post-build)\n"
            "        if: env.TEFLON_ROFIX == '1'\n")
        sub("断言表 + 两条幻影补丁的应用证据", ASSERT_PERCHANNEL, ASSERT_PERCHANNEL_MORE)
        sub("负控制表：RODIAG 由「必须缺席」改为「必须存在」", NEG_RO, NEG_NO_RO)
        sub("上传步骤名", UPLOAD_BASE, UPLOAD_DIAG)
        sub("产物名", ARTIFACT, "          name: libteflon-fix-rofix-perchannel-kpat-phantomdiag\n")
        sub("正/反控制块", ANCHOR_SUMMARY, POSITIVE_DIAG + "\n" + ANCHOR_SUMMARY)
        # ★ Verify artifact 步（step 33）里还有两处与 ROFIX 强绑定的硬门，
        #   run 39406884649 实测：`ERROR: 产物里找不到 [och-rofix]`。
        sub("Verify artifact：ROFIX 正门 → 反门",
            "          if ! strings -a \"$SO\" | grep -qF '[och-rofix]'; then\n"
            "            echo \"ERROR: 产物里找不到 [och-rofix] —— ROFIX 修复代码不在二进制里\"\n"
            "            echo \"       说明 read_outputs 范围对齐没打进去：越界读照旧，真机仍会 SIGSEGV\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"STRING-CHECK: [och-rofix] PRESENT (ROFIX really in .so)\"\n",
            "          # ★ 本变体 ROFIX=0（与 RODIAG 硬互斥）⇒ 这里必须**缺席**\n"
            "          if strings -a \"$SO\" | grep -qF '[och-rofix]'; then\n"
            "            echo \"ERROR: 产物里出现 [och-rofix] —— 与 ROFIX=0 矛盾\"\n"
            "            exit 1\n"
            "          fi\n"
            "          echo \"STRING-CHECK: [och-rofix] absent (ROFIX off in this variant)\"\n")
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
                  "clamp": "phantomclamp", "gate": "phantomgate"}[mode]
        path = os.path.join(outdir, "build-teflon-fix-rofix-perchannel-kpat-%s.yml" % suffix)
        os.makedirs(outdir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("  已写出: %s (%d B)" % (path, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
