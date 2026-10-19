#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门：报告/commit 里的**完成断言**必须附证据，否则视为未完成。

## 为什么要有这道门（och 第 N 轮）

规范条文只能靠人遵守。本项目已经**三次**栽在同一类错误上：

  1. 手抄锚点写错（`_` vs `-`），真源码 count==0，而 mock selftest 12 项全绿
  2. 通用锚点门自身写错（`newline=""` 保留 CRLF + 长度阈值 24），把正常的补丁判死
  3. ★ 把三道需要 Mesa 源码的门放进 Preflight（源码那时还没下载），
     **CI 上必然失败**，而我只做了「YAML 可解析 + 步骤在位」就报告「已接线」

第 3 次说明：**「我提醒你」这个机制是无效的，必须靠自动门拦。**
⇒ 这道门把「已接线 / 已集成 / 已接入」这类断言**变成可自动检查的东西**。

## 判定规则

扫文本（commit message / 报告 / PR 正文），命中「完成断言」后，
要求**同一段落**（该行及其后 4 行）内出现下列任一：

  · **真跑过的证据**
      - Actions run 链接：`actions/runs/<数字>`
      - run 号：`run 39146418954` / `run 号 39146418954`
      - 门的原始输出标记：`RESULT: ALL PASS` / `SELFTEST RESULT: ALL PASS`
                             / `VERIFY: PASS` / `VERDICT: PASS` / `PASS=`
                             / `exit=0`
      - 产物三元组里的 sha256：`sha256 1c4af48a` / `sha256(前8)`
  · **诚实的免责声明**（也算通过，因为它把「没验证」写明了）
      `未验证` / `未跑` / `零 CI` / `零上机` / `NOT DELIVERABLE`
      / `未验证不声称` / `不写可用结论` / `未验证/未跑` / `结构性校验`

⇒ 「已接线，但未验证」**通过**；「已接线。」**不通过**。
   门不禁止人写「已接线」，它禁止的是**没有证据的「已接线」**。

## 这道门自己也可能是错的

本项目已有两例「检查器给出错误的 FAIL，而 FAIL 会诱导人去修不存在的问题」。
⇒ 本门自带 `--selftest`，含：

  · 正控制 3 例（含「已接线，但未验证」这种诚实写法）
  · 负控制 4 例（无证据的断言、无断言的普通文本、免责声明被误当证据的边界）
  · **回归**：拿本仓真实的 commit message 过一遍（它们是已发生的文本）

## 用法

    python3 tools/check-claims-evidence.py <文本文件> [...]
    python3 tools/check-claims-evidence.py --selftest
    git log -1 --format=%B | python3 tools/check-claims-evidence.py -
"""

import argparse
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── 完成断言：出现即要求证据 ──────────────────────────────────────
# ★ 刻意**收窄**。宽断言在自由中文里假阳性太高 —— 本门第一版带「全绿」，
#   拿本仓真实 commit message 一回归就抓到 2 处**假阳性**：
#     · 「而它自己的 mock selftest 12 项全绿」—— 这是在**描述一次失败**
#     · 「两者都在本地全绿」—— 证据（RESULT: ALL PASS）在窗口之外
#   而 FAIL 会诱导人去修一个不存在的问题（见规范 §2.5）。
#   ⇒ 只收「我方完成动作 + 完成态」的高精度形态。
#   ⇒ 另外默认 **advisory**（只警告不拦），见 main()。
CLAIM_PATTERNS = [
    (re.compile(r"已\s*接线"), "已接线"),
    (re.compile(r"已\s*集成"), "已集成"),
    (re.compile(r"已\s*接入"), "已接入"),
    (re.compile(r"已\s*部署"), "已部署"),
    (re.compile(r"(?<!未)已\s*验证(?![a-zA-Z一-鿿])"), "已验证"),
    (re.compile(r"(?<!未)已\s*生效"), "已生效"),
    (re.compile(r"CI\s*接线"), "CI 接线"),
    (re.compile(r"CI\s*(?:已)?(?:通过)"), "CI 已通过"),
    (re.compile(r"端到端\s*(?:已)?通"), "端到端已通"),
]

# ── 证据 ─────────────────────────────────────────────────────────
EVIDENCE = [
    (re.compile(r"actions/runs/\d+"), "Actions run 链接"),
    (re.compile(r"\brun\s*(?:号)?\s*[:：]?\s*\d{6,}"), "run 号"),
    (re.compile(r"RESULT:\s*ALL\s*PASS"), "门输出 RESULT: ALL PASS"),
    (re.compile(r"SELFTEST\s+RESULT:\s*ALL\s*PASS"), "门输出 SELFTEST ALL PASS"),
    (re.compile(r"VERIFY:\s*PASS"), "门输出 VERIFY: PASS"),
    (re.compile(r"VERDICT:\s*PASS"), "门输出 VERDICT: PASS"),
    (re.compile(r"PASS=\d+\s+FAIL=0"), "门输出 PASS=n FAIL=0"),
    (re.compile(r"exit\s*=\s*0"), "门输出 exit=0"),
    (re.compile(r"sha256[（(]?前?8?[)）]?\s*[:：]?\s*[0-9a-f]{8,64}"), "产物 sha256"),
    (re.compile(r"\b[0-9a-f]{40}\b"), "commit sha"),
]

# ── 免责声明：把「没验证」写明了，也算通过 ────────────────────────
DISCLAIMER = [
    re.compile(r"未验证"),
    re.compile(r"未跑"),
    re.compile(r"零\s*CI"),
    re.compile(r"零\s*上机"),
    re.compile(r"NOT\s+DELIVERABLE", re.I),
    re.compile(r"未验证不声称"),
    re.compile(r"不写可用结论"),
    re.compile(r"结构性校验"),
    re.compile(r"只做了.{0,12}校验"),
    re.compile(r"尚未"),
    re.compile(r"未\s*dispatch", re.I),
    re.compile(r"结构性校验\s*[≠!]=\s*功能可用"),
    # 「列在未做清单里」的延期措辞 —— 它明确否定了完成，不该被判成完成断言。
    # 实测：c81e4ce8 的「…、CI 接线：**下一轮**」若不豁免就是假阳性。
    re.compile(r"下一轮"),
    re.compile(r"排队"),
    re.compile(r"留给\s*下一"),
    re.compile(r"待\s*(?:办|做)"),
    re.compile(r"本轮\s*不"),
    re.compile(r"不\s*做\s*（|不[做改]"),
]

WINDOW_AFTER = 4          # 断言行之后再看 4 行（证据常写在下一段的抬头）
WINDOW_BEFORE = 2


def _in_windows(lines, i):
    lo = max(0, i - WINDOW_BEFORE)
    hi = min(len(lines), i + WINDOW_AFTER + 1)
    return "\n".join(lines[lo:hi])


def scan(text):
    """返回 (违例列表, 命中统计)。违例元素 = (行号, 断言名, 该窗口内的文本片段)。"""
    lines = text.replace("\r\n", "\n").split("\n")
    viol = []
    n_claims = 0
    for i, line in enumerate(lines):
        # 规范条文里的举例不算断言（前面有 > 或 § 或「例如」）
        stripped = line.lstrip()
        is_quote = stripped.startswith(">") or stripped.startswith("- ") and False
        for rx, name in CLAIM_PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            if is_quote:
                continue
            n_claims += 1
            win = _in_windows(lines, i)
            ev = [lbl for rx2, lbl in EVIDENCE if rx2.search(win)]
            dc = [d.pattern for d in DISCLAIMER if d.search(win)]
            if not ev and not dc:
                viol.append((i + 1, name, line.strip()[:110]))
            break            # 每行只报第一个断言，避免刷屏
    return viol, n_claims


# ── 自测 ─────────────────────────────────────────────────────────
NEG = """\
## 状态
- 锚点核对**已接线**。
- 文档一致性门已加入。
"""


def selftest():
    FAILS = []

    def rep(label, ok, detail=""):
        print("  %-56s %s%s" % (label, "PASS" if ok else "FAIL",
                                "" if ok else "   <- " + detail))
        if not ok:
            FAILS.append(label)

    print("=== 正控制（必须有证据/免责声明才算通过） ===")
    pos = [
        ("带 run 链接",
         "- Preflight 已接线，见 https://github.com/EasongChung/rk_syno_pack/actions/runs/39146418954\n"),
        ("带 run 号",
         "- 全链路已闭环。run 号 39143802751 SUCCESS\n"),
        ("★诚实写法：『已接线，但未验证』",
         "- 该门已接线，但**未验证**：本轮零 CI、零上机。\n"),
        ("带门输出",
         "- 锚点核对已接入，输出 RESULT: ALL PASS\n"),
        ("带 sha256",
         "- 产物已验证，sha256 1c4af48a97868d0767457ab66779c2ba\n"),
        ("纯描述、无断言",
         "- 这一段只是在解释设计动机，没有做任何完成断言。\n"),
        ("规范条文里的举例（引用块）",
         "> 禁止写「已接线」这类无证据断言。\n"),
    ]
    for label, txt in pos:
        v, n = scan(txt)
        rep(label, not v, "误报: %s" % (v[:1],))

    print()
    print("=== 负控制（无证据的断言必须被抓） ===")
    neg = [
        ("最常见的形态：句号收尾、无任何证据", NEG),
        ("『已验证』无证据",
         "- 该模型已验证，可以上设备了。\n"),
        ("『端到端已通』无证据",
         "- 链路端到端通了。\n"),
        ("『CI 已通过』无 run 链接",
         "- CI 已通过。\n"),
        ("『已部署』无证据",
         "- 服务已部署到设备。\n"),
    ]
    for label, txt in neg:
        v, n = scan(txt)
        rep(label, bool(v), "漏报（应抓到却没抓到）")

    print()
    print("=== 负控制之二：诚实的延期写法不该被抓（防假阳性） ===")
    honest = [
        ("列在未做清单里",
         "- 开关步、文档门、workflow 派生：**下一轮**\n"),
        ("显式说明未验证",
         "- 该门已接线，但**未验证**：本轮零 CI。\n"),
        ("引用他人/既往状态",
         "> 上一轮门漏判：锚点错了、门没抓到，而 mock 全绿。\n"),
    ]
    for label, txt in honest:
        v, n = scan(txt)
        rep(label, not v, "假阳性: %s" % (v[:1],))

    print()
    print("=== 回归：拿本仓真实 commit message 过一遍 ===")
    print("    ★ 这段的价值不在「有没有红」，而在**假阳性率**。")
    print("      第一版带「全绿」时，这里抓到 2 处**假阳性**（一句在描述一次失败，")
    print("      一句的证据在窗口外）⇒ 因此默认模式改为 advisory。")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        r = subprocess.run(
            ["git", "log", "--format=%H%n%B%n----", "-6"],
            cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        msgs = [m for m in (r.stdout or "").split("----") if m.strip()]
    except Exception:                                          # noqa: BLE001
        msgs = []
    # 已发生的文本，**已知结论钉死**（不随实现漂移）。
    # 命中 = 这条 commit 里的完成断言缺证据（历史上确实发生过）。
    EXPECT_HIT = {
        "25e183d6": "真阳性：该 commit 标题写「CI 接线」，而那三道门位置放错、"
                    "CI 上必然失败 —— 正是它该抓的那次",
    }
    if not msgs:
        print("  (跳过：读不到 git log)")
    for m in msgs:
        sha = m.strip().splitlines()[0][:8]
        body = "\n".join(m.strip().splitlines()[1:])
        v, n = scan(body)
        why = EXPECT_HIT.get(sha)
        if why:
            # 已知应当被抓：抓到才算通过（抓不到说明门退化了）
            rep("commit %s 应被抓到且被抓到" % sha, bool(v),
                "应命中「%s」但没抓到 —— 门退化了" % why[:40])
        else:
            rep("commit %s（%d 处断言）无假阳性" % (sha, n), not v,
                "假阳性: " + "; ".join("L%d [%s] %s" % (a, b, c[:50])
                                       for a, b, c in v[:2]))
        for a, b, c in v[:3]:
            print("       - L%d [%s] %s" % (a, b, c))

    print()
    if FAILS:
        print("SELFTEST RESULT: %d FAIL —— %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("SELFTEST RESULT: ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="文本文件；- 表示 stdin")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="发现无证据断言时 exit 1（默认只警告不拦）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.files:
        ap.error("需要至少一个文件，或 --selftest")

    total = 0
    for path in args.files:
        if path == "-":
            text = sys.stdin.read()
            label = "<stdin>"
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            label = os.path.basename(path)
        viol, n = scan(text)
        print("=== %s：%d 处完成断言，%d 处缺证据 ===" % (label, n, len(viol)))
        for ln, name, snippet in viol:
            print("  L%-5d [%s] %s" % (ln, name, snippet))
        total += len(viol)

    print()
    if total:
        print()
        print("WARN: %d 处完成断言没有附证据 —— 这是**警告**，不是失败。" % total)
        print("      处理办法二选一：")
        print("        · 补上真跑过的 run 链接 / 门原始输出；")
        print("        · 或把话改成诚实的「未验证」。")
        if args.strict:
            print("VERDICT: FAIL（--strict）")
            return 1
        print("VERDICT: PASS-with-warnings（默认不拦）")
        print("      ⚠️ 本门**默认不拦**，因为它在自由中文上会假阳性；")
        print("         已实测的假阳性见 --selftest 的回归段。")
        print("         要它拦，必须人先复核每一条（规范 §2.5）。")
        return 0
    print()
    print("VERDICT: PASS —— 所有完成断言都附了证据或写明了未验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
