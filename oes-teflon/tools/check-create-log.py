#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机日志判读器：把 `[och-create]` / `[och-alias]` 两种取证输出放在一起算账。

用法：
    python3 oes-teflon/tools/check-create-log.py <device.log>
    python3 oes-teflon/tools/check-create-log.py <device.log> --ctx 5000   # 打印含该值的上下文
    cat device.log | python3 oes-teflon/tools/check-create-log.py -

为什么要有这个工具（三个理由，都是被数据教会的）：

1. **并排打数，避免"自己和自己矛盾"**。人读日志最容易犯的错就是分别看两处、
   各自成立，合起来不可能。本工具把 `created ... req_size=X` 与
   `created ... actual=X` 的**行数并排**打出来：CREATEDIAG 已证
   `actual == req_size` 在**每一行**都成立 ⇒ 这两个计数**必须相等**；
   不等就说明日志被截断/串行/看错，而这类错误单看一处永远看不出来。

2. **同一 idx 的时间线**。谁先谁后决定因果：`create_tensor(151, 235224)` 与
   `reference_tensor_with_offset(150→151, size=610000)` 的先后关系，
   单看任何一条都判不出来，必须按出现顺序排在一起。

3. **alias 改写 size 的检测**。`reference_tensor_with_offset` 里
   `dst->size = size` 是**无条件**写的：它只改元数据、不碰 buffer。
   本工具把「某 idx 被 create 过 size=A、又被 alias 写成 size=B（A≠B）」
   全部列出来 —— 那正是「tensor->size 与 buffer 大小脱钩」的机械形态。

退出码：发现不一致 ⇒ 1（可用于将来接进流程）；纯 `--ctx` 打印 ⇒ 0。
⚠️ 本工具只做**算账与一致性检查**，不做根因判断：它不能替你回答"为什么"，
   只能保证你看到的两组数字彼此不矛盾、并把相关行按时间摆在一起。
"""

import re
import sys

# ⚠️ `(?:sg=\d+\s+)?` 是**可选**的子图标识前缀（下一轮取证要求带），且**必须完全
#    非捕获**：写成 `(?:sg=(\d+)\s+)?` 会把后面所有组号右移一位，下面的 m.group(N)
#    就全错位了 —— 我第一版就这么写错了，当场发现。子图号另用 sg_of(line) 取。
RE_ENTER = re.compile(
    r"\[och-create\] enter\s+(?:sg=\d+\s+)?idx=(\d+) req_size=(\d+) have_res=(\S+) "
    r"have_size=(\d+) have_actual=(\d+)")
RE_EARLY = re.compile(
    r"\[och-create\]\s+EARLY-RETURN\s+(?:sg=\d+\s+)?idx=(\d+) req_size=(\d+) "
    r"tensor_size=(\d+) res=(\S+) actual=(\d+)\s*(.*)$")
RE_CREATED = re.compile(
    r"\[och-create\]\s+created\s+(?:sg=\d+\s+)?idx=(\d+) req_size=(\d+) res=(\S+) actual=(\d+)")
RE_DESTROY = re.compile(
    r"\[och-create\]\s+destroy\s+(?:sg=\d+\s+)?idx=(\d+) res=(\S+) actual=(\d+)")

_SG_RE = re.compile(r"\bsg=(\d+)")


def sg_of(line):
    m = _SG_RE.search(line)
    return int(m.group(1)) if m else None
RE_REF = re.compile(
    r"\[och-alias\] ref\s+src=(\d+) dst=(\d+) offset=(\d+) size=(\d+) "
    r"old_res=(\S+) new_res=(\S+)")
RE_MIGRATE = re.compile(
    r"\[och-alias\]\s+migrate idx=(\d+) res=(\S+) old\(size=(\d+),off=(\d+)\) "
    r"-> new\(size=(\d+),off=(\d+)\)")
RE_REC = re.compile(
    r"\[och-alias\] recreate idx=(\d+) new_size=(\d+) old_res=(\S+) "
    r"old\(size=(\d+),off=(\d+)\)")

problems = []


def bad(msg):
    problems.append(msg)
    print("  [不一致] " + msg)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ctx = None
    if "--ctx" in sys.argv:
        i = sys.argv.index("--ctx")
        ctx = sys.argv[i + 1] if i + 1 < len(sys.argv) else "5000"
    if not args:
        sys.stderr.write(__doc__)
        return 2
    if args[0] == "-":
        lines = sys.stdin.read().splitlines()
    else:
        with open(args[0], encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

    ev = []            # (lineno, kind, payload)
    counts = {}
    sg_by_line = {}
    sg_wanted = None
    if "--sg" in sys.argv and sys.argv[sys.argv.index("--sg") + 1].isdigit():
        sg_wanted = int(sys.argv[sys.argv.index("--sg") + 1])
    for no, line in enumerate(lines, 1):
        sg_by_line[no] = sg_of(line)
        if sg_wanted is not None:
            s = sg_of(line)
            if s is not None and s != sg_wanted:
                continue
        for kind, rx in (("enter", RE_ENTER), ("early", RE_EARLY),
                         ("created", RE_CREATED), ("destroy", RE_DESTROY),
                         ("ref", RE_REF), ("migrate", RE_MIGRATE),
                         ("recreate", RE_REC)):
            m = rx.search(line)
            if m:
                counts[kind] = counts.get(kind, 0) + 1
                ev.append((no, kind, m))
                break

    def key(no, idx):
        """按 (子图, idx) 分组。idx 是子图局部编号 ⇒ 不带 sg 分组就会把不同子图的
        同一个 idx 混成同一个张量。**这条是被 sg 用例当场抓出来的**：sg=7 与 sg=9
        各有一次 created idx=151，不分组就会被误判成「重复创建且无解释」。"""
        return (sg_by_line.get(no), idx)

    print("=== 0. 行数统计 ===")
    for k in ("enter", "early", "created", "destroy", "ref", "migrate", "recreate"):
        print("  %-10s %d" % (k, counts.get(k, 0)))
    print("  %-10s %d" % ("[och-create] 合计",
                          sum(counts.get(k, 0) for k in ("enter", "early", "created", "destroy"))))
    print("  %-10s %d" % ("[och-alias] 合计",
                          sum(counts.get(k, 0) for k in ("ref", "migrate", "recreate"))))
    if not ev:
        print("\nRESULT: 日志里没有任何 [och-*] 行 —— 是不是忘了 ETNA_MESA_DEBUG=ml_msgs？")
        return 1

    print()
    print("=== 0b. ★ 子图局部性警告（读下面任何 per-idx 结论前必读）===")
    has_sg = any("[och-create]" in l and " sg=" in l for l in lines)
    if not has_sg:
        print("  ⚠️ 本日志**没有子图标识**（形如 `sg=`）。`etna_ml_tensor` 的 idx 是")
        print("     **子图局部**编号：每个 partition 各建一张表，同一个 idx 在不同")
        print("     partition 里指的是**不同张量**。⇒ 第 4/5 节的「per-idx 时间线」")
        print("     与「alias 改写 size」在跨 partition 的行上**不成立**，只能当线索。")
        print("     下一轮取证务必带子图标识（本工具已支持 `sg=` 字段，见 --sg）。")
    else:
        print("  日志带子图标识 ⇒ per-idx 结论可在本子图内成立（用 --sg <id> 限定）")

    print()
    print("=== 1. created 行的 req_size vs actual（必须逐行相等）===")
    n_ceq = n_cne = 0
    c_req = {}
    c_act = {}
    for no, kind, m in ev:
        if kind != "created":
            continue
        idx, req, _res, act = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        c_req[req] = c_req.get(req, 0) + 1
        c_act[act] = c_act.get(act, 0) + 1
        if act == req:
            n_ceq += 1
        else:
            n_cne += 1
            bad("第%d行 created idx=%d req_size=%d actual=%d（两者不等）" % (no, idx, req, act))
    print("  actual == req_size : %d 行" % n_ceq)
    print("  actual != req_size : %d 行" % n_cne)

    print()
    print("=== 2. ★ 并排计数（这两个数必须相等，否则日志自相矛盾）===")
    probe = int(ctx) if ctx and ctx.isdigit() else None
    for val in sorted(set(list(c_req) + list(c_act))):
        a, b = c_req.get(val, 0), c_act.get(val, 0)
        flag = "" if a == b else "   <== 两数不等！"
        mark = "  <= 关注值" if (probe is not None and val == probe) else ""
        print("  值 %-9d req_size 侧 %3d 行 / actual 侧 %3d 行%s%s" % (val, a, b, flag, mark))
    mism = [v for v in set(list(c_req) + list(c_act)) if c_req.get(v, 0) != c_act.get(v, 0)]
    if mism:
        bad("req_size 侧与 actual 侧的计数不等，值=%s ⇒ 日志被截断/串行，或看错了地方"
            % sorted(mism))

    print()
    print("=== 3. EARLY-RETURN 标记统计 ===")
    n_bad_marker = 0
    for no, kind, m in ev:
        if kind != "early":
            continue
        tail = (m.group(6) or "").strip()
        if "不一致" in tail:
            n_bad_marker += 1
            bad("第%d行 EARLY-RETURN 标记为「size 不一致」: idx=%s req_size=%s tensor_size=%s"
                % (no, m.group(1), m.group(2), m.group(3)))
    print("  EARLY-RETURN 总数        %d" % counts.get("early", 0))
    print("  其中「size 不一致」       %d" % n_bad_marker)
    print("  其中「size 一致」         %d" % (counts.get("early", 0) - n_bad_marker))

    print()
    print("=== 4. 每个 idx 的事件时间线（谁先谁后决定因果）===")
    tl = {}
    for no, kind, m in ev:
        if kind in ("enter", "early", "created", "destroy"):
            tl.setdefault(key(no, int(m.group(1))), []).append((no, kind, m))
        elif kind == "ref":
            tl.setdefault(key(no, int(m.group(2))), []).append((no, "ref-dst", m))
            tl.setdefault(key(no, int(m.group(1))), []).append((no, "ref-src", m))
    for k in sorted(tl, key=lambda t: (str(t[0]), t[1])):
        idx = k[1]
        seq = tl[k]
        if len(seq) < 2 and not any(k == "ref-dst" for _n, k, _m in seq):
            continue
        parts = []
        for no, kind, m in seq:
            if kind == "enter":
                parts.append("L%d enter(size=%s,have=%s,have_act=%s)"
                             % (no, m.group(2), m.group(4), m.group(5)))
            elif kind == "early":
                parts.append("L%d EARLY-RETURN(size=%s,tensor=%s,actual=%s)"
                             % (no, m.group(2), m.group(3), m.group(5)))
            elif kind == "created":
                parts.append("L%d created(req=%s,actual=%s)" % (no, m.group(2), m.group(4)))
            elif kind == "destroy":
                parts.append("L%d destroy(actual=%s)" % (no, m.group(3)))
            elif kind == "ref-dst":
                parts.append("L%d REF->dst(size=%s,old_res=%s,new_res=%s)"
                             % (no, m.group(4), m.group(5), m.group(6)))
            elif kind == "ref-src":
                parts.append("L%d REF->src" % no)
        print("  sg=%-5s idx=%-4d %s" % (k[0], idx, "  ".join(parts)))

    print()
    print("=== 5. ★ alias 无条件改写 dst->size：与该 idx 先前 create 的 size 不符者 ===")
    last_size = {}      # (sg, idx) -> 最近一次 create/enter 看到的 tensor->size
    for no, kind, m in ev:
        if kind == "created":
            last_size[key(no, int(m.group(1)))] = int(m.group(2))
        elif kind == "enter":
            if int(m.group(4)):
                last_size[key(no, int(m.group(1)))] = int(m.group(4))
        elif kind == "ref":
            dst, newsz, old_res = int(m.group(2)), int(m.group(4)), m.group(5)
            prev = last_size.get(key(no, dst))
            if prev is not None and prev != newsz:
                print("  L%d sg=%s dst=%d : tensor->size 由 %d 被改写成 %d"
                      "（old_res=%s, new_res=%s）"
                      % (no, sg_by_line.get(no), dst, prev, newsz, old_res, m.group(6)))
                print("       ⇒ buffer 大小仍是 %d 那一版；元数据与 buffer 由此脱钩" % prev)
                print("       ⇒ 这类「alias 改 size、不改 buffer」就是 assert 的机械成因")
            elif prev is None:
                print("  L%d sg=%s dst=%d : 首次出现，alias 直接写 size=%d（此前无 create 记录）"
                      % (no, sg_by_line.get(no), dst, newsz))
            last_size[key(no, dst)] = newsz
    if not any(k == "ref" for _n, k, _m in ev):
        print("  （本日志没有 [och-alias] 行 ⇒ 无法做 alias/size 对账）")

    print()
    print("=== 5b. 同一 idx 被以不同 size 重复 created（assert 开着时不可能）===")
    # `create_tensor` 的早退分支是 `if (res != NULL) { assert(size == tensor->size); return; }`
    # ⇒ 同一 idx 两次以不同 size 走到「真分配」分支，在我们的构建里必然先 abort。
    #   所以日志里若出现「两次 created、size 不同、中间没有 early-return 或 alias
    #   解释」，只有三种可能：日志被截断/多线程交织/这轮构建没有 assert。
    #   —— 三种都值得当场知道，故列为不一致。
    cre = {}
    between = {}
    for no, kind, m in ev:
        if kind == "created":
            cre.setdefault(key(no, int(m.group(1))), []).append((no, int(m.group(2))))
        elif kind == "early":
            between.setdefault(key(no, int(m.group(1))), []).append(no)
        elif kind == "ref":
            between.setdefault(key(no, int(m.group(2))), []).append(no)   # 作为 dst 被改写
            between.setdefault(key(no, int(m.group(1))), []).append(no)   # 作为 src 被搬
        elif kind == "destroy":
            between.setdefault(key(no, int(m.group(1))), []).append(no)
    found_dup = False
    for k in sorted(cre, key=lambda t: (str(t[0]), t[1])):
        lst = cre[k]
        if len(lst) < 2:
            continue
        expl = between.get(k, [])
        for (n1, s1), (n2, s2) in zip(lst, lst[1:]):
            if s1 == s2:
                continue
            inter = [x for x in expl if n1 < x < n2]
            if inter:
                continue
            found_dup = True
            bad("sg=%s idx=%d 在 L%d(size=%d) 与 L%d(size=%d) 之间无 "
                "early-return/alias/destroy 解释却被重复创建"
                % (k[0], k[1], n1, s1, n2, s2))
    if not found_dup:
        print("  无（每个 idx 的多次 created 要么 size 相同，要么中间有解释事件）")

    if probe is not None:
        print()
        print("=== 6. 含 %d 的行及其上下文 ===" % probe)
        hit = [i for i, l in enumerate(lines) if str(probe) in l]
        shown = set()
        for i in hit:
            for k in range(max(0, i - 3), min(len(lines), i + 4)):
                if k not in shown:
                    shown.add(k)
                    print("  %s%5d| %s" % ("*" if k == i else " ", k + 1, lines[k]))
            print("  " + "-" * 60)

    print()
    if problems:
        print("RESULT: %d 处不一致 —— %s" % (len(problems), "; ".join(problems[:6])))
        return 1
    print("RESULT: PASS（日志内部自洽；但本工具不判断根因，只保证数字彼此不矛盾）")
    return 0


if __name__ == "__main__":
    sys.exit(main())