#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解释一个 `created idx=N req_size=S` 里的 S 是**哪几个 dims 的乘积**。

用法：
    python3 oes-teflon/tools/explain-req-size.py shapes.txt device.log
      shapes.txt 每行： <idx> <d0>x<d1>x<d2>...      （模型侧形状，d0 通常是 1）
      device.log   是 `[och-create]` / `[och-alias]` 的真机输出

为什么需要它（数据给的约束，不是猜的）：
  驱动里**每一处** size 计算都是硬编码的 `dims[1]*dims[2]*dims[3]`（4D 假设），
  真源码 etnaviv_ml.c:294/299/411/421/433/440 全是这一个式子，全文件没有按
  dims_size 遍历的地方。所以只要知道某个张量的 dims 排列，就能算出它会被算成多少；
  反过来，给定 req_size 与模型形状，就能**反推驱动看到的 dims 排列**。

输出三类命中：
  · 全维乘积        —— 说明 dims 排列是对的（D1 不成立）
  · 连续三元组乘积  —— 说明 dims[1..3] 取到了哪一段（D1：4D 硬编码取错段）
  · 其它张量的三元组 —— 说明 size 来自**别的张量**（D2：index 与 size 不同源）

⚠️ 本工具只做**算术反推**，不下根因结论：
   · 它不知道 pipe_tensor->dims 在上游被怎么排列过（那是日志要回答的）；
   · 它不知道 idx 是**子图局部**编号（不同 partition 的同一 idx 是不同张量），
     所以跨 partition 的行不可混算 —— 用 --sg <id> 只算同一子图内的行。
"""

import re
import sys

RE_CREATED = re.compile(
    r"\[och-create\]\s+created\s+idx=(\d+) req_size=(\d+) res=(\S+) actual=(\d+)")
RE_SG = re.compile(r"\[och-create\]\s+created\s+sg=(\d+)\s+idx=(\d+)\s+req_size=(\d+)")


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def triples(dims):
    out = []
    for i in range(len(dims) - 2):
        out.append(("dims[%d..%d]" % (i, i + 2), prod(dims[i:i + 3])))
    return out


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 2
    shapes = {}
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            shapes[int(parts[0])] = [int(x) for x in parts[1].split("x")]
    sg_filter = None
    if "--sg" in sys.argv:
        sg_filter = int(sys.argv[sys.argv.index("--sg") + 1])

    rows = []
    with open(sys.argv[2], encoding="utf-8", errors="replace") as f:
        for no, line in enumerate(f, 1):
            m = RE_SG.search(line)
            if m:
                if sg_filter is not None and int(m.group(1)) != sg_filter:
                    continue
                rows.append((no, int(m.group(2)), int(m.group(3))))
                continue
            m = RE_CREATED.search(line)
            if m:
                rows.append((no, int(m.group(1)), int(m.group(2))))

    if not rows:
        print("没解析到 created 行 —— 日志对吗？")
        return 1

    seen = set()
    print("=== created 行反推：req_size 是哪几个 dims 的乘积 ===")
    print("(idx 是**子图局部**编号；没有 sg= 字段时无法判断是否同一子图，")
    print(" 跨 partition 的行不可混算 —— 下面的结论只对'形状能整除'的行可信)")
    print()
    unexplained = 0
    for no, idx, req in rows:
        key = (idx, req)
        if key in seen:
            continue
        seen.add(key)
        dims = shapes.get(idx)
        if dims is None:
            print("  L%-5d idx=%-4d req=%-8d （无该 idx 的形状，无法反推）" % (no, idx, req))
            unexplained += 1
            continue
        full = prod(dims[1:]) if len(dims) > 1 else prod(dims)
        hits = [(lbl, v) for lbl, v in triples(dims) if v == req]
        print("  L%-5d idx=%-4d shape=%-22s req=%-8d" % (no, idx, "x".join(map(str, dims)), req))
        if full == req:
            print("        ⇒ 等于**除 d0 外的全维乘积** ⇒ dims 排列正确（D1 不成立）")
        if hits:
            for lbl, v in hits:
                a = lbl.split("[")[1].split("..")[0]
                b = lbl.split("..")[1].rstrip("]")
                print("        ⇒ 等于 %s = %s" % (lbl, "x".join(str(x) for x in
                                                         dims[int(a):int(b) + 1])))
        if not hits and full != req:
            others = []
            for oidx, odims in shapes.items():
                if oidx == idx:
                    continue
                for lbl, v in triples(odims):
                    if v == req:
                        others.append("t%d 的 %s（shape=%s）" % (oidx, lbl,
                                                                 "x".join(map(str, odims))))
            if others:
                print("        ⇒ 等于**别的张量**的乘积：%s" % "；".join(sorted(set(others))[:4]))
                print("           ⇒ 提示 index 与 size 可能不同源（D2），但也可能是巧合，需日志确认")
            else:
                print("        ⇒ 既不是全维乘积，也匹配不到任何形状的三元组 ⇒ 更深的错位")
                unexplained += 1
    print()
    print("RESULT: 解析 %d 行（去重后 %d 组），无法解释 %d 组"
          % (len(rows), len(seen), unexplained))
    return 0


if __name__ == "__main__":
    sys.exit(main())