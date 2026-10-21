#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diff_nn_desc.py —— 解析 [och-nn] 描述符 dump，比较「挂死作业」与「正常作业」

用法:
    python3 diff_nn_desc.py <证据目录> [挂死模型名] [正常模型名]

输入：<证据目录>/nndump-<模型名>.log 里的 `[och-nn] ...` 行（由
build-recipe/patch-etnaviv-nn-desc-dump.py 注入，见报告 28 §四）。

输出：
  1) 每个模型逐作业的几何摘要（哪个作业的参数异常）
  2) 指定两模型之间**逐字段 diff**（人类可读字段）
  3) 两份描述符**原始字节**的差异位置（位级定位用）

⚠️ 关于「原始字节 → 字段名」的映射：
   `struct etna_nn_params` 是**位域**结构体（FIELD(field, bits) 宏），
   位域在内存里的排布由编译器决定（GCC 的实现细节），**不能**在本机离线
   还原成字段名。因此本工具只做：
     · 人类可读字段的 diff（补丁里已直接打印出来）  ⇒ 能直接看出哪个参数不同
     · 原始字节的差异位置                          ⇒ 精确到位
   两者合起来即可定位：先看哪个字段不同，再看它落在原始字节的哪一段。
"""

import os
import re
import sys

RE_LINE = re.compile(r'\[och-nn\]\s+(.*)')
RE_KV = re.compile(r'([a-z_]+)=([^\s]+)')


def parse_log(path):
    jobs = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RE_LINE.search(line)
            if not m:
                continue
            body = m.group(1)
            raw = ''
            if 'raw=' in body:
                body, raw = body.split('raw=', 1)
            kv = dict(RE_KV.findall(body))
            kv['raw'] = raw.strip()
            jobs.append(kv)
    return jobs


def fmt(v):
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return "%.6g" % f
    except Exception:
        return v


def diff_fields(a, b):
    keys = []
    for k in a:
        if k != 'raw' and k not in keys:
            keys.append(k)
    for k in b:
        if k != 'raw' and k not in keys:
            keys.append(k)
    out = []
    for k in keys:
        va, vb = a.get(k, '<无>'), b.get(k, '<无>')
        if va != vb:
            out.append((k, va, vb))
    return out


def diff_raw(a, b):
    ra = a.get('raw', '')
    rb = b.get('raw', '')
    if not ra or not rb:
        return None
    n = min(len(ra), len(rb)) // 2
    ba = bytes.fromhex(ra[:n * 2])
    bb = bytes.fromhex(rb[:n * 2])
    if len(ba) != len(bb):
        return [("长度不同", len(ba), len(bb))]
    return [(i, ba[i], bb[i]) for i in range(len(ba)) if ba[i] != bb[i]]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    bad = sys.argv[2] if len(sys.argv) > 2 else 'bis-s09p03'
    good = sys.argv[3] if len(sys.argv) > 3 else 'iso-seg04'

    def load(name):
        p = os.path.join(d, "nndump-%s.log" % name)
        if not os.path.isfile(p):
            return None
        return parse_log(p)

    print("=" * 100)
    print("NN core 描述符比对")
    print("=" * 100)
    models = {}
    for n in (bad, good):
        jobs = load(n)
        models[n] = jobs
        if jobs is None:
            print("  [%s] 缺日志: %s/nndump-%s.log" % (n, d, n))
            continue
        print("\n【%s】共 %d 个作业的描述符 dump：" % (n, len(jobs)))
        for i, j in enumerate(jobs):
            print("  #%-2d in=%-14s out=%-14s kw=%-3s dw=%-2s pw=%-2s sb=%-3s tile=%-9s kpc=%-4s "
                  "img_cache=%-9s coef_cache=%-9s conv_scale=%-12s shift=%-4s"
                  % (i, j.get('in', '?'), j.get('out', '?'), j.get('kw', '?'), j.get('dw', '?'),
                     j.get('pw', '?'), j.get('superblocks', '?'), j.get('tile', '?'), j.get('kpc', '?'),
                     j.get('img_cache', '?'), j.get('coef_cache', '?'), j.get('conv_scale', '?'),
                     j.get('shift', '?')))

    if not models.get(bad) or not models.get(good):
        print("\n（缺一侧日志，无法 diff）")
        return 1

    a, b = models[bad][0], models[good][0]
    print("\n" + "=" * 100)
    print("逐字段 diff：%s 的第 1 个作业  vs  %s 的第 1 个作业" % (bad, good))
    print("=" * 100)
    df = diff_fields(a, b)
    if not df:
        print("  （人类可读字段完全相同 ⇒ 差异在未打印的位域里，见下面的原始字节 diff）")
    for k, va, vb in df:
        print("  %-14s %-22s → %-22s" % (k, fmt(va), fmt(vb)))

    dr = diff_raw(a, b)
    print("\n原始字节 diff（%d 字节的描述符）：" % (len(a.get('raw', '')) // 2))
    if dr is None:
        print("  （无 raw 数据）")
    elif not dr:
        print("  ✅ 描述符**逐字节完全相同** ⇒ 差异不在描述符里（去查系数/coeff 缓存与地址）")
    else:
        print("  差异 %d 处：" % len(dr))
        for off, x, y in dr:
            print("    byte[%3d]  0x%02x → 0x%02x   (bit %d..%d)" % (off, x, y, off * 8, off * 8 + 7))
    return 0


if __name__ == '__main__':
    sys.exit(main())
