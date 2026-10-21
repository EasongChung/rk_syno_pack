#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把纯 Python 依赖装进项目内的 `tools/vendor/`（不需要 pip）。

## 为什么需要这个脚本

本机（以及设备）的 Python 是被裁剪过的：**没有 pip、没有 apt、没有 flatc**。
`ensurepip` 也失败（系统 Python 缺 `ensurepip` 所需的组件）。

而模型手术必须能**可靠地解析/重建 TFLite**。手写整个 schema 的序列化器风险太大
（一个字段错位就会静默产出坏模型），所以用官方生成好的绑定：

| 包 | 作用 | 为什么可信 |
|---|---|---|
| `flatbuffers` | FlatBuffers 运行时 | 官方，纯 Python，26 KB |
| `tflite` | TFLite schema 的**生成代码**（`tflite.Model` 等） | 与 `flatc` 从 `schema.fbs` 生成的同源 |

## 做法

wheel 本质是 zip ⇒ **直接下载 + 解包**，不经过 pip：

1. 查 PyPI JSON API 拿 `-none-any.whl` 的 URL
2. 下载到 `tools/vendor/_wheels/`（留档，便于离线复现）
3. 解包到 `tools/vendor/`

## 用法

    python3 tools/bootstrap_vendor.py          # 装（已存在则跳过）
    python3 tools/bootstrap_vendor.py --force  # 重装

装完后任何脚本这样引用：

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
WHEELS = os.path.join(VENDOR, "_wheels")

# 只固定这两个；版本不锁死（写进 vendor 的 MANIFEST 里便于追溯）
PACKAGES = ("flatbuffers", "tflite")


def pypi_wheel_url(pkg):
    """返回该包最新版的 (version, url, filename)——只挑纯 Python wheel。"""
    with urllib.request.urlopen("https://pypi.org/pypi/%s/json" % pkg, timeout=60) as r:
        j = json.load(r)
    ver = j["info"]["version"]
    for u in j["urls"]:
        if u["filename"].endswith("-none-any.whl"):
            return ver, u["url"], u["filename"]
    raise SystemExit("ERROR: %s 没有 pure-python wheel，本脚本不处理" % pkg)


def install(pkg, force=False):
    ver, url, fname = pypi_wheel_url(pkg)
    top = pkg.replace("-", "_")
    target = os.path.join(VENDOR, top)
    if os.path.isdir(target) and not force:
        print("  [skip] %-12s 已存在（%s）" % (pkg, target))
        return ver, fname, False

    os.makedirs(WHEELS, exist_ok=True)
    dest = os.path.join(WHEELS, fname)
    if not os.path.isfile(dest):
        print("  [get ] %-12s %s" % (pkg, fname))
        # 用 curl 而非 urllib：这里不做任何校验拼接，curl 行为更可预期
        rc = os.system("curl -fsSL -o '%s' '%s'" % (dest, url))
        if rc != 0 or not os.path.isfile(dest):
            raise SystemExit("ERROR: 下载失败 %s" % url)
    else:
        print("  [cach] %-12s %s" % (pkg, fname))

    os.makedirs(VENDOR, exist_ok=True)
    with zipfile.ZipFile(dest) as z:
        names = z.namelist()
        if any(".." in n for n in names):          # 基本卫生检查
            raise SystemExit("ERROR: wheel 含可疑路径")
        z.extractall(VENDOR)
    print("  [ok  ] %-12s -> %s（%d 个条目）" % (pkg, target, len(names)))
    return ver, fname, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("=== vendors 安装到 %s ===" % VENDOR)
    rows = []
    for pkg in PACKAGES:
        ver, fname, did = install(pkg, args.force)
        rows.append((pkg, ver, fname, "installed" if did else "already present"))

    man = os.path.join(VENDOR, "MANIFEST.tsv")
    with open(man, "w", encoding="utf-8") as f:
        f.write("# package\tversion\twheel\tstatus\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("  清单写入 %s" % man)

    # 自证：装完必须能导入（不问“装没装”，问“能不能用”）
    print()
    print("=== 自证：导入测试 ===")
    sys.path.insert(0, VENDOR)
    ok = 0
    try:
        import flatbuffers
        print("  ✅ flatbuffers %s" % getattr(flatbuffers, "__version__", "?"))
        ok += 1
    except Exception as e:
        print("  ❌ flatbuffers: %s" % e)
    try:
        from tflite.Model import Model            # noqa: F401
        from tflite.Tensor import Tensor          # noqa: F401
        from tflite.Operator import Operator      # noqa: F401
        print("  ✅ tflite.Model / Tensor / Operator 可导入")
        ok += 1
    except Exception as e:
        print("  ❌ tflite: %s" % e)
    print()
    return 0 if ok == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
