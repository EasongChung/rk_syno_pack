#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等 oca 主机恢复后，自动补上「读取回复 + 重新派发」。

## 为什么需要它

2026-10-17 傍晚 oca 主机 `192.168.9.10` 整个下线（ping 100% 丢包，22/80/4096 全部
TCP 超时；本机到 NAS 正常 → 不是本地断网）。期间：
- `send-file` 与 `messages` 全部 `URLError … timed out`；
- art13 监视器自 17:24 基线后再无产出（挂载视图会随主机离线而冻结）。

这类等待不适合靠人肉轮询，所以本脚本：
1. 每 `--poll` 秒探一次 TCP（只探，不发请求，避免把远端打爆）；
2. 一旦端口通，**先把离线期间的回复抓下来**（这是最容易被漏掉的一步）；
3. 再补发派发（文件 + 消息）；
4. 每一步都落日志；**用 marker 文件保证幂等** —— 主机抖一下重新上线不会重复派发。

## ⚠️ 已知限制（实测踩到，必须写下来）

本脚本**扛不住本机重启**。2026-10-17 傍晚 oca 主机下线后我起了本脚本，
结果同一时段**我自己的容器也重启了** —— 脚本进程连同它的输出文件一起消失，
marker 也没写，于是「已挂自动恢复」这句话在一夜之间变成假的。
这类「以为挂上了、其实早就没了」最危险：它让人**不再手动去查**。

所以：

- 看到本脚本起不来，**不要**理解为「远端还没恢复」，先确认进程是否还在
  （`ps aux | grep resume_after_outage`）；
- 状态文件（marker / 日志）写在**工程目录内**而不是临时目录，正是为了让重启后
  「到底派发过没有」仍然可查；
- 远端恢复后如果本机也重启过，**以 oca 会话里的实际消息为准**，不要以本脚本的日志为准。

- `state/dispatch-ocfix.done`：本次派发完成后写入，重跑不会重发；
- `state/reply-after-outage-<ts>.txt`：每次成功抓回复都存一份，不覆盖历史。

## 用法

    python3 tools/oca/resume_after_outage.py --timeout 7200 --poll 30
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SKILL = Path("/root/.config/opencode/skills/oca")
PROJ = Path("/root/workspace/oes-teflon")
SESSION = "ses_ec086ff9affezp0FVI1ImVY7RA"
STATE = PROJ / "tools" / "oca" / "state"
REGISTRY = Path.home() / ".config" / "opencode" / "remote-instances.json"

MSG = PROJ / "tools" / "oca" / "msg-ocfix-build.txt"
FILES = [
    (PROJ / "build-recipe" / "patch-etnaviv-outchannel-fix.py",
     "附件1/2：新修复补丁 patch-etnaviv-outchannel-fix.py（write_core_interleaved 的 "
     "out_channel 映射修正 + 越界钳位，自带精确计数门 + 10 个反例）。落进 build-recipe/。"),
    (PROJ / "build-recipe" / "build-teflon.sh",
     "附件2/2：build-teflon.sh 新增 TEFLON_OCFIX=1 步骤 3g；其余六个开关全部默认关，"
     "构建流程与产物不变。落进 build-recipe/。"),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def host_port():
    """从实例注册表取 host:port（不回显任何凭据）。"""
    data = json.load(open(REGISTRY, encoding="utf-8"))
    cfg = data["instances"]["oca"]
    h = cfg["host"].split("://", 1)[-1].rstrip("/")
    if ":" in h:
        hostname, port = h.rsplit(":", 1)
        return hostname, int(port)
    return h, 80


def tcp_ok(hostname, port, timeout=6):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((hostname, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def api(args, timeout=300):
    return subprocess.run(["python3", "scripts/oc_api.py", "-p", "oca"] + args,
                          cwd=SKILL, capture_output=True, text=True, timeout=timeout)


def fetch_reply():
    """抓最近回复并存档 —— 离线期间最容易漏的一步。"""
    r = api(["messages", SESSION, "--limit", "4"])
    if r.returncode != 0:
        log(f"抓回复失败 rc={r.returncode}: {(r.stderr or '').strip()[:120]}")
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = PROJ / "tools" / "oca" / f"reply-after-outage-{ts}.txt"
    out.write_text(r.stdout, encoding="utf-8")
    log(f"回复已存档: {out.name}（{len(r.stdout)} 字节）")
    return r.stdout


def dispatch():
    """补发：两个文件 + 消息。任一步失败都返回 False，不写 marker。"""
    for local, text in FILES:
        if not local.exists():
            log(f"缺文件，跳过: {local}")
            return False
        r = api(["send-file", SESSION, str(local),
                 "--default-dest", "D:/clawtemp/o2tf-src",
                 "--text", text, "--no-watch-hint"], timeout=600)
        if r.returncode != 0:
            log(f"send-file 失败 {local.name} rc={r.returncode}: "
                f"{(r.stderr or '').strip()[:140]}")
            return False
        log(f"已发送 {local.name}")

    r = api(["prompt", SESSION, MSG.read_text(encoding="utf-8")], timeout=300)
    if r.returncode != 0:
        log(f"消息发送失败 rc={r.returncode}: {(r.stderr or '').strip()[:140]}")
        return False
    mid = ""
    for line in r.stdout.splitlines():
        if '"id"' in line:
            mid = line.strip().strip(",")
            break
    log(f"消息已发送 {mid}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="忽略 marker 重新派发")
    args = ap.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    marker = STATE / "dispatch-ocfix.done"

    if marker.exists() and not args.force:
        log(f"已派发过（{marker.name} 存在），直接退出。要重来加 --force")
        return 0

    hostname, port = host_port()
    log(f"探测 oca 主机 {hostname}:{port}，间隔 {args.poll}s，上限 {args.timeout}s")

    deadline = time.time() + args.timeout
    waited = 0
    while time.time() < deadline:
        if tcp_ok(hostname, port):
            log(f"主机已恢复（等待 {waited}s）")
            # 给服务一点时间完成启动，否则第一次请求可能又超时
            time.sleep(10)
            fetch_reply()
            if dispatch():
                marker.write_text(datetime.now().isoformat(timespec="seconds") + "\n",
                                  encoding="utf-8")
                log("派发完成，已写 marker")
                log("接下来请另起 watch 盯 oca 的回复")
                return 0
            log("派发失败，未写 marker；60s 后重试一次")
            time.sleep(60)
        else:
            waited += args.poll
            if waited % 300 == 0:
                log(f"仍不可达（已等 {waited}s）")
            time.sleep(args.poll)

    log(f"超时：{args.timeout}s 内主机始终不可达，未派发（未写 marker，可直接重跑本脚本）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
