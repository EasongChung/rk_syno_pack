#!/usr/bin/env python3
# 把指定文本文件作为 oca 会话消息发出。用 subprocess 列表传参，不经 shell 词法，
# 避免中文/反引号/引号被 shell 解析。
# 用法：python3 tools/oca/send_msg.py <消息文件路径>
import subprocess, sys
from pathlib import Path

SKILL = "/root/.config/opencode/skills/oca"
# 默认会话；oca 于 2026-10-20 切到新会话做 244-depthwise 零设备排查，
# 因此允许用第二个参数覆盖会话 ID（AGENTS.md 的 watch 纪律要求「发消息与挂 watch 同轮」）。
SESSION = "ses_ec086ff9affezp0FVI1ImVY7RA"
if len(sys.argv) > 2:
    SESSION = sys.argv[2]
msgfile = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/workspace/oes-teflon/tools/oca/msg-diag-build.txt"
text = Path(msgfile).read_text(encoding="utf-8")
print(f"==> 发送 {msgfile} ({len(text)} 字符) → session {SESSION}")

p = subprocess.run(["python3", "scripts/oc_api.py", "-p", "oca", "prompt", SESSION, text],
                   cwd=SKILL, capture_output=True, text=True, timeout=300)
sys.stdout.write(p.stdout)
sys.stderr.write(p.stderr)
sys.exit(p.returncode)