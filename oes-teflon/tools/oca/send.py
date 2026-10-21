#!/usr/bin/env python3
# 把 msg.txt 作为 oca 会话消息发出。用 subprocess 列表传参，不经 shell 词法，
# 避免中文/反引号/引号被 shell 解析。
import subprocess, sys
from pathlib import Path

SKILL = "/root/.config/opencode/skills/oca"
SESSION = "ses_ec086ff9affezp0FVI1ImVY7RA"
text = Path("/tmp/opencode/oca4/msg.txt").read_text(encoding="utf-8")

cmd = ["python3", "scripts/oc_api.py", "-p", "oca", "prompt", SESSION, text]
p = subprocess.run(cmd, cwd=SKILL, capture_output=True, text=True, timeout=300)
sys.stdout.write(p.stdout)
sys.stderr.write(p.stderr)
sys.exit(p.returncode)
