#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -L "$0" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$(readlink "$0")")" && pwd)"
fi
STUDIO_DIR="$SCRIPT_DIR"
VENV_PY="$STUDIO_DIR/../seeg-workbench/.venv/bin/python"
PORT=8780

cd "$STUDIO_DIR"
# 仅清理监听在目标端口的旧版服务端进程（-sTCP:LISTEN 避免误杀浏览器客户端进程）
lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
echo "=========================================="
echo "  Judgement Bias SEEG 标准化分析工作台"
echo "  访问地址: http://127.0.0.1:$PORT"
echo "=========================================="

(sleep 1.8 && open "http://127.0.0.1:$PORT") &
exec "$VENV_PY" -m uvicorn backend.app.main:app --host 127.0.0.1 --port "$PORT"
