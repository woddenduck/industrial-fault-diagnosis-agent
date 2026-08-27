#!/usr/bin/env bash

set -euo pipefail

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:6006}"
export VLLM_TIMEOUT="${VLLM_TIMEOUT:-120}"

HOST="${GATEWAY_HOST:-0.0.0.0}"
PORT="${GATEWAY_PORT:-6008}"
LOG_LEVEL="${GATEWAY_LOG_LEVEL:-info}"

# 获取脚本所在目录，保证从其他目录执行时也能正确找到 app。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=================================================="
echo "正在启动 LLM Gateway"
echo "工作目录：${SCRIPT_DIR}"
echo "监听地址：${HOST}"
echo "服务端口：${PORT}"
echo "日志级别：${LOG_LEVEL}"
echo "应用入口：app.main:app"
echo "=================================================="

if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "错误：当前 Python 环境缺少 FastAPI 或 Uvicorn"
    echo "请确认已经激活 llm_gateway 环境并安装 requirements.txt"
    exit 1
fi

exec python -m uvicorn app.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}" \
    --reload
