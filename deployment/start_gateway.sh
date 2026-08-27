#!/usr/bin/env bash

set -Eeuo pipefail


die() {
    echo "错误：$*" >&2
    exit 1
}


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

[[ -f "${PROFILE_FILE}" ]] || die "配置文件不存在：${PROFILE_FILE}"
[[ -r "${PROFILE_FILE}" ]] || die "配置文件不可读取：${PROFILE_FILE}"

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${GATEWAY_HOST:=0.0.0.0}"
: "${GATEWAY_PORT:=6008}"
: "${GATEWAY_CONDA_ENV:=llm_gateway}"
: "${LLM_BASE_URL:=${VLLM_BASE_URL:-http://127.0.0.1:6006}}"
: "${EMBEDDING_BASE_URL:=http://127.0.0.1:6010}"
: "${RERANKER_BASE_URL:=http://127.0.0.1:6009}"
: "${LOG_LEVEL:=INFO}"
: "${LOG_DIR:=logs}"
: "${GATEWAY_LOG_FILE:=gateway.log}"

if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}"
fi
mkdir -p "${LOG_DIR}" || die "无法创建日志目录：${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${GATEWAY_LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

[[ -f "${PROJECT_ROOT}/gateway/app/main.py" ]] \
    || die "找不到 Gateway 入口：${PROJECT_ROOT}/gateway/app/main.py"
[[ "${GATEWAY_PORT}" =~ ^[0-9]+$ ]] \
    || die "GATEWAY_PORT 必须是整数，当前值：${GATEWAY_PORT}"
(( GATEWAY_PORT >= 1 && GATEWAY_PORT <= 65535 )) \
    || die "GATEWAY_PORT 必须位于 1～65535，当前值：${GATEWAY_PORT}"

validate_http_url() {
    local name="$1"
    local value="$2"
    case "${value}" in
        http://*|https://*) ;;
        *) die "${name} 必须是 HTTP(S) 地址，当前值：${value}" ;;
    esac
}

validate_http_url "LLM_BASE_URL" "${LLM_BASE_URL}"
validate_http_url "EMBEDDING_BASE_URL" "${EMBEDDING_BASE_URL}"
validate_http_url "RERANKER_BASE_URL" "${RERANKER_BASE_URL}"

# 为既有 Gateway 配置导出 VLLM_* 兼容变量。
export VLLM_BASE_URL="${LLM_BASE_URL}"
export VLLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-120}"

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${GATEWAY_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "当前未激活 ${GATEWAY_CONDA_ENV}，并且找不到 conda 命令"
    conda run -n "${GATEWAY_CONDA_ENV}" python -c 'import uvicorn' >/dev/null 2>&1 \
        || die "Conda 环境 ${GATEWAY_CONDA_ENV} 中无法导入 uvicorn"
    RUN_PREFIX=(conda run --no-capture-output -n "${GATEWAY_CONDA_ENV}")
else
    python -c 'import uvicorn' >/dev/null 2>&1 || die "当前环境无法导入 uvicorn"
fi

cd "${PROJECT_ROOT}/gateway"

echo "=================================================="
echo "正在启动 Gateway"
echo "项目目录：${PROJECT_ROOT}"
echo "工作目录：$(pwd)"
echo "配置文件：${PROFILE_FILE}"
echo "Conda 环境：${GATEWAY_CONDA_ENV}"
echo "监听地址：${GATEWAY_HOST}"
echo "服务端口：${GATEWAY_PORT}"
echo "Gateway 地址：http://127.0.0.1:${GATEWAY_PORT}"
echo "LLM 地址：${LLM_BASE_URL}"
echo "Embedding 地址：${EMBEDDING_BASE_URL}"
echo "Reranker 地址：${RERANKER_BASE_URL}"
echo "日志文件：${LOG_FILE}"
echo "=================================================="

exec "${RUN_PREFIX[@]}" python -m uvicorn app.main:app \
    --host "${GATEWAY_HOST}" \
    --port "${GATEWAY_PORT}" \
    --log-level "${LOG_LEVEL,,}"
