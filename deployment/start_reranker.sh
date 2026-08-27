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

: "${RERANKER_MODEL_PATH:=Qwen/Qwen3-Reranker-0.6B}"
: "${RERANKER_DEVICE:=cpu}"
: "${RERANKER_HOST:=0.0.0.0}"
: "${RERANKER_PORT:=6009}"
: "${RERANKER_MAX_DOCUMENTS:=50}"
: "${RERANKER_MAX_QUERY_CHARS:=4096}"
: "${RERANKER_MAX_DOCUMENT_CHARS:=8192}"
: "${RERANKER_MAX_LENGTH:=1024}"
: "${RERANKER_BATCH_SIZE:=4}"
: "${RERANKER_LOG_LEVEL:=INFO}"
: "${RERANKER_CONDA_ENV:=reranker_service}"
: "${HF_HOME:=.cache/huggingface}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${LOG_DIR:=logs}"
: "${RERANKER_LOG_FILE:=reranker.log}"

if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}"
fi
if [[ "${HF_HOME}" != /* ]]; then
    HF_HOME="${PROJECT_ROOT}/${HF_HOME#./}"
    export HF_HOME
fi
mkdir -p "${LOG_DIR}" "${HF_HOME}" || die "无法创建日志目录或 Hugging Face 缓存目录"
LOG_FILE="${LOG_DIR}/${RERANKER_LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

[[ -f "${PROJECT_ROOT}/reranker_service/app.py" ]] \
    || die "找不到 Reranker 服务入口：${PROJECT_ROOT}/reranker_service/app.py"
[[ "${RERANKER_PORT}" =~ ^[0-9]+$ ]] \
    || die "RERANKER_PORT 必须是整数，当前值：${RERANKER_PORT}"
(( RERANKER_PORT >= 1 && RERANKER_PORT <= 65535 )) \
    || die "RERANKER_PORT 必须位于 1～65535，当前值：${RERANKER_PORT}"

if [[ "${RERANKER_DEVICE}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${RERANKER_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "当前未激活 ${RERANKER_CONDA_ENV}，并且找不到 conda 命令"
    conda run -n "${RERANKER_CONDA_ENV}" python -c 'import uvicorn' >/dev/null 2>&1 \
        || die "Conda 环境 ${RERANKER_CONDA_ENV} 中无法导入 uvicorn"
    RUN_PREFIX=(conda run --no-capture-output -n "${RERANKER_CONDA_ENV}")
else
    python -c 'import uvicorn' >/dev/null 2>&1 || die "当前环境无法导入 uvicorn"
fi

cd "${PROJECT_ROOT}"

echo "=================================================="
echo "正在启动 Reranker Service"
echo "项目目录：${PROJECT_ROOT}"
echo "配置文件：${PROFILE_FILE}"
echo "Conda 环境：${RERANKER_CONDA_ENV}"
echo "模型路径：${RERANKER_MODEL_PATH}"
echo "运行设备：${RERANKER_DEVICE}"
echo "监听地址：${RERANKER_HOST}"
echo "服务端口：${RERANKER_PORT}"
echo "服务地址：${RERANKER_BASE_URL:-http://127.0.0.1:${RERANKER_PORT}}"
echo "日志文件：${LOG_FILE}"
echo "最大候选数：${RERANKER_MAX_DOCUMENTS}"
echo "最大模型长度：${RERANKER_MAX_LENGTH}"
echo "推理批量：${RERANKER_BATCH_SIZE}"
echo "=================================================="

exec "${RUN_PREFIX[@]}" python -m uvicorn reranker_service.app:app \
    --host "${RERANKER_HOST}" \
    --port "${RERANKER_PORT}" \
    --workers 1 \
    --log-level "${RERANKER_LOG_LEVEL,,}"
