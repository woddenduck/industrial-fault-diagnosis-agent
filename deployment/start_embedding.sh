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

: "${EMBEDDING_MODEL_PATH:=BAAI/bge-m3}"
: "${EMBEDDING_DEVICE:=cpu}"
: "${EMBEDDING_HOST:=0.0.0.0}"
: "${EMBEDDING_PORT:=6010}"
: "${EMBEDDING_MAX_TEXTS:=32}"
: "${EMBEDDING_MAX_CHARS:=12000}"
: "${EMBEDDING_MAX_TOKENS:=1024}"
: "${EMBEDDING_BATCH_SIZE:=16}"
: "${EMBEDDING_LOG_LEVEL:=INFO}"
: "${EMBEDDING_CONDA_ENV:=embedding_service}"
: "${HF_HOME:=.cache/huggingface}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${LOG_DIR:=logs}"
: "${EMBEDDING_LOG_FILE:=embedding.log}"

if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}"
fi
if [[ "${HF_HOME}" != /* ]]; then
    HF_HOME="${PROJECT_ROOT}/${HF_HOME#./}"
    export HF_HOME
fi
mkdir -p "${LOG_DIR}" "${HF_HOME}" || die "无法创建日志目录或 Hugging Face 缓存目录"
LOG_FILE="${LOG_DIR}/${EMBEDDING_LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

[[ -f "${PROJECT_ROOT}/embedding_service/app.py" ]] \
    || die "找不到 Embedding 服务入口：${PROJECT_ROOT}/embedding_service/app.py"
[[ "${EMBEDDING_PORT}" =~ ^[0-9]+$ ]] \
    || die "EMBEDDING_PORT 必须是整数，当前值：${EMBEDDING_PORT}"
(( EMBEDDING_PORT >= 1 && EMBEDDING_PORT <= 65535 )) \
    || die "EMBEDDING_PORT 必须位于 1～65535，当前值：${EMBEDDING_PORT}"

if [[ "${EMBEDDING_DEVICE}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${EMBEDDING_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "当前未激活 ${EMBEDDING_CONDA_ENV}，并且找不到 conda 命令"
    conda run -n "${EMBEDDING_CONDA_ENV}" python -c 'import uvicorn' >/dev/null 2>&1 \
        || die "Conda 环境 ${EMBEDDING_CONDA_ENV} 中无法导入 uvicorn"
    RUN_PREFIX=(conda run --no-capture-output -n "${EMBEDDING_CONDA_ENV}")
else
    python -c 'import uvicorn' >/dev/null 2>&1 || die "当前环境无法导入 uvicorn"
fi

cd "${PROJECT_ROOT}"

echo "=================================================="
echo "正在启动 Embedding Service"
echo "项目目录：${PROJECT_ROOT}"
echo "配置文件：${PROFILE_FILE}"
echo "Conda 环境：${EMBEDDING_CONDA_ENV}"
echo "模型路径：${EMBEDDING_MODEL_PATH}"
echo "运行设备：${EMBEDDING_DEVICE}"
echo "监听地址：${EMBEDDING_HOST}"
echo "服务端口：${EMBEDDING_PORT}"
echo "服务地址：${EMBEDDING_BASE_URL:-http://127.0.0.1:${EMBEDDING_PORT}}"
echo "日志文件：${LOG_FILE}"
echo "最大文本数：${EMBEDDING_MAX_TEXTS}"
echo "最大字符数：${EMBEDDING_MAX_CHARS}"
echo "最大 Token：${EMBEDDING_MAX_TOKENS}"
echo "推理批量：${EMBEDDING_BATCH_SIZE}"
echo "=================================================="

exec "${RUN_PREFIX[@]}" python -m uvicorn embedding_service.app:app \
    --host "${EMBEDDING_HOST}" \
    --port "${EMBEDDING_PORT}" \
    --workers 1 \
    --log-level "${EMBEDDING_LOG_LEVEL,,}"
