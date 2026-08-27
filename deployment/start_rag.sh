#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RAG_ROOT="${PROJECT_ROOT}/industrial-rag"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

[[ -r "${PROFILE_FILE}" ]] || die "配置文件不存在或不可读：${PROFILE_FILE}"
[[ -f "${RAG_ROOT}/app/main.py" ]] || die "找不到 RAG 服务入口"

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${RAG_HOST:=0.0.0.0}"
: "${RAG_PORT:=8000}"
: "${RAG_CONDA_ENV:=rag_system}"
: "${RAG_DATA_DIR:=industrial-rag/data}"
: "${RAG_MODEL_DIR:=industrial-rag/models}"
: "${RAG_UPLOAD_DIR:=industrial-rag/data/uploads}"
: "${RAG_VECTOR_DB_DIR:=industrial-rag/data/vector_store}"
: "${RAG_LOG_DIR:=industrial-rag/logs}"

resolve_repo_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s/%s\n' "${PROJECT_ROOT}" "${value#./}"
    fi
}

export DATA_DIR="$(resolve_repo_path "${RAG_DATA_DIR}")"
export MODEL_DIR="$(resolve_repo_path "${RAG_MODEL_DIR}")"
export UPLOAD_DIR="$(resolve_repo_path "${RAG_UPLOAD_DIR}")"
export VECTOR_DB_DIR="$(resolve_repo_path "${RAG_VECTOR_DB_DIR}")"
export LOG_DIR="$(resolve_repo_path "${RAG_LOG_DIR}")"
export RAG_CONFIG_FILE="${PROFILE_FILE}"

# 空目录只负责承载运行数据，知识库仍需通过接口正式构建。
mkdir -p \
    "${DATA_DIR}" \
    "${MODEL_DIR}" \
    "${UPLOAD_DIR}" \
    "${VECTOR_DB_DIR}" \
    "${LOG_DIR}"

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${RAG_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "找不到 conda 命令"
    conda run -n "${RAG_CONDA_ENV}" python -c 'import fastapi, uvicorn' >/dev/null 2>&1 \
        || die "Conda 环境 ${RAG_CONDA_ENV} 缺少 FastAPI 或 Uvicorn"
    RUN_PREFIX=(conda run --no-capture-output -n "${RAG_CONDA_ENV}")
fi

cd "${RAG_ROOT}"

printf '正在启动 Industrial RAG：http://127.0.0.1:%s\n' "${RAG_PORT}"
printf '配置文件：%s\n' "${PROFILE_FILE}"

exec "${RUN_PREFIX[@]}" python -m uvicorn app.main:app \
    --host "${RAG_HOST}" \
    --port "${RAG_PORT}"
