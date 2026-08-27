#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

[[ -r "${PROFILE_FILE}" ]] || die "配置文件不存在或不可读：${PROFILE_FILE}"

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${AGENT_HOST:=0.0.0.0}"
: "${AGENT_PORT:=8010}"
: "${AGENT_CONDA_ENV:=agent_system}"
: "${AGENT_DATA_DIR:=data}"
: "${RAG_BASE_URL:=http://127.0.0.1:8000}"
: "${LOG_DIR:=logs}"

resolve_repo_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s/%s\n' "${PROJECT_ROOT}" "${value#./}"
    fi
}

export AGENT_DATA_DIR="$(resolve_repo_path "${AGENT_DATA_DIR}")"
export DEVICE_STATUS_FILE="$(resolve_repo_path "${DEVICE_STATUS_FILE:-${AGENT_DATA_DIR}/device_status.json}")"
export MAINTENANCE_HISTORY_FILE="$(resolve_repo_path "${MAINTENANCE_HISTORY_FILE:-${AGENT_DATA_DIR}/maintenance_history.json}")"

ABS_LOG_DIR="$(resolve_repo_path "${LOG_DIR}")"
mkdir -p "${ABS_LOG_DIR}"

[[ -f "${DEVICE_STATUS_FILE}" ]] || die "设备状态文件不存在：${DEVICE_STATUS_FILE}"
[[ -f "${MAINTENANCE_HISTORY_FILE}" ]] || die "维修记录文件不存在：${MAINTENANCE_HISTORY_FILE}"

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${AGENT_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "找不到 conda 命令"
    conda run -n "${AGENT_CONDA_ENV}" python -c 'import fastapi, uvicorn' >/dev/null 2>&1 \
        || die "Conda 环境 ${AGENT_CONDA_ENV} 缺少 FastAPI 或 Uvicorn"
    RUN_PREFIX=(conda run --no-capture-output -n "${AGENT_CONDA_ENV}")
fi

cd "${PROJECT_ROOT}"

printf '正在启动 Agent API：http://127.0.0.1:%s\n' "${AGENT_PORT}"
printf '配置文件：%s\n' "${PROFILE_FILE}"
printf '日志目录：%s\n' "${ABS_LOG_DIR}"

exec "${RUN_PREFIX[@]}" python -m agent.api
