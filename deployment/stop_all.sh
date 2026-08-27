#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

if [[ -r "${PROFILE_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(sed 's/\r$//' "${PROFILE_FILE}")
    set +a
fi

: "${RUN_DIR:=deployment/run}"
: "${STOP_TIMEOUT:=20}"

if [[ "${RUN_DIR}" = /* ]]; then
    ABS_RUN_DIR="${RUN_DIR}"
else
    ABS_RUN_DIR="${PROJECT_ROOT}/${RUN_DIR#./}"
fi

stop_service() {
    local key="$1"
    local name="$2"
    local pid_file="${ABS_RUN_DIR}/${key}.pid"

    if [[ ! -f "${pid_file}" ]]; then
        printf '[跳过] %-16s 没有 PID 文件\n' "${name}"
        return
    fi

    local pid
    pid="$(tr -d '[:space:]' <"${pid_file}")"

    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        printf '[警告] %-16s PID 文件无效：%s\n' "${name}" "${pid_file}" >&2
        rm -f "${pid_file}"
        return
    fi

    if ! kill -0 "${pid}" >/dev/null 2>&1; then
        printf '[清理] %-16s 进程已退出\n' "${name}"
        rm -f "${pid_file}"
        return
    fi

    printf '[停止] %-16s PID=%s\n' "${name}" "${pid}"
    kill -TERM -- "-${pid}" >/dev/null 2>&1 \
        || kill -TERM "${pid}" >/dev/null 2>&1 \
        || true

    local elapsed=0
    while kill -0 "${pid}" >/dev/null 2>&1 && (( elapsed < STOP_TIMEOUT )); do
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "${pid}" >/dev/null 2>&1; then
        printf '[强制] %-16s 超时后强制结束\n' "${name}" >&2
        kill -KILL -- "-${pid}" >/dev/null 2>&1 \
            || kill -KILL "${pid}" >/dev/null 2>&1 \
            || true
    fi

    rm -f "${pid_file}"
}

printf '按照依赖关系的逆序停止服务。\n'
stop_service agent 'Agent API'
stop_service industrial_rag 'Industrial RAG'
stop_service gateway 'Gateway'
stop_service reranker 'Reranker'
stop_service embedding 'Embedding'
stop_service llm 'LLM'
printf '停止流程完成。\n'
