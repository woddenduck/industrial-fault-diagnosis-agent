#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

die() {
    printf '[错误] %s\n' "$*" >&2
    exit 1
}

info() {
    printf '[信息] %s\n' "$*"
}

warn() {
    printf '[警告] %s\n' "$*" >&2
}

usage() {
    printf '%s\n' \
        '用法：' \
        '  bash deployment/start_all.sh' \
        '  bash deployment/start_all.sh --validate' \
        '  bash deployment/start_all.sh --health-only'
}

MODE="${1:-start}"
case "${MODE}" in
    start|--validate|--health-only) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

[[ -r "${PROFILE_FILE}" ]] || die "配置文件不存在或不可读：${PROFILE_FILE}"

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${LLM_PORT:=6006}"
: "${EMBEDDING_PORT:=6010}"
: "${RERANKER_PORT:=6009}"
: "${GATEWAY_PORT:=6008}"
: "${RAG_PORT:=8000}"
: "${AGENT_PORT:=8010}"
: "${LLM_BASE_URL:=http://127.0.0.1:${LLM_PORT}}"
: "${EMBEDDING_BASE_URL:=http://127.0.0.1:${EMBEDDING_PORT}}"
: "${RERANKER_BASE_URL:=http://127.0.0.1:${RERANKER_PORT}}"
: "${GATEWAY_BASE_URL:=http://127.0.0.1:${GATEWAY_PORT}}"
: "${RAG_BASE_URL:=http://127.0.0.1:${RAG_PORT}}"
: "${AGENT_BASE_URL:=http://127.0.0.1:${AGENT_PORT}}"
: "${LOG_DIR:=logs}"
: "${RUN_DIR:=deployment/run}"

resolve_repo_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s/%s\n' "${PROJECT_ROOT}" "${value#./}"
    fi
}

ABS_LOG_DIR="$(resolve_repo_path "${LOG_DIR}")"
ABS_RUN_DIR="$(resolve_repo_path "${RUN_DIR}")"

validate_port() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} 必须是整数：${value}"
    (( value >= 1 && value <= 65535 )) || die "${name} 超出有效范围：${value}"
}

declare -A USED_PORTS=()
for pair in \
    "LLM:${LLM_PORT}" \
    "Embedding:${EMBEDDING_PORT}" \
    "Reranker:${RERANKER_PORT}" \
    "Gateway:${GATEWAY_PORT}" \
    "RAG:${RAG_PORT}" \
    "Agent:${AGENT_PORT}"; do
    service="${pair%%:*}"
    port="${pair##*:}"
    validate_port "${service}_PORT" "${port}"
    [[ -z "${USED_PORTS[$port]:-}" ]] \
        || die "端口 ${port} 同时分配给 ${USED_PORTS[$port]} 和 ${service}"
    USED_PORTS[$port]="${service}"
done

for script in \
    start_llm.sh \
    start_embedding.sh \
    start_reranker.sh \
    start_gateway.sh \
    start_rag.sh \
    start_agent.sh; do
    [[ -r "${SCRIPT_DIR}/${script}" ]] || die "缺少启动脚本：${script}"
done

if [[ "${MODE}" == "--health-only" ]]; then
    exec env PROFILE_FILE="${PROFILE_FILE}" bash "${SCRIPT_DIR}/health_check.sh"
fi

printf '%s\n' \
    '============================================================' \
    '工业故障诊断 Agent｜统一启动检查' \
    "配置文件：${PROFILE_FILE}" \
    '============================================================'

printf '%-18s %s\n' 'LLM' "${LLM_BASE_URL}"
printf '%-18s %s\n' 'Embedding' "${EMBEDDING_BASE_URL}"
printf '%-18s %s\n' 'Reranker' "${RERANKER_BASE_URL}"
printf '%-18s %s\n' 'Gateway' "${GATEWAY_BASE_URL}"
printf '%-18s %s\n' 'Industrial RAG' "${RAG_BASE_URL}"
printf '%-18s %s\n' 'Agent API' "${AGENT_BASE_URL}"

if [[ "${MODE}" == "--validate" ]]; then
    printf '[通过] 配置、端口和启动脚本静态检查通过，未启动服务。\n'
    exit 0
fi

command -v curl >/dev/null 2>&1 || die "找不到 curl 命令"
command -v setsid >/dev/null 2>&1 || die "找不到 setsid 命令"

mkdir -p "${ABS_LOG_DIR}" "${ABS_RUN_DIR}"

port_in_use() {
    local port="$1"
    local python_cmd="python3"
    command -v python3 >/dev/null 2>&1 || python_cmd="python"

    "${python_cmd}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.3)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

for pair in \
    "LLM:${LLM_PORT}" \
    "Embedding:${EMBEDDING_PORT}" \
    "Reranker:${RERANKER_PORT}" \
    "Gateway:${GATEWAY_PORT}" \
    "RAG:${RAG_PORT}" \
    "Agent:${AGENT_PORT}"; do
    service="${pair%%:*}"
    port="${pair##*:}"
    port_in_use "${port}" && die "${service} 端口 ${port} 已被占用"
done

http_ready() {
    curl --silent --show-error --fail --max-time 5 "$1" >/dev/null 2>&1
}

wait_for_service() {
    local name="$1"
    local url="$2"
    local timeout="$3"
    local pid="$4"
    local log_file="$5"
    local elapsed=0

    while (( elapsed < timeout )); do
        if http_ready "${url}"; then
            printf '[就绪] %-16s %s\n' "${name}" "${url}"
            return 0
        fi

        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            warn "${name} 在健康检查通过前退出，日志末尾如下："
            tail -n 20 "${log_file}" >&2 2>/dev/null || true
            return 1
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    warn "${name} 在 ${timeout} 秒内未就绪：${url}"
    tail -n 20 "${log_file}" >&2 2>/dev/null || true
    return 1
}

launch() {
    local key="$1"
    local name="$2"
    local script="$3"
    local health_url="$4"
    local timeout="$5"
    local log_file="${ABS_LOG_DIR}/${key}.startup.log"
    local pid_file="${ABS_RUN_DIR}/${key}.pid"

    info "正在启动 ${name}"
    setsid env PROFILE_FILE="${PROFILE_FILE}" bash "${SCRIPT_DIR}/${script}" \
        >"${log_file}" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" >"${pid_file}"

    wait_for_service "${name}" "${health_url}" "${timeout}" "${pid}" "${log_file}"
}

launch llm 'LLM' start_llm.sh \
    "${LLM_BASE_URL%/}/v1/models" "${LLM_STARTUP_TIMEOUT:-300}"
launch embedding 'Embedding' start_embedding.sh \
    "${EMBEDDING_BASE_URL%/}/health" "${EMBEDDING_STARTUP_TIMEOUT:-180}"

if ! launch reranker 'Reranker' start_reranker.sh \
    "${RERANKER_BASE_URL%/}/health" "${RERANKER_STARTUP_TIMEOUT:-180}"; then
    warn 'Reranker 未就绪，系统将依赖 Gateway 的降级检索能力继续启动。'
fi

launch gateway 'Gateway' start_gateway.sh \
    "${GATEWAY_BASE_URL%/}/health" "${GATEWAY_STARTUP_TIMEOUT:-60}"
launch industrial_rag 'Industrial RAG' start_rag.sh \
    "${RAG_BASE_URL%/}/health" "${RAG_STARTUP_TIMEOUT:-90}"
launch agent 'Agent API' start_agent.sh \
    "${AGENT_BASE_URL%/}/health" "${AGENT_STARTUP_TIMEOUT:-60}"

printf '\n'
env PROFILE_FILE="${PROFILE_FILE}" bash "${SCRIPT_DIR}/health_check.sh"
