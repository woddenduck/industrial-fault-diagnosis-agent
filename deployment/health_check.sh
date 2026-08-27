#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROFILE_FILE="${PROFILE_FILE:-${PROJECT_ROOT}/.env}"

[[ -r "${PROFILE_FILE}" ]] || {
    printf '[错误] 配置文件不存在或不可读：%s\n' "${PROFILE_FILE}" >&2
    exit 1
}

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${LLM_BASE_URL:=http://127.0.0.1:6006}"
: "${EMBEDDING_BASE_URL:=http://127.0.0.1:6010}"
: "${RERANKER_BASE_URL:=http://127.0.0.1:6009}"
: "${GATEWAY_BASE_URL:=http://127.0.0.1:${GATEWAY_PORT:-6008}}"
: "${RAG_BASE_URL:=http://127.0.0.1:${RAG_PORT:-8000}}"
: "${AGENT_BASE_URL:=http://127.0.0.1:${AGENT_PORT:-8010}}"
: "${HEALTH_CONNECT_TIMEOUT:=2}"
: "${HEALTH_MAX_TIME:=5}"

command -v curl >/dev/null 2>&1 || {
    printf '[错误] 找不到 curl 命令\n' >&2
    exit 1
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

classify_json_status() {
    local file="$1"
    local python_cmd="python3"
    command -v python3 >/dev/null 2>&1 || python_cmd="python"

    "${python_cmd}" - "${file}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as stream:
        payload = json.load(stream)
except Exception:
    print("READY")
    raise SystemExit(0)

status = str(payload.get("status", "READY")).strip().upper()
if status in {"HEALTHY", "READY", "OK", "UP", "RUNNING"}:
    print("READY")
elif status in {"DEGRADED", "FAILED", "UNHEALTHY", "DOWN", "ERROR"}:
    print(status)
else:
    print("READY")
PY
}

FAILURES=0
DEGRADED=0

check_service() {
    local name="$1"
    local url="$2"
    local required="$3"
    local body_file="${TMP_DIR}/${name// /_}.json"
    local code

    code="$(curl --silent --show-error \
        --connect-timeout "${HEALTH_CONNECT_TIMEOUT}" \
        --max-time "${HEALTH_MAX_TIME}" \
        --output "${body_file}" \
        --write-out '%{http_code}' \
        "${url}" 2>/dev/null || true)"

    if [[ ! "${code}" =~ ^2[0-9][0-9]$ ]]; then
        if [[ "${required}" == "required" ]]; then
            printf '%-18s %-10s %s\n' "${name}" 'DOWN' "${url}"
            FAILURES=$((FAILURES + 1))
        else
            printf '%-18s %-10s %s\n' "${name}" 'DEGRADED' "${url}"
            DEGRADED=$((DEGRADED + 1))
        fi
        return
    fi

    local state
    state="$(classify_json_status "${body_file}")"
    case "${state}" in
        FAILED|UNHEALTHY|DOWN|ERROR)
            if [[ "${required}" == "required" ]]; then
                printf '%-18s %-10s %s\n' "${name}" "${state}" "${url}"
                FAILURES=$((FAILURES + 1))
            else
                printf '%-18s %-10s %s\n' "${name}" 'DEGRADED' "${url}"
                DEGRADED=$((DEGRADED + 1))
            fi
            ;;
        DEGRADED)
            printf '%-18s %-10s %s\n' "${name}" 'DEGRADED' "${url}"
            DEGRADED=$((DEGRADED + 1))
            ;;
        *)
            printf '%-18s %-10s %s\n' "${name}" 'READY' "${url}"
            ;;
    esac
}

printf '%s\n' \
    '============================================================' \
    '工业故障诊断 Agent｜六服务健康检查' \
    "配置文件：${PROFILE_FILE}" \
    '============================================================'

check_service 'LLM' "${LLM_BASE_URL%/}/v1/models" required
check_service 'Embedding' "${EMBEDDING_BASE_URL%/}/health" required
check_service 'Reranker' "${RERANKER_BASE_URL%/}/health" optional
check_service 'Gateway' "${GATEWAY_BASE_URL%/}/health" required
check_service 'Industrial RAG' "${RAG_BASE_URL%/}/health" required
check_service 'Agent API' "${AGENT_BASE_URL%/}/health" required

printf '%s\n' '============================================================'
if (( FAILURES > 0 )); then
    printf '总体结果：失败，%d 个必要服务异常。\n' "${FAILURES}"
    exit 1
fi

if (( DEGRADED > 0 )); then
    printf '总体结果：可用但已降级，%d 个组件异常。\n' "${DEGRADED}"
else
    printf '总体结果：全部健康。\n'
fi
