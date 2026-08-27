#!/usr/bin/env bash
set -Eeuo pipefail

# Day21 Stage3 - unified health report
# Location: industrial-rag/deployment/health_check.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RAG_CONFIG_FILE="${RAG_CONFIG_FILE:-${PROJECT_ROOT}/config/autodl.env}"

[[ -r "$RAG_CONFIG_FILE" ]] || {
    echo "[ERROR] Config file not found/readable: ${RAG_CONFIG_FILE}" >&2
    exit 1
}

set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "$RAG_CONFIG_FILE")
set +a

: "${LLM_BASE_URL:?Missing LLM_BASE_URL}"
: "${EMBEDDING_BASE_URL:?Missing EMBEDDING_BASE_URL}"
: "${RERANKER_BASE_URL:?Missing RERANKER_BASE_URL}"
: "${GATEWAY_PORT:?Missing GATEWAY_PORT}"
: "${RAG_PORT:?Missing RAG_PORT}"

HEALTH_CONNECT_TIMEOUT="${HEALTH_CONNECT_TIMEOUT:-2}"
HEALTH_MAX_TIME="${HEALTH_MAX_TIME:-5}"

command -v curl >/dev/null 2>&1 || {
    echo "[ERROR] curl command not found" >&2
    exit 1
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

classify_json_status() {
    local file="$1"
    local py=""
    if command -v python3 >/dev/null 2>&1; then
        py="python3"
    elif command -v python >/dev/null 2>&1; then
        py="python"
    else
        printf 'READY\n'
        return 0
    fi

    "$py" - "$file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    print("READY")
    raise SystemExit(0)

status = str(payload.get("status", "READY")).strip().upper()
if status in {"HEALTHY", "READY", "OK", "UP", "RUNNING"}:
    print("READY")
elif status == "DEGRADED":
    print("DEGRADED")
elif status in {"FAILED", "FAIL", "UNHEALTHY", "DOWN", "ERROR"}:
    print("FAILED")
else:
    # A 2xx endpoint with an unknown/nonstandard status is still reachable.
    print(status or "READY")
PY
}

FAILURES=0

check_service() {
    local name="$1"
    local url="$2"
    local safe_name="${name// /_}"
    local body_file="${TMP_DIR}/${safe_name}.json"
    local code=""

    code="$(curl \
        --silent --show-error \
        --connect-timeout "$HEALTH_CONNECT_TIMEOUT" \
        --max-time "$HEALTH_MAX_TIME" \
        --output "$body_file" \
        --write-out '%{http_code}' \
        "$url" 2>/dev/null || true)"

    if [[ ! "$code" =~ ^2[0-9][0-9]$ ]]; then
        printf '%-16s %-10s %s (HTTP %s)\n' "$name" "DOWN" "$url" "${code:-000}"
        FAILURES=$((FAILURES + 1))
        return 0
    fi

    local state
    state="$(classify_json_status "$body_file")"
    case "$state" in
        FAILED)
            printf '%-16s %-10s %s\n' "$name" "FAILED" "$url"
            FAILURES=$((FAILURES + 1))
            ;;
        DEGRADED)
            # Degraded is intentionally reachable/operational and is not a shell failure.
            printf '%-16s %-10s %s\n' "$name" "DEGRADED" "$url"
            ;;
        *)
            printf '%-16s %-10s %s\n' "$name" "$state" "$url"
            ;;
    esac
}

printf '============================================================\n'
printf 'Industrial RAG Health Check\n'
printf 'Config: %s\n' "$RAG_CONFIG_FILE"
printf '============================================================\n'

# vLLM /v1/models verifies that the OpenAI-compatible server is reachable and a model is exposed.
check_service "LLM" "${LLM_BASE_URL%/}/v1/models"
check_service "Embedding" "${EMBEDDING_BASE_URL%/}/health"
check_service "Reranker" "${RERANKER_BASE_URL%/}/health"
check_service "Gateway" "http://127.0.0.1:${GATEWAY_PORT}/health"
check_service "Industrial RAG" "http://127.0.0.1:${RAG_PORT}/health"

printf '============================================================\n'
if (( FAILURES == 0 )); then
    printf 'Overall: PASS (DEGRADED is allowed)\n'
    exit 0
fi
printf 'Overall: FAIL (%d service(s) unreachable/failed)\n' "$FAILURES"
exit 1