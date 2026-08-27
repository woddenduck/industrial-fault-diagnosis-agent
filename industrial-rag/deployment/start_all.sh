#!/usr/bin/env bash
set -Eeuo pipefail

# Day21 Stage3 - Industrial RAG unified startup (AutoDL/Linux)
# Location: industrial-rag/deployment/start_all.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PLATFORM_ROOT="${PLATFORM_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)}"
PLATFORM_DEPLOYMENT="${PLATFORM_ROOT}/deployment"

RAG_CONFIG_FILE="${RAG_CONFIG_FILE:-${PROJECT_ROOT}/config/autodl.env}"
MODEL_PROFILE_FILE="${MODEL_PROFILE_FILE:-${PLATFORM_DEPLOYMENT}/profile_24gb.env}"
RAG_CONDA_ENV="${RAG_CONDA_ENV:-rag_system}"

STARTUP_WAIT_INTERVAL="${STARTUP_WAIT_INTERVAL:-2}"
LLM_STARTUP_TIMEOUT="${LLM_STARTUP_TIMEOUT:-300}"
EMBEDDING_STARTUP_TIMEOUT="${EMBEDDING_STARTUP_TIMEOUT:-180}"
RERANKER_STARTUP_TIMEOUT="${RERANKER_STARTUP_TIMEOUT:-180}"
GATEWAY_STARTUP_TIMEOUT="${GATEWAY_STARTUP_TIMEOUT:-60}"
RAG_STARTUP_TIMEOUT="${RAG_STARTUP_TIMEOUT:-60}"

usage() {
    cat <<'USAGE'
Usage:
  bash deployment/start_all.sh
  bash deployment/start_all.sh --validate
  bash deployment/start_all.sh --health-only
  bash deployment/start_all.sh --check-port <SERVICE> <PORT>

Environment overrides:
  RAG_CONFIG_FILE       Industrial RAG Stage2 config file.
  MODEL_PROFILE_FILE    Parent platform model-service profile.
  PLATFORM_ROOT         Parent enterprise-llm-agent-platform directory.
  RAG_CONDA_ENV         Industrial RAG conda env (default: rag_system).
USAGE
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

info() { printf '[INFO] %s\n' "$*"; }
ok()   { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }

is_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

validate_port_number() {
    local name="$1"
    local value="$2"
    is_integer "$value" || die "${name} must be an integer, got: ${value}"
    (( value >= 1 && value <= 65535 )) || die "${name} must be in 1..65535, got: ${value}"
}

port_in_use() {
    local port="$1"
    local py=""
    if command -v python3 >/dev/null 2>&1; then
        py="python3"
    elif command -v python >/dev/null 2>&1; then
        py="python"
    else
        die "python3/python not found; cannot perform portable port preflight"
    fi

    "$py" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for family, address in (
    (socket.AF_INET, ("127.0.0.1", port)),
    (socket.AF_INET6, ("::1", port)),
):
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(address) == 0:
                raise SystemExit(0)
    except OSError:
        pass
raise SystemExit(1)
PY
}

check_port_free() {
    local service="$1"
    local port="$2"
    validate_port_number "${service}_PORT" "$port"
    if port_in_use "$port"; then
        printf '[ERROR] Port %s is already in use.\n' "$port" >&2
        printf 'Service: %s\n' "$service" >&2
        return 1
    fi
    printf '[OK] %-12s port %s is available.\n' "$service" "$port"
}

# This narrow mode is intentionally available for tests and troubleshooting.
if [[ "${1:-}" == "--check-port" ]]; then
    [[ $# -eq 3 ]] || die "--check-port requires <SERVICE> <PORT>"
    check_port_free "$2" "$3"
    exit $?
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

[[ $# -le 1 ]] || { usage >&2; exit 2; }
MODE="${1:-start}"
case "$MODE" in
    start|--validate|--health-only) ;;
    *) usage >&2; exit 2 ;;
esac

[[ -r "$RAG_CONFIG_FILE" ]] || die "RAG config file not found/readable: ${RAG_CONFIG_FILE}"
[[ -r "$MODEL_PROFILE_FILE" ]] || die "Model-service profile not found/readable: ${MODEL_PROFILE_FILE}"

load_env_file() {
    local file="$1"
    set -a
    # shellcheck disable=SC1090
    source <(sed 's/\r$//' "$file")
    set +a
}

# Parent profile owns model-specific settings; Day21 Stage2 config is loaded last
# so its ports/endpoints/directories are the source of truth for Industrial RAG.
load_env_file "$MODEL_PROFILE_FILE"
load_env_file "$RAG_CONFIG_FILE"

REQUIRED_KEYS=(
    LLM_BASE_URL EMBEDDING_BASE_URL RERANKER_BASE_URL
    LLM_PORT EMBEDDING_PORT RERANKER_PORT GATEWAY_PORT RAG_PORT
    DATA_DIR MODEL_DIR UPLOAD_DIR VECTOR_DB_DIR LOG_DIR
)
for key in "${REQUIRED_KEYS[@]}"; do
    [[ -n "${!key:-}" ]] || die "Missing required config: ${key} (${RAG_CONFIG_FILE})"
done

for pair in \
    "LLM_PORT=${LLM_PORT}" \
    "EMBEDDING_PORT=${EMBEDDING_PORT}" \
    "RERANKER_PORT=${RERANKER_PORT}" \
    "GATEWAY_PORT=${GATEWAY_PORT}" \
    "RAG_PORT=${RAG_PORT}"; do
    validate_port_number "${pair%%=*}" "${pair#*=}"
done

# Fail early on accidental duplicate service ports.
declare -A SEEN_PORTS=()
for pair in \
    "LLM:${LLM_PORT}" \
    "Embedding:${EMBEDDING_PORT}" \
    "Reranker:${RERANKER_PORT}" \
    "Gateway:${GATEWAY_PORT}" \
    "Industrial RAG:${RAG_PORT}"; do
    service="${pair%%:*}"
    port="${pair##*:}"
    if [[ -n "${SEEN_PORTS[$port]:-}" ]]; then
        die "Port ${port} is configured for both ${SEEN_PORTS[$port]} and ${service}"
    fi
    SEEN_PORTS[$port]="$service"
done

validate_http_url() {
    local key="$1"
    local value="$2"
    case "$value" in
        http://*|https://*) ;;
        *) die "${key} must be an HTTP(S) URL, got: ${value}" ;;
    esac
}
validate_http_url LLM_BASE_URL "$LLM_BASE_URL"
validate_http_url EMBEDDING_BASE_URL "$EMBEDDING_BASE_URL"
validate_http_url RERANKER_BASE_URL "$RERANKER_BASE_URL"

resolve_project_path() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s/%s\n' "$PROJECT_ROOT" "${value#./}"
    fi
}

ABS_DATA_DIR="$(resolve_project_path "$DATA_DIR")"
ABS_MODEL_DIR="$(resolve_project_path "$MODEL_DIR")"
ABS_UPLOAD_DIR="$(resolve_project_path "$UPLOAD_DIR")"
ABS_VECTOR_DB_DIR="$(resolve_project_path "$VECTOR_DB_DIR")"
ABS_LOG_DIR="$(resolve_project_path "$LOG_DIR")"
RUN_DIR="${PROJECT_ROOT}/deployment/run"
RUNTIME_PROFILE="${RUN_DIR}/runtime_profile.env"

validate_conda_env() {
    local env_name="$1"
    command -v conda >/dev/null 2>&1 || die "conda command not found"
    conda run -n "$env_name" python -c 'import sys; print(sys.executable)' >/dev/null 2>&1 \
        || die "Conda environment is unavailable or broken: ${env_name}"
    ok "Conda environment: ${env_name}"
}

validate_model_source() {
    local name="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || { warn "${name} model source is not defined in model profile; delegated launcher defaults will apply"; return 0; }

    case "$value" in
        /*|./*|../*)
            local resolved="$value"
            [[ "$value" = /* ]] || resolved="${PLATFORM_ROOT}/${value#./}"
            [[ -e "$resolved" ]] || die "${name} model path does not exist: ${resolved}"
            ok "${name} model path: ${resolved}"
            ;;
        *)
            ok "${name} model id: ${value}"
            ;;
    esac
}

preflight_common() {
    command -v curl >/dev/null 2>&1 || die "curl command not found"
    command -v conda >/dev/null 2>&1 || die "conda command not found"

    [[ -d "$PROJECT_ROOT/app" ]] || die "Industrial RAG app directory not found: ${PROJECT_ROOT}/app"
    [[ -f "$PROJECT_ROOT/app/main.py" ]] || die "Industrial RAG entry not found: ${PROJECT_ROOT}/app/main.py"
    [[ -d "$ABS_MODEL_DIR" ]] || die "MODEL_DIR does not exist: ${ABS_MODEL_DIR}"

    mkdir -p "$ABS_DATA_DIR" "$ABS_UPLOAD_DIR" "$ABS_VECTOR_DB_DIR" "$ABS_LOG_DIR" "$RUN_DIR"

    for script in start_llm.sh start_embedding.sh start_reranker.sh start_gateway.sh; do
        [[ -f "$PLATFORM_DEPLOYMENT/$script" ]] || die "Parent launcher not found: ${PLATFORM_DEPLOYMENT}/${script}"
    done

    : "${LLM_CONDA_ENV:=vllm_qwen3}"
    : "${EMBEDDING_CONDA_ENV:=embedding_service}"
    : "${RERANKER_CONDA_ENV:=reranker_service}"
    : "${GATEWAY_CONDA_ENV:=llm_gateway}"

    validate_conda_env "$LLM_CONDA_ENV"
    validate_conda_env "$EMBEDDING_CONDA_ENV"
    validate_conda_env "$RERANKER_CONDA_ENV"
    validate_conda_env "$GATEWAY_CONDA_ENV"
    validate_conda_env "$RAG_CONDA_ENV"

    validate_model_source LLM "${LLM_MODEL_PATH:-}"
    validate_model_source Embedding "${EMBEDDING_MODEL_PATH:-}"
    validate_model_source Reranker "${RERANKER_MODEL_PATH:-}"

    ok "Stage2 config: ${RAG_CONFIG_FILE}"
    ok "Parent model profile: ${MODEL_PROFILE_FILE}"
    ok "Industrial RAG project: ${PROJECT_ROOT}"
    ok "Parent platform: ${PLATFORM_ROOT}"
}

write_runtime_profile() {
    {
        printf '# Auto-generated by industrial-rag/deployment/start_all.sh\n'
        printf '# Parent model profile first; Stage2 config overrides common values.\n'
        sed 's/\r$//' "$MODEL_PROFILE_FILE"
        printf '\n# --- Day21 Stage2 overrides ---\n'
        sed 's/\r$//' "$RAG_CONFIG_FILE"
        printf '\n# --- Stage3 orchestration overrides ---\n'
        printf 'LOG_DIR=%s\n' "$ABS_LOG_DIR"
        printf 'LLM_CONDA_ENV=%s\n' "$LLM_CONDA_ENV"
        printf 'EMBEDDING_CONDA_ENV=%s\n' "$EMBEDDING_CONDA_ENV"
        printf 'RERANKER_CONDA_ENV=%s\n' "$RERANKER_CONDA_ENV"
        printf 'GATEWAY_CONDA_ENV=%s\n' "$GATEWAY_CONDA_ENV"
    } > "$RUNTIME_PROFILE"
    chmod 600 "$RUNTIME_PROFILE" 2>/dev/null || true
    ok "Runtime service profile: ${RUNTIME_PROFILE}"
}

preflight_ports() {
    # Check every port before starting anything to avoid a half-started system.
    check_port_free "LLM" "$LLM_PORT" || return 1
    check_port_free "Embedding" "$EMBEDDING_PORT" || return 1
    check_port_free "Reranker" "$RERANKER_PORT" || return 1
    check_port_free "Gateway" "$GATEWAY_PORT" || return 1
    check_port_free "Industrial RAG" "$RAG_PORT" || return 1
}

if [[ "$MODE" == "--health-only" ]]; then
    exec env RAG_CONFIG_FILE="$RAG_CONFIG_FILE" bash "$SCRIPT_DIR/health_check.sh"
fi

printf '\n============================================================\n'
printf 'Day21 Stage3 | Industrial RAG Unified Startup\n'
printf '============================================================\n'
preflight_common
write_runtime_profile

if [[ "$MODE" == "--validate" ]]; then
    ok "Validation passed. No service was started."
    exit 0
fi

preflight_ports || exit 1

http_ready() {
    local url="$1"
    curl --silent --show-error --fail --max-time 5 "$url" >/dev/null 2>&1
}

wait_for_service() {
    local name="$1"
    local url="$2"
    local timeout="$3"
    local pid="$4"
    local log_file="$5"
    local elapsed=0

    while (( elapsed < timeout )); do
        if http_ready "$url"; then
            printf '[READY] %-14s %s\n' "$name" "$url"
            return 0
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            warn "${name} exited before becoming healthy. Log tail:"
            tail -n 20 "$log_file" >&2 2>/dev/null || true
            return 1
        fi
        sleep "$STARTUP_WAIT_INTERVAL"
        elapsed=$((elapsed + STARTUP_WAIT_INTERVAL))
    done

    warn "${name} health check timed out after ${timeout}s: ${url}"
    tail -n 20 "$log_file" >&2 2>/dev/null || true
    return 1
}

launch_parent_service() {
    local name="$1"
    local conda_env="$2"
    local script="$3"
    local health_url="$4"
    local timeout="$5"
    local key="$6"
    local log_file="${ABS_LOG_DIR}/${key}.startup.log"
    local pid_file="${RUN_DIR}/${key}.pid"

    info "Starting ${name} ..."
    (
        cd "$PLATFORM_ROOT"
        exec env PROFILE_FILE="$RUNTIME_PROFILE" \
            conda run --no-capture-output -n "$conda_env" \
            bash "$script"
    ) >"$log_file" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    info "${name} PID=${pid}, log=${log_file}"

    if wait_for_service "$name" "$health_url" "$timeout" "$pid" "$log_file"; then
        return 0
    fi
    return 1
}

launch_rag() {
    local log_file="${ABS_LOG_DIR}/industrial_rag.startup.log"
    local pid_file="${RUN_DIR}/industrial_rag.pid"

    info "Starting Industrial RAG ..."
    (
        cd "$PROJECT_ROOT"
        exec env RAG_CONFIG_FILE="$RAG_CONFIG_FILE" RAG_ENV=autodl \
            conda run --no-capture-output -n "$RAG_CONDA_ENV" \
            python -m uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "$RAG_PORT"
    ) >"$log_file" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    info "Industrial RAG PID=${pid}, log=${log_file}"

    wait_for_service "Industrial RAG" "http://127.0.0.1:${RAG_PORT}/health" \
        "$RAG_STARTUP_TIMEOUT" "$pid" "$log_file"
}

# A model failure does NOT prevent Gateway startup; Gateway is allowed to become degraded.
LLM_RESULT=0
EMBEDDING_RESULT=0
RERANKER_RESULT=0
GATEWAY_RESULT=0
RAG_RESULT=0

launch_parent_service \
    "LLM" "$LLM_CONDA_ENV" "$PLATFORM_DEPLOYMENT/start_llm.sh" \
    "${LLM_BASE_URL%/}/v1/models" "$LLM_STARTUP_TIMEOUT" "llm" || LLM_RESULT=$?

launch_parent_service \
    "Embedding" "$EMBEDDING_CONDA_ENV" "$PLATFORM_DEPLOYMENT/start_embedding.sh" \
    "${EMBEDDING_BASE_URL%/}/health" "$EMBEDDING_STARTUP_TIMEOUT" "embedding" || EMBEDDING_RESULT=$?

launch_parent_service \
    "Reranker" "$RERANKER_CONDA_ENV" "$PLATFORM_DEPLOYMENT/start_reranker.sh" \
    "${RERANKER_BASE_URL%/}/health" "$RERANKER_STARTUP_TIMEOUT" "reranker" || RERANKER_RESULT=$?

launch_parent_service \
    "Gateway" "$GATEWAY_CONDA_ENV" "$PLATFORM_DEPLOYMENT/start_gateway.sh" \
    "http://127.0.0.1:${GATEWAY_PORT}/health" "$GATEWAY_STARTUP_TIMEOUT" "gateway" || GATEWAY_RESULT=$?

launch_rag || RAG_RESULT=$?

printf '\n============================================================\n'
printf 'Final health report\n'
printf '============================================================\n'
set +e
env RAG_CONFIG_FILE="$RAG_CONFIG_FILE" bash "$SCRIPT_DIR/health_check.sh"
HEALTH_RC=$?
set -e

printf '\nStartup phase result (0 means health endpoint became reachable):\n'
printf '  LLM            %s\n' "$LLM_RESULT"
printf '  Embedding      %s\n' "$EMBEDDING_RESULT"
printf '  Reranker       %s\n' "$RERANKER_RESULT"
printf '  Gateway        %s\n' "$GATEWAY_RESULT"
printf '  Industrial RAG %s\n' "$RAG_RESULT"
printf '============================================================\n'

if (( GATEWAY_RESULT != 0 || RAG_RESULT != 0 )); then
    die "Gateway or Industrial RAG failed to become reachable; inspect ${ABS_LOG_DIR}"
fi

if (( LLM_RESULT != 0 || EMBEDDING_RESULT != 0 || RERANKER_RESULT != 0 )); then
    warn "One or more model services failed. Gateway/RAG may be DEGRADED."
fi

# health_check returns success for reachable HEALTHY/READY/OK/DEGRADED states.
exit "$HEALTH_RC"