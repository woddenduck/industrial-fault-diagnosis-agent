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

# 自动导出 profile 中的变量，并兼容 Windows CRLF。
set -a
# shellcheck disable=SC1090
source <(sed 's/\r$//' "${PROFILE_FILE}")
set +a

: "${LLM_START_MODE:=local}"
: "${LLM_MODEL_PATH:?profile 中必须配置 LLM_MODEL_PATH}"
: "${LLM_SERVED_MODEL_NAME:=Qwen/Qwen3-8B}"
: "${LLM_HOST:=0.0.0.0}"
: "${LLM_PORT:=6006}"
: "${LLM_MAX_MODEL_LEN:=4096}"
: "${LLM_GPU_MEMORY_UTILIZATION:=0.85}"
: "${LLM_DTYPE:=auto}"
: "${LLM_CONDA_ENV:=vllm_qwen3}"
: "${LOG_DIR:=logs}"
: "${LLM_LOG_FILE:=llm.log}"

if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}"
fi
mkdir -p "${LOG_DIR}" || die "无法创建日志目录：${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${LLM_LOG_FILE}"

# 同时写入终端和固定日志文件。
exec > >(tee -a "${LOG_FILE}") 2>&1

validate_port() {
    local port="$1"
    [[ "${port}" =~ ^[0-9]+$ ]] || die "LLM_PORT 必须是整数，当前值：${port}"
    (( port >= 1 && port <= 65535 )) || die "LLM_PORT 必须位于 1～65535，当前值：${port}"
}

validate_port "${LLM_PORT}"

if [[ "${LLM_START_MODE}" == "remote" ]]; then
    echo "=================================================="
    echo "当前 profile 将 LLM 配置为远程服务，本机不启动 vLLM。"
    echo "远程服务地址：${LLM_BASE_URL:-未配置}"
    echo "请在 AutoDL 上使用 24GB profile 启动 LLM。"
    echo "=================================================="
    exit 0
fi

[[ "${LLM_START_MODE}" == "local" ]] || \
    die "LLM_START_MODE 只能是 local 或 remote，当前值：${LLM_START_MODE}"

# LLM_MODEL_PATH 同时支持：
# 1. 本地模型目录
# 2. Hugging Face 模型 ID，例如 Qwen/Qwen3-8B

if [[ -d "${LLM_MODEL_PATH}" ]]; then

    echo "检测到本地模型目录：${LLM_MODEL_PATH}"

    [[ -f "${LLM_MODEL_PATH}/config.json" ]] \
        || die "模型目录中缺少 config.json：${LLM_MODEL_PATH}"

    compgen -G "${LLM_MODEL_PATH}/*.safetensors" >/dev/null \
        || die "模型目录中没有找到 safetensors 权重文件：${LLM_MODEL_PATH}"

else

    echo "检测到 Hugging Face 模型 ID：${LLM_MODEL_PATH}"
    echo "vLLM 将自动下载或读取 Hugging Face 缓存中的模型。"

fi

RUN_PREFIX=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${LLM_CONDA_ENV}" ]]; then
    command -v conda >/dev/null 2>&1 || die "当前未激活 ${LLM_CONDA_ENV}，并且找不到 conda 命令"

    # 不要使用 `conda run ... bash -lc` 检查命令。
    # 登录 Shell 可能重新覆盖 Conda 注入的 PATH，导致环境存在且已安装
    # vLLM 时仍被误判为“找不到 vllm”。这里直接在目标环境的
    # Python 进程中检查 PATH。
    if ! conda run -n "${LLM_CONDA_ENV}" \
        python -c 'import shutil, sys; sys.exit(0 if shutil.which("vllm") else 1)'; then
        echo "诊断信息：" >&2
        conda run -n "${LLM_CONDA_ENV}" \
            python -c 'import sys, shutil; print("python=", sys.executable); print("vllm=", shutil.which("vllm"))' \
            >&2 || true
        die "Conda 环境 ${LLM_CONDA_ENV} 中找不到 vllm 命令。请执行：conda run -n ${LLM_CONDA_ENV} python -m pip show vllm"
    fi

    RUN_PREFIX=(conda run --no-capture-output -n "${LLM_CONDA_ENV}")
else
    command -v vllm >/dev/null 2>&1 \
        || die "当前 Conda 环境 ${LLM_CONDA_ENV} 中找不到 vllm 命令"
fi

cd "${PROJECT_ROOT}"

echo "=================================================="
echo "正在启动 vLLM 服务"
echo "项目目录：${PROJECT_ROOT}"
echo "配置文件：${PROFILE_FILE}"
echo "Conda 环境：${LLM_CONDA_ENV}"
echo "模型路径：${LLM_MODEL_PATH}"
echo "对外模型名称：${LLM_SERVED_MODEL_NAME}"
echo "监听地址：${LLM_HOST}"
echo "服务端口：${LLM_PORT}"
echo "服务地址：${LLM_BASE_URL:-http://127.0.0.1:${LLM_PORT}}"
echo "最大上下文长度：${LLM_MAX_MODEL_LEN}"
echo "GPU 显存使用比例：${LLM_GPU_MEMORY_UTILIZATION}"
echo "数据类型：${LLM_DTYPE}"
echo "日志文件：${LOG_FILE}"
echo "=================================================="

exec "${RUN_PREFIX[@]}" vllm serve "${LLM_MODEL_PATH}" \
    --host "${LLM_HOST}" \
    --port "${LLM_PORT}" \
    --served-model-name "${LLM_SERVED_MODEL_NAME}" \
    --max-model-len "${LLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION}" \
    --dtype "${LLM_DTYPE}"
