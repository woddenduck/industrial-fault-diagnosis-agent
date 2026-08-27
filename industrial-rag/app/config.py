"""
Day21 Stage2 - Application configuration layer.

Design goals
------------
1. Configuration comes from config/*.env or real OS environment variables.
2. Real OS environment variables override values in the selected env file.
3. Missing/invalid configuration fails immediately with ConfigurationError.
4. This module parses configuration only; filesystem lifecycle belongs to common.paths.
5. Importing this module does not mutate os.environ.

Environment selection
---------------------
Priority:
    RAG_CONFIG_FILE
        > RAG_ENV
        > development

Examples:
    RAG_ENV=autodl python tests/test_config.py
    RAG_CONFIG_FILE=config/local.env python tests/test_config.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from dotenv import dotenv_values


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_ENVIRONMENTS = frozenset({"development", "local", "autodl"})

CONFIG_KEYS = (
    "LLM_BASE_URL",
    "EMBEDDING_BASE_URL",
    "RERANKER_BASE_URL",
    "LLM_PORT",
    "EMBEDDING_PORT",
    "RERANKER_PORT",
    "GATEWAY_PORT",
    "RAG_PORT",
    "DATA_DIR",
    "MODEL_DIR",
    "UPLOAD_DIR",
    "VECTOR_DB_DIR",
    "LOG_DIR",
    "RETRIEVAL_TOP_K",
    "RERANK_TOP_K",
    "SIMILARITY_THRESHOLD",
    "HTTP_TIMEOUT",
    "LLM_TIMEOUT",
    "MODEL_MAX_CONTEXT",
    "CONTEXT_TOKEN_BUDGET",
    "HISTORY_TOKEN_BUDGET",
)


def _resolve_config_file() -> tuple[str, Path]:
    """
    Resolve the selected env file.

    RAG_CONFIG_FILE is intended for explicit testing/temporary configuration.
    RAG_ENV selects config/{development|local|autodl}.env.
    """
    explicit_file = os.getenv("RAG_CONFIG_FILE")
    if explicit_file:
        path = Path(explicit_file).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return "custom", path.resolve(strict=False)

    env_name = os.getenv("RAG_ENV", "development").strip().lower()
    if env_name not in SUPPORTED_ENVIRONMENTS:
        allowed = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ConfigurationError(
            f"Unsupported RAG_ENV: {env_name!r}. Allowed values: {allowed}"
        )

    path = PROJECT_ROOT / "config" / f"{env_name}.env"
    return env_name, path.resolve(strict=False)


ENVIRONMENT, CONFIG_FILE = _resolve_config_file()

if not CONFIG_FILE.is_file():
    raise ConfigurationError(f"Configuration file does not exist: {CONFIG_FILE}")


def _load_raw_config() -> dict[str, str]:
    """
    Load file values, then overlay matching OS environment variables.

    We intentionally do not call load_dotenv(); that would mutate os.environ and
    makes configuration tests harder to isolate.
    """
    file_values = dotenv_values(CONFIG_FILE)

    merged: dict[str, str] = {
        key: str(value).strip()
        for key, value in file_values.items()
        if key in CONFIG_KEYS and value is not None
    }

    for key in CONFIG_KEYS:
        value = os.getenv(key)
        if value is not None:
            merged[key] = value.strip()

    return merged


_RAW_CONFIG = _load_raw_config()


def _required(config: Mapping[str, str], name: str) -> str:
    value = config.get(name)
    if value is None or not value.strip():
        raise ConfigurationError(
            f"Missing required configuration: {name}"
        )
    return value.strip()


def _positive_int(config: Mapping[str, str], name: str) -> int:
    raw = _required(config, name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid integer configuration: {name}={raw!r}"
        ) from exc

    if value <= 0:
        raise ConfigurationError(
            f"Configuration must be > 0: {name}={value}"
        )
    return value


def _port(config: Mapping[str, str], name: str) -> int:
    value = _positive_int(config, name)
    if value > 65535:
        raise ConfigurationError(
            f"Invalid TCP port: {name}={value}"
        )
    return value


def _positive_float(config: Mapping[str, str], name: str) -> float:
    raw = _required(config, name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid float configuration: {name}={raw!r}"
        ) from exc

    if value <= 0:
        raise ConfigurationError(
            f"Configuration must be > 0: {name}={value}"
        )
    return value


def _non_negative_float(config: Mapping[str, str], name: str) -> float:
    raw = _required(config, name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid float configuration: {name}={raw!r}"
        ) from exc

    if value < 0:
        raise ConfigurationError(
            f"Configuration must be >= 0: {name}={value}"
        )
    return value


def _http_url(config: Mapping[str, str], name: str) -> str:
    raw = _required(config, name).rstrip("/")
    parsed = urlparse(raw)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(
            f"Invalid HTTP URL configuration: {name}={raw!r}"
        )

    return raw


@dataclass(frozen=True, slots=True)
class Settings:
    # Services
    llm_base_url: str
    embedding_base_url: str
    reranker_base_url: str

    # Ports
    llm_port: int
    embedding_port: int
    reranker_port: int
    gateway_port: int
    rag_port: int

    # Filesystem (kept as configured strings here; converted to Path in common.paths)
    data_dir: str
    model_dir: str
    upload_dir: str
    vector_db_dir: str
    log_dir: str

    # Retrieval
    retrieval_top_k: int
    rerank_top_k: int
    similarity_threshold: float

    # Network
    http_timeout: float
    llm_timeout: float

    # Context
    model_max_context: int
    context_token_budget: int
    history_token_budget: int


def load_settings(config: Mapping[str, str] | None = None) -> Settings:
    """Parse and validate raw configuration into immutable typed settings."""
    raw = _RAW_CONFIG if config is None else config

    settings_obj = Settings(
        llm_base_url=_http_url(raw, "LLM_BASE_URL"),
        embedding_base_url=_http_url(raw, "EMBEDDING_BASE_URL"),
        reranker_base_url=_http_url(raw, "RERANKER_BASE_URL"),

        llm_port=_port(raw, "LLM_PORT"),
        embedding_port=_port(raw, "EMBEDDING_PORT"),
        reranker_port=_port(raw, "RERANKER_PORT"),
        gateway_port=_port(raw, "GATEWAY_PORT"),
        rag_port=_port(raw, "RAG_PORT"),

        data_dir=_required(raw, "DATA_DIR"),
        model_dir=_required(raw, "MODEL_DIR"),
        upload_dir=_required(raw, "UPLOAD_DIR"),
        vector_db_dir=_required(raw, "VECTOR_DB_DIR"),
        log_dir=_required(raw, "LOG_DIR"),

        retrieval_top_k=_positive_int(raw, "RETRIEVAL_TOP_K"),
        rerank_top_k=_positive_int(raw, "RERANK_TOP_K"),
        similarity_threshold=_non_negative_float(raw, "SIMILARITY_THRESHOLD"),

        http_timeout=_positive_float(raw, "HTTP_TIMEOUT"),
        llm_timeout=_positive_float(raw, "LLM_TIMEOUT"),

        model_max_context=_positive_int(raw, "MODEL_MAX_CONTEXT"),
        context_token_budget=_positive_int(raw, "CONTEXT_TOKEN_BUDGET"),
        history_token_budget=_positive_int(raw, "HISTORY_TOKEN_BUDGET"),
    )

    if settings_obj.rerank_top_k > settings_obj.retrieval_top_k:
        raise ConfigurationError(
            "RERANK_TOP_K cannot be greater than RETRIEVAL_TOP_K: "
            f"{settings_obj.rerank_top_k} > {settings_obj.retrieval_top_k}"
        )

    if settings_obj.context_token_budget > settings_obj.model_max_context:
        raise ConfigurationError(
            "CONTEXT_TOKEN_BUDGET cannot be greater than MODEL_MAX_CONTEXT: "
            f"{settings_obj.context_token_budget} > {settings_obj.model_max_context}"
        )

    if settings_obj.history_token_budget > settings_obj.model_max_context:
        raise ConfigurationError(
            "HISTORY_TOKEN_BUDGET cannot be greater than MODEL_MAX_CONTEXT: "
            f"{settings_obj.history_token_budget} > {settings_obj.model_max_context}"
        )

    return settings_obj


settings = load_settings()


# Uppercase aliases make migration from older modules less disruptive.
LLM_BASE_URL = settings.llm_base_url
EMBEDDING_BASE_URL = settings.embedding_base_url
RERANKER_BASE_URL = settings.reranker_base_url

LLM_PORT = settings.llm_port
EMBEDDING_PORT = settings.embedding_port
RERANKER_PORT = settings.reranker_port
GATEWAY_PORT = settings.gateway_port
RAG_PORT = settings.rag_port

RETRIEVAL_TOP_K = settings.retrieval_top_k
RERANK_TOP_K = settings.rerank_top_k
SIMILARITY_THRESHOLD = settings.similarity_threshold

HTTP_TIMEOUT = settings.http_timeout
LLM_TIMEOUT = settings.llm_timeout

MODEL_MAX_CONTEXT = settings.model_max_context
CONTEXT_TOKEN_BUDGET = settings.context_token_budget
HISTORY_TOKEN_BUDGET = settings.history_token_budget

# ============================================================
# FastAPI 入口兼容导出
# ============================================================
#
# 正式配置统一由 Settings + config/*.env 管理，app/main.py 仍直接导入：
#
#   API_HOST
#   API_PORT
#   APP_NAME
#   APP_VERSION
#   CHAT_MODEL
#   EMBEDDING_MODEL
#   RERANKER_MODEL
#   GATEWAY_BASE_URL
#   HEALTH_TIMEOUT
#   MODEL_QUERY_TIMEOUT
#   RAG_DIR
#   TEMP_UPLOAD_DIR
#   VECTOR_STORE_DIR
#   ensure_runtime_directories
#
# 这里集中提供对应导出，避免在业务入口重复解析环境变量。
#
# 注意：
#   1. 新代码优先使用 settings / common.paths。
#   2. 这里不重新定义端口配置来源。
#   3. 这里不在 import 阶段创建目录。
# ============================================================


def _resolve_compat_path(raw_value: str) -> Path:
    """
    将路径配置转换为绝对 Path。

    相对路径始终相对于 PROJECT_ROOT，
    不依赖当前 shell 的 cwd。
    """
    path = Path(raw_value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve(strict=False)


# ------------------------------------------------------------
# FastAPI application metadata
# ------------------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "Industrial Fault Diagnosis RAG API",
).strip()

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
).strip()


# ------------------------------------------------------------
# Industrial RAG API
# ------------------------------------------------------------
#
# API_PORT 是 FastAPI 入口使用的 RAG_PORT 别名。
# ------------------------------------------------------------

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
).strip() or "0.0.0.0"

API_PORT = RAG_PORT


# ------------------------------------------------------------
# Gateway
# ------------------------------------------------------------
#
# Gateway 地址从统一端口配置派生，不在此处写死。
# ------------------------------------------------------------

GATEWAY_BASE_URL = os.getenv(
    "GATEWAY_BASE_URL",
    f"http://127.0.0.1:{GATEWAY_PORT}",
).strip().rstrip("/")


# ------------------------------------------------------------
# Runtime model display names
# ------------------------------------------------------------
#
# 这三个值主要用于 GET /models 的 configured / fallback 信息。
# 实际推理服务仍然由 Gateway / vLLM / Embedding / Reranker
# 的运行时服务决定。
# ------------------------------------------------------------

CHAT_MODEL = (
    os.getenv(
        "CHAT_MODEL",
        "Qwen/Qwen3-8B",
    ).strip()
    or None
)

EMBEDDING_MODEL = (
    os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-m3",
    ).strip()
    or None
)

RERANKER_MODEL = (
    os.getenv(
        "RERANKER_MODEL",
        "Qwen/Qwen3-Reranker-0.6B",
    ).strip()
    or None
)


# ------------------------------------------------------------
# HTTP timeout compatibility
# ------------------------------------------------------------
#
# HEALTH_TIMEOUT 与 MODEL_QUERY_TIMEOUT 统一映射到 HTTP_TIMEOUT。
# ------------------------------------------------------------

HEALTH_TIMEOUT = HTTP_TIMEOUT
MODEL_QUERY_TIMEOUT = HTTP_TIMEOUT


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

BASE_DIR = PROJECT_ROOT

APP_DIR = (
    PROJECT_ROOT
    / "app"
).resolve(strict=False)

RAG_DIR = (
    PROJECT_ROOT
    / "rag"
).resolve(strict=False)

DATA_DIR = _resolve_compat_path(
    settings.data_dir
)

MODEL_DIR = _resolve_compat_path(
    settings.model_dir
)

UPLOAD_DIR = _resolve_compat_path(
    settings.upload_dir
)

VECTOR_DB_DIR = _resolve_compat_path(
    settings.vector_db_dir
)

LOG_DIR = _resolve_compat_path(
    settings.log_dir
)


# FastAPI 入口使用 VECTOR_STORE_DIR，底层正式名称为 VECTOR_DB_DIR。
VECTOR_STORE_DIR = VECTOR_DB_DIR


# HTTP 上传的请求级临时目录。
TEMP_UPLOAD_DIR = (
    DATA_DIR
    / ".tmp_uploads"
).resolve(strict=False)


# ------------------------------------------------------------
# Runtime directory initialization compatibility
# ------------------------------------------------------------

def ensure_runtime_directories() -> None:
    """
    FastAPI 启动时的运行目录初始化入口。

    正式目录生命周期管理由 common.paths 负责。

    initialize_paths():
        1. 检查 DATA_DIR
        2. 检查 MODEL_DIR
        3. 检查 VECTOR_DB_DIR
        4. 创建 UPLOAD_DIR
        5. 创建 LOG_DIR
        6. 创建 REPORTS_DIR

    最后再创建 FastAPI 上传临时目录。
    """

    # 延迟导入非常重要：
    #
    # common.paths 本身需要 import app.config。
    # 如果在 app.config 顶层直接 import common.paths，
    # 会产生循环导入。
    from common.paths import initialize_paths

    initialize_paths()

    TEMP_UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
