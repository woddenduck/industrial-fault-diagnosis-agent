"""Gateway 的统一配置模块。

所有服务地址、超时、批量限制和模型路径都必须在这里读取。

业务模块中禁止：

1. 重复调用 os.getenv()；
2. 写死服务地址；
3. 在多个文件中分别定义同一配置。

为兼容既有部署，仍接受 `VLLM_*` 环境变量和
`settings.vllm_*` 属性；新部署应优先使用 `LLM_*` 配置。
"""

import os
from urllib.parse import urlsplit


def _get_raw_value(
    name: str,
    default: str | int | float,
    *,
    fallback_names: tuple[str, ...] = (),
) -> str:
    """按照优先级读取环境变量原始值。

    读取顺序：

    1. 当前正式变量名；
    2. 兼容变量名；
    3. 默认值。
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        for fallback_name in fallback_names:
            raw_value = os.getenv(fallback_name)

            if raw_value is not None:
                break

    if raw_value is None:
        raw_value = str(default)

    return raw_value


def _get_string(
    name: str,
    default: str,
    *,
    fallback_names: tuple[str, ...] = (),
) -> str:
    """读取非空字符串环境变量。"""

    value = _get_raw_value(
        name=name,
        default=default,
        fallback_names=fallback_names,
    ).strip()

    if not value:
        raise RuntimeError(
            f"环境变量 {name} 不能为空",
        )

    return value


def _get_positive_int(
    name: str,
    default: int,
    *,
    fallback_names: tuple[str, ...] = (),
) -> int:
    """读取正整数环境变量。"""

    raw_value = _get_raw_value(
        name=name,
        default=default,
        fallback_names=fallback_names,
    )

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"环境变量 {name} 必须是整数，当前值为：{raw_value}",
        ) from exc

    if value <= 0:
        raise RuntimeError(
            f"环境变量 {name} 必须大于 0，当前值为：{value}",
        )

    return value


def _get_positive_float(
    name: str,
    default: float,
    *,
    fallback_names: tuple[str, ...] = (),
) -> float:
    """读取正浮点数环境变量。"""

    raw_value = _get_raw_value(
        name=name,
        default=default,
        fallback_names=fallback_names,
    )

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"环境变量 {name} 必须是数字，当前值为：{raw_value}",
        ) from exc

    if value <= 0:
        raise RuntimeError(
            f"环境变量 {name} 必须大于 0，当前值为：{value}",
        )

    return value


def _normalize_http_base_url(
    name: str,
    value: str,
    *,
    remove_v1_suffix: bool = False,
) -> str:
    """校验并规范化 HTTP 服务根地址。"""

    base_url = value.strip().rstrip("/")

    # 兼容将 vLLM 地址配置成：
    # http://127.0.0.1:6006/v1
    #
    # 客户端会自行拼接 /v1/models、/v1/chat/completions，
    # 因此这里需要删除末尾的 /v1，避免出现 /v1/v1/...。
    if remove_v1_suffix and base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    parsed = urlsplit(base_url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"环境变量 {name} 必须是合法的 HTTP(S) 地址，"
            f"例如 http://127.0.0.1:6006，"
            f"当前值为：{base_url}",
        )

    return base_url


def _get_base_url(
    name: str,
    default: str,
    *,
    fallback_names: tuple[str, ...] = (),
    remove_v1_suffix: bool = False,
) -> str:
    """读取并规范化 HTTP 服务根地址。"""

    value = _get_string(
        name=name,
        default=default,
        fallback_names=fallback_names,
    )

    return _normalize_http_base_url(
        name=name,
        value=value,
        remove_v1_suffix=remove_v1_suffix,
    )


def normalize_vllm_base_url(value: str) -> str:
    """规范化 vLLM 根地址。

    保留该函数是为了兼容原 config.py 中可能存在的直接调用。
    """

    return _normalize_http_base_url(
        name="VLLM_BASE_URL",
        value=value,
        remove_v1_suffix=True,
    )


# ==================================================
# Gateway
# ==================================================

GATEWAY_HOST = _get_string(
    name="GATEWAY_HOST",
    default="0.0.0.0",
)

GATEWAY_PORT = _get_positive_int(
    name="GATEWAY_PORT",
    default=6008,
)


# ==================================================
# 模型服务客户端公共配置
# ==================================================

# 健康检查必须快速失败，避免 /health 长时间阻塞。
HEALTH_CONNECT_TIMEOUT_SECONDS = _get_positive_float(
    name="HEALTH_CONNECT_TIMEOUT_SECONDS",
    default=2.0,
)

HEALTH_READ_TIMEOUT_SECONDS = _get_positive_float(
    name="HEALTH_READ_TIMEOUT_SECONDS",
    default=3.0,
)

HEALTH_TOTAL_TIMEOUT_SECONDS = _get_positive_float(
    name="HEALTH_TOTAL_TIMEOUT_SECONDS",
    default=5.0,
)

# 仅供健康检查、模型列表、Embedding 和 Reranker 的瞬时连接失败重试。
MODEL_SERVICE_RETRY_DELAY_SECONDS = _get_positive_float(
    name="MODEL_SERVICE_RETRY_DELAY_SECONDS",
    default=0.2,
)


# ==================================================
# LLM
# ==================================================

# 新配置优先使用 LLM_BASE_URL。
#
# 如果没有设置 LLM_BASE_URL，则读取兼容变量 VLLM_BASE_URL。
LLM_BASE_URL = _get_base_url(
    name="LLM_BASE_URL",
    default="http://127.0.0.1:6006",
    fallback_names=(
        "VLLM_BASE_URL",
    ),
    remove_v1_suffix=True,
)


# 统一超时配置。
#
# 读取优先级：
#
# 1. LLM_TIMEOUT_SECONDS
# 2. VLLM_TIMEOUT_SECONDS
# 3. VLLM_TIMEOUT
# 4. 默认值 120 秒
LLM_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_TIMEOUT_SECONDS",
    default=120.0,
    fallback_names=(
        "VLLM_TIMEOUT_SECONDS",
        "VLLM_TIMEOUT",
    ),
)


# 连接超时：Gateway 与 vLLM 建立 TCP/HTTP 连接的最长等待时间。
LLM_CONNECT_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_CONNECT_TIMEOUT_SECONDS",
    default=5.0,
    fallback_names=(
        "VLLM_CONNECT_TIMEOUT",
    ),
)


# 读取超时：连接建立后，相邻两次响应数据之间允许等待的时间。
#
# 如果没有单独配置读取超时，则使用统一的
# LLM_TIMEOUT_SECONDS。
LLM_READ_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_READ_TIMEOUT_SECONDS",
    default=LLM_TIMEOUT_SECONDS,
    fallback_names=(
        "VLLM_READ_TIMEOUT",
    ),
)


# 写入超时：Gateway 向 vLLM 发送请求体时允许等待的最长时间。
LLM_WRITE_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_WRITE_TIMEOUT_SECONDS",
    default=10.0,
    fallback_names=(
        "VLLM_WRITE_TIMEOUT",
    ),
)


# 连接池超时：从 HTTP 连接池中获取可用连接的最长等待时间。
LLM_POOL_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_POOL_TIMEOUT_SECONDS",
    default=5.0,
    fallback_names=(
        "VLLM_POOL_TIMEOUT",
    ),
)


# 总请求超时：Gateway 调用 vLLM 到整个请求结束的最长时间。
LLM_TOTAL_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_TOTAL_TIMEOUT_SECONDS",
    default=180.0,
    fallback_names=(
        "VLLM_TOTAL_TIMEOUT",
    ),
)


LLM_STREAM_READ_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_STREAM_READ_TIMEOUT_SECONDS",
    default=600.0,
    fallback_names=(
        "VLLM_STREAM_READ_TIMEOUT",
    ),
)

LLM_STREAM_TOTAL_TIMEOUT_SECONDS = _get_positive_float(
    name="LLM_STREAM_TOTAL_TIMEOUT_SECONDS",
    default=600.0,
    fallback_names=(
        "VLLM_STREAM_TOTAL_TIMEOUT",
    ),
)


LLM_MODEL_PATH = _get_string(
    name="LLM_MODEL_PATH",
    default="Qwen/Qwen3-8B",
)

LLM_SERVED_MODEL_NAME = _get_string(
    name="LLM_SERVED_MODEL_NAME",
    default="Qwen/Qwen3-8B",
)


# ==================================================
# vLLM 配置兼容别名
# ==================================================

# 兼容以下旧导入：
#
# from app.config import VLLM_BASE_URL
# from app.config import VLLM_TIMEOUT_SECONDS
#
VLLM_BASE_URL = LLM_BASE_URL
VLLM_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS


# 兼容可能直接导入分项超时常量的代码。
VLLM_CONNECT_TIMEOUT = LLM_CONNECT_TIMEOUT_SECONDS
VLLM_READ_TIMEOUT = LLM_READ_TIMEOUT_SECONDS
VLLM_WRITE_TIMEOUT = LLM_WRITE_TIMEOUT_SECONDS
VLLM_POOL_TIMEOUT = LLM_POOL_TIMEOUT_SECONDS
VLLM_TOTAL_TIMEOUT = LLM_TOTAL_TIMEOUT_SECONDS


VLLM_STREAM_READ_TIMEOUT = LLM_STREAM_READ_TIMEOUT_SECONDS
VLLM_STREAM_TOTAL_TIMEOUT = LLM_STREAM_TOTAL_TIMEOUT_SECONDS


# ==================================================
# Embedding
# ==================================================

EMBEDDING_BASE_URL = _get_base_url(
    name="EMBEDDING_BASE_URL",
    default="http://127.0.0.1:6010",
)

EMBEDDING_TIMEOUT_SECONDS = _get_positive_float(
    name="EMBEDDING_TIMEOUT_SECONDS",
    default=30.0,
)

EMBEDDING_CONNECT_TIMEOUT_SECONDS = _get_positive_float(
    name="EMBEDDING_CONNECT_TIMEOUT_SECONDS",
    default=5.0,
)

EMBEDDING_READ_TIMEOUT_SECONDS = _get_positive_float(
    name="EMBEDDING_READ_TIMEOUT_SECONDS",
    default=EMBEDDING_TIMEOUT_SECONDS,
)

EMBEDDING_WRITE_TIMEOUT_SECONDS = _get_positive_float(
    name="EMBEDDING_WRITE_TIMEOUT_SECONDS",
    default=30.0,
)

EMBEDDING_POOL_TIMEOUT_SECONDS = _get_positive_float(
    name="EMBEDDING_POOL_TIMEOUT_SECONDS",
    default=5.0,
)

EMBEDDING_MODEL_PATH = _get_string(
    name="EMBEDDING_MODEL_PATH",
    default="BAAI/bge-m3",
)

EMBEDDING_DEVICE = _get_string(
    name="EMBEDDING_DEVICE",
    default="cpu",
)

EMBEDDING_MAX_BATCH_SIZE = _get_positive_int(
    name="EMBEDDING_MAX_BATCH_SIZE",
    default=32,
)

EMBEDDING_MAX_TEXT_LENGTH = _get_positive_int(
    name="EMBEDDING_MAX_TEXT_LENGTH",
    default=4096,
)


# ==================================================
# Reranker
# ==================================================

RERANKER_BASE_URL = _get_base_url(
    name="RERANKER_BASE_URL",
    default="http://127.0.0.1:6009",
)

RERANKER_TIMEOUT_SECONDS = _get_positive_float(
    name="RERANKER_TIMEOUT_SECONDS",
    default=30.0,
)

RERANKER_CONNECT_TIMEOUT_SECONDS = _get_positive_float(
    name="RERANKER_CONNECT_TIMEOUT_SECONDS",
    default=5.0,
)

RERANKER_READ_TIMEOUT_SECONDS = _get_positive_float(
    name="RERANKER_READ_TIMEOUT_SECONDS",
    default=RERANKER_TIMEOUT_SECONDS,
)

RERANKER_WRITE_TIMEOUT_SECONDS = _get_positive_float(
    name="RERANKER_WRITE_TIMEOUT_SECONDS",
    default=30.0,
)

RERANKER_POOL_TIMEOUT_SECONDS = _get_positive_float(
    name="RERANKER_POOL_TIMEOUT_SECONDS",
    default=5.0,
)

RERANKER_MODEL_PATH = _get_string(
    name="RERANKER_MODEL_PATH",
    default="Qwen/Qwen3-Reranker-0.6B",
)

RERANKER_DEVICE = _get_string(
    name="RERANKER_DEVICE",
    default="cpu",
)

RERANKER_MAX_DOCUMENTS = _get_positive_int(
    name="RERANKER_MAX_DOCUMENTS",
    default=50,
)


# ==================================================
# Logging
# ==================================================

LOG_DIR = _get_string(
    name="LOG_DIR",
    default="./logs",
)

LOG_LEVEL = _get_string(
    name="LOG_LEVEL",
    default="INFO",
).upper()


# ==================================================
# Settings 对象兼容层
# ==================================================

class Settings:
    """Gateway 配置对象。

    保留该类是为了兼容原项目中以下写法：

        from app.config import settings

        settings.vllm_base_url
        settings.vllm_connect_timeout
        settings.vllm_read_timeout
        settings.vllm_write_timeout
        settings.vllm_pool_timeout
        settings.vllm_total_timeout
        settings.vllm_timeout
    """

    def __init__(self) -> None:
        # Gateway
        self.gateway_host = GATEWAY_HOST
        self.gateway_port = GATEWAY_PORT

        # 模型服务客户端公共配置
        self.health_connect_timeout = HEALTH_CONNECT_TIMEOUT_SECONDS
        self.health_read_timeout = HEALTH_READ_TIMEOUT_SECONDS
        self.health_total_timeout = HEALTH_TOTAL_TIMEOUT_SECONDS
        self.model_service_retry_delay = (
            MODEL_SERVICE_RETRY_DELAY_SECONDS
        )

        # 新 LLM 命名
        self.llm_base_url = LLM_BASE_URL
        self.llm_timeout_seconds = LLM_TIMEOUT_SECONDS
        self.llm_connect_timeout = LLM_CONNECT_TIMEOUT_SECONDS
        self.llm_read_timeout = LLM_READ_TIMEOUT_SECONDS
        self.llm_write_timeout = LLM_WRITE_TIMEOUT_SECONDS
        self.llm_pool_timeout = LLM_POOL_TIMEOUT_SECONDS
        self.llm_total_timeout = LLM_TOTAL_TIMEOUT_SECONDS
        self.llm_stream_read_timeout = (
            LLM_STREAM_READ_TIMEOUT_SECONDS
        )
        self.llm_stream_total_timeout = (
            LLM_STREAM_TOTAL_TIMEOUT_SECONDS
        )
        self.llm_model_path = LLM_MODEL_PATH
        self.llm_served_model_name = LLM_SERVED_MODEL_NAME

        # 原 vLLM 属性兼容
        self.vllm_base_url = LLM_BASE_URL
        self.vllm_connect_timeout = LLM_CONNECT_TIMEOUT_SECONDS
        self.vllm_read_timeout = LLM_READ_TIMEOUT_SECONDS
        self.vllm_write_timeout = LLM_WRITE_TIMEOUT_SECONDS
        self.vllm_pool_timeout = LLM_POOL_TIMEOUT_SECONDS
        self.vllm_total_timeout = LLM_TOTAL_TIMEOUT_SECONDS
        self.vllm_stream_read_timeout = (
            LLM_STREAM_READ_TIMEOUT_SECONDS
        )
        self.vllm_stream_total_timeout = (
            LLM_STREAM_TOTAL_TIMEOUT_SECONDS
        )

        # 保留原 config.py 中的旧属性。
        #
        # 原有代码通常将 vllm_timeout 当作读取超时使用，
        # 因此这里继续指向 read timeout。
        self.vllm_timeout = self.vllm_read_timeout

        # Embedding
        self.embedding_base_url = EMBEDDING_BASE_URL
        self.embedding_timeout_seconds = EMBEDDING_TIMEOUT_SECONDS
        self.embedding_connect_timeout = (
            EMBEDDING_CONNECT_TIMEOUT_SECONDS
        )
        self.embedding_read_timeout = EMBEDDING_READ_TIMEOUT_SECONDS
        self.embedding_write_timeout = EMBEDDING_WRITE_TIMEOUT_SECONDS
        self.embedding_pool_timeout = EMBEDDING_POOL_TIMEOUT_SECONDS
        self.embedding_model_path = EMBEDDING_MODEL_PATH
        self.embedding_device = EMBEDDING_DEVICE
        self.embedding_max_batch_size = EMBEDDING_MAX_BATCH_SIZE
        self.embedding_max_text_length = EMBEDDING_MAX_TEXT_LENGTH

        # Reranker
        self.reranker_base_url = RERANKER_BASE_URL
        self.reranker_timeout_seconds = RERANKER_TIMEOUT_SECONDS
        self.reranker_connect_timeout = (
            RERANKER_CONNECT_TIMEOUT_SECONDS
        )
        self.reranker_read_timeout = RERANKER_READ_TIMEOUT_SECONDS
        self.reranker_write_timeout = RERANKER_WRITE_TIMEOUT_SECONDS
        self.reranker_pool_timeout = RERANKER_POOL_TIMEOUT_SECONDS
        self.reranker_model_path = RERANKER_MODEL_PATH
        self.reranker_device = RERANKER_DEVICE
        self.reranker_max_documents = RERANKER_MAX_DOCUMENTS

        # Logging
        self.log_dir = LOG_DIR
        self.log_level = LOG_LEVEL


# 保留原来的全局配置对象。
settings = Settings()
