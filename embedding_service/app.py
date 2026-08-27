"""
独立 Embedding 模型服务。

职责：
1. 在服务启动时加载 Embedding 模型；
2. 接收单条或多条文本；
3. 批量生成文本向量；
4. 支持向量归一化；
5. 校验批量大小、空文本、字符长度和 Token 长度；
6. 返回向量数量、维度和向量列表。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
)
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 基础日志
# ============================================================

LOG_LEVEL = os.getenv(
    "EMBEDDING_LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("embedding_service")


# ============================================================
# 2. 配置读取
# ============================================================


def _get_positive_int(
    name: str,
    default: int,
) -> int:
    """读取正整数环境变量。"""

    raw_value = os.getenv(name, str(default))

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


EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "BAAI/bge-m3",
).strip()

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
).strip()

EMBEDDING_MAX_TEXTS = _get_positive_int(
    "EMBEDDING_MAX_TEXTS",
    32,
)

EMBEDDING_MAX_CHARS = _get_positive_int(
    "EMBEDDING_MAX_CHARS",
    12000,
)

EMBEDDING_MAX_TOKENS = _get_positive_int(
    "EMBEDDING_MAX_TOKENS",
    2048,
)

EMBEDDING_BATCH_SIZE = _get_positive_int(
    "EMBEDDING_BATCH_SIZE",
    8,
)

if not EMBEDDING_MODEL_PATH:
    raise RuntimeError(
        "环境变量 EMBEDDING_MODEL_PATH 不能为空",
    )

if not EMBEDDING_DEVICE:
    raise RuntimeError(
        "环境变量 EMBEDDING_DEVICE 不能为空",
    )


# 防止多个请求同时进入 CPU 编码，造成线程争抢和内存突增。
# 当前阶段先保证稳定性，后续再设计并发控制。
encode_lock = threading.Lock()


# ============================================================
# 3. Pydantic 请求和响应模型
# ============================================================


class ServiceSchema(BaseModel):
    """Embedding 服务公共数据模型。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class EmbedRequest(ServiceSchema):
    """POST /embed 的请求体。"""

    texts: list[StrictStr] = Field(
        min_length=1,
        max_length=EMBEDDING_MAX_TEXTS,
        description="需要编码的文本，可以包含一条或多条",
    )

    normalize: StrictBool = Field(
        default=True,
        description="是否对返回向量执行 L2 归一化",
    )

    @field_validator(
        "texts",
        mode="before",
    )
    @classmethod
    def accept_single_text(
        cls,
        value: object,
    ) -> object:
        """允许 texts 直接传入单个字符串。

        以下两种格式都可以：

        {"texts": "一段文本"}

        {"texts": ["第一段文本", "第二段文本"]}
        """

        if isinstance(value, str):
            return [value]

        return value

    @field_validator("texts")
    @classmethod
    def validate_texts(
        cls,
        texts: list[str],
    ) -> list[str]:
        """清理文本并拒绝空字符串和超长字符串。"""

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            cleaned_text = text.strip()

            if not cleaned_text:
                raise ValueError(
                    f"texts[{index}] 不能为空或只包含空格",
                )

            if len(cleaned_text) > EMBEDDING_MAX_CHARS:
                raise ValueError(
                    f"texts[{index}] 字符数为 "
                    f"{len(cleaned_text)}，"
                    f"超过限制 {EMBEDDING_MAX_CHARS}",
                )

            cleaned_texts.append(cleaned_text)

        return cleaned_texts


class EmbedResponse(ServiceSchema):
    """POST /embed 的正常响应。"""

    count: int = Field(
        ge=1,
        description="实际处理的文本数量",
    )

    dimension: int = Field(
        ge=1,
        description="单个向量的维度",
    )

    embeddings: list[list[float]]


class HealthResponse(ServiceSchema):
    """GET /health 的响应。"""

    status: Literal["ready"]
    model: str
    device: str
    dimension: int

    max_texts: int
    max_chars: int
    max_tokens: int
    batch_size: int

    model_load_seconds: float
    memory_increase_mb: float


# ============================================================
# 4. 模型辅助函数
# ============================================================


def _get_embedding_dimension(
    model: SentenceTransformer,
) -> int:
    """兼容不同版本的 Sentence Transformers。"""

    if hasattr(model, "get_embedding_dimension"):
        dimension = model.get_embedding_dimension()
    else:
        dimension = model.get_sentence_embedding_dimension()

    if dimension is None:
        raise RuntimeError(
            "无法从模型中获取向量维度",
        )

    return int(dimension)


# ============================================================
# 5. 模型生命周期
# ============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """在应用启动时加载一次模型。"""

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    logger.info(
        "开始加载 Embedding 模型：model=%s, device=%s",
        EMBEDDING_MODEL_PATH,
        EMBEDDING_DEVICE,
    )

    load_start = time.perf_counter()

    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            device=EMBEDDING_DEVICE,
        )

        model.eval()

    except Exception:
        logger.exception(
            "Embedding 模型加载失败：model=%s, device=%s",
            EMBEDDING_MODEL_PATH,
            EMBEDDING_DEVICE,
        )
        raise

    load_seconds = time.perf_counter() - load_start

    memory_after = process.memory_info().rss

    memory_increase_mb = max(
        0.0,
        (memory_after - memory_before) / 1024 / 1024,
    )

    dimension = _get_embedding_dimension(model)

    model_max_tokens = int(
        getattr(
            model,
            "max_seq_length",
            EMBEDDING_MAX_TOKENS,
        )
        or EMBEDDING_MAX_TOKENS
    )

    effective_max_tokens = min(
        EMBEDDING_MAX_TOKENS,
        model_max_tokens,
    )

    actual_device = str(model.device)

    app.state.embedding_model = model
    app.state.dimension = dimension
    app.state.actual_device = actual_device
    app.state.effective_max_tokens = effective_max_tokens
    app.state.model_load_seconds = round(
        load_seconds,
        4,
    )
    app.state.memory_increase_mb = round(
        memory_increase_mb,
        2,
    )

    logger.info(
        "Embedding 模型加载完成："
        "model=%s, device=%s, dimension=%s, "
        "max_tokens=%s, load_seconds=%.4f, "
        "memory_increase_mb=%.2f",
        EMBEDDING_MODEL_PATH,
        actual_device,
        dimension,
        effective_max_tokens,
        load_seconds,
        memory_increase_mb,
    )

    try:
        yield
    finally:
        logger.info(
            "Embedding 服务正在关闭",
        )

        app.state.embedding_model = None


# ============================================================
# 6. FastAPI 应用
# ============================================================


app = FastAPI(
    title="Embedding Service",
    description="独立运行的文本向量化服务",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# 7. 健康检查接口
# ============================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="检查 Embedding 服务状态",
)
def health_check() -> HealthResponse:
    """返回模型、设备、维度和资源信息。"""

    model = getattr(
        app.state,
        "embedding_model",
        None,
    )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding 模型尚未加载完成",
        )

    return HealthResponse(
        status="ready",
        model=EMBEDDING_MODEL_PATH,
        device=app.state.actual_device,
        dimension=app.state.dimension,
        max_texts=EMBEDDING_MAX_TEXTS,
        max_chars=EMBEDDING_MAX_CHARS,
        max_tokens=app.state.effective_max_tokens,
        batch_size=EMBEDDING_BATCH_SIZE,
        model_load_seconds=app.state.model_load_seconds,
        memory_increase_mb=app.state.memory_increase_mb,
    )


# ============================================================
# 8. Embedding 接口
# ============================================================


@app.post(
    "/embed",
    response_model=EmbedResponse,
    tags=["Embedding"],
    summary="生成文本向量",
)
def create_embeddings(
    request: EmbedRequest,
) -> EmbedResponse:
    """批量生成文本向量。"""

    model: SentenceTransformer | None = getattr(
        app.state,
        "embedding_model",
        None,
    )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding 模型尚未加载完成",
        )

    request_start = time.perf_counter()

    try:
        with encode_lock:
            # 不截断文本，先主动检查真实 Token 长度。
            tokenized = model.tokenizer(
                request.texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            token_lengths = [
                len(input_ids)
                for input_ids in tokenized["input_ids"]
            ]

            max_tokens = app.state.effective_max_tokens

            for index, token_length in enumerate(token_lengths):
                if token_length > max_tokens:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"texts[{index}] 的 Token 数为 "
                            f"{token_length}，超过限制 "
                            f"{max_tokens}"
                        ),
                    )

            embeddings = model.encode(
                request.texts,
                batch_size=min(
                    EMBEDDING_BATCH_SIZE,
                    len(request.texts),
                ),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(
                    request.normalize,
                ),
            )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Embedding 编码失败：count=%s, normalize=%s",
            len(request.texts),
            request.normalize,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding 编码失败",
        ) from exc

    embeddings_array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # 防止单条文本在某些版本中被返回为一维数组。
    if embeddings_array.ndim == 1:
        embeddings_array = embeddings_array.reshape(
            1,
            -1,
        )

    if embeddings_array.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型返回了非法的向量形状",
        )

    if not np.isfinite(embeddings_array).all():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型返回的向量包含 NaN 或无穷值",
        )

    count = int(embeddings_array.shape[0])
    dimension = int(embeddings_array.shape[1])

    if count != len(request.texts):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "返回向量数量与输入文本数量不一致"
            ),
        )

    if dimension != app.state.dimension:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"返回向量维度异常："
                f"expected={app.state.dimension}, "
                f"actual={dimension}"
            ),
        )

    norms = np.linalg.norm(
        embeddings_array,
        axis=1,
    )

    latency = time.perf_counter() - request_start

    logger.info(
        "Embedding 请求完成："
        "count=%s, dimension=%s, normalize=%s, "
        "token_lengths=%s, norm_min=%.6f, "
        "norm_max=%.6f, latency_seconds=%.4f",
        count,
        dimension,
        request.normalize,
        token_lengths,
        float(norms.min()),
        float(norms.max()),
        latency,
    )

    return EmbedResponse(
        count=count,
        dimension=dimension,
        embeddings=embeddings_array.tolist(),
    )
