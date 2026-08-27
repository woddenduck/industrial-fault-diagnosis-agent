"""Embedding 模型服务客户端。"""

import math
from typing import Any

import httpx

from app.clients.base import BaseModelServiceClient
from app.clients.errors import ModelServiceResponseError
from app.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_CONNECT_TIMEOUT_SECONDS,
    EMBEDDING_POOL_TIMEOUT_SECONDS,
    EMBEDDING_READ_TIMEOUT_SECONDS,
    EMBEDDING_WRITE_TIMEOUT_SECONDS,
    HEALTH_CONNECT_TIMEOUT_SECONDS,
    HEALTH_READ_TIMEOUT_SECONDS,
)


HEALTH_TIMEOUT = httpx.Timeout(
    connect=HEALTH_CONNECT_TIMEOUT_SECONDS,
    read=HEALTH_READ_TIMEOUT_SECONDS,
    write=5.0,
    pool=5.0,
)

EMBEDDING_TIMEOUT = httpx.Timeout(
    connect=EMBEDDING_CONNECT_TIMEOUT_SECONDS,
    read=EMBEDDING_READ_TIMEOUT_SECONDS,
    write=EMBEDDING_WRITE_TIMEOUT_SECONDS,
    pool=EMBEDDING_POOL_TIMEOUT_SECONDS,
)


class EmbeddingClient(BaseModelServiceClient):
    """封装所有 Embedding 服务调用。"""

    def __init__(
        self,
        base_url: str = EMBEDDING_BASE_URL,
    ) -> None:
        super().__init__(
            service_name="embedding",
            base_url=base_url,
        )

    async def health(
        self,
        request_id: str,
    ) -> bool:
        """检查 Embedding 服务状态。"""

        return await self._request_status(
            method="GET",
            path="/health",
            request_id=request_id,
            timeout=HEALTH_TIMEOUT,
            retry_on_connect=True,
        )

    async def embed(
        self,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """调用 Embedding 服务生成向量，并校验响应结构。"""

        expected_texts = payload.get("texts")
        if not isinstance(expected_texts, list) or not expected_texts:
            raise RuntimeError("EmbeddingClient 收到的 texts 必须是非空列表")

        expected_count = len(expected_texts)

        result = await self._request_json(
            method="POST",
            path="/embed",
            request_id=request_id,
            timeout=EMBEDDING_TIMEOUT,
            json_data=payload,
            # 仅连接失败时最多重试一次。
            retry_on_connect=True,
        )

        embeddings = result.get("embeddings")
        dimension = result.get("dimension")
        count = result.get("count")

        if not isinstance(embeddings, list) or not embeddings:
            raise ModelServiceResponseError(
                service=self.service_name,
                message="Embedding 响应缺少非空 embeddings 数组",
            )

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ModelServiceResponseError(
                service=self.service_name,
                message="Embedding 响应中的 dimension 非法",
            )

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ModelServiceResponseError(
                service=self.service_name,
                message="Embedding 响应中的 count 非法",
            )

        if count != len(embeddings):
            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    "Embedding 响应中的 count "
                    "与 embeddings 数量不一致"
                ),
            )

        if count != expected_count:
            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    "Embedding 响应数量与请求文本数量不一致"
                ),
            )

        for vector_index, vector in enumerate(embeddings):
            if not isinstance(vector, list):
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message=(
                        f"Embedding 响应中的 embeddings[{vector_index}] "
                        "不是数组"
                    ),
                )

            if len(vector) != dimension:
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message=(
                        f"Embedding 响应中的 embeddings[{vector_index}] "
                        "维度与 dimension 不一致"
                    ),
                )

            for value in vector:
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ModelServiceResponseError(
                        service=self.service_name,
                        message="Embedding 响应中存在非法向量数值",
                    )

        return result


embedding_client = EmbeddingClient()
