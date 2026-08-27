"""Reranker 模型服务客户端。"""

import math
from typing import Any

import httpx

from app.clients.base import BaseModelServiceClient
from app.clients.errors import ModelServiceResponseError
from app.config import (
    HEALTH_CONNECT_TIMEOUT_SECONDS,
    HEALTH_READ_TIMEOUT_SECONDS,
    RERANKER_BASE_URL,
    RERANKER_CONNECT_TIMEOUT_SECONDS,
    RERANKER_POOL_TIMEOUT_SECONDS,
    RERANKER_READ_TIMEOUT_SECONDS,
    RERANKER_WRITE_TIMEOUT_SECONDS,
)


HEALTH_TIMEOUT = httpx.Timeout(
    connect=HEALTH_CONNECT_TIMEOUT_SECONDS,
    read=HEALTH_READ_TIMEOUT_SECONDS,
    write=5.0,
    pool=5.0,
)

RERANKER_TIMEOUT = httpx.Timeout(
    connect=RERANKER_CONNECT_TIMEOUT_SECONDS,
    read=RERANKER_READ_TIMEOUT_SECONDS,
    write=RERANKER_WRITE_TIMEOUT_SECONDS,
    pool=RERANKER_POOL_TIMEOUT_SECONDS,
)


class RerankerClient(BaseModelServiceClient):
    """封装所有 Reranker 服务调用。"""

    def __init__(
        self,
        base_url: str = RERANKER_BASE_URL,
    ) -> None:
        super().__init__(
            service_name="reranker",
            base_url=base_url,
        )

    async def health(
        self,
        request_id: str,
    ) -> bool:
        """检查 Reranker 服务状态。"""

        return await self._request_status(
            method="GET",
            path="/health",
            request_id=request_id,
            timeout=HEALTH_TIMEOUT,
            retry_on_connect=True,
        )

    async def rerank(
        self,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """调用 Reranker 服务，并校验索引、文档和分数。"""

        expected_documents = payload.get("documents")
        expected_top_k = payload.get("top_k")

        if not isinstance(expected_documents, list) or not expected_documents:
            raise RuntimeError(
                "RerankerClient 收到的 documents 必须是非空列表"
            )

        if (
            not isinstance(expected_top_k, int)
            or isinstance(expected_top_k, bool)
            or expected_top_k <= 0
        ):
            raise RuntimeError("RerankerClient 收到的 top_k 非法")

        result = await self._request_json(
            method="POST",
            path="/rerank",
            request_id=request_id,
            timeout=RERANKER_TIMEOUT,
            json_data=payload,
            # 仅连接失败时最多重试一次。
            retry_on_connect=True,
        )

        results = result.get("results")

        if not isinstance(results, list):
            raise ModelServiceResponseError(
                service=self.service_name,
                message="Reranker 响应缺少 results 数组",
            )

        if len(results) != expected_top_k:
            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    "Reranker 响应结果数量与请求 top_k 不一致"
                ),
            )

        seen_indices: set[int] = set()

        for item in results:
            if not isinstance(item, dict):
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message="Reranker results 中存在非法元素",
                )

            index = item.get("index")
            score = item.get("score")
            document = item.get("document")

            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(expected_documents)
            ):
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message="Reranker 结果中的 index 非法或越界",
                )

            if index in seen_indices:
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message="Reranker 响应中存在重复 index",
                )
            seen_indices.add(index)

            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message="Reranker 结果中的 score 非法",
                )

            if not isinstance(document, str) or not document.strip():
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message="Reranker 结果中的 document 非法",
                )

            if document != expected_documents[index]:
                raise ModelServiceResponseError(
                    service=self.service_name,
                    message=(
                        "Reranker 结果中的 document "
                        "与原始 index 对应文档不一致"
                    ),
                )

        return result


reranker_client = RerankerClient()