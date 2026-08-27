"""
Industrial RAG HTTP Client。

职责：
1. 调用 Industrial RAG POST /chat；
2. 区分 HTTP 成功与业务成功；
3. 解析 answered / rejected；
4. 标准化 HTTP 和网络错误；
5. 保留 RAG request_id、sources 和降级信息；
6. 对可安全重试的连接错误执行有限重试。

本模块不负责：
- LangGraph 路由；
- 生成最终诊断；
- 执行风险判断。
"""

from __future__ import annotations

from typing import Any

import httpx

from agent.config import (
    RAG_BASE_URL,
    RAG_MAX_RETRIES,
    RAG_TIMEOUT,
)


class RAGClient:
    """Industrial RAG /chat 同步客户端。"""

    RETRYABLE_HTTP_STATUS = {
        502,
        503,
    }

    ERROR_CODE_BY_STATUS = {
        400: "RAG_INVALID_REQUEST",
        404: "KNOWLEDGE_BASE_NOT_FOUND",
        409: "KNOWLEDGE_BASE_NOT_READY",
        422: "RAG_VALIDATION_FAILED",
        502: "RAG_UPSTREAM_UNAVAILABLE",
        503: "RAG_UNAVAILABLE",
        504: "RAG_UPSTREAM_TIMEOUT",
    }

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.base_url = (
            base_url or RAG_BASE_URL
        ).rstrip("/")

        self.chat_url = (
            f"{self.base_url}/chat"
        )

        self.timeout = (
            RAG_TIMEOUT
            if timeout is None
            else float(timeout)
        )

        self.max_retries = (
            RAG_MAX_RETRIES
            if max_retries is None
            else int(max_retries)
        )

        if self.timeout <= 0:
            raise ValueError(
                "RAG timeout 必须大于 0"
            )

        if self.max_retries < 0:
            raise ValueError(
                "RAG max_retries 不能小于 0"
            )

    def _post(
        self,
        payload: dict[str, Any],
    ) -> httpx.Response:
        """
        单次 HTTP 请求。

        单独定义该方法，便于单元测试替换，
        不需要启动真实 RAG 服务。
        """

        with httpx.Client(
            timeout=self.timeout,
        ) as client:
            return client.post(
                self.chat_url,
                json=payload,
            )

    @staticmethod
    def _extract_error_message(
        data: dict[str, Any],
        default: str,
    ) -> str:
        """从 FastAPI 或统一错误响应中提取信息。"""

        for key in (
            "detail",
            "message",
            "error",
        ):
            value = data.get(key)

            if isinstance(value, str):
                value = value.strip()

                if value:
                    return value

            if isinstance(value, (dict, list)):
                return str(value)

        return default

    @staticmethod
    def _normalize_sources(
        raw_sources: Any,
    ) -> list[dict[str, Any]]:
        """只保留 RAG 已承诺的稳定引用字段。"""

        if not isinstance(raw_sources, list):
            return []

        sources: list[dict[str, Any]] = []

        for source in raw_sources:
            if not isinstance(source, dict):
                continue

            sources.append(
                {
                    "document": str(
                        source.get(
                            "document",
                            "未知文档",
                        )
                    ),
                    "page": str(
                        source.get(
                            "page",
                            "未知",
                        )
                    ),
                    "section": str(
                        source.get(
                            "section",
                            "未知",
                        )
                    ),
                    "chunk_id": source.get(
                        "chunk_id"
                    ),
                }
            )

        return sources

    @staticmethod
    def _failure(
        *,
        code: str,
        message: str,
        retryable: bool,
        attempts: int,
        http_status: int | None = None,
        rag_request_id: str = "",
    ) -> dict[str, Any]:
        """生成统一失败结果。"""

        return {
            "success": False,
            "status": "error",
            "answer": "",
            "sources": [],
            "llm_called": False,
            "evidence_sufficient": False,
            "allow_llm": False,
            "degraded": False,

            "rag_request_id": rag_request_id,
            "http_status": http_status,
            "attempts": attempts,

            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        attempts: int,
    ) -> dict[str, Any]:
        """将 RAG HTTP Response 转换为稳定结果。"""

        try:
            data = response.json()

        except (ValueError, TypeError) as exc:
            return self._failure(
                code="RAG_INVALID_RESPONSE",
                message=(
                    "Industrial RAG 返回的不是合法 JSON："
                    f"{exc}"
                ),
                retryable=False,
                attempts=attempts,
                http_status=response.status_code,
            )

        if not isinstance(data, dict):
            return self._failure(
                code="RAG_INVALID_RESPONSE",
                message=(
                    "Industrial RAG 响应顶层必须是 JSON Object"
                ),
                retryable=False,
                attempts=attempts,
                http_status=response.status_code,
            )

        rag_request_id = str(
            data.get("request_id") or ""
        )

        if response.status_code != 200:
            code = self.ERROR_CODE_BY_STATUS.get(
                response.status_code,
                "RAG_HTTP_ERROR",
            )

            message = self._extract_error_message(
                data,
                (
                    "Industrial RAG 请求失败："
                    f"HTTP {response.status_code}"
                ),
            )

            return self._failure(
                code=code,
                message=message,
                retryable=(
                    response.status_code
                    in self.RETRYABLE_HTTP_STATUS
                ),
                attempts=attempts,
                http_status=response.status_code,
                rag_request_id=rag_request_id,
            )

        business_status = data.get("status")

        if business_status not in {
            "answered",
            "rejected",
        }:
            return self._failure(
                code="RAG_INVALID_RESPONSE",
                message=(
                    "Industrial RAG 返回了未知业务状态："
                    f"{business_status!r}"
                ),
                retryable=False,
                attempts=attempts,
                http_status=response.status_code,
                rag_request_id=rag_request_id,
            )

        answer = data.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            return self._failure(
                code="RAG_INVALID_RESPONSE",
                message="Industrial RAG 返回了空 answer",
                retryable=False,
                attempts=attempts,
                http_status=response.status_code,
                rag_request_id=rag_request_id,
            )

        decision = data.get("decision")

        if not isinstance(decision, dict):
            decision = {}

        sources = self._normalize_sources(
            data.get("sources")
        )

        if business_status == "answered":
            evidence_sufficient = bool(
                decision.get(
                    "evidence_sufficient",
                    True,
                )
            )
            allow_llm = bool(
                decision.get(
                    "allow_llm",
                    data.get("llm_called", True),
                )
            )

        else:
            evidence_sufficient = False
            allow_llm = False
            sources = []

        return {
            # success 表示 RAG 正确处理了请求。
            # rejected 也是正常业务结果。
            "success": True,
            "status": business_status,
            "answer": answer.strip(),
            "sources": sources,
            "llm_called": bool(
                data.get("llm_called", False)
            ),
            "evidence_sufficient":
                evidence_sufficient,
            "allow_llm": allow_llm,

            "decision": decision,
            "degraded": bool(
                data.get("degraded", False)
            ),
            "retrieval_mode": data.get(
                "retrieval_mode"
            ),
            "rerank_executed": bool(
                data.get(
                    "rerank_executed",
                    False,
                )
            ),
            "rerank_error": data.get(
                "rerank_error"
            ),
            "latency": (
                data.get("latency")
                if isinstance(
                    data.get("latency"),
                    dict,
                )
                else {}
            ),

            "rag_request_id": rag_request_id,
            "knowledge_base_id": data.get(
                "knowledge_base_id"
            ),
            "device_model": data.get(
                "device_model"
            ),

            "http_status": response.status_code,
            "attempts": attempts,
            "error": None,
        }

    def retrieve(
        self,
        question: str,
        knowledge_base_id: str,
        device_model: str,
        *,
        history: list[dict[str, Any]] | None = None,
        history_summary: str | None = None,
    ) -> dict[str, Any]:
        """
        调用 Industrial RAG /chat。

        返回稳定的 answered、rejected 或 error 业务结果。
        """

        question = (
            question.strip()
            if isinstance(question, str)
            else ""
        )
        knowledge_base_id = (
            knowledge_base_id.strip()
            if isinstance(
                knowledge_base_id,
                str,
            )
            else ""
        )
        device_model = (
            device_model.strip().upper()
            if isinstance(device_model, str)
            else ""
        )

        missing_fields = []

        if not question:
            missing_fields.append("question")

        if not knowledge_base_id:
            missing_fields.append(
                "knowledge_base_id"
            )

        if not device_model:
            missing_fields.append(
                "device_model"
            )

        if missing_fields:
            return self._failure(
                code="INVALID_RAG_REQUEST",
                message=(
                    "RAG 请求缺少必要字段："
                    + ", ".join(missing_fields)
                ),
                retryable=False,
                attempts=0,
            )

        payload = {
            "question": question,
            "knowledge_base_id":
                knowledge_base_id,
            "device_model": device_model,
            "history": [
                dict(item)
                for item in (history or [])
                if isinstance(item, dict)
            ],
            "history_summary": (
                history_summary.strip()
                if isinstance(
                    history_summary,
                    str,
                )
                and history_summary.strip()
                else None
            ),
        }

        total_attempts = (
            self.max_retries + 1
        )

        for attempt_index in range(
            total_attempts
        ):
            attempts = attempt_index + 1

            try:
                response = self._post(
                    payload
                )

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                if attempt_index < self.max_retries:
                    continue

                return self._failure(
                    code="RAG_UNAVAILABLE",
                    message=(
                        "无法连接 Industrial RAG："
                        f"{exc}"
                    ),
                    retryable=True,
                    attempts=attempts,
                )

            except httpx.ReadTimeout as exc:
                # 真实 RAG 已经可能完成检索或 LLM 调用，
                # 不自动重复执行耗时请求。
                return self._failure(
                    code="RAG_TIMEOUT",
                    message=(
                        "等待 Industrial RAG 响应超时："
                        f"{exc}"
                    ),
                    retryable=False,
                    attempts=attempts,
                )

            except httpx.TimeoutException as exc:
                return self._failure(
                    code="RAG_TIMEOUT",
                    message=(
                        "Industrial RAG 请求超时："
                        f"{exc}"
                    ),
                    retryable=False,
                    attempts=attempts,
                )

            except httpx.RequestError as exc:
                return self._failure(
                    code="RAG_REQUEST_ERROR",
                    message=(
                        "Industrial RAG 请求失败："
                        f"{exc}"
                    ),
                    retryable=False,
                    attempts=attempts,
                )

            if (
                response.status_code
                in self.RETRYABLE_HTTP_STATUS
                and attempt_index
                < self.max_retries
            ):
                continue

            return self._parse_response(
                response,
                attempts=attempts,
            )

        return self._failure(
            code="RAG_UNKNOWN_ERROR",
            message="Industrial RAG 请求异常结束",
            retryable=False,
            attempts=total_attempts,
        )


default_rag_client = RAGClient()
