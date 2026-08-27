"""
vLLM HTTP 客户端模块。

负责封装 Gateway 对 vLLM OpenAI 兼容接口的访问，包括：
- 查询 /v1/models；
- 调用 /v1/chat/completions；
- 普通请求；
- 真正的 SSE 流式请求；
- 连接、读取和总请求超时；
- 分类型异常转换；
- 客户端断开后的上游连接释放。

聊天请求不会自动重试，避免重复生成和浪费 GPU 资源。
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.clients.errors import ModelServiceError
from app.config import settings


logger = logging.getLogger("uvicorn.error")

DisconnectChecker = Callable[[], Awaitable[bool]]


class VLLMServiceError(ModelServiceError):
    """所有可转换为 Gateway 标准错误的 vLLM 异常基类。

    保留 Day 12 的异常属性，同时接入 Day 13 的统一异常父类。
    """

    status_code = 502
    code = "LLM_UPSTREAM_ERROR"
    public_message = "模型服务返回异常"
    retryable = False

    def __init__(self, detail: str = "") -> None:
        self.detail = detail or self.public_message
        super().__init__(
            service="llm",
            message=self.detail,
            http_status_code=self.status_code,
            retryable=self.retryable,
        )


class VLLMRequestError(VLLMServiceError):
    """请求已通过 Gateway 校验，但仍被 vLLM 判定为非法。"""

    status_code = 400
    code = "INVALID_LLM_REQUEST"
    public_message = "请求参数无法被模型服务处理"


class VLLMModelNotFoundError(VLLMServiceError):
    """请求的模型未在 vLLM 中加载。"""

    status_code = 404
    code = "MODEL_NOT_FOUND"
    public_message = "请求的模型不存在或尚未加载"


class VLLMConnectionError(VLLMServiceError):
    """Gateway 无法连接到 vLLM。"""

    status_code = 503
    code = "LLM_UNAVAILABLE"
    public_message = "模型服务暂时不可用"
    retryable = True


class VLLMTimeoutError(VLLMServiceError):
    """连接、读取或总请求时间超过限制。"""

    status_code = 504
    code = "LLM_TIMEOUT"
    public_message = "模型服务响应超时"


class VLLMUpstreamError(VLLMServiceError):
    """vLLM 返回 5xx、非法 JSON 或不符合预期的响应。"""

    status_code = 502
    code = "LLM_UPSTREAM_ERROR"
    public_message = "模型服务返回异常"


@dataclass
class StreamState:
    """供 Gateway 在流结束后记录最终状态。"""

    error_code: str | None = None
    client_disconnected: bool = False


class VLLMClient:
    """封装所有访问 vLLM OpenAI 兼容接口的逻辑。"""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float,
        read_timeout: float,
        write_timeout: float,
        pool_timeout: float,
        total_timeout: float,
        health_connect_timeout: float,
        health_read_timeout: float,
        health_total_timeout: float,
        stream_read_timeout: float,
        stream_total_timeout: float,
        retry_delay: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.read_timeout = read_timeout
        self.total_timeout = total_timeout
        self.health_total_timeout = health_total_timeout
        self.stream_read_timeout = stream_read_timeout
        self.stream_total_timeout = stream_total_timeout
        self.retry_delay = retry_delay

        timeout_config = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self.health_timeout = httpx.Timeout(
            connect=health_connect_timeout,
            read=health_read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        stream_timeout_config = httpx.Timeout(
            connect=connect_timeout,
            read=stream_read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )

        # 普通请求和流式请求使用独立连接池，以便采用不同读取超时。
        # httpx 默认不会自动重试；仅 health/get_models 显式有限重试。
        self._client = httpx.AsyncClient(
            timeout=timeout_config,
            trust_env=False,
            follow_redirects=True,
        )
        self._stream_client = httpx.AsyncClient(
            timeout=stream_timeout_config,
            trust_env=False,
            follow_redirects=True,
        )

    def _url(self, path: str) -> str:
        """构造完整的 vLLM 接口地址。"""

        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _safe_reason(value: str, limit: int = 300) -> str:
        """压缩并截断内部错误原因，避免日志包含大段上游内容。"""

        compact = " ".join(value.split())
        return compact[:limit] or "unknown"

    @classmethod
    def _response_error(cls, response: httpx.Response) -> str:
        """尽量提取 vLLM 返回的简短错误信息，仅用于服务端日志。"""

        body = response.text.strip()
        if not body:
            return f"HTTP {response.status_code}"

        try:
            payload = response.json()
        except ValueError:
            return cls._safe_reason(
                f"HTTP {response.status_code}: {body}"
            )

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return cls._safe_reason(
                        f"HTTP {response.status_code}: {message}"
                    )

            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return cls._safe_reason(
                    f"HTTP {response.status_code}: {message}"
                )

        return cls._safe_reason(
            f"HTTP {response.status_code}: {body}"
        )

    @staticmethod
    def _looks_like_model_not_found(
        response: httpx.Response,
        reason: str,
    ) -> bool:
        """识别 vLLM 的模型不存在错误。"""

        if response.status_code == 404:
            return True

        lowered = reason.lower()
        model_words = ("model", "模型")
        missing_words = (
            "not found",
            "does not exist",
            "not exist",
            "not served",
            "unknown",
            "不存在",
            "未加载",
        )
        return (
            response.status_code in {400, 422}
            and any(word in lowered for word in model_words)
            and any(word in lowered for word in missing_words)
        )

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        operation: str,
        model_request: bool,
    ) -> None:
        """将上游 HTTP 状态转换为细分异常。"""

        if not response.is_error:
            return

        reason = self._response_error(response)
        detail = f"{operation}失败，{reason}"

        if model_request and self._looks_like_model_not_found(
            response,
            reason,
        ):
            raise VLLMModelNotFoundError(detail)

        if response.status_code in {400, 422}:
            raise VLLMRequestError(detail)

        raise VLLMUpstreamError(detail)

    async def _await_request(
        self,
        awaitable: Awaitable[httpx.Response],
        *,
        operation: str,
        total_timeout: float | None = None,
    ) -> httpx.Response:
        """增加总请求超时并转换网络异常。"""

        effective_total_timeout = (
            self.total_timeout
            if total_timeout is None
            else total_timeout
        )

        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=effective_total_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise VLLMTimeoutError(
                f"{operation}超过总请求超时 "
                f"{effective_total_timeout:.1f}s"
            ) from exc
        except httpx.TimeoutException as exc:
            raise VLLMTimeoutError(
                f"{operation}发生 {type(exc).__name__}"
            ) from exc
        except httpx.RequestError as exc:
            raise VLLMConnectionError(
                f"{operation}无法连接 vLLM：{type(exc).__name__}"
            ) from exc

    async def _await_idempotent_request(
        self,
        request_factory: Callable[[], Awaitable[httpx.Response]],
        *,
        operation: str,
        request_id: str,
        total_timeout: float,
    ) -> httpx.Response:
        """健康检查和模型列表遇到瞬时连接问题时最多重试一次。"""

        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._await_request(
                    request_factory(),
                    operation=operation,
                    total_timeout=total_timeout,
                )
            except VLLMConnectionError:
                if attempt >= max_attempts:
                    raise
            except VLLMTimeoutError as exc:
                if (
                    attempt >= max_attempts
                    or not isinstance(exc.__cause__, httpx.ConnectTimeout)
                ):
                    raise

            logger.warning(
                "request_id=%s operation=%s attempt=%s/%s "
                "发生瞬时连接问题，准备有限重试",
                request_id,
                operation,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(self.retry_delay)

        raise RuntimeError("vLLM 有限重试流程异常结束")

    async def health(self, request_id: str) -> bool:
        """检查 vLLM 是否可访问且至少加载了一个模型。"""

        await self.get_models(request_id)
        return True

    async def get_models(self, request_id: str) -> list[str]:
        """获取 vLLM 当前实际加载的模型列表。"""

        url = self._url("/v1/models")
        logger.info(
            "request_id=%s 调用 vLLM 模型列表接口：url=%s",
            request_id,
            url,
        )

        response = await self._await_idempotent_request(
            lambda: self._client.get(
                url,
                timeout=self.health_timeout,
            ),
            operation="获取模型列表",
            request_id=request_id,
            total_timeout=self.health_total_timeout,
        )
        self._raise_for_status(
            response,
            operation="获取模型列表",
            model_request=False,
        )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise VLLMUpstreamError(
                "vLLM 模型列表接口返回的不是合法 JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise VLLMUpstreamError(
                "vLLM 模型列表响应不是 JSON 对象"
            )

        model_data = payload.get("data")
        if not isinstance(model_data, list):
            raise VLLMUpstreamError(
                "vLLM 模型列表响应缺少合法的 data 字段"
            )

        model_ids = [
            item["id"].strip()
            for item in model_data
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
            )
        ]

        if not model_ids:
            raise VLLMUpstreamError("vLLM 没有返回任何可用模型")

        return model_ids

    async def chat(
        self,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """向 vLLM 发送普通聊天请求。"""

        url = self._url("/v1/chat/completions")
        #request_payload = {**payload, "stream": False}
        request_payload = {
            **payload,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        logger.info(
            "request_id=%s 调用 vLLM 普通接口：url=%s, model=%s",
            request_id,
            url,
            request_payload.get("model"),
        )

        response = await self._await_request(
            self._client.post(url, json=request_payload),
            operation="普通聊天请求",
        )
        self._raise_for_status(
            response,
            operation="普通聊天请求",
            model_request=True,
        )

        try:
            result: Any = response.json()
        except ValueError as exc:
            raise VLLMUpstreamError(
                "vLLM 聊天接口返回的不是合法 JSON"
            ) from exc

        if not isinstance(result, dict):
            raise VLLMUpstreamError("vLLM 聊天响应不是 JSON 对象")

        return result

    async def _read_next_stream_chunk(
        self,
        iterator: AsyncIterator[bytes],
        *,
        started_at: float,
    ) -> bytes:
        """读取下一个非空数据块，同时执行读取和总请求超时控制。"""

        while True:
            remaining = self.stream_total_timeout - (
                time.perf_counter() - started_at
            )
            if remaining <= 0:
                raise VLLMTimeoutError(
                    f"流式请求超过总请求超时 {self.stream_total_timeout:.1f}s"
                )

            try:
                chunk = await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=remaining,
                )
            except StopAsyncIteration:
                raise
            except asyncio.TimeoutError as exc:
                raise VLLMTimeoutError(
                    f"流式请求超过总请求超时 {self.stream_total_timeout:.1f}s"
                ) from exc
            except httpx.TimeoutException as exc:
                raise VLLMTimeoutError(
                    f"读取 vLLM 流式响应超时：{type(exc).__name__}"
                ) from exc
            except httpx.RequestError as exc:
                raise VLLMUpstreamError(
                    f"读取 vLLM 流式响应失败：{type(exc).__name__}"
                ) from exc

            if chunk:
                return chunk

    @staticmethod
    def _stream_error_event(
        request_id: str,
        *,
        code: str,
        message: str,
    ) -> bytes:
        """构造与普通错误响应一致的 SSE 错误事件。"""

        data = json.dumps(
            {
                "request_id": request_id,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
            ensure_ascii=False,
        )
        return f"\n\nevent: error\ndata: {data}\n\n".encode("utf-8")

    async def chat_stream(
        self,
        payload: dict[str, Any],
        request_id: str,
        is_disconnected: DisconnectChecker,
    ) -> tuple[AsyncIterator[bytes], StreamState]:
        """
        建立 vLLM SSE 流，并返回逐块转发迭代器和流状态。

        Gateway 会在发送下游响应头前检查：
        - 上游连接是否成功；
        - HTTP 状态是否正常；
        - Content-Type 是否为 SSE；
        - 首个数据块是否能在超时范围内到达。

        因此，连接失败、模型不存在以及首块超时仍可转换为标准 JSON
        状态码。响应头发出后的中途异常只能通过 SSE error 事件通知客户端。
        """

        url = self._url("/v1/chat/completions")
        #request_payload = {**payload, "stream": True}
        request_payload = {
            **payload,
            "stream": True,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        started_at = time.perf_counter()
        state = StreamState()

        logger.info(
            "request_id=%s 建立 vLLM 流式连接：url=%s, model=%s",
            request_id,
            url,
            request_payload.get("model"),
        )

        request = self._stream_client.build_request(
            "POST",
            url,
            json=request_payload,
            headers={"Accept": "text/event-stream"},
        )

        response: httpx.Response | None = None
        try:
            response = await self._await_request(
                self._stream_client.send(request, stream=True),
                operation="建立流式连接",
                total_timeout=self.stream_total_timeout,
            )

            if response.is_error:
                try:
                    await asyncio.wait_for(
                        response.aread(),
                        timeout=min(
                            self.stream_read_timeout,
                            self.stream_total_timeout,
                        ),
                    )
                except (asyncio.TimeoutError, httpx.TimeoutException):
                    raise VLLMTimeoutError(
                        "读取 vLLM 流式错误响应超时"
                    )
                except httpx.RequestError as exc:
                    raise VLLMUpstreamError(
                        "读取 vLLM 流式错误响应失败"
                    ) from exc

                self._raise_for_status(
                    response,
                    operation="流式聊天请求",
                    model_request=True,
                )

            content_type = response.headers.get(
                "content-type",
                "",
            ).lower()
            if "text/event-stream" not in content_type:
                try:
                    await asyncio.wait_for(
                        response.aread(),
                        timeout=min(
                            self.stream_read_timeout,
                            self.stream_total_timeout,
                        ),
                    )
                except (asyncio.TimeoutError, httpx.TimeoutException):
                    raise VLLMTimeoutError(
                        "读取 vLLM 非流式响应超时"
                    )
                except httpx.RequestError as exc:
                    raise VLLMUpstreamError(
                        "读取 vLLM 非流式响应失败"
                    ) from exc

                body = self._safe_reason(response.text or "<empty>")
                raise VLLMUpstreamError(
                    "vLLM 流式接口未返回 text/event-stream；"
                    f"content_type={content_type or 'unknown'}；"
                    f"body={body}"
                )

            raw_iterator = response.aiter_raw()
            try:
                first_chunk = await self._read_next_stream_chunk(
                    raw_iterator,
                    started_at=started_at,
                )
            except StopAsyncIteration as exc:
                raise VLLMUpstreamError(
                    "vLLM 流式响应在首个数据块前结束"
                ) from exc

        except Exception:
            if response is not None:
                await response.aclose()
            raise

        logger.info(
            "request_id=%s 收到首个流式数据块：ttfc=%.3fs",
            request_id,
            time.perf_counter() - started_at,
        )

        async def relay_stream() -> AsyncIterator[bytes]:
            """原样转发上游 SSE 字节，并在结束时释放连接。"""

            total_bytes = 0

            try:
                if await is_disconnected():
                    state.client_disconnected = True
                    return

                total_bytes += len(first_chunk)
                yield first_chunk

                while True:
                    if await is_disconnected():
                        state.client_disconnected = True
                        logger.info(
                            "request_id=%s 客户端已断开，停止读取 vLLM 流",
                            request_id,
                        )
                        break

                    try:
                        chunk = await self._read_next_stream_chunk(
                            raw_iterator,
                            started_at=started_at,
                        )
                    except StopAsyncIteration:
                        break

                    total_bytes += len(chunk)
                    yield chunk

            except asyncio.CancelledError:
                state.client_disconnected = True
                logger.info(
                    "request_id=%s 流式任务被取消，准备关闭上游连接",
                    request_id,
                )
                raise
            except VLLMTimeoutError as exc:
                state.error_code = exc.code
                logger.warning(
                    "request_id=%s 流式请求失败：error_type=%s, detail=%s",
                    request_id,
                    exc.code,
                    self._safe_reason(exc.detail),
                )
                yield self._stream_error_event(
                    request_id,
                    code=exc.code,
                    message=exc.public_message,
                )
                yield b"data: [DONE]\n\n"
            except VLLMServiceError as exc:
                state.error_code = exc.code
                logger.warning(
                    "request_id=%s 流式请求失败：error_type=%s, detail=%s",
                    request_id,
                    exc.code,
                    self._safe_reason(exc.detail),
                )
                yield self._stream_error_event(
                    request_id,
                    code=exc.code,
                    message=exc.public_message,
                )
                yield b"data: [DONE]\n\n"
            finally:
                await response.aclose()
                logger.info(
                    "request_id=%s vLLM 流式连接关闭："
                    "client_disconnected=%s, bytes=%s, duration=%.3fs",
                    request_id,
                    state.client_disconnected,
                    total_bytes,
                    time.perf_counter() - started_at,
                )

        return relay_stream(), state

    async def close(self) -> None:
        """关闭普通和流式异步 HTTP 客户端连接池。"""

        if not self._client.is_closed:
            await self._client.aclose()
        if not self._stream_client.is_closed:
            await self._stream_client.aclose()


vllm_client = VLLMClient(
    base_url=settings.vllm_base_url,
    connect_timeout=settings.vllm_connect_timeout,
    read_timeout=settings.vllm_read_timeout,
    write_timeout=settings.vllm_write_timeout,
    pool_timeout=settings.vllm_pool_timeout,
    total_timeout=settings.vllm_total_timeout,
    health_connect_timeout=settings.health_connect_timeout,
    health_read_timeout=settings.health_read_timeout,
    health_total_timeout=settings.health_total_timeout,
    stream_read_timeout=settings.vllm_stream_read_timeout,
    stream_total_timeout=settings.vllm_stream_total_timeout,
    retry_delay=settings.model_service_retry_delay,
)
