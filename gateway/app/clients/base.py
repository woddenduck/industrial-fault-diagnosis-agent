"""模型服务客户端公共基础实现。"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

import httpx

from app.clients.errors import (
    ModelServiceConnectionError,
    ModelServiceResponseError,
    ModelServiceTimeoutError,
)
from app.config import MODEL_SERVICE_RETRY_DELAY_SECONDS


logger = logging.getLogger(__name__)


class BaseModelServiceClient:
    """封装模型服务公共 HTTP 请求逻辑。"""

    def __init__(
        self,
        *,
        service_name: str,
        base_url: str,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")

    def _build_url(
        self,
        path: str,
    ) -> str:
        """拼接完整请求地址。"""

        normalized_path = path.lstrip("/")

        return f"{self.base_url}/{normalized_path}"

    def _build_headers(
        self,
        request_id: str,
    ) -> dict[str, str]:
        """构造模型服务请求头。"""

        return {
            "X-Request-ID": request_id,
            "Accept": "application/json",
        }

    @staticmethod
    def _map_upstream_status(
        upstream_status_code: int,
    ) -> int:
        """将底层服务状态码映射为 Gateway 状态码。"""

        if upstream_status_code in {
            400,
            404,
            409,
            422,
            429,
        }:
            return upstream_status_code

        if upstream_status_code >= 500:
            return 502

        return 502

    async def _request(
        self,
        *,
        method: str,
        path: str,
        request_id: str,
        timeout: httpx.Timeout,
        json_data: Optional[dict[str, Any]] = None,
        retry_on_connect: bool = False,
    ) -> httpx.Response:
        """发送普通 HTTP 请求。"""

        url = self._build_url(path)

        max_attempts = 2 if retry_on_connect else 1

        for attempt in range(1, max_attempts + 1):
            start_time = time.perf_counter()

            logger.info(
                "request_id=%s service=%s method=%s url=%s "
                "attempt=%s/%s 开始调用模型服务",
                request_id,
                self.service_name,
                method,
                url,
                attempt,
                max_attempts,
            )

            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=False,
                ) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=json_data,
                        headers=self._build_headers(request_id),
                    )

                response.raise_for_status()

                latency = time.perf_counter() - start_time

                logger.info(
                    "request_id=%s service=%s method=%s "
                    "status_code=%s latency=%.3f 调用成功",
                    request_id,
                    self.service_name,
                    method,
                    response.status_code,
                    latency,
                )

                return response

            except httpx.ConnectTimeout as exc:
                latency = time.perf_counter() - start_time

                logger.warning(
                    "request_id=%s service=%s url=%s "
                    "attempt=%s/%s latency=%.3f 连接超时",
                    request_id,
                    self.service_name,
                    url,
                    attempt,
                    max_attempts,
                    latency,
                )

                if attempt < max_attempts:
                    await asyncio.sleep(
                        MODEL_SERVICE_RETRY_DELAY_SECONDS,
                    )
                    continue

                raise ModelServiceTimeoutError(
                    service=self.service_name,
                    message=(
                        f"连接 {self.service_name} 服务超时"
                    ),
                    retryable=True,
                ) from exc

            except httpx.ConnectError as exc:
                latency = time.perf_counter() - start_time

                logger.warning(
                    "request_id=%s service=%s url=%s "
                    "attempt=%s/%s latency=%.3f 连接失败",
                    request_id,
                    self.service_name,
                    url,
                    attempt,
                    max_attempts,
                    latency,
                )

                if attempt < max_attempts:
                    await asyncio.sleep(
                        MODEL_SERVICE_RETRY_DELAY_SECONDS,
                    )
                    continue

                raise ModelServiceConnectionError(
                    service=self.service_name,
                    message=(
                        f"无法连接 {self.service_name} 服务"
                    ),
                ) from exc

            except httpx.TimeoutException as exc:
                latency = time.perf_counter() - start_time

                logger.warning(
                    "request_id=%s service=%s url=%s "
                    "latency=%.3f 请求超时",
                    request_id,
                    self.service_name,
                    url,
                    latency,
                )

                raise ModelServiceTimeoutError(
                    service=self.service_name,
                    message=(
                        f"调用 {self.service_name} 服务超时"
                    ),
                    retryable=False,
                ) from exc

            except httpx.HTTPStatusError as exc:
                upstream_status_code = (
                    exc.response.status_code
                )

                response_body = exc.response.text[:500]

                logger.warning(
                    "request_id=%s service=%s url=%s "
                    "upstream_status=%s response=%s",
                    request_id,
                    self.service_name,
                    url,
                    upstream_status_code,
                    response_body,
                )

                raise ModelServiceResponseError(
                    service=self.service_name,
                    message=(
                        f"{self.service_name} 服务返回异常状态码："
                        f"{upstream_status_code}"
                    ),
                    http_status_code=self._map_upstream_status(
                        upstream_status_code,
                    ),
                    upstream_status_code=upstream_status_code,
                    retryable=False,
                ) from exc

            except httpx.RequestError as exc:
                logger.warning(
                    "request_id=%s service=%s url=%s "
                    "发生网络请求异常：%s",
                    request_id,
                    self.service_name,
                    url,
                    exc,
                )

                raise ModelServiceConnectionError(
                    service=self.service_name,
                    message=(
                        f"访问 {self.service_name} 服务失败"
                    ),
                ) from exc

        raise RuntimeError("模型服务请求流程异常结束")

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        request_id: str,
        timeout: httpx.Timeout,
        json_data: Optional[dict[str, Any]] = None,
        retry_on_connect: bool = False,
    ) -> dict[str, Any]:
        """发送请求并解析 JSON 对象响应。"""

        response = await self._request(
            method=method,
            path=path,
            request_id=request_id,
            timeout=timeout,
            json_data=json_data,
            retry_on_connect=retry_on_connect,
        )

        try:
            result = response.json()
        except ValueError as exc:
            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    f"{self.service_name} 服务返回的不是合法 JSON"
                ),
            ) from exc

        if not isinstance(result, dict):
            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    f"{self.service_name} 服务响应必须是 JSON 对象"
                ),
            )

        return result

    async def _request_status(
        self,
        *,
        method: str,
        path: str,
        request_id: str,
        timeout: httpx.Timeout,
        retry_on_connect: bool = False,
    ) -> bool:
        """发送只关心状态码的请求。"""

        await self._request(
            method=method,
            path=path,
            request_id=request_id,
            timeout=timeout,
            retry_on_connect=retry_on_connect,
        )

        return True

    async def _stream_bytes(
        self,
        *,
        method: str,
        path: str,
        request_id: str,
        timeout: httpx.Timeout,
        json_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """发送流式请求并逐块返回字节。"""

        url = self._build_url(path)
        start_time = time.perf_counter()

        logger.info(
            "request_id=%s service=%s url=%s 开始流式调用",
            request_id,
            self.service_name,
            url,
        )

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=self._build_headers(request_id),
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk

            latency = time.perf_counter() - start_time

            logger.info(
                "request_id=%s service=%s latency=%.3f "
                "流式调用结束",
                request_id,
                self.service_name,
                latency,
            )

        except httpx.ConnectTimeout as exc:
            raise ModelServiceTimeoutError(
                service=self.service_name,
                message=(
                    f"连接 {self.service_name} 流式服务超时"
                ),
                retryable=True,
            ) from exc

        except httpx.ConnectError as exc:
            raise ModelServiceConnectionError(
                service=self.service_name,
                message=(
                    f"无法连接 {self.service_name} 流式服务"
                ),
            ) from exc

        except httpx.TimeoutException as exc:
            raise ModelServiceTimeoutError(
                service=self.service_name,
                message=(
                    f"{self.service_name} 流式响应超时"
                ),
                retryable=False,
            ) from exc

        except httpx.HTTPStatusError as exc:
            upstream_status_code = (
                exc.response.status_code
            )

            raise ModelServiceResponseError(
                service=self.service_name,
                message=(
                    f"{self.service_name} 流式服务返回异常："
                    f"{upstream_status_code}"
                ),
                http_status_code=self._map_upstream_status(
                    upstream_status_code,
                ),
                upstream_status_code=upstream_status_code,
            ) from exc

        except httpx.RequestError as exc:
            raise ModelServiceConnectionError(
                service=self.service_name,
                message=(
                    f"访问 {self.service_name} 流式服务失败"
                ),
            ) from exc
