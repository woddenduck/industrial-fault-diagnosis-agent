"""模型服务客户端统一异常。"""

from typing import Optional


class ModelServiceError(RuntimeError):
    """所有模型服务异常的统一父类。"""

    def __init__(
        self,
        *,
        service: str,
        message: str,
        http_status_code: int,
        retryable: bool,
        upstream_status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)

        self.service = service
        self.message = message
        self.http_status_code = http_status_code
        self.retryable = retryable
        self.upstream_status_code = upstream_status_code


class ModelServiceConnectionError(ModelServiceError):
    """无法连接模型服务。"""

    def __init__(
        self,
        *,
        service: str,
        message: str,
    ) -> None:
        super().__init__(
            service=service,
            message=message,
            http_status_code=503,
            retryable=True,
        )


class ModelServiceTimeoutError(ModelServiceError):
    """访问模型服务超时。"""

    def __init__(
        self,
        *,
        service: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            service=service,
            message=message,
            http_status_code=504,
            retryable=retryable,
        )


class ModelServiceResponseError(ModelServiceError):
    """模型服务返回异常状态码或非法响应。"""

    def __init__(
        self,
        *,
        service: str,
        message: str,
        http_status_code: int = 502,
        upstream_status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            service=service,
            message=message,
            http_status_code=http_status_code,
            retryable=retryable,
            upstream_status_code=upstream_status_code,
        )
