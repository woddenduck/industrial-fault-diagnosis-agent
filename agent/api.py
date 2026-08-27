"""工业设备故障诊断 Agent 的 FastAPI 服务入口。

本模块负责：

1. 对外提供统一诊断接口。
2. 注入和返回 Request ID。
3. 将 Agent 业务状态转换为 HTTP 状态码。
4. 提供 Graph 与 Industrial RAG 健康状态。
5. 将未捕获异常转换为稳定 JSON 响应。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import httpx
from fastapi import (
    FastAPI,
    Request,
    Response,
)
from fastapi.responses import JSONResponse

from agent.agent_runner import (
    AgentRunner,
    default_agent_runner,
)
from agent.config import (
    AGENT_HOST,
    AGENT_PORT,
    RAG_HEALTH_URL,
    RAG_TIMEOUT,
)
from agent.schemas import (
    AgentHealthResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    GraphInfoResponse,
)


SERVICE_NAME = (
    "Industrial Fault Diagnosis Agent"
)
SERVICE_VERSION = "1.0.0"


GRAPH_NODES = [
    "intent",
    "check_information",
    "ask_user",
    "device_status",
    "maintenance_history",
    "retrieve",
    "diagnosis",
    "risk_check",
    "report",
    "human_review",
    "final_answer",
]


SUPPORTED_INTENTS = [
    "device_status",
    "maintenance_history",
    "knowledge_query",
    "diagnosis",
]


ERROR_HTTP_STATUS = {
    "DEVICE_NOT_FOUND": 404,
    "KNOWLEDGE_BASE_NOT_FOUND": 404,
    "RAG_UNAVAILABLE": 503,
    "RAG_UPSTREAM_UNAVAILABLE": 503,
    "LLM_UNAVAILABLE": 503,
    "RAG_TIMEOUT": 504,
    "RAG_UPSTREAM_TIMEOUT": 504,
}


REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)


def normalize_request_id(
    value: str | None,
) -> str:
    """清理客户端 Request ID，无效时重新生成。"""

    if value is not None:
        normalized = str(value).strip()

        if REQUEST_ID_PATTERN.fullmatch(
            normalized
        ):
            return normalized

    return uuid4().hex


def resolve_http_status(
    result: DiagnoseResponse,
) -> int:
    """将 Agent 业务结果转换为 HTTP 状态码。"""

    if result.status != "failed":
        return 200

    # result.error 对外展示时可能是可读错误信息，
    # 稳定的机器错误码保存在 errors 列表中。
    error_code = str(result.error or "").strip()

    if error_code not in ERROR_HTTP_STATUS:
        for item in result.errors:
            code = str(item.code or "").strip()
            if code:
                error_code = code
                break

    return ERROR_HTTP_STATUS.get(error_code, 500)


def check_rag_health() -> dict[str, Any]:
    """检查 Industrial RAG 健康状态。

    RAG 不可用时，Agent API 本身仍然可以启动，
    但整体健康状态会标记为 degraded。
    """

    timeout = min(
        max(float(RAG_TIMEOUT), 1.0),
        5.0,
    )

    try:
        with httpx.Client(
            timeout=timeout
        ) as client:
            response = client.get(
                RAG_HEALTH_URL
            )

    except httpx.TimeoutException:
        return {
            "status": "unavailable",
            "endpoint": RAG_HEALTH_URL,
            "error": "RAG_HEALTH_TIMEOUT",
        }

    except httpx.HTTPError as exc:
        return {
            "status": "unavailable",
            "endpoint": RAG_HEALTH_URL,
            "error": type(exc).__name__,
        }

    try:
        data = response.json()
    except ValueError:
        data = {}

    rag_status = ""

    if isinstance(data, dict):
        rag_status = str(
            data.get("status") or ""
        ).strip().lower()

    is_healthy = (
        response.status_code == 200
        and rag_status
        in {
            "healthy",
            "ok",
            "ready",
        }
    )

    return {
        "status": (
            "healthy"
            if is_healthy
            else "unhealthy"
        ),
        "endpoint": RAG_HEALTH_URL,
        "http_status": response.status_code,
        "detail": (
            data
            if isinstance(data, dict)
            else {}
        ),
    }


def create_app(
    *,
    runner: AgentRunner | None = None,
    rag_health_checker: (
        Callable[[], dict[str, Any]]
        | None
    ) = None,
) -> FastAPI:
    """创建 Agent FastAPI 应用。

    参数可以在测试中替换，从而避免调用真实
    Graph 和 Industrial RAG。
    """

    active_runner = (
        runner
        if runner is not None
        else default_agent_runner
    )

    active_rag_health_checker = (
        rag_health_checker
        if rag_health_checker is not None
        else check_rag_health
    )

    api = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        description=(
            "面向工业设备状态查询、维修历史查询、"
            "知识检索和综合故障诊断的统一 Agent API。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @api.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next,
    ):
        """为每个 HTTP 请求建立统一 Request ID。"""

        request_id = normalize_request_id(
            request.headers.get(
                "X-Request-ID"
            )
        )

        request.state.request_id = request_id

        response = await call_next(request)

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    @api.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exception: Exception,
    ) -> JSONResponse:
        """将未捕获异常转换成稳定失败响应。"""

        request_id = getattr(
            request.state,
            "request_id",
            uuid4().hex,
        )

        result = DiagnoseResponse(
            request_id=request_id,
            status="failed",
            intent="unknown",
            rag_status="not_called",
            risk_level="unknown",
            final_answer=(
                "Agent API 执行失败，请稍后重试。"
            ),
            next_action="retry_later",
            error="AGENT_API_ERROR",
            errors=[
                {
                    "code": "AGENT_API_ERROR",
                    "message": (
                        "Agent API 出现未处理异常："
                        f"{type(exception).__name__}"
                    ),
                    "node": "agent_api",
                    "retryable": True,
                }
            ],
            execution_trace=[],
            elapsed_ms=0.0,
        )

        return JSONResponse(
            status_code=500,
            content=result.model_dump(
                mode="json"
            ),
            headers={
                "X-Request-ID": request_id
            },
        )

    @api.get(
        "/health",
        response_model=AgentHealthResponse,
        tags=["系统"],
        summary="检查 Agent 服务健康状态",
    )
    def health() -> AgentHealthResponse:
        rag_health = (
            active_rag_health_checker()
        )

        rag_status = str(
            rag_health.get("status") or ""
        ).lower()

        overall_status = (
            "healthy"
            if rag_status == "healthy"
            else "degraded"
        )

        return AgentHealthResponse(
            status=overall_status,
            graph="ready",
            rag=rag_health,
        )

    @api.get(
        "/v1/graph",
        response_model=GraphInfoResponse,
        tags=["系统"],
        summary="查看 Agent Graph 信息",
    )
    def graph_info() -> GraphInfoResponse:
        return GraphInfoResponse(
            name=(
                "industrial_fault_diagnosis_graph"
            ),
            nodes=GRAPH_NODES,
            intents=SUPPORTED_INTENTS,
        )

    @api.post(
        "/v1/diagnose",
        response_model=DiagnoseResponse,
        tags=["诊断"],
        summary="执行工业设备故障诊断",
    )
    def diagnose(
        payload: DiagnoseRequest,
        request: Request,
        response: Response,
    ) -> DiagnoseResponse:
        request_id = getattr(
            request.state,
            "request_id",
            uuid4().hex,
        )

        result = active_runner.run(
            payload,
            request_id=request_id,
        )

        response.status_code = (
            resolve_http_status(result)
        )

        response.headers[
            "X-Request-ID"
        ] = result.request_id

        return result

    return api


app = create_app()


def main() -> None:
    """直接启动 Agent API。"""

    import uvicorn

    uvicorn.run(
        app,
        host=AGENT_HOST,
        port=AGENT_PORT,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "app",
    "create_app",
    "check_rag_health",
    "normalize_request_id",
    "resolve_http_status",
]
