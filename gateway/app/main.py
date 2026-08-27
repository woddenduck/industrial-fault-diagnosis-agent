import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.clients.embedding_client import embedding_client
from app.clients.errors import ModelServiceError
from app.clients.reranker_client import reranker_client
from app.clients.vllm_client import (
    VLLMUpstreamError,
    vllm_client,
)
from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthServices,
    ModelInfo,
    ModelsResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    TokenUsage,
)


logger = logging.getLogger("uvicorn.error")


def _request_id(request: Request) -> str:
    """获取中间件生成的 request_id。"""

    return str(getattr(request.state, "request_id", "unknown"))


def _safe_log_text(value: object, limit: int = 300) -> str:
    """生成适合写入日志的短文本。"""

    compact = " ".join(str(value).split())
    return compact[:limit] or "unknown"


def _service_type_for_path(path: str) -> str:
    """根据接口路径确定本次请求对应的服务类型。

    该判断在参数校验前执行，因此 422 错误日志也能携带
    正确的 service_type。
    """

    if path == "/embed":
        return "embedding"
    if path == "/rerank":
        return "reranker"
    if path in {"/chat", "/models"}:
        return "llm"
    return "gateway"


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    service: str,
) -> JSONResponse:
    """构造统一错误响应，并记录错误类型供访问日志使用。"""

    request.state.error_type = code
    request_id = _request_id(request)

    payload = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=code,
            message=message,
        ),
        service=service,
    )

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 Gateway 生命周期中的共享资源。"""

    logger.info(
        "Gateway 启动：llm=%s, embedding=%s, reranker=%s, "
        "llm_connect_timeout=%.1fs, llm_read_timeout=%.1fs, "
        "llm_total_timeout=%.1fs",
        settings.vllm_base_url,
        settings.embedding_base_url,
        settings.reranker_base_url,
        settings.vllm_connect_timeout,
        settings.vllm_read_timeout,
        settings.vllm_total_timeout,
    )
    try:
        yield
    finally:
        await vllm_client.close()


app = FastAPI(
    title="LLM Gateway",
    description="统一管理大模型服务调用的 FastAPI Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
) -> Response:
    """
    在参数校验和路由执行前创建 request_id，并记录基础访问日志。

    不读取请求体，因此不会把完整 Prompt 或模型回答写入日志。
    """

    request.state.request_id = uuid.uuid4().hex
    request.state.start_time = time.perf_counter()
    request.state.service_type = _service_type_for_path(request.url.path)
    request.state.model = "-"
    request.state.stream = "-"
    request.state.error_type = None
    request.state.defer_access_log = False

    logger.info(
        "request_id=%s method=%s path=%s service_type=%s "
        "event=request_started",
        request.state.request_id,
        request.method,
        request.url.path,
        request.state.service_type,
    )

    response: Response | None = None
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    except Exception:
        # 未知异常会在中间件退出后交给全局 Exception 处理器。
        # 先标记错误类型，确保本次访问日志不会显示为成功。
        request.state.error_type = (
            request.state.error_type or "GATEWAY_INTERNAL_ERROR"
        )
        raise
    finally:
        # 流式请求需要等数据流真正结束后再记录完整耗时。
        if not request.state.defer_access_log:
            status_code = 500 if response is None else response.status_code
            latency = time.perf_counter() - request.state.start_time
            error_type = request.state.error_type or "-"
            event = (
                "request_failed"
                if status_code >= 400
                else "request_finished"
            )

            # /health 返回 degraded 时 HTTP 仍为 200，但需要用 warning
            # 和 error_type 明确记录依赖服务异常。
            log_method = (
                logger.info
                if status_code < 400 and error_type == "-"
                else logger.warning
            )
            log_method(
                "request_id=%s method=%s path=%s service_type=%s "
                "event=%s status_code=%s latency=%.3fs "
                "error_type=%s model=%s stream=%s",
                request.state.request_id,
                request.method,
                request.url.path,
                request.state.service_type,
                event,
                status_code,
                latency,
                error_type,
                request.state.model,
                request.state.stream,
            )


def _model_service_error_code(exc: ModelServiceError) -> str:
    """生成稳定、可检索的模型服务错误代码。"""

    existing_code = getattr(exc, "code", None)
    if isinstance(existing_code, str) and existing_code:
        return existing_code

    service = exc.service.upper()
    suffix_by_status = {
        400: "INVALID_REQUEST",
        404: "NOT_FOUND",
        422: "INVALID_REQUEST",
        502: "UPSTREAM_ERROR",
        503: "UNAVAILABLE",
        504: "TIMEOUT",
    }
    suffix = suffix_by_status.get(
        exc.http_status_code,
        "SERVICE_ERROR",
    )
    return f"{service}_{suffix}"


def _model_service_public_message(exc: ModelServiceError) -> str:
    """生成不泄露底层细节的客户端错误信息。"""

    existing_message = getattr(exc, "public_message", None)
    if isinstance(existing_message, str) and existing_message:
        return existing_message

    service_names = {
        "llm": "模型",
        "embedding": "Embedding",
        "reranker": "Reranker",
    }
    service_name = service_names.get(exc.service, exc.service)

    if exc.http_status_code == 503:
        return f"{service_name}服务暂时不可用"
    if exc.http_status_code == 504:
        return f"{service_name}服务响应超时"
    if exc.http_status_code in {400, 422}:
        return f"请求参数无法被{service_name}服务处理"
    if exc.http_status_code == 404:
        return f"请求的{service_name}资源不存在"
    return f"{service_name}服务返回异常"


@app.exception_handler(ModelServiceError)
async def handle_model_service_error(
    request: Request,
    exc: ModelServiceError,
) -> JSONResponse:
    """统一处理 LLM、Embedding 和 Reranker 客户端异常。"""

    code = _model_service_error_code(exc)
    detail = getattr(exc, "detail", exc.message)

    logger.warning(
        "request_id=%s service=%s 模型服务调用失败："
        "error_type=%s, path=%s, status_code=%s, "
        "retryable=%s, detail=%s",
        _request_id(request),
        exc.service,
        code,
        request.url.path,
        exc.http_status_code,
        exc.retryable,
        _safe_log_text(detail),
    )

    return _error_response(
        request,
        status_code=exc.http_status_code,
        code=code,
        message=_model_service_public_message(exc),
        service=exc.service,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """将 FastAPI 默认 422 错误转换为统一格式。"""

    # 仅记录错误字段和类型，不记录 exc.errors() 中可能包含的原始输入。
    error_fields = [
        {
            "loc": ".".join(str(item) for item in error.get("loc", [])),
            "type": str(error.get("type", "unknown")),
        }
        for error in exc.errors()
    ]
    logger.warning(
        "request_id=%s 请求参数校验失败：path=%s, errors=%s",
        _request_id(request),
        request.url.path,
        error_fields,
    )

    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="请求参数校验失败，请检查请求参数",
        service="gateway",
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """统一处理路由不存在、方法不允许等 HTTP 错误。"""

    if exc.status_code == status.HTTP_404_NOT_FOUND:
        code = "NOT_FOUND"
        message = "请求的接口不存在"
    elif exc.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        code = "METHOD_NOT_ALLOWED"
        message = "请求方法不被允许"
    elif exc.status_code == status.HTTP_400_BAD_REQUEST:
        code = "BAD_REQUEST"
        message = "请求格式错误"
    else:
        code = "HTTP_ERROR"
        message = "请求处理失败"

    logger.warning(
        "request_id=%s HTTP 错误：path=%s, status_code=%s, error_type=%s",
        _request_id(request),
        request.url.path,
        exc.status_code,
        code,
    )

    return _error_response(
        request,
        status_code=exc.status_code,
        code=code,
        message=message,
        service="gateway",
    )


@app.exception_handler(Exception)
async def unknown_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """兜底处理未知异常，客户端只收到安全的 500 响应。"""

    # 基础日志只记录异常类型，不记录异常正文、堆栈、本地路径或配置值。
    logger.error(
        "request_id=%s Gateway 未知异常：path=%s, exception=%s",
        _request_id(request),
        request.url.path,
        type(exc).__name__,
    )

    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="GATEWAY_INTERNAL_ERROR",
        message="Gateway 内部错误",
        service="gateway",
    )


async def _check_model_service_health(
    *,
    service: str,
    check: Callable[[str], Awaitable[bool]],
    request_id: str,
) -> tuple[str, str | None]:
    """执行单个服务健康检查，任何异常都只影响该服务状态。"""

    try:
        result = await asyncio.wait_for(
            check(request_id),
            timeout=settings.health_total_timeout,
        )
        if result is not True:
            raise RuntimeError("健康检查未返回 True")
        return "available", None

    except asyncio.TimeoutError:
        code = f"{service.upper()}_HEALTH_TIMEOUT"
        logger.warning(
            "request_id=%s path=/health service_type=%s "
            "event=health_check_failed error_type=%s",
            request_id,
            service,
            code,
        )
        return "unavailable", code

    except ModelServiceError as exc:
        code = _model_service_error_code(exc)
        logger.warning(
            "request_id=%s path=/health service_type=%s "
            "event=health_check_failed error_type=%s "
            "retryable=%s",
            request_id,
            service,
            code,
            exc.retryable,
        )
        return "unavailable", code

    except Exception as exc:
        # 单个健康检查实现自身发生未知异常时，也不能让整个
        # /health 或 Gateway 进程崩溃。只记录异常类型，不记录正文。
        code = f"{service.upper()}_HEALTH_CHECK_ERROR"
        logger.error(
            "request_id=%s path=/health service_type=%s "
            "event=health_check_failed error_type=%s exception=%s",
            request_id,
            service,
            code,
            type(exc).__name__,
        )
        return "unavailable", code


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="检查 Gateway 和三个模型服务状态",
)
async def health_check(
    request: Request,
) -> HealthResponse:
    """并发检查 LLM、Embedding 和 Reranker 服务。

    Gateway 自身仍能正常响应时，即使某个下游服务关闭，
    也返回 HTTP 200，并通过 body.status=degraded 表达依赖降级。
    这样不会把下游故障误判成 Gateway 进程故障。
    """

    request_id = _request_id(request)

    llm_result, embedding_result, reranker_result = await asyncio.gather(
        _check_model_service_health(
            service="llm",
            check=vllm_client.health,
            request_id=request_id,
        ),
        _check_model_service_health(
            service="embedding",
            check=embedding_client.health,
            request_id=request_id,
        ),
        _check_model_service_health(
            service="reranker",
            check=reranker_client.health,
            request_id=request_id,
        ),
    )

    services = HealthServices(
        llm=llm_result[0],
        embedding=embedding_result[0],
        reranker=reranker_result[0],
    )

    healthy = all(
        service_status == "available"
        for service_status in (
            services.llm,
            services.embedding,
            services.reranker,
        )
    )

    if not healthy:
        request.state.error_type = "DEPENDENCY_UNAVAILABLE"

    return HealthResponse(
        status="healthy" if healthy else "degraded",
        services=services,
    )


@app.get(
    "/models",
    response_model=ModelsResponse,
    responses={
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Models"],
    summary="获取可用模型列表",
)
async def list_models(request: Request) -> ModelsResponse:
    """从 vLLM 动态获取当前实际加载的模型。"""

    model_ids = await vllm_client.get_models(_request_id(request))
    return ModelsResponse(
        models=[
            ModelInfo(id=model_id, status="available")
            for model_id in model_ids
        ]
    )




@app.post(
    "/embed",
    response_model=EmbeddingResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Embedding"],
    summary="生成文本向量",
)
async def embed(
    embed_request: EmbeddingRequest,
    http_request: Request,
) -> EmbeddingResponse:
    """调用独立 Embedding 服务并返回 Gateway 统一响应。"""

    request_id = _request_id(http_request)
    start_time = http_request.state.start_time
    http_request.state.service_type = "embedding"
    http_request.state.model = "embedding"

    logger.info(
        "request_id=%s path=/embed service_type=embedding "
        "event=request_metadata text_count=%s normalize=%s "
        "input_chars=%s",
        request_id,
        len(embed_request.texts),
        embed_request.normalize,
        sum(len(text) for text in embed_request.texts),
    )

    raw_result = await embedding_client.embed(
        payload=embed_request.model_dump(mode="json"),
        request_id=request_id,
    )

    latency = round(time.perf_counter() - start_time, 3)

    response = EmbeddingResponse(
        request_id=request_id,
        embeddings=raw_result["embeddings"],
        dimension=raw_result["dimension"],
        count=raw_result["count"],
        latency=latency,
    )

    logger.info(
        "request_id=%s path=/embed service_type=embedding "
        "event=upstream_finished count=%s dimension=%s "
        "latency=%.3fs",
        request_id,
        response.count,
        response.dimension,
        latency,
    )

    return response


@app.post(
    "/rerank",
    response_model=RerankResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Reranker"],
    summary="对候选文档重新排序",
)
async def rerank(
    rerank_request: RerankRequest,
    http_request: Request,
) -> RerankResponse:
    """调用独立 Reranker 服务并返回 Gateway 统一响应。"""

    request_id = _request_id(http_request)
    start_time = http_request.state.start_time
    http_request.state.service_type = "reranker"
    http_request.state.model = "reranker"

    logger.info(
        "request_id=%s path=/rerank service_type=reranker "
        "event=request_metadata document_count=%s top_k=%s "
        "query_chars=%s document_chars=%s",
        request_id,
        len(rerank_request.documents),
        rerank_request.top_k,
        len(rerank_request.query),
        sum(len(document) for document in rerank_request.documents),
    )

    # 要求上游返回全部候选文档分数，由 Gateway 统一排序和截断。
    upstream_payload = rerank_request.model_dump(mode="json")
    upstream_payload["top_k"] = len(rerank_request.documents)

    raw_result = await reranker_client.rerank(
        payload=upstream_payload,
        request_id=request_id,
    )

    # 分数降序；分数相同时按原始索引升序，保证结果稳定。
    sorted_items = sorted(
        raw_result["results"],
        key=lambda item: (
            -float(item["score"]),
            int(item["index"]),
        ),
    )
    selected_items = sorted_items[:rerank_request.top_k]

    results = [
        RerankResult(
            index=item["index"],
            score=float(item["score"]),
            document=item["document"],
        )
        for item in selected_items
    ]

    latency = round(time.perf_counter() - start_time, 3)

    response = RerankResponse(
        request_id=request_id,
        results=results,
        latency=latency,
    )

    logger.info(
        "request_id=%s path=/rerank service_type=reranker "
        "event=upstream_finished document_count=%s returned=%s "
        "latency=%.3fs",
        request_id,
        len(rerank_request.documents),
        len(response.results),
        latency,
    )

    return response

def _safe_non_negative_int(value: object) -> int:
    """将 vLLM usage 字段安全转换为非负整数。"""

    try:
        result = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(result, 0)


def _build_chat_payload(
    chat_request: ChatRequest,
    *,
    stream: bool,
) -> dict[str, object]:
    """将 Gateway 请求模型转换为 vLLM 请求格式。"""

    return {
        "model": chat_request.model,
        "messages": [
            message.model_dump(mode="json")
            for message in chat_request.messages
        ],
        "temperature": chat_request.temperature,
        "max_tokens": chat_request.max_tokens,
        "stream": stream,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {
            "description": (
                "stream=false 时返回 ChatResponse JSON；"
                "stream=true 时返回 text/event-stream SSE 数据流"
            )
        },
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Chat"],
    summary="发送普通或流式聊天请求",
)
async def chat(
    chat_request: ChatRequest,
    http_request: Request,
) -> ChatResponse | StreamingResponse:
    """
    统一聊天接口。

    - stream=false：等待 vLLM 完整生成后返回统一 JSON；
    - stream=true：不缓存完整回答，原样实时转发 vLLM 的 SSE 数据块。
    """

    request_id = _request_id(http_request)
    start_time = http_request.state.start_time
    http_request.state.service_type = "llm"
    http_request.state.model = chat_request.model
    http_request.state.stream = str(chat_request.stream).lower()

    logger.info(
        "request_id=%s path=/chat service_type=llm "
        "event=request_metadata model=%s stream=%s "
        "message_count=%s input_chars=%s",
        request_id,
        chat_request.model,
        chat_request.stream,
        len(chat_request.messages),
        sum(len(message.content) for message in chat_request.messages),
    )

    if chat_request.stream:
        payload = _build_chat_payload(chat_request, stream=True)

        # 先连接上游并预取首个数据块。此阶段失败仍能返回 404/502/503/504 JSON。
        stream_iterator, stream_state = await vllm_client.chat_stream(
            payload=payload,
            request_id=request_id,
            is_disconnected=http_request.is_disconnected,
        )

        http_request.state.defer_access_log = True

        async def logged_stream() -> AsyncIterator[bytes]:
            """转发流式数据，并在流真正结束后记录完整访问日志。"""

            try:
                async for chunk in stream_iterator:
                    yield chunk
            finally:
                error_type = stream_state.error_code
                if error_type is None and stream_state.client_disconnected:
                    error_type = "CLIENT_DISCONNECTED"
                error_type = error_type or "-"

                latency = time.perf_counter() - start_time
                log_method = (
                    logger.info if error_type == "-" else logger.warning
                )
                event = (
                    "request_failed"
                    if error_type != "-"
                    else "request_finished"
                )
                log_method(
                    "request_id=%s method=%s path=%s service_type=llm "
                    "event=%s status_code=200 latency=%.3fs "
                    "error_type=%s model=%s stream=true",
                    request_id,
                    http_request.method,
                    http_request.url.path,
                    event,
                    latency,
                    error_type,
                    chat_request.model,
                )

        return StreamingResponse(
            content=logged_stream(),
            status_code=status.HTTP_200_OK,
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    payload = _build_chat_payload(chat_request, stream=False)
    raw_result = await vllm_client.chat(
        payload=payload,
        request_id=request_id,
    )

    try:
        content = raw_result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMUpstreamError(
            "vLLM 返回格式异常，缺少 choices[0].message.content"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise VLLMUpstreamError("vLLM 返回的回答为空")

    raw_usage = raw_result.get("usage")
    if not isinstance(raw_usage, dict):
        raw_usage = {}

    prompt_tokens = _safe_non_negative_int(
        raw_usage.get("prompt_tokens")
    )
    completion_tokens = _safe_non_negative_int(
        raw_usage.get("completion_tokens")
    )
    total_tokens = _safe_non_negative_int(
        raw_usage.get("total_tokens")
    )
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    latency = round(time.perf_counter() - start_time, 3)

    return ChatResponse(
        request_id=request_id,
        model=str(raw_result.get("model") or chat_request.model),
        content=content.strip(),
        usage=usage,
        latency=latency,
    )
