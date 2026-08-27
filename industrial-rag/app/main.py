"""
Day 20 - Stage 7
FastAPI Business Entry - Fault Injection Ready

职责：
- HTTP 请求接收；
- Pydantic Schema 校验；
- 调用已经验收通过的 Service；
- 转换为 HTTP Response；
- 提供 /health 与 /models。

明确不负责：
- BM25 算法；
- Vector Search 算法；
- RRF；
- Reranker 算法；
- Context Builder 算法；
- Prompt 规则；
- LLM 推理逻辑。
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.config import (
    API_HOST,
    API_PORT,
    APP_NAME,
    APP_VERSION,
    CHAT_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    GATEWAY_BASE_URL,
    HEALTH_TIMEOUT,
    LLM_BASE_URL,
    MODEL_QUERY_TIMEOUT,
    RAG_DIR,
    RERANKER_BASE_URL,
    RERANKER_MODEL,
    TEMP_UPLOAD_DIR,
    VECTOR_STORE_DIR,
    ensure_runtime_directories,
)
from app.schemas import (
    ChatRequest,
    HealthResponse,
    KnowledgeBaseCreateRequest,
    ModelsResponse,
)


# ============================================================
# 1. Import Path Bootstrap
# ============================================================
#
# 你当前 Day18/Day20 的 RAG 模块中仍有：
#     from prompt_builder import ...
#     from config import ...
# 这类直接按 rag/ 目录导入的写法。
#
# 因此 FastAPI 启动时把 rag/ 加入 sys.path，
# 保持与已经验收通过的脚本运行方式兼容。
# ============================================================

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# ============================================================
# 2. Existing Services / RAG Modules
# ============================================================

from app.services.document_service import save_document  # noqa: E402
from app.services.knowledge_builder import (  # noqa: E402
    create_and_build_knowledge_base,
)
from app.services import knowledge_service  # noqa: E402
from app.services.qa_service import LLMTimeoutError, QAService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

import rag.context_builder as context_builder  # noqa: E402
import rag.hybrid_retriever as hybrid_retriever  # noqa: E402
import rag.prompt_builder as prompt_builder  # noqa: E402
import rag.query_rewriter as query_rewriter  # noqa: E402
import rag.rerank_pipeline as rerank_pipeline  # noqa: E402

try:
    import vector_store  # noqa: E402
except ImportError:
    vector_store = None


# ============================================================
# 3. Service Wiring
# ============================================================

retrieval_service = RetrievalService(
    kb_manager=knowledge_service,
    query_rewriter=query_rewriter,
    hybrid_retriever=hybrid_retriever,
    reranker=rerank_pipeline,
    context_builder=context_builder,
)

qa_service = QAService(
    retrieval_service=retrieval_service,
    prompt_builder=prompt_builder.build_prompt,
)


# ============================================================
# 4. FastAPI App
# ============================================================

ensure_runtime_directories()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Day20 工业设备故障诊断 RAG V1："
        "Document Upload -> Knowledge Base Build -> Retrieval -> QA"
    ),
)


# ============================================================
# 5. Helpers
# ============================================================

def _safe_filename(filename: str | None) -> str:
    """
    只保留 basename，防止上传文件名携带 ../ 等路径。
    """

    name = Path(filename or "").name.strip()

    if not name:
        raise ValueError("上传文件名不能为空")

    return name


def _write_upload_to_temp(
    upload_file: UploadFile,
) -> tuple[Path, Path]:
    """
    将 FastAPI UploadFile 落到临时目录。

    你的 document_service.save_document() 接收 file_path，
    所以 Controller 在这里完成 HTTP File -> Local Path 的适配。

    返回：
        temp_file_path, request_temp_dir
    """

    filename = _safe_filename(upload_file.filename)

    request_dir = (
        TEMP_UPLOAD_DIR
        / f"upload_{uuid.uuid4().hex}"
    )
    request_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    temp_path = request_dir / filename

    try:
        upload_file.file.seek(0)

        with temp_path.open("wb") as target:
            shutil.copyfileobj(
                upload_file.file,
                target,
                length=1024 * 1024,
            )

        return temp_path, request_dir

    except Exception:
        shutil.rmtree(
            request_dir,
            ignore_errors=True,
        )
        raise


def _cleanup_temp_dir(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(
            path,
            ignore_errors=True,
        )


def _http_reachability(
    name: str,
    url: str,
) -> dict[str, Any]:
    """
    Health 判断的是“这个组件是否可访问”。

    注意 Gateway /health 自己可能因为 Reranker Down 报 degraded。
    但 Gateway 进程本身仍然可用，所以只要 HTTP 2xx，就把 Gateway
    组件本身记为 healthy；Reranker 的降级由独立检查负责。
    """

    try:
        response = requests.get(
            url,
            timeout=HEALTH_TIMEOUT,
        )

        if 200 <= response.status_code < 300:
            detail = f"{name} reachable"

            try:
                data = response.json()

                if isinstance(data, dict):
                    reported_status = data.get(
                        "status"
                    )

                    if reported_status:
                        detail += (
                            f"; reported_status="
                            f"{reported_status}"
                        )

            except ValueError:
                pass

            return {
                "status": "healthy",
                "detail": detail,
                "url": url,
            }

        return {
            "status": "unhealthy",
            "detail": (
                f"{name} returned "
                f"HTTP {response.status_code}"
            ),
            "url": url,
        }

    except requests.RequestException as exc:
        return {
            "status": "unhealthy",
            "detail": str(exc),
            "url": url,
        }


def _check_vector_store() -> dict[str, Any]:
    """
    优先通过现有 rag/vector_store.py 真正访问 Collection；
    如果当前模块没有 get_collection()，才退化为持久化目录检查。
    """

    try:
        if vector_store is not None:
            get_collection = getattr(
                vector_store,
                "get_collection",
                None,
            )

            if callable(get_collection):
                collection = get_collection()

                if collection is None:
                    raise RuntimeError(
                        "get_collection() 返回 None"
                    )

                detail = "Vector Store collection reachable"

                count_fn = getattr(
                    collection,
                    "count",
                    None,
                )

                if callable(count_fn):
                    detail += (
                        f"; records={count_fn()}"
                    )

                return {
                    "status": "healthy",
                    "detail": detail,
                    "url": None,
                }

        path = VECTOR_STORE_DIR

        if not path.exists():
            raise FileNotFoundError(
                f"Vector Store 目录不存在：{path}"
            )

        if not path.is_dir():
            raise RuntimeError(
                f"Vector Store 路径不是目录：{path}"
            )

        return {
            "status": "healthy",
            "detail": (
                "Vector Store directory exists: "
                f"{path.resolve()}"
            ),
            "url": None,
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "detail": str(exc),
            "url": None,
        }


def _first_openai_model_id(
    data: Any,
) -> str | None:
    """
    解析 vLLM /v1/models：
        {"data": [{"id": "Qwen/Qwen3-8B"}]}
    """

    if not isinstance(data, dict):
        return None

    items = data.get("data")

    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue

        model_id = item.get("id")

        if isinstance(model_id, str):
            model_id = model_id.strip()

            if model_id:
                return model_id

    return None


def _find_named_model(
    data: Any,
    aliases: tuple[str, ...],
) -> str | None:
    """
    尽量兼容 Gateway /models 的不同 JSON 结构。

    支持例如：
        {"llm": "..."}
        {"models": {"embedding": "..."}}
        {"services": {"reranker": {"model": "..."}}}
    """

    alias_set = {
        alias.lower()
        for alias in aliases
    }

    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = str(key).lower()

            if key_lower in alias_set:
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value

                if isinstance(value, dict):
                    for model_key in (
                        "model",
                        "model_id",
                        "name",
                        "id",
                    ):
                        candidate = value.get(
                            model_key
                        )

                        if (
                            isinstance(candidate, str)
                            and candidate.strip()
                        ):
                            return candidate.strip()

            nested = _find_named_model(
                value,
                aliases,
            )

            if nested:
                return nested

    elif isinstance(data, list):
        for item in data:
            nested = _find_named_model(
                item,
                aliases,
            )

            if nested:
                return nested

    return None


def _extract_generic_model_name(
    data: Any,
) -> str | None:
    """
    从 Embedding / Reranker 的 health/config JSON 中尽量提取模型名。

    兼容常见字段：
        model / model_id / model_name / model_path / name
    """

    if isinstance(data, dict):
        for key in (
            "model",
            "model_id",
            "model_name",
            "model_path",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in data.values():
            found = _extract_generic_model_name(value)
            if found:
                return found

    elif isinstance(data, list):
        for item in data:
            found = _extract_generic_model_name(item)
            if found:
                return found

    return None


def _get_json_or_none(url: str) -> Any:
    try:
        response = requests.get(
            url,
            timeout=MODEL_QUERY_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _normalize_path_values(
    value: Any,
) -> Any:
    """
    Knowledge Builder 返回值中包含 pathlib.Path。
    FastAPI 通常能编码 Path，这里再显式转换一次，
    保证响应是稳定 JSON。
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            key: _normalize_path_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _normalize_path_values(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _normalize_path_values(item)
            for item in value
        ]

    return value


# ============================================================
# 6. Root
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# 7. POST /documents/upload
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    device_model: str = Form(...),
    document_type: str = Form(...),
    version: str | None = Form(default=None),
):
    """
    HTTP UploadFile
        -> 临时本地文件
        -> document_service.save_document(file_path, ...)
        -> document_id

    Document Service 自己继续负责：
    - 扩展名校验；
    - 空文件校验；
    - SHA256；
    - 重复文档检测；
    - data/documents.json 持久化。
    """

    temp_dir: Path | None = None

    try:
        temp_path, temp_dir = await run_in_threadpool(
            _write_upload_to_temp,
            file,
        )

        result = await run_in_threadpool(
            save_document,
            str(temp_path),
            device_model,
            document_type,
            version,
        )

        return jsonable_encoder(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"文档上传失败：{exc}",
        ) from exc

    finally:
        await run_in_threadpool(
            _cleanup_temp_dir,
            temp_dir,
        )
        await file.close()


# ============================================================
# 8. POST /knowledge-bases
# ============================================================

@app.post("/knowledge-bases")
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
):
    """
    一次请求完成：
        Knowledge Base 元数据创建
        -> PDF Parse
        -> Chunk
        -> Embedding
        -> Chroma
        -> KB 专属 BM25
        -> Stage3 一致性验收
        -> status=ready

    这里使用 knowledge_builder.create_and_build_knowledge_base()，
    因为 Retrieval Service 只允许 ready KB 进入检索。
    """

    try:
        result = await run_in_threadpool(
            create_and_build_knowledge_base,
            name=request.knowledge_name,
            device_model=request.device_model,
            document_ids=request.document_ids,
        )

        return jsonable_encoder(
            _normalize_path_values(result)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge Base 构建失败：{exc}",
        ) from exc


# ============================================================
# 9. POST /chat
# ============================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
):
    """
    Controller 只负责把 HTTP Request 映射到 QAService.answer()。

    Stage7 额外负责：
    - 为本次业务请求生成 request_id；
    - LLM Timeout -> HTTP 504 + 统一错误结构；
    - 普通上游失败 -> HTTP 502 + request_id；
    - 正常响应也返回 request_id，便于日志关联。

    RAG 主链路仍由现有 Service 完成：
        QA Service
        -> Retrieval Service
        -> Query Rewrite
        -> Hybrid
        -> Reranker / Fallback
        -> Context Builder
        -> Evidence Decision
        -> Prompt Builder
        -> Gateway /chat
    """

    request_id = uuid.uuid4().hex
    http_request.state.request_id = request_id

    try:
        result = await run_in_threadpool(
            qa_service.answer,
            query=request.question,
            knowledge_base_id=(
                request.knowledge_base_id
            ),
            device_model=request.device_model,
            history=request.history,
            history_summary=(
                request.history_summary
            ),
        )

        if not isinstance(result, dict):
            raise TypeError(
                "QAService.answer() 返回值必须是 dict"
            )

        result["request_id"] = request_id
        return jsonable_encoder(result)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except LLMTimeoutError as exc:
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "error": "LLM_TIMEOUT",
                "message": str(exc),
                "request_id": request_id,
            },
            headers={
                "X-Request-ID": request_id,
            },
        )

    except RuntimeError as exc:
        message = str(exc)

        if (
            "Knowledge Base 当前不可用于检索"
            in message
        ):
            raise HTTPException(
                status_code=409,
                detail=message,
            ) from exc

        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "error": "UPSTREAM_SERVICE_ERROR",
                "message": message,
                "request_id": request_id,
            },
            headers={
                "X-Request-ID": request_id,
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"问答服务执行失败：{exc}",
                "request_id": request_id,
            },
            headers={
                "X-Request-ID": request_id,
            },
        )


# ============================================================
# 10. GET /health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> dict[str, Any]:
    """
    状态规则：

    healthy:
        Gateway / LLM / Embedding / Reranker / Vector Store 全部正常

    degraded:
        核心服务正常，但 Reranker 不可用
        （现有 Rerank Pipeline 可以 fallback）

    unhealthy:
        Gateway / LLM / Embedding / Vector Store 任一不可用
    """

    components = {
        "gateway": _http_reachability(
            "Gateway",
            f"{GATEWAY_BASE_URL}/health",
        ),
        "llm": _http_reachability(
            "LLM",
            f"{LLM_BASE_URL}/v1/models",
        ),
        "embedding": _http_reachability(
            "Embedding",
            f"{EMBEDDING_BASE_URL}/health",
        ),
        "reranker": _http_reachability(
            "Reranker",
            f"{RERANKER_BASE_URL}/health",
        ),
        "vector_store": _check_vector_store(),
    }

    critical_names = (
        "gateway",
        "llm",
        "embedding",
        "vector_store",
    )

    critical_failed = any(
        components[name]["status"]
        != "healthy"
        for name in critical_names
    )

    reranker_failed = (
        components["reranker"]["status"]
        != "healthy"
    )

    if critical_failed:
        overall = "unhealthy"
    elif reranker_failed:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "components": components,
    }


# ============================================================
# 11. GET /models
# ============================================================

@app.get(
    "/models",
    response_model=ModelsResponse,
)
def models() -> dict[str, Any]:
    """
    Runtime 优先：
    - LLM 直接查询 vLLM /v1/models；
    - Embedding / Reranker 优先从 Gateway /models 解析；
    - 如果 Gateway JSON 结构暂时无法解析，则退回环境变量，
      同时把 Gateway 原始 JSON 完整返回，方便现场确认。
    """

    gateway_data: Any = None
    runtime_llm: str | None = None

    # LLM：vLLM 提供标准 /v1/models，直接读取真实运行模型。
    try:
        response = requests.get(
            f"{LLM_BASE_URL}/v1/models",
            timeout=MODEL_QUERY_TIMEOUT,
        )
        response.raise_for_status()
        runtime_llm = _first_openai_model_id(
            response.json()
        )
    except (requests.RequestException, ValueError):
        runtime_llm = None

    # Gateway：保留原始 /models，作为现场辅助信息。
    try:
        response = requests.get(
            f"{GATEWAY_BASE_URL}/models",
            timeout=MODEL_QUERY_TIMEOUT,
        )
        response.raise_for_status()
        gateway_data = response.json()
    except requests.RequestException as exc:
        gateway_data = {
            "status": "unavailable",
            "detail": str(exc),
        }
    except ValueError as exc:
        gateway_data = {
            "status": "invalid_json",
            "detail": str(exc),
        }

    # Embedding / Reranker：优先直接看各自 /health 中是否报告模型名。
    embedding_health = _get_json_or_none(
        f"{EMBEDDING_BASE_URL}/health"
    )
    reranker_health = _get_json_or_none(
        f"{RERANKER_BASE_URL}/health"
    )

    runtime_embedding = (
        _extract_generic_model_name(embedding_health)
        or _find_named_model(
            gateway_data,
            ("embedding", "embed", "embedding_model"),
        )
        or EMBEDDING_MODEL
    )

    runtime_reranker = (
        _extract_generic_model_name(reranker_health)
        or _find_named_model(
            gateway_data,
            ("reranker", "rerank", "reranker_model"),
        )
        or RERANKER_MODEL
    )

    return {
        "llm": runtime_llm or CHAT_MODEL,
        "embedding": runtime_embedding,
        "reranker": runtime_reranker,
        "configured": {
            "llm": CHAT_MODEL,
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
        },
        "gateway": gateway_data,
    }



# ============================================================
# 12. Local Start
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )