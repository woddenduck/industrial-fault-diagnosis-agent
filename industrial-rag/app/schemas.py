"""Industrial RAG 的 HTTP 请求与响应契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeBaseCreateRequest(BaseModel):
    """POST /knowledge-bases 请求体。"""

    knowledge_name: str = Field(
        ...,
        min_length=1,
        description="知识库名称，例如：G120C故障知识库",
    )
    device_model: str = Field(
        ...,
        min_length=1,
        description="设备型号，例如：G120C",
    )
    document_ids: list[str] = Field(
        ...,
        min_length=1,
        description="已经通过 /documents/upload 上传得到的 document_id 列表",
    )


class ChatRequest(BaseModel):
    """POST /chat 请求体。"""

    question: str = Field(
        ...,
        min_length=1,
        description="用户问题",
    )
    knowledge_base_id: str = Field(
        ...,
        min_length=1,
        description="目标 Knowledge Base ID",
    )
    device_model: str = Field(
        ...,
        min_length=1,
        description="设备型号；必须与 Knowledge Base 的 device_model 一致",
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="可选多轮历史",
    )
    history_summary: str | None = Field(
        default=None,
        description="可选历史摘要",
    )


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None
    url: str | None = None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, ComponentHealth]


class ModelsResponse(BaseModel):
    llm: str | None = None
    embedding: str | None = None
    reranker: str | None = None
    configured: dict[str, str | None] = Field(default_factory=dict)
    gateway: Any = None
