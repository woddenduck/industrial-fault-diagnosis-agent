"""
Industrial Fault Diagnosis Agent
API 请求与响应数据契约。

本模块只负责：

- API 输入验证；
- API 输出结构；
- OpenAPI Schema；
- 请求字段标准化。

本模块不负责：

- 调用 LangGraph；
- 调用 Tool；
- 调用 RAG；
- HTTP 状态码映射。
"""

from __future__ import annotations

import re

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# Literal Contracts
# ============================================================

AgentIntent = Literal[
    "device_status",
    "maintenance_history",
    "knowledge_query",
    "diagnosis",
    "unknown",
]

AgentStatus = Literal[
    "completed",
    "needs_input",
    "insufficient_evidence",
    "human_review_required",
    "failed",
]

RiskLevel = Literal[
    "unknown",
    "low",
    "medium",
    "high",
    "critical",
]

RAGStatus = Literal[
    "not_called",
    "answered",
    "rejected",
    "error",
]

HistoryRole = Literal[
    "user",
    "assistant",
]


DEVICE_ID_PATTERN = re.compile(
    r"^DEVICE-\d{3}$"
)


# ============================================================
# Base Schema
# ============================================================

class APIModel(BaseModel):
    """
    Agent API Schema 基类。

    extra="forbid"：
        拒绝未定义字段，防止调用方拼错字段名。

    str_strip_whitespace=True：
        自动清理字符串首尾空格。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ============================================================
# Request Schemas
# ============================================================

class HistoryMessage(APIModel):
    """
    Agent 与 RAG 共用的历史消息。
    """

    role: HistoryRole = Field(
        ...,
        description=(
            "消息角色，只允许 user 或 assistant"
        ),
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="历史消息正文",
    )


class DiagnoseRequest(APIModel):
    """
    POST /v1/diagnose 请求体。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "用户的设备查询或故障诊断问题"
        ),
        examples=[
            (
                "请综合诊断 DEVICE-001 "
                "当前温度异常"
            )
        ],
    )

    device_id: str | None = Field(
        default=None,
        pattern=r"^DEVICE-\d{3}$",
        description=(
            "设备编号，例如 DEVICE-001；"
            "部分知识查询可以不提供"
        ),
        examples=[
            "DEVICE-001"
        ],
    )

    device_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "设备型号，例如 G120C"
        ),
        examples=[
            "G120C"
        ],
    )

    knowledge_base_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Industrial RAG Knowledge Base ID"
        ),
        examples=[
            "kb_ab50652fe3a4"
        ],
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "最近的对话历史，最多 20 条消息"
        ),
    )

    history_summary: str | None = Field(
        default=None,
        max_length=8000,
        description="更早对话的摘要",
    )

    create_report: bool = Field(
        default=False,
        description=(
            "是否在综合诊断后创建结构化报告"
        ),
    )

    @field_validator(
        "query",
    )
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        """
        str_strip_whitespace 会先处理首尾空格。

        这里再次确保纯空白字符串不能进入系统。
        """

        if not value:
            raise ValueError(
                "query cannot be empty"
            )

        return value

    @field_validator(
        "device_id",
        mode="before",
    )
    @classmethod
    def normalize_device_id(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            return value

        normalized = (
            value
            .strip()
            .upper()
        )

        if not normalized:
            return None

        return normalized

    @field_validator(
        "device_model",
        mode="before",
    )
    @classmethod
    def normalize_device_model(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            return value

        normalized = (
            value
            .strip()
            .upper()
        )

        if not normalized:
            return None

        return normalized

    @field_validator(
        "knowledge_base_id",
        "history_summary",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            return value

        normalized = value.strip()

        if not normalized:
            return None

        return normalized


# ============================================================
# Response Child Schemas
# ============================================================

class SourceResponse(APIModel):
    """
    RAG 证据来源。
    """

    document: str = "未知文档"
    page: str = "未知"
    section: str = "未知"
    chunk_id: str | None = None


class AgentErrorResponse(APIModel):
    """
    AgentState.errors 中的单个错误。
    """

    code: str
    message: str
    node: str | None = None
    retryable: bool = False


class TraceResponse(APIModel):
    """
    单个节点执行轨迹。
    """

    node: str
    status: str

    duration_ms: float | None = Field(
        default=None,
        ge=0,
    )

    detail: str | None = None


# ============================================================
# Main Response Schema
# ============================================================

class DiagnoseResponse(APIModel):
    """
    POST /v1/diagnose 正常响应。

    即使业务结果为：

    - needs_input
    - insufficient_evidence
    - human_review_required

    仍然使用本响应结构。
    """

    request_id: str = Field(
        ...,
        min_length=1,
    )

    status: AgentStatus

    intent: AgentIntent = "unknown"

    device_id: str = ""
    device_model: str = ""
    knowledge_base_id: str = ""

    # 真实业务数据
    device_status: dict[str, Any] = Field(
        default_factory=dict,
    )

    maintenance_history: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    # RAG 结果
    rag_status: RAGStatus = "not_called"

    rag_request_id: str = ""

    rag_decision: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    degraded: bool = False

    sources: list[
        SourceResponse
    ] = Field(
        default_factory=list,
    )

    # 诊断与风险结果
    diagnosis: str = ""

    risk_level: RiskLevel = "unknown"

    human_review_required: bool = False

    report: dict[
        str,
        Any,
    ] | None = None

    # 最终输出
    final_answer: str = ""

    next_action: str = ""

    error: str = ""

    errors: list[
        AgentErrorResponse
    ] = Field(
        default_factory=list,
    )

    execution_trace: list[
        TraceResponse
    ] = Field(
        default_factory=list,
    )

    elapsed_ms: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Agent Graph 总执行时间，单位毫秒"
        ),
    )


# ============================================================
# Health and Graph Information
# ============================================================

class AgentHealthResponse(APIModel):
    """
    GET /health 响应。
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    graph: Literal[
        "ready",
        "error",
    ]

    rag: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class GraphInfoResponse(APIModel):
    """
    可选的 Graph 信息响应。

    当前保留该 Schema，
    便于后续调试或 README 展示。
    """

    name: str

    nodes: list[str] = Field(
        default_factory=list,
    )

    intents: list[
        AgentIntent
    ] = Field(
        default_factory=list,
    )