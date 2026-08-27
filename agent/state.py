from __future__ import annotations

from typing import Any, Literal, TypedDict
from uuid import uuid4


AgentIntent = Literal[
    "device_status",
    "maintenance_history",
    "knowledge_query",
    "diagnosis",
    "unknown",
]

AgentStatus = Literal[
    "running",
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


class SourceItem(TypedDict, total=False):
    document: str
    page: str
    section: str
    chunk_id: str


class AgentErrorItem(TypedDict, total=False):
    code: str
    message: str
    node: str
    retryable: bool


class TraceItem(TypedDict, total=False):
    node: str
    status: str
    duration_ms: float
    detail: str


class AgentState(TypedDict, total=False):
    # 请求信息
    request_id: str
    user_query: str
    device_id: str
    device_model: str
    knowledge_base_id: str
    history: list[dict[str, Any]]
    history_summary: str | None
    create_report: bool

    # 意图和流程控制
    intent: AgentIntent
    missing_fields: list[str]
    next_action: str
    status: AgentStatus

    # 业务 Tool 结果
    device_status: dict[str, Any]
    maintenance_history: dict[str, Any]

    # RAG 结果
    rag_status: str
    rag_answer: str
    rag_request_id: str
    rag_decision: dict[str, Any]
    sources: list[SourceItem]
    degraded: bool

    # 诊断结果
    diagnosis: str
    risk_level: RiskLevel
    human_review_required: bool
    report: dict[str, Any] | None
    final_answer: str

    # 错误与轨迹
    error: str
    errors: list[AgentErrorItem]
    execution_trace: list[TraceItem]

    # 临时兼容字段，完成生产 Graph 后删除
    retrieved_documents: list[dict[str, Any]]
    mock_diagnosis: str


def create_initial_state(
    *,
    user_query: str,
    device_id: str = "",
    device_model: str = "",
    knowledge_base_id: str = "",
    history: list[dict[str, Any]] | None = None,
    history_summary: str | None = None,
    create_report: bool = False,
    request_id: str | None = None,
) -> AgentState:
    """创建一次 Agent 请求的独立初始状态。"""

    return {
        "request_id": request_id or uuid4().hex,
        "user_query": str(user_query).strip(),
        "device_id": str(device_id).strip(),
        "device_model": str(device_model).strip().upper(),
        "knowledge_base_id": str(
            knowledge_base_id
        ).strip(),
        "history": [
            dict(item)
            for item in (history or [])
            if isinstance(item, dict)
        ],
        "history_summary": (
            history_summary.strip()
            if isinstance(history_summary, str)
            and history_summary.strip()
            else None
        ),
        "create_report": bool(create_report),

        "intent": "unknown",
        "missing_fields": [],
        "next_action": "initialize",
        "status": "running",

        "device_status": {},
        "maintenance_history": {},

        "rag_status": "not_called",
        "rag_answer": "",
        "rag_request_id": "",
        "rag_decision": {},
        "sources": [],
        "degraded": False,

        "diagnosis": "",
        "risk_level": "unknown",
        "human_review_required": False,
        "report": None,
        "final_answer": "",

        "error": "",
        "errors": [],
        "execution_trace": [],

        # 旧 Graph 临时兼容
        "retrieved_documents": [],
        "mock_diagnosis": "",
    }