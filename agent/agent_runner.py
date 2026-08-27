"""
工业设备故障诊断 Agent 统一运行入口。

本模块只负责：

1. 将 API 请求转换为 AgentState。
2. 调用已经编译完成的 LangGraph。
3. 将 Graph 最终状态转换为公开响应。
4. 统一处理 Graph 运行异常。

意图识别、工具调用、RAG 检索、风险判断等业务逻辑，
全部由 LangGraph 和对应节点负责。
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from agent.config import (
    DEFAULT_DEVICE_MODEL,
    DEFAULT_KNOWLEDGE_BASE_ID,
)
from agent.graph import industrial_graph
from agent.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
)
from agent.state import (
    AgentState,
    create_initial_state,
)


AGENT_RUNTIME_ERROR = "AGENT_RUNTIME_ERROR"
INVALID_GRAPH_RESULT = "INVALID_GRAPH_RESULT"
INVALID_TERMINAL_STATE = "INVALID_TERMINAL_STATE"


PUBLIC_STATUSES = {
    "completed",
    "needs_input",
    "insufficient_evidence",
    "human_review_required",
    "failed",
}


class GraphInvoker(Protocol):
    """约束正式 Graph 和测试替身必须提供 invoke 方法。"""

    def invoke(
        self,
        input: AgentState,
    ) -> Mapping[str, Any]:
        ...


class AgentRunner:
    """工业故障诊断 Graph 的统一运行边界。"""

    def __init__(
        self,
        graph: GraphInvoker | None = None,
    ) -> None:
        self._graph = (
            graph
            if graph is not None
            else industrial_graph
        )

    def run(
        self,
        request: DiagnoseRequest | Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> DiagnoseResponse:
        """执行一次独立的 Agent 请求。"""

        validated_request = self._validate_request(
            request
        )

        resolved_request_id = self._resolve_request_id(
            request_id
        )

        initial_state = self._build_initial_state(
            validated_request,
            request_id=resolved_request_id,
        )

        started_at = perf_counter()

        try:
            graph_result = self._graph.invoke(
                initial_state
            )
        except Exception as exc:
            return self._build_failure_response(
                initial_state=initial_state,
                error_code=AGENT_RUNTIME_ERROR,
                exception=exc,
                elapsed_ms=self._elapsed_ms(started_at),
            )

        elapsed_ms = self._elapsed_ms(started_at)

        if not isinstance(graph_result, Mapping):
            return self._build_failure_response(
                initial_state=initial_state,
                error_code=INVALID_GRAPH_RESULT,
                exception=TypeError(
                    "industrial_graph.invoke() "
                    "没有返回字典结构"
                ),
                elapsed_ms=elapsed_ms,
            )

        try:
            return self._build_response(
                graph_result=graph_result,
                fallback_state=initial_state,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            return self._build_failure_response(
                initial_state=initial_state,
                error_code=INVALID_GRAPH_RESULT,
                exception=exc,
                elapsed_ms=elapsed_ms,
            )

    @staticmethod
    def _validate_request(
        request: DiagnoseRequest | Mapping[str, Any],
    ) -> DiagnoseRequest:
        """保证 Runner 内部始终使用合法请求模型。"""

        if isinstance(request, DiagnoseRequest):
            return request

        if isinstance(request, Mapping):
            return DiagnoseRequest.model_validate(
                dict(request)
            )

        raise TypeError(
            "request 必须是 DiagnoseRequest "
            f"或字典结构，当前类型为："
            f"{type(request).__name__}"
        )

    @staticmethod
    def _resolve_request_id(
        request_id: str | None,
    ) -> str:
        """保留外部请求编号；没有提供时自动生成。"""

        if request_id is not None:
            normalized = str(request_id).strip()

            if normalized:
                return normalized

        return uuid4().hex

    @staticmethod
    def _build_initial_state(
        request: DiagnoseRequest,
        *,
        request_id: str,
    ) -> AgentState:
        """将 API 请求转换为完整的 Agent 初始状态。"""

        history = [
            item.model_dump(mode="python")
            for item in request.history
        ]

        device_model = (
            request.device_model
            or DEFAULT_DEVICE_MODEL
        )

        knowledge_base_id = (
            request.knowledge_base_id
            or DEFAULT_KNOWLEDGE_BASE_ID
        )

        return create_initial_state(
            request_id=request_id,
            user_query=request.query,
            device_id=request.device_id or "",
            device_model=device_model,
            knowledge_base_id=knowledge_base_id,
            history=history,
            history_summary=request.history_summary,
            create_report=request.create_report,
        )

    def _build_response(
        self,
        *,
        graph_result: Mapping[str, Any],
        fallback_state: AgentState,
        elapsed_ms: float,
    ) -> DiagnoseResponse:
        """将 Graph 最终状态转换为公开响应。"""

        state = dict(graph_result)

        status, terminal_error = (
            self._resolve_terminal_status(state)
        )

        errors = self._normalize_errors(
            state.get("errors")
        )

        if terminal_error is not None:
            errors.append(terminal_error)

        request_id = self._text(
            state.get("request_id")
            or fallback_state.get("request_id")
        )

        final_answer = self._text(
            state.get("final_answer")
        )

        error = self._text(
            state.get("error")
        )

        if terminal_error is not None:
            error = terminal_error["code"]

            if not final_answer:
                final_answer = (
                    "Agent 工作流未产生有效终止状态，"
                    "请稍后重试。"
                )

        return DiagnoseResponse(
            request_id=request_id,
            status=status,
            intent=self._text(
                state.get("intent"),
                default="unknown",
            ),
            device_id=self._text(
                state.get("device_id")
            ),
            device_model=self._text(
                state.get("device_model")
            ),
            knowledge_base_id=self._text(
                state.get("knowledge_base_id")
            ),
            device_status=self._dict(
                state.get("device_status")
            ),
            maintenance_history=self._dict(
                state.get("maintenance_history")
            ),
            rag_status=self._text(
                state.get("rag_status"),
                default="not_called",
            ),
            rag_request_id=self._text(
                state.get("rag_request_id")
            ),
            rag_decision=self._dict(
                state.get("rag_decision")
            ),
            degraded=bool(
                state.get("degraded", False)
            ),
            sources=self._normalize_sources(
                state.get("sources")
            ),
            diagnosis=self._text(
                state.get("diagnosis")
            ),
            risk_level=self._text(
                state.get("risk_level"),
                default="unknown",
            ),
            human_review_required=bool(
                state.get(
                    "human_review_required",
                    False,
                )
            ),
            report=self._optional_dict(
                state.get("report")
            ),
            final_answer=final_answer,
            next_action=self._text(
                state.get("next_action")
            ),
            error=error,
            errors=errors,
            execution_trace=self._normalize_trace(
                state.get("execution_trace")
            ),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _resolve_terminal_status(
        state: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        """确认 Graph 是否进入了合法终止状态。"""

        status = AgentRunner._text(
            state.get("status")
        )

        if status in PUBLIC_STATUSES:
            return status, None

        if bool(
            state.get("human_review_required")
        ):
            return "human_review_required", None

        missing_fields = state.get(
            "missing_fields"
        )

        has_missing_fields = (
            isinstance(missing_fields, list)
            and bool(missing_fields)
        )

        if (
            has_missing_fields
            or state.get("next_action") == "ask_user"
        ):
            return "needs_input", None

        if AgentRunner._text(
            state.get("error")
        ):
            return "failed", None

        return (
            "failed",
            {
                "code": INVALID_TERMINAL_STATE,
                "message": (
                    "Graph 没有进入合法终止状态"
                ),
                "node": "agent_runner",
                "retryable": False,
            },
        )

    @staticmethod
    def _build_failure_response(
        *,
        initial_state: AgentState,
        error_code: str,
        exception: Exception,
        elapsed_ms: float,
    ) -> DiagnoseResponse:
        """将运行异常转换成稳定失败响应。"""

        exception_name = type(exception).__name__

        return DiagnoseResponse(
            request_id=AgentRunner._text(
                initial_state.get("request_id")
            ),
            status="failed",
            intent="unknown",
            device_id=AgentRunner._text(
                initial_state.get("device_id")
            ),
            device_model=AgentRunner._text(
                initial_state.get("device_model")
            ),
            knowledge_base_id=AgentRunner._text(
                initial_state.get(
                    "knowledge_base_id"
                )
            ),
            rag_status="not_called",
            risk_level="unknown",
            final_answer=(
                "Agent 服务执行失败，请稍后重试。"
            ),
            next_action="retry_later",
            error=error_code,
            errors=[
                {
                    "code": error_code,
                    "message": (
                        "Agent Graph 执行失败："
                        f"{exception_name}"
                    ),
                    "node": "agent_runner",
                    "retryable": True,
                }
            ],
            execution_trace=[
                {
                    "node": "agent_runner",
                    "status": "error",
                    "duration_ms": elapsed_ms,
                    "detail": (
                        "graph_exception="
                        f"{exception_name}"
                    ),
                }
            ],
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _normalize_sources(
        value: Any,
    ) -> list[dict[str, Any]]:
        """整理知识库证据来源。"""

        if not isinstance(value, list):
            return []

        sources: list[dict[str, Any]] = []

        for item in value:
            if not isinstance(item, Mapping):
                continue

            sources.append(
                {
                    "document": AgentRunner._text(
                        item.get("document"),
                        default="未知文档",
                    ),
                    "page": AgentRunner._text(
                        item.get("page"),
                        default="未知",
                    ),
                    "section": AgentRunner._text(
                        item.get("section"),
                        default="未知",
                    ),
                    "chunk_id": AgentRunner._text(
                        item.get("chunk_id")
                    ),
                }
            )

        return sources

    @staticmethod
    def _normalize_errors(
        value: Any,
    ) -> list[dict[str, Any]]:
        """整理 Graph 产生的结构化错误。"""

        if not isinstance(value, list):
            return []

        errors: list[dict[str, Any]] = []

        for item in value:
            if not isinstance(item, Mapping):
                continue

            errors.append(
                {
                    "code": AgentRunner._text(
                        item.get("code"),
                        default="AGENT_ERROR",
                    ),
                    "message": AgentRunner._text(
                        item.get("message"),
                        default="Agent 执行错误",
                    ),
                    "node": (
                        AgentRunner._text(
                            item.get("node")
                        )
                        or None
                    ),
                    "retryable": bool(
                        item.get("retryable", False)
                    ),
                }
            )

        return errors

    @staticmethod
    def _normalize_trace(
        value: Any,
    ) -> list[dict[str, Any]]:
        """整理 Graph 节点执行轨迹。"""

        if not isinstance(value, list):
            return []

        trace: list[dict[str, Any]] = []

        for item in value:
            if not isinstance(item, Mapping):
                continue

            duration = item.get("duration_ms")

            duration_ms = (
                float(duration)
                if isinstance(duration, (int, float))
                else None
            )

            trace.append(
                {
                    "node": AgentRunner._text(
                        item.get("node"),
                        default="unknown_node",
                    ),
                    "status": AgentRunner._text(
                        item.get("status"),
                        default="unknown",
                    ),
                    "duration_ms": duration_ms,
                    "detail": (
                        AgentRunner._text(
                            item.get("detail")
                        )
                        or None
                    ),
                }
            )

        return trace

    @staticmethod
    def _dict(
        value: Any,
    ) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)

        return {}

    @staticmethod
    def _optional_dict(
        value: Any,
    ) -> dict[str, Any] | None:
        if isinstance(value, Mapping):
            return dict(value)

        return None

    @staticmethod
    def _text(
        value: Any,
        *,
        default: str = "",
    ) -> str:
        if value is None:
            return default

        text = str(value).strip()

        if not text:
            return default

        return text

    @staticmethod
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return round(
            (perf_counter() - started_at) * 1000,
            3,
        )


default_agent_runner = AgentRunner()


def run_agent(
    request: DiagnoseRequest | Mapping[str, Any],
    *,
    request_id: str | None = None,
) -> DiagnoseResponse:
    """执行一次工业设备故障诊断请求。"""

    return default_agent_runner.run(
        request,
        request_id=request_id,
    )


__all__ = [
    "AgentRunner",
    "GraphInvoker",
    "default_agent_runner",
    "run_agent",
]