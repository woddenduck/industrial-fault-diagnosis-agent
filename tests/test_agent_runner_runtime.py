"""关卡 6.2：Agent Runner 运行边界验收。

本测试直接通过 Python 运行，不依赖 pytest。

测试使用假的 Graph，不会访问：

1. Industrial RAG
2. Gateway
3. LLM
4. Embedding
5. Reranker
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.agent_runner import AgentRunner
from agent.config import (
    DEFAULT_DEVICE_MODEL,
    DEFAULT_KNOWLEDGE_BASE_ID,
)
from agent.schemas import DiagnoseRequest


SEPARATOR = "=" * 96
_NOT_SET = object()


class FakeGraph:
    """用于观察 Runner 输入和模拟 Graph 输出。"""

    def __init__(
        self,
        *,
        updates: dict[str, Any] | None = None,
        result: Any = _NOT_SET,
        exception: Exception | None = None,
    ) -> None:
        self.updates = updates or {}
        self.result = result
        self.exception = exception

        self.call_count = 0
        self.last_input: dict[str, Any] | None = None

    def invoke(
        self,
        input: dict[str, Any],
    ) -> Any:
        self.call_count += 1
        self.last_input = deepcopy(dict(input))

        if self.exception is not None:
            raise self.exception

        if self.result is not _NOT_SET:
            return self.result

        final_state = deepcopy(dict(input))
        final_state.update(deepcopy(self.updates))

        return final_state


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            key: to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    return value


def print_json(
    title: str,
    value: Any,
) -> None:
    print(f"\n[{title}]")

    print(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def check(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[CHECK PASS] {message}")


def completed_updates() -> dict[str, Any]:
    """构造一次正常诊断完成后的 Graph 状态。"""

    return {
        "status": "completed",
        "intent": "diagnosis",
        "next_action": "completed",
        "device_status": {
            "device_id": "DEVICE-001",
            "status": "normal",
            "temperature": 42.5,
        },
        "maintenance_history": {
            "device_id": "DEVICE-001",
            "record_count": 1,
            "records": [],
        },
        "rag_status": "answered",
        "rag_request_id": "rag-runtime-001",
        "rag_decision": {
            "evidence_sufficient": True,
            "allow_llm": True,
            "reason": "SAFE_CONTEXT_READY",
        },
        "sources": [
            {
                "document": "G120C_manual.pdf",
                "page": "414-415",
                "section": "9.5 故障和警告列表",
                "chunk_id": "chunk-runtime-001",
            }
        ],
        "diagnosis": "设备当前状态正常，建议继续观察温度变化。",
        "risk_level": "medium",
        "human_review_required": False,
        "report": {
            "report_id": "REPORT-RUNTIME-001",
            "severity": "medium",
        },
        "final_answer": "综合诊断已经完成。",
        "error": "",
        "errors": [],
        "execution_trace": [
            {
                "node": "intent_node",
                "status": "success",
                "detail": "intent=diagnosis",
            },
            {
                "node": "final_answer_node",
                "status": "success",
                "detail": "status=completed",
            },
        ],
    }


def case_complete_request_mapping() -> None:
    request = DiagnoseRequest(
        query="  请综合诊断 DEVICE-001 当前温度异常  ",
        device_id=" device-001 ",
        device_model=" g120c ",
        knowledge_base_id=" kb_runtime_001 ",
        history=[
            {
                "role": "user",
                "content": "设备之前出现过温度异常。",
            }
        ],
        history_summary="正在排查设备温度问题。",
        create_report=True,
    )

    graph = FakeGraph(
        updates=completed_updates()
    )

    runner = AgentRunner(graph=graph)

    response = runner.run(
        request,
        request_id=" runtime-request-001 ",
    )

    print_json("Graph Input", graph.last_input)
    print_json("Runner Response", response)

    check(
        graph.call_count == 1,
        "Graph 只执行一次",
    )

    check(
        graph.last_input is not None,
        "Graph 收到初始状态",
    )

    initial_state = graph.last_input or {}

    check(
        initial_state["request_id"]
        == "runtime-request-001",
        "Request ID 清理首尾空白后写入 State",
    )

    check(
        initial_state["user_query"]
        == "请综合诊断 DEVICE-001 当前温度异常",
        "query 正确映射为 user_query",
    )

    check(
        initial_state["device_id"]
        == "DEVICE-001",
        "device_id 正确写入 State",
    )

    check(
        initial_state["device_model"]
        == "G120C",
        "device_model 正确写入 State",
    )

    check(
        initial_state["knowledge_base_id"]
        == "kb_runtime_001",
        "knowledge_base_id 正确写入 State",
    )

    check(
        len(initial_state["history"]) == 1,
        "对话历史转换为普通字典列表",
    )

    check(
        initial_state["create_report"] is True,
        "报告生成开关正确写入 State",
    )

    check(
        response.status == "completed",
        "Runner 返回 completed",
    )

    check(
        response.intent == "diagnosis",
        "Runner 保留诊断意图",
    )

    check(
        response.rag_status == "answered",
        "Runner 保留 RAG 状态",
    )

    check(
        response.rag_request_id
        == "rag-runtime-001",
        "Runner 保留 RAG Request ID",
    )

    check(
        len(response.sources) == 1,
        "Runner 返回结构化证据来源",
    )

    check(
        response.elapsed_ms >= 0,
        "Runner 返回毫秒级执行耗时",
    )


def case_default_configuration() -> None:
    request = DiagnoseRequest(
        query="查询变频器过热原因"
    )

    graph = FakeGraph(
        updates={
            "status": "completed",
            "intent": "knowledge_query",
            "next_action": "completed",
            "final_answer": "知识查询完成。",
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        request,
        request_id="runtime-request-002",
    )

    print_json("Graph Input", graph.last_input)
    print_json("Runner Response", response)

    initial_state = graph.last_input or {}

    check(
        initial_state["device_model"]
        == DEFAULT_DEVICE_MODEL,
        "未传设备型号时使用默认配置",
    )

    check(
        initial_state["knowledge_base_id"]
        == DEFAULT_KNOWLEDGE_BASE_ID,
        "未传知识库 ID 时使用默认配置",
    )

    check(
        response.status == "completed",
        "默认配置场景执行完成",
    )


def case_needs_input() -> None:
    graph = FakeGraph(
        updates={
            "status": "running",
            "intent": "device_status",
            "missing_fields": ["device_id"],
            "next_action": "ask_user",
            "final_answer": "请提供设备编号。",
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="检查设备当前状态"
        ),
        request_id="runtime-request-003",
    )

    print_json("Graph Output", graph.updates)
    print_json("Runner Response", response)

    check(
        response.status == "needs_input",
        "缺少必要信息时推断为 needs_input",
    )

    check(
        response.next_action == "ask_user",
        "保留向用户补充信息的后续动作",
    )

    check(
        response.final_answer == "请提供设备编号。",
        "保留补充信息提示",
    )


def case_insufficient_evidence() -> None:
    graph = FakeGraph(
        updates={
            "status": "insufficient_evidence",
            "intent": "knowledge_query",
            "rag_status": "rejected",
            "rag_decision": {
                "evidence_sufficient": False,
                "allow_llm": False,
                "reason": "INSUFFICIENT_EVIDENCE",
            },
            "next_action": "provide_more_information",
            "final_answer": (
                "当前知识库中没有找到足够可靠的证据。"
            ),
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="查询一个知识库中不存在的问题"
        ),
        request_id="runtime-request-004",
    )

    print_json("Graph Output", graph.updates)
    print_json("Runner Response", response)

    check(
        response.status
        == "insufficient_evidence",
        "保留证据不足业务状态",
    )

    check(
        response.rag_status == "rejected",
        "保留 RAG 拒答状态",
    )

    check(
        response.rag_decision.get(
            "allow_llm"
        ) is False,
        "保留禁止调用 LLM 的证据决策",
    )

    check(
        response.error == "",
        "证据不足不是技术错误",
    )


def case_human_review() -> None:
    graph = FakeGraph(
        updates={
            "status": "running",
            "intent": "diagnosis",
            "next_action": "awaiting_human_review",
            "diagnosis": "存在严重过热和火灾风险。",
            "risk_level": "critical",
            "human_review_required": True,
            "final_answer": (
                "请由具备权限的工程师进行现场复核。"
            ),
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="设备严重过热并伴有冒烟"
        ),
        request_id="runtime-request-005",
    )

    print_json("Graph Output", graph.updates)
    print_json("Runner Response", response)

    check(
        response.status
        == "human_review_required",
        "人工复核标记优先转换为人工复核状态",
    )

    check(
        response.risk_level == "critical",
        "保留 critical 风险等级",
    )

    check(
        response.human_review_required is True,
        "保留人工复核布尔标记",
    )

    check(
        response.next_action
        == "awaiting_human_review",
        "保留等待人工复核动作",
    )


def case_graph_business_failure() -> None:
    graph = FakeGraph(
        updates={
            "status": "failed",
            "intent": "diagnosis",
            "rag_status": "error",
            "next_action": "retry_later",
            "error": "RAG_TIMEOUT",
            "errors": [
                {
                    "code": "RAG_TIMEOUT",
                    "message": "RAG 请求超时",
                    "node": "retrieve_node",
                    "retryable": True,
                }
            ],
            "final_answer": (
                "知识检索服务超时，请稍后重试。"
            ),
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="诊断 DEVICE-001",
            device_id="DEVICE-001",
        ),
        request_id="runtime-request-006",
    )

    print_json("Graph Output", graph.updates)
    print_json("Runner Response", response)

    check(
        response.status == "failed",
        "保留 Graph 返回的 failed 状态",
    )

    check(
        response.error == "RAG_TIMEOUT",
        "保留稳定业务错误码",
    )

    check(
        len(response.errors) == 1,
        "保留结构化错误列表",
    )

    check(
        response.errors[0].node
        == "retrieve_node",
        "保留错误发生节点",
    )

    check(
        response.errors[0].retryable is True,
        "保留错误可重试属性",
    )


def case_graph_exception() -> None:
    graph = FakeGraph(
        exception=TimeoutError(
            "模拟 Graph 执行超时"
        )
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="诊断 DEVICE-001",
            device_id="DEVICE-001",
        ),
        request_id="runtime-request-007",
    )

    print_json("Runner Response", response)

    check(
        response.status == "failed",
        "Graph 异常被转换为 failed",
    )

    check(
        response.error == "AGENT_RUNTIME_ERROR",
        "Graph 异常使用统一运行错误码",
    )

    check(
        response.request_id
        == "runtime-request-007",
        "异常响应保留原 Request ID",
    )

    check(
        response.next_action == "retry_later",
        "异常响应提示稍后重试",
    )

    check(
        response.execution_trace[0].node
        == "agent_runner",
        "运行异常记录在 Runner 轨迹中",
    )


def case_invalid_graph_result() -> None:
    graph = FakeGraph(
        result="这不是合法的 Graph State"
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="查询设备状态"
        ),
        request_id="runtime-request-008",
    )

    print_json("Raw Graph Result", graph.result)
    print_json("Runner Response", response)

    check(
        response.status == "failed",
        "非字典 Graph 结果被安全收口",
    )

    check(
        response.error == "INVALID_GRAPH_RESULT",
        "非字典结果使用稳定错误码",
    )

    check(
        response.final_answer != "",
        "非法 Graph 结果仍返回用户可读提示",
    )


def case_invalid_terminal_state() -> None:
    graph = FakeGraph(
        updates={
            "status": "running",
            "intent": "diagnosis",
            "next_action": "diagnosis",
            "error": "",
            "missing_fields": [],
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="诊断 DEVICE-001",
            device_id="DEVICE-001",
        ),
        request_id="runtime-request-009",
    )

    print_json("Graph Output", graph.updates)
    print_json("Runner Response", response)

    check(
        response.status == "failed",
        "未结束的 Graph 被转换为 failed",
    )

    check(
        response.error
        == "INVALID_TERMINAL_STATE",
        "非法终态使用稳定错误码",
    )

    check(
        any(
            item.code
            == "INVALID_TERMINAL_STATE"
            for item in response.errors
        ),
        "结构化错误中包含非法终态记录",
    )


def case_internal_fields_not_exposed() -> None:
    updates = completed_updates()

    updates.update(
        {
            "user_query": "内部用户问题",
            "history": [
                {
                    "role": "user",
                    "content": "内部对话历史",
                }
            ],
            "history_summary": "内部摘要",
            "rag_answer": "内部 RAG 原始回答",
            "missing_fields": ["internal_field"],
            "retrieved_documents": [
                {
                    "raw_content": "内部检索文档"
                }
            ],
            "mock_diagnosis": "内部模拟诊断",
            "internal_prompt": "内部提示词",
        }
    )

    graph = FakeGraph(updates=updates)

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="诊断 DEVICE-001",
            device_id="DEVICE-001",
        ),
        request_id="runtime-request-010",
    )

    response_data = response.model_dump(
        mode="json"
    )

    print_json("Runner Response", response_data)

    forbidden_fields = {
        "user_query",
        "history",
        "history_summary",
        "rag_answer",
        "missing_fields",
        "retrieved_documents",
        "mock_diagnosis",
        "internal_prompt",
    }

    exposed_fields = (
        forbidden_fields
        & set(response_data.keys())
    )

    check(
        not exposed_fields,
        f"内部字段没有泄漏：{sorted(exposed_fields)}",
    )

    check(
        response.status == "completed",
        "字段过滤不影响正常响应",
    )


def case_invalid_request_rejected_before_graph() -> None:
    graph = FakeGraph(
        updates=completed_updates()
    )

    runner = AgentRunner(graph=graph)

    invalid_payload = {
        "query": "   ",
        "device_id": "DEVICE-001",
    }

    print_json("Invalid Request", invalid_payload)

    try:
        runner.run(invalid_payload)
    except ValidationError as exc:
        print_json(
            "Validation Errors",
            exc.errors(include_url=False),
        )
    else:
        raise AssertionError(
            "非法请求没有产生 ValidationError"
        )

    check(
        graph.call_count == 0,
        "请求校验失败时不会调用 Graph",
    )


def case_request_id_generated() -> None:
    graph = FakeGraph(
        updates={
            "status": "completed",
            "intent": "device_status",
            "next_action": "completed",
            "final_answer": "设备状态查询完成。",
        }
    )

    response = AgentRunner(
        graph=graph
    ).run(
        DiagnoseRequest(
            query="查询 DEVICE-001 当前状态",
            device_id="DEVICE-001",
        ),
        request_id="   ",
    )

    print_json("Graph Input", graph.last_input)
    print_json("Runner Response", response)

    check(
        bool(response.request_id),
        "空 Request ID 会自动生成",
    )

    check(
        len(response.request_id) == 32,
        "自动生成的 Request ID 为 32 位十六进制字符串",
    )

    try:
        int(response.request_id, 16)
    except ValueError as exc:
        raise AssertionError(
            "自动生成的 Request ID 不是十六进制字符串"
        ) from exc

    print(
        "[CHECK PASS] 自动生成的 Request ID "
        "可以解析为十六进制"
    )

    check(
        graph.last_input is not None
        and graph.last_input["request_id"]
        == response.request_id,
        "State 和响应使用同一个 Request ID",
    )


TEST_CASES = [
    (
        "R01",
        "完整请求正确转换并返回完成响应",
        case_complete_request_mapping,
    ),
    (
        "R02",
        "缺省字段使用统一配置",
        case_default_configuration,
    ),
    (
        "R03",
        "缺少必要信息时返回 needs_input",
        case_needs_input,
    ),
    (
        "R04",
        "RAG 证据不足状态正常返回",
        case_insufficient_evidence,
    ),
    (
        "R05",
        "高风险结果进入人工复核",
        case_human_review,
    ),
    (
        "R06",
        "Graph 业务失败状态完整保留",
        case_graph_business_failure,
    ),
    (
        "R07",
        "Graph 抛出异常时安全收口",
        case_graph_exception,
    ),
    (
        "R08",
        "Graph 返回非法类型时安全收口",
        case_invalid_graph_result,
    ),
    (
        "R09",
        "Graph 未进入合法终态时安全收口",
        case_invalid_terminal_state,
    ),
    (
        "R10",
        "内部 State 字段不会泄漏",
        case_internal_fields_not_exposed,
    ),
    (
        "R11",
        "非法请求不会进入 Graph",
        case_invalid_request_rejected_before_graph,
    ),
    (
        "R12",
        "缺少 Request ID 时自动生成",
        case_request_id_generated,
    ),
]


def main() -> int:
    print(
        "Industrial Fault Diagnosis Agent | "
        "Gate 6.2 Agent Runner Runtime"
    )
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    passed = 0
    failed = 0

    for case_id, title, test_func in TEST_CASES:
        print(f"\n{SEPARATOR}")
        print(f"{case_id} | {title}")
        print(SEPARATOR)

        try:
            test_func()
        except Exception as exc:
            failed += 1

            print(
                f"\n[RESULT] FAIL | "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            passed += 1
            print("\n[RESULT] PASS")

    print(f"\n{SEPARATOR}")
    print(
        "AGENT RUNNER SUMMARY | "
        f"PASS={passed} | "
        f"FAIL={failed} | "
        f"TOTAL={len(TEST_CASES)}"
    )
    print(SEPARATOR)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())