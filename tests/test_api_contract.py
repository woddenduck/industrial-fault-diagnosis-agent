"""Agent API Schema 契约离线验收。

可由 Pytest 自动收集，也可以在项目根目录直接运行：

    python tests/test_api_contract.py

直接运行时会打印每个用例、载荷、校验结果和最终汇总。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.schemas import DiagnoseRequest, DiagnoseResponse, HistoryMessage


SEPARATOR = "=" * 96


def to_jsonable(value: Any) -> Any:
    """Convert Pydantic models and nested values to JSON-compatible data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def print_json(title: str, value: Any) -> None:
    print(f"\n[{title}]")
    print(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[CHECK PASS] {message}")


def expect_validation_error(
    factory: Callable[[], Any],
    expected_location: tuple[str | int, ...],
) -> ValidationError:
    """Assert that model creation fails at the expected field location."""
    try:
        factory()
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        print_json("Validation Errors", errors)
        locations = [tuple(error.get("loc", ())) for error in errors]
        check(
            any(location[: len(expected_location)] == expected_location for location in locations),
            f"校验错误定位到 {expected_location}",
        )
        return exc

    raise AssertionError("预期产生 ValidationError，但数据被错误接受")


def test_minimal_request() -> None:
    payload = {"query": "  查询设备当前状态  "}
    print_json("Input", payload)

    request = DiagnoseRequest.model_validate(payload)
    print_json("Parsed Request", request)

    check(request.query == "查询设备当前状态", "query 自动清理首尾空白")
    check(request.device_id is None, "device_id 缺省值为 None")
    check(request.device_model is None, "device_model 缺省值为 None")
    check(request.history == [], "history 缺省值为空列表")
    check(request.create_report is False, "create_report 缺省值为 False")


def test_full_request() -> None:
    payload = {
        "query": "  请综合诊断 DEVICE-001 当前温度异常  ",
        "device_id": " device-001 ",
        "device_model": " g120c ",
        "knowledge_base_id": " kb_ab50652fe3a4 ",
        "history": [
            {"role": "user", "content": "  设备曾经出现过高温告警。  "},
            {"role": "assistant", "content": "  请提供设备编号。  "},
        ],
        "history_summary": "  用户正在排查设备温度问题。  ",
        "create_report": True,
    }
    print_json("Input", payload)

    request = DiagnoseRequest.model_validate(payload)
    print_json("Parsed Request", request)

    check(request.device_id == "DEVICE-001", "device_id 转换为大写标准形式")
    check(request.device_model == "G120C", "device_model 转换为大写标准形式")
    check(request.knowledge_base_id == "kb_ab50652fe3a4", "知识库 ID 清理首尾空白")
    check(len(request.history) == 2, "保留两条合法历史消息")
    check(all(isinstance(item, HistoryMessage) for item in request.history), "history 转换为 HistoryMessage")
    check(request.history[0].content == "设备曾经出现过高温告警。", "历史消息正文清理首尾空白")
    check(request.history_summary == "用户正在排查设备温度问题。", "历史摘要清理首尾空白")
    check(request.create_report is True, "保留报告生成开关")


def test_blank_query_rejected() -> None:
    payload = {"query": "   "}
    print_json("Input", payload)
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("query",))


def test_oversized_query_rejected() -> None:
    payload = {"query": "故" * 4001}
    print_json("Input Summary", {"query_length": len(payload["query"])})
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("query",))


def test_invalid_device_id_rejected() -> None:
    payload = {"query": "查询设备状态", "device_id": "pump-01"}
    print_json("Input", payload)
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("device_id",))


def test_empty_optional_values_become_none() -> None:
    payload = {
        "query": "查询设备状态",
        "device_id": "   ",
        "device_model": "   ",
        "knowledge_base_id": "   ",
        "history_summary": "   ",
    }
    print_json("Input", payload)

    request = DiagnoseRequest.model_validate(payload)
    print_json("Parsed Request", request)

    check(request.device_id is None, "空 device_id 归一化为 None")
    check(request.device_model is None, "空 device_model 归一化为 None")
    check(request.knowledge_base_id is None, "空 knowledge_base_id 归一化为 None")
    check(request.history_summary is None, "空 history_summary 归一化为 None")


def test_extra_request_field_rejected() -> None:
    payload = {"query": "查询设备状态", "unexpected_field": "not-allowed"}
    print_json("Input", payload)
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("unexpected_field",))


def test_invalid_history_role_rejected() -> None:
    payload = {
        "query": "继续诊断",
        "history": [{"role": "system", "content": "不允许客户端注入系统消息"}],
    }
    print_json("Input", payload)
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("history", 0, "role"))


def test_blank_history_content_rejected() -> None:
    payload = {
        "query": "继续诊断",
        "history": [{"role": "user", "content": "   "}],
    }
    print_json("Input", payload)
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("history", 0, "content"))


def test_too_many_history_messages_rejected() -> None:
    payload = {
        "query": "继续诊断",
        "history": [
            {"role": "user", "content": f"历史消息 {index}"}
            for index in range(21)
        ],
    }
    print_json("Input Summary", {"history_count": len(payload["history"])})
    expect_validation_error(lambda: DiagnoseRequest.model_validate(payload), ("history",))


def build_completed_response_payload() -> dict[str, Any]:
    return {
        "request_id": "req-contract-001",
        "status": "completed",
        "intent": "diagnosis",
        "device_id": "DEVICE-001",
        "device_model": "G120C",
        "knowledge_base_id": "kb_ab50652fe3a4",
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
        "rag_request_id": "rag-contract-001",
        "rag_decision": {
            "evidence_sufficient": True,
            "allow_llm": True,
            "reason": "SAFE_CONTEXT_READY",
        },
        "degraded": False,
        "sources": [
            {
                "document": "G120C_manual.pdf",
                "page": "414-415",
                "section": "9.5 故障和警告列表",
                "chunk_id": "chunk-001",
            }
        ],
        "diagnosis": "设备温度目前处于正常范围，建议继续观察趋势。",
        "risk_level": "medium",
        "human_review_required": False,
        "report": {"report_id": "REPORT-001", "severity": "medium"},
        "final_answer": "诊断已完成。",
        "next_action": "completed",
        "error": "",
        "errors": [],
        "execution_trace": [
            {
                "node": "intent_node",
                "status": "success",
                "duration_ms": 1.25,
                "detail": "intent=diagnosis",
            }
        ],
        "elapsed_ms": 14637.0,
    }


def test_completed_response() -> None:
    payload = build_completed_response_payload()
    print_json("Input", payload)

    response = DiagnoseResponse.model_validate(payload)
    print_json("Parsed Response", response)

    check(response.request_id == "req-contract-001", "保留 Agent request_id")
    check(response.status == "completed", "接受 completed 业务状态")
    check(response.rag_status == "answered", "保留 RAG 业务状态")
    check(response.rag_request_id == "rag-contract-001", "保留 RAG request_id")
    check(len(response.sources) == 1, "证据来源转换为结构化响应")
    check(len(response.execution_trace) == 1, "执行轨迹转换为结构化响应")
    check(response.elapsed_ms == 14637.0, "保留毫秒级执行耗时")


def test_human_review_response() -> None:
    payload = build_completed_response_payload()
    payload.update(
        {
            "request_id": "req-contract-002",
            "status": "human_review_required",
            "risk_level": "critical",
            "human_review_required": True,
            "next_action": "awaiting_human_review",
            "final_answer": "检测到严重危险语义，请等待授权工程师复核。",
        }
    )
    print_json("Input", payload)

    response = DiagnoseResponse.model_validate(payload)
    print_json("Parsed Response", response)

    check(response.status == "human_review_required", "接受人工复核业务状态")
    check(response.risk_level == "critical", "接受 critical 风险等级")
    check(response.human_review_required is True, "保留人工复核标记")
    check(response.next_action == "awaiting_human_review", "保留人工复核后续动作")


def test_failed_response_with_structured_error() -> None:
    payload = {
        "request_id": "req-contract-003",
        "status": "failed",
        "intent": "diagnosis",
        "rag_status": "error",
        "risk_level": "unknown",
        "final_answer": "诊断服务暂时不可用，请稍后重试。",
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
        "elapsed_ms": 30000.0,
    }
    print_json("Input", payload)

    response = DiagnoseResponse.model_validate(payload)
    print_json("Parsed Response", response)

    check(response.status == "failed", "接受 failed 技术状态")
    check(response.rag_status == "error", "失败响应保留 RAG error 状态")
    check(len(response.errors) == 1, "保留结构化错误列表")
    check(response.errors[0].code == "RAG_TIMEOUT", "保留稳定错误码")
    check(response.errors[0].retryable is True, "保留错误可重试标记")


def test_invalid_response_status_rejected() -> None:
    payload = build_completed_response_payload()
    payload["status"] = "success"
    print_json("Input Summary", {"status": payload["status"]})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("status",))


def test_invalid_intent_rejected() -> None:
    payload = build_completed_response_payload()
    payload["intent"] = "delete_device"
    print_json("Input Summary", {"intent": payload["intent"]})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("intent",))


def test_invalid_rag_status_rejected() -> None:
    payload = build_completed_response_payload()
    payload["rag_status"] = "success"
    print_json("Input Summary", {"rag_status": payload["rag_status"]})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("rag_status",))


def test_invalid_risk_level_rejected() -> None:
    payload = build_completed_response_payload()
    payload["risk_level"] = "dangerous"
    print_json("Input Summary", {"risk_level": payload["risk_level"]})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("risk_level",))


def test_missing_request_id_rejected() -> None:
    payload = build_completed_response_payload()
    payload.pop("request_id")
    print_json("Input Summary", {"request_id_present": False})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("request_id",))


def test_extra_response_field_rejected() -> None:
    payload = build_completed_response_payload()
    payload["internal_prompt"] = "不应暴露的内部字段"
    print_json("Input Summary", {"extra_field": "internal_prompt"})
    expect_validation_error(lambda: DiagnoseResponse.model_validate(payload), ("internal_prompt",))


TEST_CASES: list[tuple[str, str, Callable[[], None]]] = [
    ("C01", "最小合法请求采用稳定默认值", test_minimal_request),
    ("C02", "完整请求完成清洗和嵌套模型转换", test_full_request),
    ("C03", "纯空白 query 被拒绝", test_blank_query_rejected),
    ("C04", "超过 4000 字符的 query 被拒绝", test_oversized_query_rejected),
    ("C05", "非标准 device_id 被拒绝", test_invalid_device_id_rejected),
    ("C06", "空白可选字符串统一转换为 None", test_empty_optional_values_become_none),
    ("C07", "请求中的未知字段被拒绝", test_extra_request_field_rejected),
    ("C08", "history 禁止 system 角色", test_invalid_history_role_rejected),
    ("C09", "history 禁止空白正文", test_blank_history_content_rejected),
    ("C10", "history 最多接受 20 条消息", test_too_many_history_messages_rejected),
    ("C11", "完整 completed 响应通过契约校验", test_completed_response),
    ("C12", "critical 响应保留人工复核状态", test_human_review_response),
    ("C13", "failed 响应保留结构化错误", test_failed_response_with_structured_error),
    ("C14", "非法响应 status 被拒绝", test_invalid_response_status_rejected),
    ("C15", "非法 intent 被拒绝", test_invalid_intent_rejected),
    ("C16", "非法 rag_status 被拒绝", test_invalid_rag_status_rejected),
    ("C17", "非法 risk_level 被拒绝", test_invalid_risk_level_rejected),
    ("C18", "缺失 request_id 的响应被拒绝", test_missing_request_id_rejected),
    ("C19", "响应中的未知字段被拒绝", test_extra_response_field_rejected),
]


def main() -> int:
    print("Industrial Fault Diagnosis Agent | Gate 6.1 API Contract")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    passed = 0
    failed = 0

    for test_id, title, test_func in TEST_CASES:
        print(f"\n{SEPARATOR}")
        print(f"{test_id} | {title}")
        print(SEPARATOR)

        try:
            test_func()
        except Exception as exc:  # noqa: BLE001 - test runner must report every case
            failed += 1
            print(f"\n[RESULT] FAIL | {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print("\n[RESULT] PASS")

    print(f"\n{SEPARATOR}")
    print(f"API CONTRACT SUMMARY | PASS={passed} | FAIL={failed} | TOTAL={len(TEST_CASES)}")
    print(SEPARATOR)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
