"""关卡 6.3：Agent FastAPI 接口验收。

本测试直接使用 Python 运行，不依赖 pytest。

测试使用：

1. FastAPI TestClient
2. 假 Agent Runner
3. 假 RAG 健康检查

不会访问真实 Graph、RAG 或模型服务。
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.api import create_app
from agent.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
)


SEPARATOR = "=" * 96


class FakeRunner:
    """模拟 Agent Runner，并记录 API 传入的数据。"""

    def __init__(
        self,
        *,
        response_template: dict[str, Any] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.response_template = (
            response_template
            if response_template is not None
            else completed_response_template()
        )

        self.exception = exception

        self.call_count = 0
        self.last_request: DiagnoseRequest | None = None
        self.last_request_id: str | None = None

    def run(
        self,
        request: DiagnoseRequest,
        *,
        request_id: str | None = None,
    ) -> DiagnoseResponse:
        self.call_count += 1
        self.last_request = request
        self.last_request_id = request_id

        if self.exception is not None:
            raise self.exception

        response_data = deepcopy(
            self.response_template
        )

        response_data["request_id"] = (
            request_id or "fake-request-id"
        )

        return DiagnoseResponse.model_validate(
            response_data
        )


def completed_response_template() -> dict[str, Any]:
    """构造默认的正常诊断响应。"""

    return {
        "status": "completed",
        "intent": "diagnosis",
        "device_id": "DEVICE-001",
        "device_model": "G120C",
        "knowledge_base_id": "kb_test_001",
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
        "rag_request_id": "rag-api-test-001",
        "rag_decision": {
            "evidence_sufficient": True,
            "allow_llm": True,
        },
        "degraded": False,
        "sources": [
            {
                "document": "G120C_manual.pdf",
                "page": "414-415",
                "section": "9.5 故障和警告列表",
                "chunk_id": "chunk-api-001",
            }
        ],
        "diagnosis": "设备当前状态正常。",
        "risk_level": "medium",
        "human_review_required": False,
        "report": None,
        "final_answer": "诊断已经完成。",
        "next_action": "completed",
        "error": "",
        "errors": [],
        "execution_trace": [
            {
                "node": "final_answer_node",
                "status": "success",
                "detail": "status=completed",
            }
        ],
        "elapsed_ms": 15.5,
    }


def make_client(
    runner: FakeRunner,
    *,
    rag_health: dict[str, Any] | None = None,
) -> TestClient:
    """创建不会访问真实外部服务的测试客户端。"""

    health_result = (
        rag_health
        if rag_health is not None
        else {
            "status": "healthy",
            "http_status": 200,
            "detail": {
                "status": "healthy"
            },
        }
    )

    app = create_app(
        runner=runner,
        rag_health_checker=lambda: deepcopy(
            health_result
        ),
    )

    return TestClient(
        app,
        raise_server_exceptions=False,
    )


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


def print_http_response(response) -> None:
    print(f"\n[HTTP Status]\n{response.status_code}")

    print(
        "\n[X-Request-ID]\n"
        f"{response.headers.get('X-Request-ID')}"
    )

    try:
        body = response.json()
    except ValueError:
        body = response.text

    print_json("Response Body", body)


def check(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[CHECK PASS] {message}")


def check_generated_request_id(
    request_id: str | None,
) -> None:
    check(
        request_id is not None,
        "响应包含 X-Request-ID",
    )

    normalized = request_id or ""

    check(
        len(normalized) == 32,
        "自动生成的 Request ID 长度为 32",
    )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise AssertionError(
            "自动生成的 Request ID 不是十六进制"
        ) from exc

    print(
        "[CHECK PASS] 自动生成的 Request ID "
        "可以解析为十六进制"
    )


def case_health_healthy() -> None:
    runner = FakeRunner()

    with make_client(runner) as client:
        response = client.get("/health")

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "健康检查返回 HTTP 200",
    )

    check(
        body["status"] == "healthy",
        "RAG 健康时 Agent 状态为 healthy",
    )

    check(
        body["graph"] == "ready",
        "Graph 状态为 ready",
    )

    check(
        body["rag"]["status"] == "healthy",
        "返回 RAG 健康状态",
    )

    check(
        runner.call_count == 0,
        "健康检查不会执行诊断 Runner",
    )


def case_health_degraded() -> None:
    runner = FakeRunner()

    with make_client(
        runner,
        rag_health={
            "status": "unavailable",
            "error": "RAG_HEALTH_TIMEOUT",
        },
    ) as client:
        response = client.get("/health")

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "降级状态仍返回 HTTP 200",
    )

    check(
        body["status"] == "degraded",
        "RAG 不可用时 Agent 标记为 degraded",
    )

    check(
        body["graph"] == "ready",
        "RAG 不可用不影响 Graph 加载状态",
    )

    check(
        body["rag"]["error"]
        == "RAG_HEALTH_TIMEOUT",
        "返回明确的 RAG 健康错误码",
    )


def case_graph_info() -> None:
    runner = FakeRunner()

    with make_client(runner) as client:
        response = client.get("/v1/graph")

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "Graph 信息接口返回 HTTP 200",
    )

    check(
        body["name"]
        == "industrial_fault_diagnosis_graph",
        "返回正式 Graph 名称",
    )

    check(
        "retrieve" in body["nodes"],
        "Graph 节点包含 retrieve",
    )

    check(
        "risk_check" in body["nodes"],
        "Graph 节点包含 risk_check",
    )

    check(
        "human_review" in body["nodes"],
        "Graph 节点包含 human_review",
    )

    check(
        "diagnosis" in body["intents"],
        "支持综合诊断意图",
    )


def case_completed_diagnosis() -> None:
    runner = FakeRunner()

    request_payload = {
        "query": "请综合诊断 DEVICE-001 当前状态",
        "device_id": "device-001",
        "device_model": "g120c",
        "knowledge_base_id": "kb_test_001",
        "create_report": False,
    }

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json=request_payload,
            headers={
                "X-Request-ID": "api-request-001"
            },
        )

    print_json("Request Body", request_payload)
    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "正常诊断返回 HTTP 200",
    )

    check(
        body["status"] == "completed",
        "返回 completed 业务状态",
    )

    check(
        body["request_id"]
        == "api-request-001",
        "响应体保留客户端 Request ID",
    )

    check(
        response.headers["X-Request-ID"]
        == "api-request-001",
        "响应头保留客户端 Request ID",
    )

    check(
        runner.call_count == 1,
        "诊断接口只调用一次 Runner",
    )

    check(
        isinstance(
            runner.last_request,
            DiagnoseRequest,
        ),
        "API 向 Runner 传递 DiagnoseRequest",
    )

    check(
        runner.last_request is not None
        and runner.last_request.device_id
        == "DEVICE-001",
        "请求进入 Runner 前完成设备编号清洗",
    )

    check(
        runner.last_request_id
        == "api-request-001",
        "API 将 Request ID 传入 Runner",
    )


def case_request_id_generated() -> None:
    runner = FakeRunner()

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "查询 DEVICE-001 当前状态",
                "device_id": "DEVICE-001",
            },
        )

    print_http_response(response)

    header_request_id = response.headers.get(
        "X-Request-ID"
    )

    check_generated_request_id(
        header_request_id
    )

    body = response.json()

    check(
        body["request_id"]
        == header_request_id,
        "响应头和响应体使用相同 Request ID",
    )

    check(
        runner.last_request_id
        == header_request_id,
        "Runner 使用同一个 Request ID",
    )


def case_invalid_request_id_replaced() -> None:
    runner = FakeRunner()

    invalid_request_id = (
        "invalid request id with spaces"
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "查询 DEVICE-001 当前状态",
                "device_id": "DEVICE-001",
            },
            headers={
                "X-Request-ID": invalid_request_id
            },
        )

    print_http_response(response)

    generated_request_id = (
        response.headers.get("X-Request-ID")
    )

    check(
        generated_request_id
        != invalid_request_id,
        "非法 Request ID 不会被继续使用",
    )

    check_generated_request_id(
        generated_request_id
    )

    check(
        runner.last_request_id
        == generated_request_id,
        "替换后的 Request ID 传入 Runner",
    )


def case_validation_error() -> None:
    runner = FakeRunner()

    invalid_payload = {
        "query": "   ",
        "device_id": "DEVICE-001",
    }

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json=invalid_payload,
            headers={
                "X-Request-ID": "api-request-422"
            },
        )

    print_json("Invalid Request", invalid_payload)
    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 422,
        "请求 Schema 错误返回 HTTP 422",
    )

    check(
        isinstance(body.get("detail"), list),
        "返回 FastAPI 结构化校验错误",
    )

    check(
        response.headers["X-Request-ID"]
        == "api-request-422",
        "422 响应仍然包含 Request ID",
    )

    check(
        runner.call_count == 0,
        "请求校验失败时不会调用 Runner",
    )


def case_needs_input() -> None:
    runner = FakeRunner(
        response_template={
            "status": "needs_input",
            "intent": "device_status",
            "final_answer": "请提供设备编号。",
            "next_action": "ask_user",
            "rag_status": "not_called",
            "risk_level": "unknown",
            "error": "",
            "errors": [],
            "execution_trace": [],
            "elapsed_ms": 1.0,
        }
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "检查设备状态"
            },
        )

    print_http_response(response)

    check(
        response.status_code == 200,
        "needs_input 属于正常业务响应",
    )

    check(
        response.json()["status"]
        == "needs_input",
        "响应体保留 needs_input",
    )

    check(
        response.json()["next_action"]
        == "ask_user",
        "提示客户端继续补充信息",
    )


def case_insufficient_evidence() -> None:
    runner = FakeRunner(
        response_template={
            "status": "insufficient_evidence",
            "intent": "knowledge_query",
            "rag_status": "rejected",
            "rag_decision": {
                "evidence_sufficient": False,
                "allow_llm": False,
            },
            "final_answer": (
                "当前知识库没有足够可靠的证据。"
            ),
            "next_action": (
                "provide_more_information"
            ),
            "risk_level": "unknown",
            "error": "",
            "errors": [],
            "execution_trace": [],
            "elapsed_ms": 2.0,
        }
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "查询未知故障"
            },
        )

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "证据不足不是 HTTP 技术错误",
    )

    check(
        body["status"]
        == "insufficient_evidence",
        "响应保留证据不足状态",
    )

    check(
        body["rag_status"] == "rejected",
        "响应保留 RAG 拒答状态",
    )


def case_human_review() -> None:
    runner = FakeRunner(
        response_template={
            "status": "human_review_required",
            "intent": "diagnosis",
            "diagnosis": "存在严重过热风险。",
            "risk_level": "critical",
            "human_review_required": True,
            "final_answer": (
                "请等待授权工程师现场复核。"
            ),
            "next_action": (
                "awaiting_human_review"
            ),
            "rag_status": "answered",
            "error": "",
            "errors": [],
            "execution_trace": [],
            "elapsed_ms": 3.0,
        }
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "设备严重过热并伴有冒烟",
                "device_id": "DEVICE-001",
            },
        )

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 200,
        "人工复核状态返回 HTTP 200",
    )

    check(
        body["status"]
        == "human_review_required",
        "保留人工复核业务状态",
    )

    check(
        body["risk_level"] == "critical",
        "保留 critical 风险等级",
    )

    check(
        body["human_review_required"] is True,
        "保留人工复核标记",
    )


def assert_failed_http_mapping(
    *,
    error_code: str,
    expected_http_status: int,
) -> None:
    runner = FakeRunner(
        response_template={
            "status": "failed",
            "intent": "diagnosis",
            "rag_status": "error",
            "risk_level": "unknown",
            "final_answer": "请求执行失败。",
            "next_action": "retry_later",
            "error": error_code,
            "errors": [
                {
                    "code": error_code,
                    "message": "模拟错误",
                    "node": "test_node",
                    "retryable": True,
                }
            ],
            "execution_trace": [],
            "elapsed_ms": 4.0,
        }
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "诊断 DEVICE-001",
                "device_id": "DEVICE-001",
            },
            headers={
                "X-Request-ID": (
                    f"mapping-{error_code.lower()}"
                )
            },
        )

    print_json(
        "Mapping Input",
        {
            "error_code": error_code,
            "expected_http_status": (
                expected_http_status
            ),
        },
    )

    print_http_response(response)

    body = response.json()

    check(
        response.status_code
        == expected_http_status,
        (
            f"{error_code} 映射为 "
            f"HTTP {expected_http_status}"
        ),
    )

    check(
        body["status"] == "failed",
        "HTTP 错误响应保留 failed 状态",
    )

    check(
        body["error"] == error_code,
        "HTTP 错误响应保留原错误码",
    )


def case_device_not_found() -> None:
    assert_failed_http_mapping(
        error_code="DEVICE_NOT_FOUND",
        expected_http_status=404,
    )


def case_rag_unavailable() -> None:
    assert_failed_http_mapping(
        error_code="RAG_UNAVAILABLE",
        expected_http_status=503,
    )


def case_rag_timeout() -> None:
    assert_failed_http_mapping(
        error_code="RAG_TIMEOUT",
        expected_http_status=504,
    )


def case_unknown_failure() -> None:
    assert_failed_http_mapping(
        error_code="AGENT_RUNTIME_ERROR",
        expected_http_status=500,
    )


def case_unhandled_api_exception() -> None:
    runner = FakeRunner(
        exception=RuntimeError(
            "模拟 API 未处理异常"
        )
    )

    with make_client(runner) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "诊断 DEVICE-001",
                "device_id": "DEVICE-001",
            },
            headers={
                "X-Request-ID": "api-exception-001"
            },
        )

    print_http_response(response)

    body = response.json()

    check(
        response.status_code == 500,
        "未处理异常返回 HTTP 500",
    )

    check(
        body["status"] == "failed",
        "未处理异常返回 failed",
    )

    check(
        body["error"] == "AGENT_API_ERROR",
        "未处理异常使用统一 API 错误码",
    )

    check(
        body["request_id"]
        == "api-exception-001",
        "异常响应保留 Request ID",
    )

    check(
        response.headers["X-Request-ID"]
        == "api-exception-001",
        "异常响应头包含 Request ID",
    )

    check(
        body["errors"][0]["node"]
        == "agent_api",
        "异常定位到 agent_api",
    )


TEST_CASES = [
    (
        "A01",
        "RAG 正常时健康检查返回 healthy",
        case_health_healthy,
    ),
    (
        "A02",
        "RAG 不可用时健康检查返回 degraded",
        case_health_degraded,
    ),
    (
        "A03",
        "Graph 信息接口返回节点和意图",
        case_graph_info,
    ),
    (
        "A04",
        "正常诊断返回 HTTP 200",
        case_completed_diagnosis,
    ),
    (
        "A05",
        "缺少 Request ID 时自动生成",
        case_request_id_generated,
    ),
    (
        "A06",
        "非法 Request ID 被安全替换",
        case_invalid_request_id_replaced,
    ),
    (
        "A07",
        "非法请求返回 HTTP 422",
        case_validation_error,
    ),
    (
        "A08",
        "needs_input 返回 HTTP 200",
        case_needs_input,
    ),
    (
        "A09",
        "证据不足返回 HTTP 200",
        case_insufficient_evidence,
    ),
    (
        "A10",
        "人工复核返回 HTTP 200",
        case_human_review,
    ),
    (
        "A11",
        "设备不存在映射为 HTTP 404",
        case_device_not_found,
    ),
    (
        "A12",
        "RAG 不可用映射为 HTTP 503",
        case_rag_unavailable,
    ),
    (
        "A13",
        "RAG 超时映射为 HTTP 504",
        case_rag_timeout,
    ),
    (
        "A14",
        "未知运行失败映射为 HTTP 500",
        case_unknown_failure,
    ),
    (
        "A15",
        "API 未处理异常统一收口",
        case_unhandled_api_exception,
    ),
]


def main() -> int:
    print(
        "Industrial Fault Diagnosis Agent | "
        "Gate 6.3 FastAPI"
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
        "AGENT API SUMMARY | "
        f"PASS={passed} | "
        f"FAIL={failed} | "
        f"TOTAL={len(TEST_CASES)}"
    )

    print(SEPARATOR)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())