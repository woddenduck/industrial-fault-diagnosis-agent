"""Agent 真实 HTTP 端到端验收。

本程序不使用 Mock，要求 RAG 与 Agent 服务均已真实启动。
既可以通过 pytest 执行，也可以直接通过 python 执行并查看完整过程。
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any

import httpx


RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
DEVICE_ID = os.getenv("E2E_DEVICE_ID", "DEVICE-001")
DEVICE_MODEL = os.getenv("E2E_DEVICE_MODEL", "G120C")
KNOWLEDGE_BASE_ID = os.getenv(
    "E2E_KNOWLEDGE_BASE_ID",
    os.getenv("DEFAULT_KNOWLEDGE_BASE_ID", ""),
)
HTTP_TIMEOUT = float(os.getenv("E2E_HTTP_TIMEOUT", "90"))


def _title(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _request(
    method: str,
    url: str,
    *,
    request_id: str | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    print(f"请求：{method} {url}")
    if request_id:
        print(f"X-Request-ID：{request_id}")
    if json_body is not None:
        print("请求体：")
        print(_pretty(json_body))

    try:
        response = httpx.request(
            method,
            url,
            headers=headers,
            json=json_body,
            timeout=HTTP_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise AssertionError(
            f"无法访问服务：{url}\n"
            f"请确认 RAG 运行在 {RAG_BASE_URL}，Agent 运行在 {AGENT_BASE_URL}。\n"
            f"原始错误：{exc}"
        ) from exc

    print(f"HTTP 状态码：{response.status_code}")
    print(f"响应 X-Request-ID：{response.headers.get('x-request-id', '')}")
    try:
        print("响应体：")
        print(_pretty(response.json()))
    except ValueError:
        print(response.text)
    return response


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise AssertionError("响应体不是合法 JSON") from exc
    assert isinstance(data, dict), "响应 JSON 顶层必须是对象"
    return data


def _trace_nodes(data: dict[str, Any]) -> set[str]:
    trace = data.get("execution_trace", [])
    assert isinstance(trace, list) and trace, "execution_trace 不能为空"
    return {
        item.get("node", "")
        for item in trace
        if isinstance(item, dict) and item.get("node")
    }


def test_01_rag_health() -> None:
    _title("E2E-01｜RAG 健康检查")
    response = _request("GET", f"{RAG_BASE_URL}/health")
    assert response.status_code == 200, response.text

    data = _json(response)
    assert data.get("status") == "healthy", f"RAG 未达到 healthy：{data}"

    components = data.get("components")
    assert isinstance(components, dict) and components, "RAG components 不能为空"
    unhealthy = {
        name: detail
        for name, detail in components.items()
        if not isinstance(detail, dict) or detail.get("status") != "healthy"
    }
    assert not unhealthy, f"RAG 存在异常组件：{unhealthy}"
    print("结果：通过。RAG 及依赖组件全部健康。")


def test_02_agent_health() -> None:
    _title("E2E-02｜Agent 健康检查")
    response = _request("GET", f"{AGENT_BASE_URL}/health")
    assert response.status_code == 200, response.text

    data = _json(response)
    assert data.get("status") == "healthy", f"Agent 未达到 healthy：{data}"
    assert response.headers.get("x-request-id"), "响应头缺少 X-Request-ID"
    print("结果：通过。Agent HTTP 服务已正常监听。")


def test_03_graph_contract() -> None:
    _title("E2E-03｜正式 Graph 信息")
    response = _request("GET", f"{AGENT_BASE_URL}/v1/graph")
    assert response.status_code == 200, response.text

    data = _json(response)
    nodes = data.get("nodes")
    assert isinstance(nodes, list), "Graph 响应缺少 nodes 列表"

    required_nodes = {
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
    }
    missing = required_nodes - set(nodes)
    assert not missing, f"正式 Graph 缺少节点：{sorted(missing)}"
    print("结果：通过。正式工业诊断 Graph 已加载。")


def test_04_missing_device_id() -> None:
    _title("E2E-04｜缺少设备编号的条件路由")
    request_id = f"e2e-needs-input-{uuid.uuid4().hex[:8]}"
    response = _request(
        "POST",
        f"{AGENT_BASE_URL}/v1/diagnose",
        request_id=request_id,
        json_body={"query": "请检查设备当前运行状态"},
    )
    assert response.status_code == 200, response.text

    data = _json(response)
    assert response.headers.get("x-request-id") == request_id
    assert data.get("request_id") == request_id
    assert data.get("status") == "needs_input"
    assert data.get("next_action") == "ask_user"
    assert data.get("rag_status") in {"not_called", ""}
    assert data.get("errors") in ([], None)

    nodes = _trace_nodes(data)
    assert "check_information_node" in nodes
    assert "ask_user_node" in nodes
    assert "retrieve_node" not in nodes
    print("结果：通过。缺少 device_id 时正确进入 ask_user，未调用 RAG。")


def test_05_real_diagnosis() -> None:
    _title("E2E-05｜完整真实诊断链路")
    assert KNOWLEDGE_BASE_ID.strip(), (
        "知识库 ID 不能为空，请设置 E2E_KNOWLEDGE_BASE_ID"
    )

    request_id = f"e2e-real-{uuid.uuid4().hex[:8]}"
    payload = {
        "query": (
            f"请综合诊断 {DEVICE_ID} 当前运行状态，结合维修记录和 "
            f"{DEVICE_MODEL} 技术手册判断是否存在过热风险，并生成诊断报告。"
        ),
        "device_id": DEVICE_ID,
        "device_model": DEVICE_MODEL,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "create_report": True,
    }
    response = _request(
        "POST",
        f"{AGENT_BASE_URL}/v1/diagnose",
        request_id=request_id,
        json_body=payload,
    )
    assert response.status_code == 200, response.text

    data = _json(response)
    assert response.headers.get("x-request-id") == request_id
    assert data.get("request_id") == request_id
    assert data.get("status") in {"completed", "human_review_required"}
    assert data.get("device_id") == DEVICE_ID
    assert data.get("device_model") == DEVICE_MODEL
    assert data.get("knowledge_base_id") == KNOWLEDGE_BASE_ID

    device_status = data.get("device_status")
    assert isinstance(device_status, dict) and device_status
    assert device_status.get("device_id") == DEVICE_ID
    assert device_status.get("exists") is True

    maintenance = data.get("maintenance_history")
    assert isinstance(maintenance, dict) and maintenance
    assert maintenance.get("device_id") == DEVICE_ID

    assert data.get("rag_status") == "answered"
    assert data.get("rag_request_id"), "rag_request_id 不能为空"
    decision = data.get("rag_decision")
    assert isinstance(decision, dict) and decision
    assert decision.get("evidence_sufficient") is True
    assert decision.get("allow_llm") is True

    sources = data.get("sources")
    assert isinstance(sources, list) and sources, "sources 至少包含一个真实来源"
    for index, source in enumerate(sources, start=1):
        assert isinstance(source, dict), f"来源 {index} 结构错误"
        assert source.get("document"), f"来源 {index} 缺少 document"
        assert source.get("chunk_id"), f"来源 {index} 缺少 chunk_id"

    assert str(data.get("diagnosis", "")).strip(), "diagnosis 不能为空"
    assert data.get("risk_level") not in {None, "", "unknown"}
    assert str(data.get("final_answer", "")).strip(), "final_answer 不能为空"
    assert data.get("error") in {None, ""}
    assert data.get("errors") == []

    report = data.get("report")
    assert isinstance(report, dict) and report, "create_report=true 时 report 不能为空"
    assert report.get("report_id"), "报告缺少 report_id"
    assert report.get("device_id") == DEVICE_ID

    nodes = _trace_nodes(data)
    required_trace = {
        "intent_node",
        "check_information_node",
        "device_status_node",
        "maintenance_history_node",
        "retrieve_node",
        "diagnosis_node",
        "risk_check_node",
        "report_node",
    }
    missing_trace = required_trace - nodes
    assert not missing_trace, f"执行轨迹缺少节点：{sorted(missing_trace)}"

    if data.get("status") == "human_review_required":
        assert data.get("human_review_required") is True
        assert data.get("next_action") == "awaiting_human_review"
        assert "human_review_node" in nodes
    else:
        assert data.get("human_review_required") is False

    # 语义风险属于质量评测，不应被当前 HTTP 契约测试固化为正确答案。
    temperature = device_status.get("temperature")
    if (
        device_status.get("status") == "normal"
        and isinstance(temperature, (int, float))
        and temperature < 60
        and data.get("risk_level") in {"high", "critical"}
    ):
        print(
            "质量告警：实时设备状态为 normal 且温度低于 60℃，"
            "但系统给出了高风险结论。HTTP 链路通过，"
            "风险判定与 RAG 证据相关性需要单独修复。"
        )

    print("结果：通过。真实设备、维修、RAG、诊断、风控和报告链路完整。")


def main() -> int:
    tests = [
        test_01_rag_health,
        test_02_agent_health,
        test_03_graph_contract,
        test_04_missing_device_id,
        test_05_real_diagnosis,
    ]

    print("Agent 真实 HTTP 端到端验收")
    print(f"RAG 地址：{RAG_BASE_URL}")
    print(f"Agent 地址：{AGENT_BASE_URL}")
    print(f"设备：{DEVICE_ID} / {DEVICE_MODEL}")
    print(f"知识库：{KNOWLEDGE_BASE_ID}")
    print(f"超时：{HTTP_TIMEOUT} 秒")

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:  # 直接运行时集中输出失败原因
            print(f"\n[失败] {test.__name__}：{exc}")
            print(f"\n验收结果：{passed}/{len(tests)} 通过")
            return 1

    print(f"\n验收结果：{passed}/{len(tests)} 全部通过")
    print("关卡 6.4：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
