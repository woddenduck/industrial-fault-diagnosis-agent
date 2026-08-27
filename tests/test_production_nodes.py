"""关卡 4：生产节点直接运行验收。

运行方式（必须从项目根目录执行）：
    python tests/test_production_nodes.py

本文件不依赖 pytest。每个场景都会打印输入、节点输出和逐项检查结果。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent.nodes as nodes
from agent.routing import route_after_risk_check
from agent.state import create_initial_state


SEPARATOR = "=" * 88
passed = 0
failed = 0


def show(label: str, value: Any) -> None:
    print(f"\n[{label}]")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[CHECK PASS] {message}")


def merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    return {**state, **update}


def run_case(case_id: str, title: str, body: Callable[[], None]) -> None:
    global passed, failed
    print(f"\n{SEPARATOR}\n{case_id} | {title}\n{SEPARATOR}")
    try:
        body()
    except Exception as exc:
        failed += 1
        print(f"\n[RESULT] FAIL: {type(exc).__name__}: {exc}")
    else:
        passed += 1
        print("\n[RESULT] PASS")


def case_tool_success() -> None:
    original = nodes.get_device_status
    nodes.get_device_status = lambda device_id: {
        "success": True,
        "tool": "get_device_status",
        "data": {
            "device_id": device_id,
            "status": "warning",
            "temperature": 82.5,
            "vibration": 4.8,
            "running": True,
        },
        "error": None,
    }
    try:
        state = create_initial_state(user_query="状态", device_id="DEVICE-001")
        update = nodes.device_status_node(state)
    finally:
        nodes.get_device_status = original
    show("Initial State", state)
    show("Node Update", update)
    check(update["device_status"]["device_id"] == "DEVICE-001", "保存解包后的业务 data")
    check("success" not in update["device_status"], "没有把工具响应外壳写入状态")
    check(update["status"] == "running", "成功后流程保持 running")


def case_tool_business_failure() -> None:
    original = nodes.get_device_status
    nodes.get_device_status = lambda _device_id: {
        "success": False,
        "tool": "get_device_status",
        "data": None,
        "error": {"code": "DEVICE_NOT_FOUND", "message": "设备不存在"},
    }
    try:
        state = create_initial_state(user_query="状态", device_id="DEVICE-999")
        update = nodes.device_status_node(state)
    finally:
        nodes.get_device_status = original
    show("Initial State", state)
    show("Node Update", update)
    check(update["status"] == "failed", "success=false 被识别为节点失败")
    check(update["errors"][-1]["code"] == "DEVICE_NOT_FOUND", "保留工具业务错误码")
    check("device_status" not in update, "失败数据不会污染 device_status")


def case_maintenance_success() -> None:
    original = nodes.query_maintenance_history
    nodes.query_maintenance_history = lambda device_id: {
        "success": True,
        "tool": "query_maintenance_history",
        "data": {
            "device_id": device_id,
            "record_count": 1,
            "records": [{"date": "2026-08-20", "action": "更换风扇"}],
            "interpretation": "Maintenance records were found.",
        },
        "error": None,
    }
    try:
        state = create_initial_state(user_query="维修历史", device_id="DEVICE-002")
        update = nodes.maintenance_history_node(state)
    finally:
        nodes.query_maintenance_history = original
    show("Node Update", update)
    check(update["maintenance_history"]["record_count"] == 1, "维修记录正确进入状态")


class FakeRAG:
    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.call: dict[str, Any] = {}

    def retrieve(self, question, knowledge_base_id, device_model, **kwargs):
        self.call = {
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "device_model": device_model,
            **kwargs,
        }
        return self.result


def run_with_fake_rag(state: dict[str, Any], result: dict[str, Any]):
    original = nodes.rag_client
    fake = FakeRAG(result)
    nodes.rag_client = fake
    try:
        update = nodes.retrieve_node(state)
    finally:
        nodes.rag_client = original
    return update, fake.call


def case_rag_answered() -> None:
    state = create_initial_state(
        user_query="F3002 报警如何处理？",
        device_model="G120C",
        knowledge_base_id="kb_demo",
        history=[{"role": "user", "content": "设备过热"}],
        history_summary="此前出现温升",
    )
    result = {
        "success": True,
        "status": "answered",
        "answer": "检查风道和冷却风扇。",
        "sources": [{"document": "G120C 手册", "page": "88", "section": "报警"}],
        "decision": {"evidence_sufficient": True, "allow_llm": True},
        "degraded": False,
        "rag_request_id": "rag-001",
        "error": None,
    }
    update, call = run_with_fake_rag(state, result)
    show("RAG Request", call)
    show("Node Update", update)
    check(update["rag_status"] == "answered", "保留 answered 业务状态")
    check(update["rag_answer"] == result["answer"], "保留 RAG 答案")
    check(update["sources"] == result["sources"], "保留证据来源")
    check(call["history_summary"] == "此前出现温升", "透传历史摘要")


def case_rag_rejected() -> None:
    state = create_initial_state(
        user_query="未知问题", device_model="G120C", knowledge_base_id="kb_demo"
    )
    result = {
        "success": True,
        "status": "rejected",
        "answer": "当前证据不足，无法可靠回答。",
        "sources": [],
        "decision": {"evidence_sufficient": False, "allow_llm": False},
        "degraded": False,
        "rag_request_id": "rag-002",
        "error": None,
    }
    update, _ = run_with_fake_rag(state, result)
    show("Node Update", update)
    check(update["rag_status"] == "rejected", "rejected 不会被误判为技术失败")
    check(update["status"] == "insufficient_evidence", "进入证据不足状态")
    check(update["next_action"] == "final_answer", "停止无依据的后续诊断")
    check(not update.get("errors"), "业务拒绝不写入技术错误列表")


def case_rag_error() -> None:
    state = create_initial_state(
        user_query="报警", device_model="G120C", knowledge_base_id="kb_demo"
    )
    result = {
        "success": False,
        "status": "error",
        "answer": "",
        "sources": [],
        "rag_request_id": "",
        "error": {"code": "RAG_TIMEOUT", "message": "等待响应超时", "retryable": False},
    }
    update, _ = run_with_fake_rag(state, result)
    show("Node Update", update)
    check(update["rag_status"] == "error", "技术错误使用 rag_status=error")
    check(update["status"] == "failed", "技术错误终止当前请求")
    check(update["errors"][-1]["code"] == "RAG_TIMEOUT", "保留 RAG 错误码")


def case_diagnosis_aggregation() -> None:
    state = create_initial_state(
        user_query="综合诊断", device_id="DEVICE-001", device_model="G120C",
        knowledge_base_id="kb_demo",
    )
    state.update(
        {
            "device_status": {
                "device_id": "DEVICE-001", "status": "warning",
                "temperature": 82.5, "vibration": 4.8, "running": True,
            },
            "maintenance_history": {
                "record_count": 2, "records": [],
                "interpretation": "Maintenance records were found.",
            },
            "rag_status": "answered",
            "rag_answer": "手册证据表明应检查散热风道。",
        }
    )
    update = nodes.diagnosis_node(state)
    show("Node Update", update)
    diagnosis = update["diagnosis"]
    check("82.5" in diagnosis, "诊断包含实时状态")
    check("2 条维修记录" in diagnosis, "诊断包含维修历史")
    check("检查散热风道" in diagnosis, "诊断包含有依据的 RAG 结论")
    check("mock_diagnosis" not in update, "不再使用模拟诊断字段")


def case_high_risk() -> None:
    state = create_initial_state(user_query="诊断", device_id="DEVICE-001")
    state.update({"diagnosis": "建议解除保护后写入PLC参数。", "status": "running"})
    update = nodes.risk_check_node(state)
    merged = merge(state, update)
    show("Node Update", update)
    check(update["risk_level"] == "high", "危险控制建议被判定为 high")
    check(update["human_review_required"] is True, "高风险要求人工复核")
    check(route_after_risk_check(merged) == "human_review", "风险路由进入人工复核")


def case_critical_risk() -> None:
    state = create_initial_state(user_query="诊断", device_id="DEVICE-001")
    state.update({"diagnosis": "制动电阻过热，存在火灾风险。", "status": "running"})
    update = nodes.risk_check_node(state)
    merged = merge(state, update)
    show("Node Update", update)
    check(update["risk_level"] == "critical", "人身/火灾危险被判定为 critical")
    check(route_after_risk_check(merged) == "human_review", "critical 同样进入人工复核")


def case_final_answer() -> None:
    state = create_initial_state(
        user_query="F3002 是什么？", device_model="G120C", knowledge_base_id="kb_demo"
    )
    state.update(
        {
            "intent": "knowledge_query",
            "rag_status": "answered",
            "rag_answer": "F3002 表示直流母线过压。",
            "sources": [{"document": "G120C 手册", "page": "120", "section": "故障码"}],
        }
    )
    update = nodes.final_answer_node(state)
    show("Node Update", update)
    check("F3002 表示" in update["final_answer"], "最终回答包含业务答案")
    check("G120C 手册" in update["final_answer"], "最终回答包含证据来源")
    check(update["status"] == "completed", "正常请求完成")


def case_dynamic_ask_user() -> None:
    state = create_initial_state(user_query="怎么处理？")
    state.update(
        {
            "intent": "knowledge_query",
            "missing_fields": ["knowledge_base_id"],
            "status": "needs_input",
        }
    )
    update = nodes.ask_user_node(state)
    show("Node Update", update)
    check("知识库编号" in update["final_answer"], "按真实缺失字段生成追问")
    check("设备编号" not in update["final_answer"], "不再固定追问 device_id")


def case_report_skip() -> None:
    state = create_initial_state(user_query="诊断", device_id="DEVICE-001", create_report=False)
    update = nodes.report_node(state)
    show("Node Update", update)
    check(update["next_action"] == "final_answer", "未请求报告时安全跳过")
    check("report" not in update, "跳过时不制造空报告")


def case_report_success() -> None:
    original = nodes.create_diagnostic_report
    captured: dict[str, Any] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "tool": "create_diagnostic_report",
            "data": {"report_id": "REPORT-DEMO", **kwargs},
            "error": None,
        }

    nodes.create_diagnostic_report = fake_report
    try:
        state = create_initial_state(
            user_query="诊断并生成报告", device_id="DEVICE-001", create_report=True
        )
        state.update(
            {
                "diagnosis": "冷却风道可能堵塞。",
                "risk_level": "medium",
                "human_review_required": False,
            }
        )
        update = nodes.report_node(state)
    finally:
        nodes.create_diagnostic_report = original
    show("Report Tool Input", captured)
    show("Node Update", update)
    check(update["report"]["report_id"] == "REPORT-DEMO", "报告写入 AgentState")
    check(captured["severity"] == "medium", "风险等级传入报告工具")


CASES = [
    ("N01", "设备状态工具成功并解包业务数据", case_tool_success),
    ("N02", "工具业务失败不会被当成成功", case_tool_business_failure),
    ("N03", "维修历史工具成功", case_maintenance_success),
    ("N04", "RAG answered 契约映射", case_rag_answered),
    ("N05", "RAG rejected 证据门禁", case_rag_rejected),
    ("N06", "RAG 技术错误标准化", case_rag_error),
    ("N07", "三类证据聚合诊断", case_diagnosis_aggregation),
    ("N08", "高风险控制建议进入人工复核", case_high_risk),
    ("N09", "critical 风险进入人工复核", case_critical_risk),
    ("N10", "生成带引用的最终回答", case_final_answer),
    ("N11", "按缺失字段动态追问", case_dynamic_ask_user),
    ("N12", "未请求报告时跳过", case_report_skip),
    ("N13", "可选诊断报告生成", case_report_success),
]


if __name__ == "__main__":
    print("Industrial Fault Diagnosis Agent | Gate 4 Production Nodes")
    print(f"Project root: {PROJECT_ROOT}")
    for case_id, title, body in CASES:
        run_case(case_id, title, body)

    print(f"\n{SEPARATOR}")
    print(f"SUMMARY | PASS={passed} | FAIL={failed} | TOTAL={passed + failed}")
    print(SEPARATOR)
    if failed:
        raise SystemExit(1)
