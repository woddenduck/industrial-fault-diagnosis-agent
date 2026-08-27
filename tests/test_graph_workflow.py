"""
关卡 5.3：正式 LangGraph 工作流验收。

运行方式：

    python tests/test_graph_workflow.py

说明：

- 不使用 pytest；
- 运行真实 industrial_graph；
- 使用可控替身代替 Tool 和 RAG；
- 打印服务调用、执行轨迹和最终状态；
- 不需要启动真实 RAG 服务。
"""

from __future__ import annotations

import copy
import json
import sys

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import agent.nodes as nodes

from agent.graph import industrial_graph

from agent.state import (
    create_initial_state,
)


SEPARATOR = "=" * 96

passed = 0
failed = 0


# ============================================================
# 输出与检查
# ============================================================

def show(
    label: str,
    value: Any,
) -> None:

    print(f"\n[{label}]")

    print(
        json.dumps(
            value,
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
        raise AssertionError(
            message
        )

    print(
        f"[CHECK PASS] {message}"
    )


def get_trace_nodes(
    result: dict[str, Any],
) -> list[str]:

    return [
        str(item.get("node"))
        for item
        in result.get(
            "execution_trace",
            [],
        )
    ]


def show_final_result(
    result: dict[str, Any],
) -> None:

    important_fields = {
        "request_id":
            result.get("request_id"),

        "intent":
            result.get("intent"),

        "device_id":
            result.get("device_id"),

        "status":
            result.get("status"),

        "next_action":
            result.get("next_action"),

        "rag_status":
            result.get("rag_status"),

        "risk_level":
            result.get("risk_level"),

        "human_review_required":
            result.get(
                "human_review_required"
            ),

        "report":
            result.get("report"),

        "final_answer":
            result.get("final_answer"),

        "error":
            result.get("error"),

        "errors":
            result.get("errors"),
    }

    show(
        "Final Result",
        important_fields,
    )


# ============================================================
# 模拟 RAG 结果
# ============================================================

def answered_rag(
    answer: str,
) -> dict[str, Any]:

    return {
        "success": True,
        "status": "answered",
        "answer": answer,

        "sources": [
            {
                "document":
                    "G120C 手册",

                "page":
                    "120",

                "section":
                    "故障诊断",

                "chunk_id":
                    "chunk-001",
            }
        ],

        "decision": {
            "evidence_sufficient": True,
            "allow_llm": True,
        },

        "degraded": False,

        "rag_request_id":
            "rag-graph-001",

        "error": None,
    }


def rejected_rag() -> dict[str, Any]:

    return {
        "success": True,
        "status": "rejected",

        "answer": (
            "当前知识库证据不足，"
            "无法可靠回答。"
        ),

        "sources": [],

        "decision": {
            "evidence_sufficient": False,
            "allow_llm": False,
        },

        "degraded": False,

        "rag_request_id":
            "rag-graph-002",

        "error": None,
    }


def error_rag() -> dict[str, Any]:

    return {
        "success": False,
        "status": "error",
        "answer": "",
        "sources": [],
        "rag_request_id": "",

        "error": {
            "code":
                "RAG_TIMEOUT",

            "message":
                "等待 Industrial RAG 响应超时",

            "retryable":
                False,
        },
    }


# ============================================================
# 可控服务替身
# ============================================================

class FakeServices:

    def __init__(
        self,
        *,
        status_result:
            dict[str, Any] | None = None,

        maintenance_result:
            dict[str, Any] | None = None,

        rag_result:
            dict[str, Any] | None = None,

        report_result:
            dict[str, Any] | None = None,
    ):

        self.calls = {
            "status": 0,
            "maintenance": 0,
            "rag": 0,
            "report": 0,
        }

        self.requests: list[
            dict[str, Any]
        ] = []

        self.status_result = (
            status_result
            if status_result is not None
            else {
                "success": True,

                "tool":
                    "get_device_status",

                "data": {
                    "device_id":
                        "DEVICE-001",

                    "status":
                        "warning",

                    "temperature":
                        82.5,

                    "vibration":
                        4.8,

                    "running":
                        True,
                },

                "error": None,
            }
        )

        self.maintenance_result = (
            maintenance_result
            if maintenance_result is not None
            else {
                "success": True,

                "tool":
                    "query_maintenance_history",

                "data": {
                    "device_id":
                        "DEVICE-001",

                    "record_count":
                        1,

                    "records": [
                        {
                            "date":
                                "2026-08-20",

                            "action":
                                "清理风道",
                        }
                    ],

                    "interpretation":
                        "Maintenance records were found.",
                },

                "error": None,
            }
        )

        self.rag_result = (
            rag_result
            if rag_result is not None
            else answered_rag(
                "建议检查并清理冷却风道。"
            )
        )

        self.report_result = (
            report_result
            if report_result is not None
            else {
                "success": True,

                "tool":
                    "create_diagnostic_report",

                "data": {
                    "report_id":
                        "REPORT-GRAPH-DEMO",
                },

                "error": None,
            }
        )

    def get_status(
        self,
        device_id: str,
    ) -> dict[str, Any]:

        self.calls["status"] += 1

        result = copy.deepcopy(
            self.status_result
        )

        if (
            result.get("success")
            and isinstance(
                result.get("data"),
                dict,
            )
        ):
            result["data"][
                "device_id"
            ] = device_id

        return result

    def get_maintenance(
        self,
        device_id: str,
    ) -> dict[str, Any]:

        self.calls[
            "maintenance"
        ] += 1

        result = copy.deepcopy(
            self.maintenance_result
        )

        if (
            result.get("success")
            and isinstance(
                result.get("data"),
                dict,
            )
        ):
            result["data"][
                "device_id"
            ] = device_id

        return result

    def retrieve(
        self,
        question,
        knowledge_base_id,
        device_model,
        **kwargs,
    ) -> dict[str, Any]:

        self.calls["rag"] += 1

        self.requests.append(
            {
                "type":
                    "rag",

                "question":
                    question,

                "knowledge_base_id":
                    knowledge_base_id,

                "device_model":
                    device_model,

                **kwargs,
            }
        )

        return copy.deepcopy(
            self.rag_result
        )

    def create_report(
        self,
        **kwargs,
    ) -> dict[str, Any]:

        self.calls["report"] += 1

        self.requests.append(
            {
                "type":
                    "report",

                "arguments":
                    kwargs,
            }
        )

        result = copy.deepcopy(
            self.report_result
        )

        if (
            result.get("success")
            and isinstance(
                result.get("data"),
                dict,
            )
        ):
            result["data"].update(
                {
                    "device_id":
                        kwargs["device_id"],

                    "severity":
                        kwargs["severity"],

                    "summary":
                        kwargs["summary"],
                }
            )

        return result


class FakeRAGClient:

    def __init__(
        self,
        services: FakeServices,
    ):

        self.services = services

    def retrieve(
        self,
        *args,
        **kwargs,
    ) -> dict[str, Any]:

        return self.services.retrieve(
            *args,
            **kwargs,
        )


# ============================================================
# 临时替换节点依赖
# ============================================================

@contextmanager
def patched_services(
    services: FakeServices,
):

    original_status = (
        nodes.get_device_status
    )

    original_maintenance = (
        nodes.query_maintenance_history
    )

    original_rag_client = (
        nodes.rag_client
    )

    original_report = (
        nodes.create_diagnostic_report
    )

    nodes.get_device_status = (
        services.get_status
    )

    nodes.query_maintenance_history = (
        services.get_maintenance
    )

    nodes.rag_client = FakeRAGClient(
        services
    )

    nodes.create_diagnostic_report = (
        services.create_report
    )

    try:
        yield

    finally:

        nodes.get_device_status = (
            original_status
        )

        nodes.query_maintenance_history = (
            original_maintenance
        )

        nodes.rag_client = (
            original_rag_client
        )

        nodes.create_diagnostic_report = (
            original_report
        )


# ============================================================
# Graph 调用
# ============================================================

def invoke_graph(
    state: dict[str, Any],
    services: FakeServices,
) -> dict[str, Any]:

    show(
        "Initial State",
        state,
    )

    with patched_services(
        services
    ):

        result = industrial_graph.invoke(
            state
        )

    show(
        "Service Calls",
        services.calls,
    )

    show(
        "Service Requests",
        services.requests,
    )

    show(
        "Execution Trace",
        result.get(
            "execution_trace",
            [],
        ),
    )

    show_final_result(
        result
    )

    return result


def diagnosis_state(
    *,
    create_report: bool = False,
) -> dict[str, Any]:

    return create_initial_state(
        user_query=(
            "诊断 DEVICE-001 当前异常，"
            "并结合维修历史判断原因"
        ),

        device_model="G120C",

        knowledge_base_id="kb_demo",

        create_report=create_report,
    )


# ============================================================
# 测试场景
# ============================================================

def case_missing_information() -> None:

    services = FakeServices()

    state = create_initial_state(
        user_query="检查设备状态"
    )

    result = invoke_graph(
        state,
        services,
    )

    check(
        result["status"]
        == "needs_input",
        "缺少设备编号时进入 needs_input",
    )

    check(
        result["next_action"]
        == "ask_user",
        "流程停在 ask_user",
    )

    check(
        "设备编号"
        in result["final_answer"],
        "最终回答要求补充设备编号",
    )

    check(
        sum(
            services.calls.values()
        ) == 0,
        "缺少信息时不调用任何服务",
    )


def case_device_status_path() -> None:

    services = FakeServices()

    state = create_initial_state(
        user_query=(
            "检查 DEVICE-001 "
            "当前运行状态"
        )
    )

    result = invoke_graph(
        state,
        services,
    )

    check(
        result["intent"]
        == "device_status",
        "正确识别设备状态查询",
    )

    check(
        services.calls == {
            "status": 1,
            "maintenance": 0,
            "rag": 0,
            "report": 0,
        },
        "设备状态查询只调用状态工具",
    )

    check(
        result["status"]
        == "completed",
        "状态查询正常完成",
    )

    check(
        "82.5"
        in result["final_answer"],
        "最终回答包含设备温度",
    )


def case_maintenance_path() -> None:

    services = FakeServices()

    state = create_initial_state(
        user_query=(
            "查询 DEVICE-001 "
            "的维修历史"
        )
    )

    result = invoke_graph(
        state,
        services,
    )

    check(
        result["intent"]
        == "maintenance_history",
        "正确识别维修历史查询",
    )

    check(
        services.calls == {
            "status": 0,
            "maintenance": 1,
            "rag": 0,
            "report": 0,
        },
        "维修查询只调用维修工具",
    )

    check(
        "清理风道"
        in result["final_answer"],
        "最终回答包含维修记录",
    )


def case_knowledge_answered() -> None:

    services = FakeServices(
        rag_result=answered_rag(
            "F3002 表示直流母线过压。"
        )
    )

    state = create_initial_state(
        user_query=(
            "G120C 的 F3002 "
            "故障码是什么含义？"
        ),

        device_model="G120C",

        knowledge_base_id="kb_demo",
    )

    result = invoke_graph(
        state,
        services,
    )

    check(
        result["intent"]
        == "knowledge_query",
        "正确识别知识库查询",
    )

    check(
        services.calls == {
            "status": 0,
            "maintenance": 0,
            "rag": 1,
            "report": 0,
        },
        "知识查询只调用 RAG",
    )

    check(
        result["rag_status"]
        == "answered",
        "RAG answered 正确进入最终回答",
    )

    check(
        "G120C 手册"
        in result["final_answer"],
        "最终回答包含证据来源",
    )


def case_knowledge_rejected() -> None:

    services = FakeServices(
        rag_result=rejected_rag()
    )

    state = create_initial_state(
        user_query=(
            "G120C 手册中不存在的"
            "问题如何处理？"
        ),

        device_model="G120C",

        knowledge_base_id="kb_demo",
    )

    result = invoke_graph(
        state,
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        result["rag_status"]
        == "rejected",
        "保留 RAG rejected 状态",
    )

    check(
        result["status"]
        == "insufficient_evidence",
        "证据不足时安全拒答",
    )

    check(
        "diagnosis_node"
        not in trace,
        "拒答后不进入诊断节点",
    )


def case_diagnosis_low_risk() -> None:

    services = FakeServices(
        rag_result=answered_rag(
            "建议检查并清理冷却风道。"
        )
    )

    result = invoke_graph(
        diagnosis_state(),
        services,
    )

    expected_trace = [
        "intent_node",
        "check_information_node",
        "device_status_node",
        "maintenance_history_node",
        "retrieve_node",
        "diagnosis_node",
        "risk_check_node",
        "final_answer_node",
    ]

    check(
        result["intent"]
        == "diagnosis",
        "正确识别综合诊断",
    )

    check(
        services.calls == {
            "status": 1,
            "maintenance": 1,
            "rag": 1,
            "report": 0,
        },
        "综合诊断调用三类只读能力",
    )

    check(
        get_trace_nodes(result)
        == expected_trace,
        "完整诊断执行顺序正确",
    )

    check(
        result["status"]
        == "completed",
        "低风险诊断正常完成",
    )


def case_diagnosis_high_risk() -> None:

    services = FakeServices(
        rag_result=answered_rag(
            "建议立即停机并断电，"
            "由工程师现场检查。"
        )
    )

    result = invoke_graph(
        diagnosis_state(),
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        result["risk_level"]
        == "high",
        "危险操作建议识别为 high",
    )

    check(
        result["status"]
        == "human_review_required",
        "高风险进入人工复核状态",
    )

    check(
        result["next_action"]
        == "awaiting_human_review",
        "流程等待授权人员处理",
    )

    check(
        "human_review_node"
        in trace,
        "实际执行人工复核节点",
    )


def case_optional_report() -> None:

    services = FakeServices(
        rag_result=answered_rag(
            "建议检查并清理冷却风道。"
        )
    )

    result = invoke_graph(
        diagnosis_state(
            create_report=True
        ),
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        services.calls["report"]
        == 1,
        "create_report=true 时生成一次报告",
    )

    check(
        result["report"]["report_id"]
        == "REPORT-GRAPH-DEMO",
        "报告写入最终状态",
    )

    check(
        "report_node"
        in trace,
        "执行轨迹包含报告节点",
    )

    check(
        result["status"]
        == "completed",
        "报告生成后正常完成",
    )


def case_tool_failure_short_circuit() -> None:

    services = FakeServices(
        status_result={
            "success": False,

            "tool":
                "get_device_status",

            "data":
                None,

            "error": {
                "code":
                    "DEVICE_NOT_FOUND",

                "message":
                    "设备不存在",
            },
        }
    )

    result = invoke_graph(
        diagnosis_state(),
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        result["status"]
        == "failed",
        "工具失败保留 failed 状态",
    )

    check(
        services.calls == {
            "status": 1,
            "maintenance": 0,
            "rag": 0,
            "report": 0,
        },
        "设备状态失败后立即短路",
    )

    check(
        "maintenance_history_node"
        not in trace,
        "失败后不执行维修历史节点",
    )

    check(
        "设备不存在"
        in result["final_answer"],
        "最终回答解释失败原因",
    )


def case_rag_error_short_circuit() -> None:

    services = FakeServices(
        rag_result=error_rag()
    )

    state = create_initial_state(
        user_query=(
            "G120C 报警码"
            "含义是什么？"
        ),

        device_model="G120C",

        knowledge_base_id="kb_demo",
    )

    result = invoke_graph(
        state,
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        result["status"]
        == "failed",
        "RAG 技术错误保留 failed 状态",
    )

    check(
        result["errors"][-1]["code"]
        == "RAG_TIMEOUT",
        "RAG 错误码贯穿完整 Graph",
    )

    check(
        "diagnosis_node"
        not in trace,
        "RAG 错误后不进入诊断节点",
    )


def case_critical_report_then_review() -> None:

    services = FakeServices(
        rag_result=answered_rag(
            "现场出现冒烟和触电风险，"
            "需要立即停机。"
        )
    )

    result = invoke_graph(
        diagnosis_state(
            create_report=True
        ),
        services,
    )

    trace = get_trace_nodes(
        result
    )

    check(
        result["risk_level"]
        == "critical",
        "人身或火灾风险识别为 critical",
    )

    check(
        services.calls["report"]
        == 1,
        "critical 场景仍生成报告",
    )

    check(
        trace.index("report_node")
        <
        trace.index("human_review_node"),
        "先生成报告再进入人工复核",
    )

    check(
        result["status"]
        == "human_review_required",
        "最终保持人工复核状态",
    )


# ============================================================
# 测试入口
# ============================================================

CASES = [
    (
        "G01",
        "缺失信息在调用服务前停止",
        case_missing_information,
    ),

    (
        "G02",
        "设备状态最短路径",
        case_device_status_path,
    ),

    (
        "G03",
        "维修历史最短路径",
        case_maintenance_path,
    ),

    (
        "G04",
        "知识查询 answered 路径",
        case_knowledge_answered,
    ),

    (
        "G05",
        "知识查询 rejected 证据门禁",
        case_knowledge_rejected,
    ),

    (
        "G06",
        "完整低风险诊断路径",
        case_diagnosis_low_risk,
    ),

    (
        "G07",
        "高风险进入人工复核",
        case_diagnosis_high_risk,
    ),

    (
        "G08",
        "低风险诊断生成报告",
        case_optional_report,
    ),

    (
        "G09",
        "Tool 失败后立即短路",
        case_tool_failure_short_circuit,
    ),

    (
        "G10",
        "RAG 技术错误后立即短路",
        case_rag_error_short_circuit,
    ),

    (
        "G11",
        "critical 报告后进入人工复核",
        case_critical_report_then_review,
    ),
]


def run_case(
    case_id: str,
    title: str,
    body: Callable[[], None],
) -> None:

    global passed
    global failed

    print(
        f"\n{SEPARATOR}\n"
        f"{case_id} | {title}\n"
        f"{SEPARATOR}"
    )

    try:

        body()

    except Exception as exc:

        failed += 1

        print(
            "\n[RESULT] FAIL: "
            f"{type(exc).__name__}: {exc}"
        )

    else:

        passed += 1

        print(
            "\n[RESULT] PASS"
        )


def main() -> None:

    print(
        "Industrial Fault Diagnosis Agent "
        "| Gate 5.3 LangGraph Workflow"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    for (
        case_id,
        title,
        body,
    ) in CASES:

        run_case(
            case_id,
            title,
            body,
        )

    print(
        f"\n{SEPARATOR}"
    )

    print(
        f"SUMMARY | "
        f"PASS={passed} | "
        f"FAIL={failed} | "
        f"TOTAL={passed + failed}"
    )

    print(
        SEPARATOR
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()