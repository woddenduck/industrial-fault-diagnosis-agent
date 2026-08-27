"""
关卡 5.1：Graph 条件路由直接运行验收。

运行方式：

    python tests/test_graph_routing.py

本测试不调用 Tool、RAG 或 LangGraph，
只验证 routing.py 中各条件路由函数。
"""

from __future__ import annotations

import json
import sys
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


from agent.routing import (
    route_after_device_status,
    route_after_maintenance_history,
    route_after_retrieve,
    route_after_diagnosis,
    route_after_risk_check,
    route_after_report,
)


SEPARATOR = "=" * 88

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
        raise AssertionError(message)

    print(
        f"[CHECK PASS] {message}"
    )


def run_case(
    case_id: str,
    title: str,
    router: Callable[[dict[str, Any]], str],
    state: dict[str, Any],
    expected: str,
) -> None:

    global passed
    global failed

    print(
        f"\n{SEPARATOR}\n"
        f"{case_id} | {title}\n"
        f"{SEPARATOR}"
    )

    show(
        "Router",
        router.__name__,
    )

    show(
        "Input State",
        state,
    )

    try:

        actual = router(
            state
        )

        show(
            "Routing Result",
            {
                "actual": actual,
                "expected": expected,
            },
        )

        check(
            actual == expected,
            (
                f"路由结果正确："
                f"{actual!r}"
            ),
        )

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


# ============================================================
# 测试数据
# ============================================================

CASES = [

    # --------------------------------------------------------
    # route_after_device_status
    # --------------------------------------------------------

    {
        "case_id": "R01",
        "title": "单独设备状态查询完成后生成回答",
        "router": route_after_device_status,
        "state": {
            "intent": "device_status",
            "status": "running",
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R02",
        "title": "综合诊断完成状态查询后继续查询维修历史",
        "router": route_after_device_status,
        "state": {
            "intent": "diagnosis",
            "status": "running",
        },
        "expected": "maintenance_history",
    },

    {
        "case_id": "R03",
        "title": "设备状态工具失败后立即收口",
        "router": route_after_device_status,
        "state": {
            "intent": "diagnosis",
            "status": "failed",
            "error": "DEVICE_NOT_FOUND",
        },
        "expected": "final_answer",
    },

    # --------------------------------------------------------
    # route_after_maintenance_history
    # --------------------------------------------------------

    {
        "case_id": "R04",
        "title": "单独维修历史查询完成后生成回答",
        "router": route_after_maintenance_history,
        "state": {
            "intent": "maintenance_history",
            "status": "running",
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R05",
        "title": "综合诊断完成维修查询后继续调用 RAG",
        "router": route_after_maintenance_history,
        "state": {
            "intent": "diagnosis",
            "status": "running",
        },
        "expected": "retrieve",
    },

    {
        "case_id": "R06",
        "title": "维修历史工具失败后立即收口",
        "router": route_after_maintenance_history,
        "state": {
            "intent": "diagnosis",
            "status": "failed",
            "error": "DATA_READ_ERROR",
        },
        "expected": "final_answer",
    },

    # --------------------------------------------------------
    # route_after_retrieve
    # --------------------------------------------------------

    {
        "case_id": "R07",
        "title": "普通知识查询 answered 后直接回答",
        "router": route_after_retrieve,
        "state": {
            "intent": "knowledge_query",
            "status": "running",
            "rag_status": "answered",
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R08",
        "title": "综合诊断 RAG answered 后进入诊断聚合",
        "router": route_after_retrieve,
        "state": {
            "intent": "diagnosis",
            "status": "running",
            "rag_status": "answered",
        },
        "expected": "diagnosis",
    },

    {
        "case_id": "R09",
        "title": "RAG rejected 后停止诊断",
        "router": route_after_retrieve,
        "state": {
            "intent": "diagnosis",
            "status": "insufficient_evidence",
            "rag_status": "rejected",
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R10",
        "title": "RAG 技术错误后停止诊断",
        "router": route_after_retrieve,
        "state": {
            "intent": "diagnosis",
            "status": "failed",
            "rag_status": "error",
            "error": "RAG_TIMEOUT",
        },
        "expected": "final_answer",
    },

    # --------------------------------------------------------
    # route_after_diagnosis
    # --------------------------------------------------------

    {
        "case_id": "R11",
        "title": "诊断聚合成功后进入风险判断",
        "router": route_after_diagnosis,
        "state": {
            "status": "running",
            "diagnosis": "建议检查冷却风道。",
        },
        "expected": "risk_check",
    },

    {
        "case_id": "R12",
        "title": "诊断节点失败后直接收口",
        "router": route_after_diagnosis,
        "state": {
            "status": "failed",
            "error": "诊断节点执行失败",
        },
        "expected": "final_answer",
    },

    # --------------------------------------------------------
    # route_after_risk_check
    # --------------------------------------------------------

    {
        "case_id": "R13",
        "title": "低风险且不生成报告时直接回答",
        "router": route_after_risk_check,
        "state": {
            "risk_level": "low",
            "create_report": False,
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R14",
        "title": "中风险且请求报告时进入报告节点",
        "router": route_after_risk_check,
        "state": {
            "risk_level": "medium",
            "create_report": True,
        },
        "expected": "report",
    },

    {
        "case_id": "R15",
        "title": "高风险且不生成报告时进入人工复核",
        "router": route_after_risk_check,
        "state": {
            "risk_level": "high",
            "create_report": False,
        },
        "expected": "human_review",
    },

    {
        "case_id": "R16",
        "title": "高风险且请求报告时先生成报告",
        "router": route_after_risk_check,
        "state": {
            "risk_level": "high",
            "create_report": True,
        },
        "expected": "report",
    },

    {
        "case_id": "R17",
        "title": "critical 且不生成报告时进入人工复核",
        "router": route_after_risk_check,
        "state": {
            "risk_level": "critical",
            "create_report": False,
        },
        "expected": "human_review",
    },

    # --------------------------------------------------------
    # route_after_report
    # --------------------------------------------------------

    {
        "case_id": "R18",
        "title": "普通风险报告生成成功后回答",
        "router": route_after_report,
        "state": {
            "status": "running",
            "risk_level": "low",
            "human_review_required": False,
            "report": {
                "report_id": "REPORT-001",
            },
        },
        "expected": "final_answer",
    },

    {
        "case_id": "R19",
        "title": "高风险报告生成成功后进入人工复核",
        "router": route_after_report,
        "state": {
            "status": "human_review_required",
            "risk_level": "high",
            "human_review_required": True,
            "report": {
                "report_id": "REPORT-002",
            },
        },
        "expected": "human_review",
    },

    {
        "case_id": "R20",
        "title": "报告生成失败后进入最终错误回答",
        "router": route_after_report,
        "state": {
            "status": "failed",
            "risk_level": "medium",
            "human_review_required": False,
            "error": "REPORT_GENERATION_ERROR",
        },
        "expected": "final_answer",
    },
]


# ============================================================
# 主程序
# ============================================================

def main() -> None:

    print(
        "Industrial Fault Diagnosis Agent "
        "| Gate 5.1 Graph Routing"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    for case in CASES:

        run_case(
            case_id=case["case_id"],
            title=case["title"],
            router=case["router"],
            state=case["state"],
            expected=case["expected"],
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