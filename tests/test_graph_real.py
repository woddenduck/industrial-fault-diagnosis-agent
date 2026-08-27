"""
关卡 5.4：真实 Industrial Agent Graph 端到端验收。

运行前要求：

1. Industrial RAG 已启动；
2. Gateway、LLM、Embedding、Reranker 服务正常；
3. data/device_status.json 存在 DEVICE-001；
4. data/maintenance_history.json 可读取；
5. 环境变量 RAG_KB_ID 已设置。

运行方式：

    export RAG_KB_ID=你的知识库ID
    python tests/test_graph_real.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

from pathlib import Path
from typing import Any


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


from agent.graph import (
    industrial_graph,
)

from agent.state import (
    create_initial_state,
)


SEPARATOR = "=" * 96


# ============================================================
# 环境配置
# ============================================================

KNOWLEDGE_BASE_ID = (
    os.getenv(
        "RAG_KB_ID",
        "",
    )
    .strip()
)

DEVICE_ID = (
    os.getenv(
        "AGENT_TEST_DEVICE_ID",
        "DEVICE-001",
    )
    .strip()
    .upper()
)

DEVICE_MODEL = (
    os.getenv(
        "AGENT_TEST_DEVICE_MODEL",
        "G120C",
    )
    .strip()
    .upper()
)

CREATE_REPORT = (
    os.getenv(
        "AGENT_TEST_CREATE_REPORT",
        "1",
    )
    == "1"
)

USER_QUERY = (
    os.getenv(
        "AGENT_TEST_QUERY",
        (
            f"请综合诊断 {DEVICE_ID} 当前温度异常，"
            f"结合维修历史和 {DEVICE_MODEL} 手册，"
            "分析可能原因并给出安全排查建议。"
        ),
    )
    .strip()
)


# ============================================================
# 输出函数
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


def contains_in_order(
    actual: list[str],
    expected: list[str],
) -> bool:
    """
    检查 expected 是否按照顺序出现在 actual 中。

    允许 actual 中包含额外节点，
    但不允许核心节点顺序错误。
    """

    current_index = 0

    for node_name in actual:

        if (
            current_index
            < len(expected)
            and node_name
            == expected[current_index]
        ):
            current_index += 1

    return (
        current_index
        == len(expected)
    )


# ============================================================
# 验收器
# ============================================================

class Acceptance:

    def __init__(self):

        self.passed = 0
        self.failed = 0

    def check(
        self,
        condition: bool,
        message: str,
        actual: Any = None,
    ) -> None:

        if condition:

            self.passed += 1

            print(
                f"[CHECK PASS] {message}"
            )

            return

        self.failed += 1

        print(
            f"[CHECK FAIL] {message}"
        )

        if actual is not None:

            show(
                "Actual Value",
                actual,
            )

    def summary(self) -> None:

        total = (
            self.passed
            + self.failed
        )

        print(
            f"\n{SEPARATOR}"
        )

        print(
            "REAL GRAPH SUMMARY | "
            f"PASS={self.passed} | "
            f"FAIL={self.failed} | "
            f"TOTAL={total}"
        )

        print(
            SEPARATOR
        )

        if self.failed:
            raise SystemExit(1)


# ============================================================
# 主测试
# ============================================================

def main() -> None:

    print(
        "Industrial Fault Diagnosis Agent "
        "| Gate 5.4 Real Graph"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    if not KNOWLEDGE_BASE_ID:

        print(
            "\n[CONFIG ERROR]"
        )

        print(
            "缺少环境变量 RAG_KB_ID。"
        )

        print(
            "\n请先执行："
        )

        print(
            "export RAG_KB_ID=你的真实知识库ID"
        )

        raise SystemExit(2)

    initial_state = create_initial_state(
        user_query=USER_QUERY,

        device_id=DEVICE_ID,

        device_model=DEVICE_MODEL,

        knowledge_base_id=(
            KNOWLEDGE_BASE_ID
        ),

        history=[],

        history_summary=None,

        create_report=CREATE_REPORT,
    )

    show(
        "Real Test Configuration",
        {
            "knowledge_base_id":
                KNOWLEDGE_BASE_ID,

            "device_id":
                DEVICE_ID,

            "device_model":
                DEVICE_MODEL,

            "create_report":
                CREATE_REPORT,

            "user_query":
                USER_QUERY,
        },
    )

    show(
        "Initial State",
        initial_state,
    )

    started_at = time.perf_counter()

    try:

        result = industrial_graph.invoke(
            initial_state
        )

    except Exception as exc:

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        print(
            "\n[GRAPH EXECUTION ERROR]"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"Elapsed: "
            f"{elapsed_seconds:.3f}s"
        )

        traceback.print_exc()

        raise SystemExit(1)

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    trace = result.get(
        "execution_trace",
        [],
    )

    trace_nodes = get_trace_nodes(
        result
    )

    show(
        "Execution Trace",
        trace,
    )

    show(
        "Device Status",
        result.get(
            "device_status",
            {},
        ),
    )

    show(
        "Maintenance History",
        result.get(
            "maintenance_history",
            {},
        ),
    )

    show(
        "RAG Result",
        {
            "rag_status":
                result.get("rag_status"),

            "rag_request_id":
                result.get(
                    "rag_request_id"
                ),

            "rag_decision":
                result.get(
                    "rag_decision"
                ),

            "degraded":
                result.get("degraded"),

            "sources":
                result.get("sources"),
        },
    )

    show(
        "Diagnosis Result",
        {
            "diagnosis":
                result.get("diagnosis"),

            "risk_level":
                result.get("risk_level"),

            "human_review_required":
                result.get(
                    "human_review_required"
                ),

            "report":
                result.get("report"),
        },
    )

    show(
        "Final Result",
        {
            "request_id":
                result.get("request_id"),

            "intent":
                result.get("intent"),

            "status":
                result.get("status"),

            "next_action":
                result.get(
                    "next_action"
                ),

            "final_answer":
                result.get(
                    "final_answer"
                ),

            "error":
                result.get("error"),

            "errors":
                result.get("errors"),

            "elapsed_seconds":
                round(
                    elapsed_seconds,
                    3,
                ),
        },
    )

    acceptance = Acceptance()

    print(
        f"\n{SEPARATOR}"
    )

    print(
        "REAL GRAPH CHECKS"
    )

    print(
        SEPARATOR
    )

    acceptance.check(
        result.get("intent")
        == "diagnosis",
        "识别为综合诊断意图",
        result.get("intent"),
    )

    acceptance.check(
        result.get("device_id")
        == DEVICE_ID,
        "设备编号在 Graph 中保持一致",
        result.get("device_id"),
    )

    acceptance.check(
        bool(
            result.get(
                "device_status"
            )
        ),
        "真实设备状态工具返回数据",
        result.get("device_status"),
    )

    acceptance.check(
        isinstance(
            result.get(
                "maintenance_history"
            ),
            dict,
        ),
        "真实维修历史工具返回结构化数据",
        result.get(
            "maintenance_history"
        ),
    )

    acceptance.check(
        result.get("rag_status")
        == "answered",
        "真实 Industrial RAG 返回 answered",
        result.get("rag_status"),
    )

    acceptance.check(
        bool(
            result.get(
                "rag_request_id"
            )
        ),
        "保留真实 RAG request_id",
        result.get(
            "rag_request_id"
        ),
    )

    acceptance.check(
        len(
            result.get(
                "sources",
                [],
            )
        )
        > 0,
        "真实 RAG 返回证据来源",
        result.get("sources"),
    )

    acceptance.check(
        bool(
            str(
                result.get(
                    "diagnosis",
                    "",
                )
            ).strip()
        ),
        "生成综合诊断结果",
        result.get("diagnosis"),
    )

    acceptance.check(
        result.get(
            "risk_level"
        )
        in {
            "low",
            "medium",
            "high",
            "critical",
        },
        "完成工业风险分级",
        result.get("risk_level"),
    )


    diagnosis_text = str(
        result.get(
            "diagnosis",
            "",
        )
    )

    hazard_keywords = (
        "火灾",
        "起火",
        "冒烟",
        "爆炸",
        "触电",
        "电弧",
    )

    hazard_detected = any(
        keyword in diagnosis_text
        for keyword in hazard_keywords
    )

    acceptance.check(
        (
                not hazard_detected
                or result.get("risk_level")
                in {
                    "high",
                    "critical",
                }
        ),
        "诊断包含严重危险语义时不得判定为低中风险",
        {
            "hazard_detected":
                hazard_detected,

            "risk_level":
                result.get("risk_level"),
        },
    )

    acceptance.check(
        (
                not hazard_detected
                or result.get(
            "human_review_required"
        )
                is True
        ),
        "诊断包含严重危险语义时必须进入人工复核",
        {
            "hazard_detected":
                hazard_detected,

            "human_review_required":
                result.get(
                    "human_review_required"
                ),
        },
    )


    if CREATE_REPORT:

        acceptance.check(
            isinstance(
                result.get("report"),
                dict,
            )
            and bool(
                result.get(
                    "report",
                    {},
                ).get(
                    "report_id"
                )
            ),
            "生成真实诊断报告",
            result.get("report"),
        )

    else:

        acceptance.check(
            result.get("report")
            is None,
            "未请求生成诊断报告",
            result.get("report"),
        )

    acceptance.check(
        result.get("status")
        in {
            "completed",
            "human_review_required",
        },
        "Graph 以正常结果或人工复核状态结束",
        result.get("status"),
    )

    acceptance.check(
        bool(
            str(
                result.get(
                    "final_answer",
                    "",
                )
            ).strip()
        ),
        "生成最终用户回答",
        result.get("final_answer"),
    )

    acceptance.check(
        not result.get("error"),
        "Graph 没有技术错误",
        result.get("error"),
    )

    core_trace = [
        "intent_node",
        "check_information_node",
        "device_status_node",
        "maintenance_history_node",
        "retrieve_node",
        "diagnosis_node",
        "risk_check_node",
    ]

    acceptance.check(
        contains_in_order(
            trace_nodes,
            core_trace,
        ),
        "核心诊断节点按照正确顺序执行",
        trace_nodes,
    )

    if CREATE_REPORT:

        acceptance.check(
            "report_node"
            in trace_nodes,
            "执行轨迹包含报告节点",
            trace_nodes,
        )

    else:

        acceptance.check(
            "report_node"
            not in trace_nodes,
            "未请求报告时不执行报告节点",
            trace_nodes,
        )

    terminal_nodes = {
        "final_answer_node",
        "human_review_node",
    }

    acceptance.check(
        bool(
            terminal_nodes.intersection(
                trace_nodes
            )
        ),
        "Graph 执行最终回答或人工复核节点",
        trace_nodes,
    )

    acceptance.summary()


if __name__ == "__main__":
    main()