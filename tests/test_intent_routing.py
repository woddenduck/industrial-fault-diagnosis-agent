"""
Intent Recognition 与 Conditional Routing 验收。

直接运行：

    python tests/test_intent_routing.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


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


from agent.nodes import (
    check_information_node,
    intent_node,
)
from agent.routing import (
    route_by_intent,
)
from agent.state import (
    create_initial_state,
)


KB_ID = "kb_ab50652fe3a4"


def pretty(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def check_equal(
    actual: Any,
    expected: Any,
    description: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{description}\n"
            f"期望：{expected!r}\n"
            f"实际：{actual!r}"
        )

    print(
        f"[CHECK PASS] {description}: "
        f"{actual!r}"
    )


def execute_case(
    *,
    query: str,
    expected_intent: str,
    expected_route: str,
    expected_missing: list[str],
    device_id: str = "",
    device_model: str = "",
    knowledge_base_id: str = "",
    expected_device_id: str | None = None,
) -> None:
    state = create_initial_state(
        user_query=query,
        device_id=device_id,
        device_model=device_model,
        knowledge_base_id=(
            knowledge_base_id
        ),
    )

    print("\n[Initial State]")
    print(pretty(state))

    intent_update = intent_node(
        state
    )

    print("\n[Intent Update]")
    print(pretty(intent_update))

    state.update(intent_update)

    information_update = (
        check_information_node(
            state
        )
    )

    print("\n[Information Update]")
    print(pretty(information_update))

    state.update(
        information_update
    )

    route = route_by_intent(
        state
    )

    print("\n[Routing Result]")
    print(route)

    check_equal(
        state["intent"],
        expected_intent,
        "意图识别正确",
    )

    check_equal(
        state["missing_fields"],
        expected_missing,
        "缺失字段判断正确",
    )

    check_equal(
        route,
        expected_route,
        "条件路由正确",
    )

    if expected_device_id is not None:
        check_equal(
            state["device_id"],
            expected_device_id,
            "设备编号处理正确",
        )


def main() -> None:
    cases = [
        {
            "id": "I01",
            "title": "从问题中提取设备编号并查询状态",
            "args": {
                "query": (
                    "检查 DEVICE-001 当前运行情况"
                ),
                "expected_intent":
                    "device_status",
                "expected_route":
                    "device_status",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-001",
            },
        },
        {
            "id": "I02",
            "title": "状态查询缺少设备编号",
            "args": {
                "query": "检查设备状态",
                "expected_intent":
                    "device_status",
                "expected_route":
                    "ask_user",
                "expected_missing": [
                    "device_id"
                ],
                "expected_device_id": "",
            },
        },
        {
            "id": "I03",
            "title": "维修历史查询",
            "args": {
                "query": (
                    "查询 DEVICE-002 的维修历史"
                ),
                "expected_intent":
                    "maintenance_history",
                "expected_route":
                    "maintenance_history",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-002",
            },
        },
        {
            "id": "I04",
            "title": "无设备编号的知识库查询",
            "args": {
                "query": (
                    "变频器温度太高时"
                    "应该检查哪些方面？"
                ),
                "knowledge_base_id": KB_ID,
                "expected_intent":
                    "knowledge_query",
                "expected_route":
                    "knowledge_query",
                "expected_missing": [],
            },
        },
        {
            "id": "I05",
            "title": "完整综合诊断",
            "args": {
                "query": (
                    "结合维修记录分析 "
                    "DEVICE-002 温度异常原因"
                ),
                "knowledge_base_id": KB_ID,
                "expected_intent":
                    "diagnosis",
                "expected_route":
                    "diagnosis",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-002",
            },
        },
        {
            "id": "I06",
            "title": "知识查询缺少KB ID",
            "args": {
                "query": "F30021怎么处理？",
                "expected_intent":
                    "knowledge_query",
                "expected_route":
                    "ask_user",
                "expected_missing": [
                    "knowledge_base_id"
                ],
            },
        },
        {
            "id": "I07",
            "title": "显式设备编号优先",
            "args": {
                "query": (
                    "检查 DEVICE-001 当前状态"
                ),
                "device_id": "DEVICE-002",
                "expected_intent":
                    "device_status",
                "expected_route":
                    "device_status",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-002",
            },
        },
        {
            "id": "I08",
            "title": "状态与维修历史组合查询",
            "args": {
                "query": (
                    "查询 DEVICE-001 当前运行情况"
                    "和维修历史"
                ),
                "knowledge_base_id": KB_ID,
                "expected_intent":
                    "diagnosis",
                "expected_route":
                    "diagnosis",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-001",
            },
        },
        {
            "id": "I09",
            "title": "无法识别的普通问候",
            "args": {
                "query": "你好",
                "expected_intent":
                    "unknown",
                "expected_route":
                    "ask_user",
                "expected_missing": [
                    "intent"
                ],
            },
        },
        {
            "id": "I10",
            "title": "小写设备编号规范化",
            "args": {
                "query": (
                    "检查 device-003 当前运行状态"
                ),
                "expected_intent":
                    "device_status",
                "expected_route":
                    "device_status",
                "expected_missing": [],
                "expected_device_id":
                    "DEVICE-003",
            },
        },
    ]

    passed = 0
    failed = 0

    for case in cases:
        print("\n")
        print("=" * 80)
        print(
            f"{case['id']} | "
            f"{case['title']}"
        )
        print("=" * 80)

        try:
            execute_case(
                **case["args"]
            )

        except Exception as exc:
            failed += 1

            print("\n[RESULT] FAIL")
            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            traceback.print_exc()

        else:
            passed += 1
            print("\n[RESULT] PASS")

    print("\n")
    print("=" * 80)
    print("Intent Routing Acceptance Summary")
    print("=" * 80)
    print(f"Total : {len(cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)

    if failed:
        print("FINAL RESULT: FAIL")
        raise SystemExit(1)

    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    main()