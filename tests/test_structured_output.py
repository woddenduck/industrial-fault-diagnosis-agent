"""
Day22 Stage 3 - Structured Output Acceptance Test

验收目标：

S01
    DEVICE-001现在是什么状态？
        ↓
    type == tool_call
        ↓
    tool_name == get_device_status
        ↓
    arguments.device_id == DEVICE-001

S02
    查询 DEVICE-002 的维护记录
        ↓
    query_maintenance_history

S03
    普通问候
        ↓
    final_answer

S04
    JSON Parse 必须是真正 json.loads()
    不允许字符串包含判断

S05
    非法 Structured Output 必须拒绝

S06
    Tool arguments 必须经过 Schema Validation

最重要：
    整个 Stage 3 不执行任何 Python Tool。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict


# ============================================================
# Make project root importable
# ============================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from agent.structured_output import (  # noqa: E402
    DecisionValidationError,
    JSONParseError,
    call_llm,
    generate_decision,
    get_registered_tool_names,
    parse_structured_output,
    validate_decision,
)


# ============================================================
# Console Helpers
# ============================================================


LINE = "=" * 78


def print_section(
    title: str,
) -> None:

    print()
    print(LINE)
    print(title)
    print(LINE)


def print_decision(
    data: Dict[str, Any],
) -> None:

    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# S01
# Real LLM - Device Status
# ============================================================


def test_s01_device_status_tool_call():
    """
    Stage 3 最核心验收。

    DEVICE-001现在是什么状态？

    必须解析结构字段。

    绝对不允许：

        assert "get_device_status" in raw_response
    """

    print_section(
        "S01 | Device Status → Tool Call"
    )

    question = (
        "DEVICE-001现在是什么状态？"
    )

    print(
        f"User       : {question}"
    )

    result = generate_decision(
        question
    )

    print(
        "Parsed Decision:"
    )

    print_decision(
        result
    )

    assert (
        result["type"]
        == "tool_call"
    )

    assert (
        result["tool_name"]
        == "get_device_status"
    )

    assert (
        result["arguments"]["device_id"]
        == "DEVICE-001"
    )

    print()
    print(
        "type       :",
        result["type"],
    )

    print(
        "tool_name  :",
        result["tool_name"],
    )

    print(
        "arguments  :",
        result["arguments"],
    )

    print(
        "PASS       : True"
    )


# ============================================================
# S02
# Real LLM - Maintenance History
# ============================================================


def test_s02_maintenance_history_tool_call():

    print_section(
        "S02 | Maintenance History → Tool Call"
    )

    question = (
        "请查询DEVICE-002的维护记录。"
    )

    print(
        f"User       : {question}"
    )

    result = generate_decision(
        question
    )

    print(
        "Parsed Decision:"
    )

    print_decision(
        result
    )

    assert (
        result["type"]
        == "tool_call"
    )

    assert (
        result["tool_name"]
        == "query_maintenance_history"
    )

    assert (
        result["arguments"]["device_id"]
        == "DEVICE-002"
    )

    print()
    print(
        "type       :",
        result["type"],
    )

    print(
        "tool_name  :",
        result["tool_name"],
    )

    print(
        "arguments  :",
        result["arguments"],
    )

    print(
        "PASS       : True"
    )


# ============================================================
# S03
# Real LLM - Final Answer
# ============================================================


def test_s03_greeting_final_answer():

    print_section(
        "S03 | Greeting → Final Answer"
    )

    question = "你好。"

    print(
        f"User       : {question}"
    )

    result = generate_decision(
        question
    )

    print(
        "Parsed Decision:"
    )

    print_decision(
        result
    )

    assert (
        result["type"]
        == "final_answer"
    )

    assert isinstance(
        result["content"],
        str,
    )

    assert (
        result["content"].strip()
    )

    # 不测试固定自然语言：
    #
    # assert result["content"] == "你好..."
    #
    # 因为 Stage 3 要验证结构，
    # 不是要求模型逐字输出固定文本。

    print()
    print(
        "type       :",
        result["type"],
    )

    print(
        "content    :",
        result["content"],
    )

    print(
        "PASS       : True"
    )


# ============================================================
# S04
# JSON Parse
# ============================================================


def test_s04_json_parse():

    print_section(
        "S04 | Strict JSON Parsing"
    )

    raw_response = """
{
  "type": "tool_call",
  "tool_name": "get_device_status",
  "arguments": {
    "device_id": "DEVICE-001"
  }
}
""".strip()

    result = parse_structured_output(
        raw_response
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["type"]
        == "tool_call"
    )

    assert (
        result["tool_name"]
        == "get_device_status"
    )

    assert (
        result["arguments"]["device_id"]
        == "DEVICE-001"
    )

    print(
        "json.loads : PASS"
    )

    print(
        "type       :",
        result["type"],
    )

    print(
        "tool_name  :",
        result["tool_name"],
    )

    print(
        "device_id  :",
        result["arguments"]["device_id"],
    )

    print(
        "PASS       : True"
    )


# ============================================================
# S05
# Invalid JSON / Invalid Decision
# ============================================================


def test_s05_invalid_structured_output_rejected():

    print_section(
        "S05 | Invalid Structured Output Rejected"
    )

    invalid_json = (
        "我需要调用 get_device_status "
        "查询 DEVICE-001"
    )

    json_rejected = False

    try:

        parse_structured_output(
            invalid_json
        )

    except JSONParseError:

        json_rejected = True

    assert json_rejected is True

    print(
        "Natural language rejected : PASS"
    )

    # --------------------------------------------------------

    invalid_type = {
        "type": "unknown",
        "content": "hello",
    }

    type_rejected = False

    try:

        validate_decision(
            invalid_type
        )

    except DecisionValidationError:

        type_rejected = True

    assert type_rejected is True

    print(
        "Unknown type rejected      : PASS"
    )

    # --------------------------------------------------------

    unknown_tool = {
        "type": "tool_call",
        "tool_name": "delete_all_devices",
        "arguments": {},
    }

    tool_rejected = False

    try:

        validate_decision(
            unknown_tool
        )

    except DecisionValidationError:

        tool_rejected = True

    assert tool_rejected is True

    print(
        "Unknown tool rejected      : PASS"
    )

    print(
        "PASS                       : True"
    )


# ============================================================
# S06
# Tool Argument Validation
# ============================================================


def test_s06_tool_argument_validation():

    print_section(
        "S06 | Registry + Arguments Validation"
    )

    registered_tools = (
        get_registered_tool_names()
    )

    print(
        "Registered Tools:"
    )

    for tool_name in registered_tools:
        print(
            f"  - {tool_name}"
        )

    assert (
        "get_device_status"
        in registered_tools
    )

    valid_call = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
        },
    }

    result = validate_decision(
        valid_call
    )

    assert (
        result["type"]
        == "tool_call"
    )

    # --------------------------------------------------------
    # 缺少 device_id
    # --------------------------------------------------------

    invalid_call = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {},
    }

    rejected = False

    try:

        validate_decision(
            invalid_call
        )

    except DecisionValidationError as exc:

        rejected = True

        print()
        print(
            "Expected Validation Error:"
        )

        print(
            exc
        )

    assert rejected is True

    print()
    print(
        "Valid arguments   : PASS"
    )

    print(
        "Invalid arguments : REJECTED"
    )

    print(
        "PASS              : True"
    )


# ============================================================
# Extra Critical Test
# Tool must NOT be executed
# ============================================================


def test_stage3_only_returns_decision():

    print_section(
        "Stage Boundary | No Tool Execution"
    )

    # 这是纯 Structured Output Validation。
    #
    # 如果 Stage 3 正确，
    # validate_decision() 只会验证该结构，
    # 不会得到真正设备状态。

    decision = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
        },
    }

    result = validate_decision(
        decision
    )

    assert (
        result
        == decision
    )

    # Stage 3 不应该返回这些字段。
    assert (
        "tool_result"
        not in result
    )

    assert (
        "status"
        not in result
    )

    assert (
        "device_status"
        not in result
    )

    print(
        "Decision validated : True"
    )

    print(
        "Tool executed       : False"
    )

    print(
        "PASS                : True"
    )


# ============================================================
# Standalone Acceptance Runner
# ============================================================


def main():

    tests: list[
        tuple[str, Callable[[], None]]
    ] = [
        (
            "S01 Device Status",
            test_s01_device_status_tool_call,
        ),
        (
            "S02 Maintenance History",
            test_s02_maintenance_history_tool_call,
        ),
        (
            "S03 Final Answer",
            test_s03_greeting_final_answer,
        ),
        (
            "S04 JSON Parse",
            test_s04_json_parse,
        ),
        (
            "S05 Invalid Output",
            test_s05_invalid_structured_output_rejected,
        ),
        (
            "S06 Validation",
            test_s06_tool_argument_validation,
        ),
        (
            "Stage Boundary",
            test_stage3_only_returns_decision,
        ),
    ]

    print()
    print(LINE)
    print(
        "Day22 Stage 3 | Structured Output Acceptance"
    )
    print(LINE)

    passed = 0
    failed = 0

    failures = []

    for name, test_func in tests:

        try:

            test_func()

            passed += 1

        except Exception as exc:

            failed += 1

            failures.append(
                (
                    name,
                    exc,
                )
            )

            print()
            print(
                f"[FAIL] {name}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print(LINE)
    print(
        "Stage 3 Acceptance Result"
    )
    print(LINE)

    print(
        f"PASS  : {passed}"
    )

    print(
        f"FAIL  : {failed}"
    )

    print(
        f"TOTAL : {len(tests)}"
    )

    if failures:

        print()
        print(
            "Failed Cases:"
        )

        for name, exc in failures:

            print(
                f"  - {name}: "
                f"{type(exc).__name__}: {exc}"
            )

        print()
        print(
            "Stage 3 Result: FAILED"
        )

        raise SystemExit(1)

    print()
    print(
        "Stage 3 Result: ALL PASSED"
    )

    print()
    print(
        "Structured Output"
        " → JSON Parse"
        " → Registry"
        " → Validation"
        " → STOP"
    )


if __name__ == "__main__":
    main()