"""
tests/test_tool_validation.py

Day22 Stage 2
Tool Schema + Registry + Validation 验收。

运行：

    pytest -v tests/test_tool_validation.py

本测试不调用 LLM。
"""

from typing import Any

import pytest

import agent.validation as validation_module

from agent.registry import (
    TOOL_REGISTRY,
    get_registered_tool,
    is_registered_tool,
    list_registered_tools,
)

from agent.tool_schemas import TOOL_SCHEMAS

from agent.validation import (
    INVALID_ARGUMENTS,
    INVALID_JSON,
    INVALID_TYPE,
    UNKNOWN_TOOL,
    VALIDATION_FAILED,
    execute_tool_call,
    validate_agent_output,
    validate_device_id,
    validate_tool_call,
)


# ============================================================
# Helpers
# ============================================================

def make_device_status_call(
    device_id: Any = "DEVICE-001",
) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": device_id,
        },
    }


# ============================================================
# Registry
# ============================================================

def test_registry_contains_expected_tools():
    """
    Registry 必须只暴露明确允许的 Tool。
    """

    expected = {
        "get_device_status",
        "query_maintenance_history",
        "create_diagnostic_report",
    }

    assert set(TOOL_REGISTRY.keys()) == expected
    assert set(list_registered_tools()) == expected


def test_registry_and_schema_are_aligned():
    """
    每个 Registry Tool 都必须存在 Schema，
    每个 Schema 也必须存在真实 Tool。
    """

    assert set(TOOL_REGISTRY.keys()) == set(
        TOOL_SCHEMAS.keys()
    )


def test_get_registered_tool():
    """
    已注册 Tool 可以找到，
    未注册 Tool 找不到。
    """

    assert (
        get_registered_tool(
            "get_device_status"
        )
        is not None
    )

    assert (
        get_registered_tool(
            "delete_database"
        )
        is None
    )

    assert is_registered_tool(
        "get_device_status"
    )

    assert not is_registered_tool(
        "delete_database"
    )


# ============================================================
# V01 - Valid Tool Call
# ============================================================

def test_valid_device_status_call():
    payload = make_device_status_call(
        "DEVICE-001"
    )

    result = validate_agent_output(payload)

    assert result["valid"] is True
    assert result["error_code"] is None


# ============================================================
# V02 - Unknown Tool
# ============================================================

def test_unknown_tool():
    """
    Stage 2 强制验收：

        delete_database

    必须：
        UNKNOWN_TOOL
    """

    payload = {
        "type": "tool_call",
        "tool_name": "delete_database",
        "arguments": {},
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert result["error_code"] == UNKNOWN_TOOL


def test_unknown_tool_without_arguments():
    """
    对 Stage 2 文档中的最小攻击案例：

        {
            "tool_name": "delete_database"
        }

    直接验证 Tool Call 层时，
    应该优先得到 UNKNOWN_TOOL。
    """

    result = validate_tool_call(
        {
            "tool_name": "delete_database",
        }
    )

    assert result["valid"] is False
    assert result["error_code"] == UNKNOWN_TOOL


# ============================================================
# V03 - Wrong Argument Type
# ============================================================

def test_device_id_wrong_type():
    """
    Stage 2 强制验收：

        device_id = 123

    必须：
        VALIDATION_FAILED
    """

    payload = make_device_status_call(123)

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V04 - Missing Required Argument
# ============================================================

def test_missing_device_id():
    payload = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {},
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V05 - additionalProperties
# ============================================================

def test_additional_property_rejected():
    """
    Stage 2 强制验收：

        {
            "device_id": "DEVICE-001",
            "command": "shutdown"
        }

    必须：
        VALIDATION_FAILED
    """

    payload = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
            "command": "shutdown",
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V06 - Invalid Business ID
# ============================================================

@pytest.mark.parametrize(
    "device_id",
    [
        "",
        "abc",
        "DEVICE",
        "DEVICE-",
        "DEVICE-1",
        "device-001",
        "DEVICE-ABC",
    ],
)
def test_invalid_device_id(device_id):
    assert validate_device_id(device_id) is False

    payload = make_device_status_call(device_id)

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V07 - Path Traversal
# ============================================================

def test_path_traversal_device_id():
    payload = make_device_status_call(
        "../../DEVICE-001"
    )

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V08 - Command Injection
# ============================================================

@pytest.mark.parametrize(
    "device_id",
    [
        "DEVICE-001; rm -rf /",
        "DEVICE-001 && shutdown",
        "DEVICE-001 | cat /etc/passwd",
        "DEVICE-001\nshutdown",
        "DEVICE-001$(whoami)",
    ],
)
def test_command_injection_device_id(device_id):
    payload = make_device_status_call(device_id)

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# V09 - arguments must be object
# ============================================================

@pytest.mark.parametrize(
    "arguments",
    [
        "DEVICE-001",
        ["DEVICE-001"],
        123,
        True,
        None,
    ],
)
def test_arguments_must_be_object(arguments):
    payload = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": arguments,
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False

    # 外层 Envelope 或 arguments 专项验证
    # 都属于非法输入。
    assert result["error_code"] in {
        INVALID_ARGUMENTS,
        VALIDATION_FAILED,
    }


# ============================================================
# V10 - Invalid JSON
# ============================================================

def test_invalid_json():
    raw_output = """
    {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments":
    }
    """

    result = validate_agent_output(raw_output)

    assert result["valid"] is False
    assert result["error_code"] == INVALID_JSON


# ============================================================
# V11 - Invalid Type
# ============================================================

@pytest.mark.parametrize(
    "output_type",
    [
        "execute_shell",
        "run_python",
        "call_function",
        "",
        None,
    ],
)
def test_invalid_output_type(output_type):
    payload = {
        "type": output_type,
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert result["error_code"] == INVALID_TYPE


def test_missing_output_type():
    payload = {
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert result["error_code"] == INVALID_TYPE


# ============================================================
# V12 - Nonexistent But Valid Device ID
# ============================================================

def test_nonexistent_but_format_valid_device_id():
    """
    这是非常重要的职责边界测试。

    DEVICE-999 在格式上完全合法。

    Validation 不应该负责判断：
        DEVICE-999 是否真的存在。

    是否存在是 Tool / Business Data Layer 的职责。
    """

    payload = make_device_status_call(
        "DEVICE-999"
    )

    result = validate_agent_output(payload)

    assert result["valid"] is True
    assert result["error_code"] is None


# ============================================================
# Maintenance Tool
# ============================================================

def test_valid_maintenance_history_call():
    payload = {
        "type": "tool_call",
        "tool_name": "query_maintenance_history",
        "arguments": {
            "device_id": "DEVICE-002",
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is True


def test_maintenance_history_extra_argument():
    payload = {
        "type": "tool_call",
        "tool_name": "query_maintenance_history",
        "arguments": {
            "device_id": "DEVICE-002",
            "delete_history": True,
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# Diagnostic Report Tool
# ============================================================

def test_valid_diagnostic_report_call():
    payload = {
        "type": "tool_call",
        "tool_name": "create_diagnostic_report",
        "arguments": {
            "device_id": "DEVICE-004",
            "summary": (
                "设备存在严重温度异常。"
            ),
            "severity": "critical",
            "findings": [
                "设备当前状态为 critical",
                "温度明显高于正常范围",
            ],
            "recommendations": [
                "立即停止设备运行",
                "检查散热系统",
            ],
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is True


@pytest.mark.parametrize(
    "severity",
    [
        "dangerous",
        "urgent",
        "CRITICAL",
        "",
        123,
    ],
)
def test_invalid_report_severity(severity):
    payload = {
        "type": "tool_call",
        "tool_name": "create_diagnostic_report",
        "arguments": {
            "device_id": "DEVICE-004",
            "summary": "设备存在异常。",
            "severity": severity,
            "findings": [
                "温度异常",
            ],
            "recommendations": [
                "检查设备",
            ],
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


def test_report_missing_required_parameter():
    payload = {
        "type": "tool_call",
        "tool_name": "create_diagnostic_report",
        "arguments": {
            "device_id": "DEVICE-004",
            "severity": "critical",
            "findings": [
                "温度异常",
            ],
            "recommendations": [
                "立即检查",
            ],

            # summary 故意缺失
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


def test_report_blank_summary():
    payload = {
        "type": "tool_call",
        "tool_name": "create_diagnostic_report",
        "arguments": {
            "device_id": "DEVICE-004",
            "summary": "   ",
            "severity": "high",
            "findings": [
                "温度异常",
            ],
            "recommendations": [
                "检查散热系统",
            ],
        },
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# final_answer
# ============================================================

def test_valid_final_answer():
    payload = {
        "type": "final_answer",
        "answer": "设备当前运行正常。",
    }

    result = validate_agent_output(payload)

    assert result["valid"] is True
    assert result["error_code"] is None


def test_blank_final_answer():
    payload = {
        "type": "final_answer",
        "answer": "   ",
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


def test_final_answer_extra_property():
    payload = {
        "type": "final_answer",
        "answer": "设备正常。",
        "command": "shutdown",
    }

    result = validate_agent_output(payload)

    assert result["valid"] is False
    assert (
        result["error_code"]
        == VALIDATION_FAILED
    )


# ============================================================
# Critical Security Test
# ============================================================

def test_invalid_call_never_executes_tool(
    monkeypatch,
):
    """
    Stage 2 最重要的安全测试之一：

    Validation 失败时，
    Python Tool 绝对不能执行。
    """

    executed = {
        "called": False,
    }

    def fake_get_device_status(
        device_id: str,
    ) -> dict[str, Any]:

        executed["called"] = True

        return {
            "success": True,
            "tool": "get_device_status",
            "data": {
                "device_id": device_id,
            },
            "error": None,
        }

    monkeypatch.setitem(
        validation_module.TOOL_REGISTRY,
        "get_device_status",
        fake_get_device_status,
    )

    malicious_payload = {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001",
            "command": "shutdown",
        },
    }

    result = execute_tool_call(
        malicious_payload
    )

    assert result["success"] is False

    assert result["error"]["code"] == (
        VALIDATION_FAILED
    )

    # 最重要的断言
    assert executed["called"] is False


def test_valid_call_executes_after_validation(
    monkeypatch,
):
    """
    与上一测试相反：

    合法请求：
        Validation PASS
            ↓
        Registry
            ↓
        Tool Execution
    """

    executed = {
        "called": False,
        "device_id": None,
    }

    def fake_get_device_status(
        device_id: str,
    ) -> dict[str, Any]:

        executed["called"] = True
        executed["device_id"] = device_id

        return {
            "success": True,
            "tool": "get_device_status",
            "data": {
                "device_id": device_id,
                "status": "normal",
            },
            "error": None,
        }

    monkeypatch.setitem(
        validation_module.TOOL_REGISTRY,
        "get_device_status",
        fake_get_device_status,
    )

    payload = make_device_status_call(
        "DEVICE-001"
    )

    result = execute_tool_call(payload)

    assert result["success"] is True

    assert executed["called"] is True

    assert (
        executed["device_id"]
        == "DEVICE-001"
    )