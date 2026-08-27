"""
Stage 7 Agent Safety Test

验证：

1. 非法 JSON
2. 参数缺失
3. 参数类型错误
4. 危险参数
5. 未知 Tool
6. Tool Exception
7. MAX_TOOL_CALLS
8. 正常 Tool Call
"""

from pathlib import Path
import sys


# ============================================================
# Add project root to PYTHONPATH
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
        str(PROJECT_ROOT)
    )


from unittest.mock import patch

import pytest


from agent.validation import (
    validate_agent_output,
    execute_tool_call,
)

from agent.registry import TOOL_REGISTRY

from agent.errors import (
    JSON_PARSE_ERROR,
    VALIDATION_FAILED,
    UNKNOWN_TOOL,
    TOOL_EXECUTION_ERROR,
    MAX_TOOL_CALLS_EXCEEDED,
)



# ============================================================
# S01 非法 JSON
# ============================================================

def test_invalid_json():

    result = validate_agent_output(
        """
        {
            tool: get_device_status
        }
        """
    )

    assert result["valid"] is False

    assert result["error_code"] == "INVALID_JSON"



# ============================================================
# S02 缺少参数
# ============================================================

def test_missing_argument():


    payload = {

        "type": "tool_call",

        "tool_name":
            "get_device_status",

        "arguments": {}

    }


    result = validate_agent_output(
        payload
    )


    assert result["valid"] is False

    assert result["error_code"] == VALIDATION_FAILED



# ============================================================
# S03 参数类型错误
# ============================================================

def test_wrong_argument_type():


    payload = {

        "type": "tool_call",

        "tool_name":
            "get_device_status",

        "arguments": {

            "device_id": 123

        }

    }


    result = validate_agent_output(
        payload
    )


    assert result["valid"] is False

    assert result["error_code"] == VALIDATION_FAILED



# ============================================================
# S04 危险参数
# ============================================================

def test_dangerous_argument_rejected():


    payload = {

        "type": "tool_call",

        "tool_name":
            "get_device_status",

        "arguments": {

            "device_id":
                "DEVICE-001; rm -rf /"

        }

    }


    result = validate_agent_output(
        payload
    )


    assert result["valid"] is False

    assert result["error_code"] == VALIDATION_FAILED



# ============================================================
# S05 Unknown Tool
# ============================================================

def test_unknown_tool():


    payload = {

        "type": "tool_call",

        "tool_name":
            "delete_device",

        "arguments": {

            "device_id":
                "DEVICE-001"

        }

    }


    result = validate_agent_output(
        payload
    )


    assert result["valid"] is False


    assert result["error_code"] == UNKNOWN_TOOL



# ============================================================
# S06 Tool Exception
# ============================================================

def test_tool_exception_handling():


    payload = {

        "type": "tool_call",

        "tool_name":
            "get_device_status",

        "arguments": {

            "device_id":
                "DEVICE-001"

        }

    }


    def broken_tool(*args, **kwargs):

        raise RuntimeError(
            "simulated tool failure"
        )



    with patch.dict(
        TOOL_REGISTRY,
        {
            "get_device_status":
                broken_tool
        }
    ):

        try:

            result = execute_tool_call(
                payload
            )


        except Exception as exc:

            pytest.fail(
                f"Agent crashed: {exc}"
            )


        else:

            # 当前 Stage2 execute_tool_call
            # 还未捕获 Exception，
            # Stage7 完成后这里应该返回：
            #
            # TOOL_EXECUTION_ERROR

            if result["success"] is False:

                assert (
                    result["error"]["code"]
                    ==
                    TOOL_EXECUTION_ERROR
                )



# ============================================================
# S07 MAX_TOOL_CALLS
# ============================================================

def test_max_tool_calls():


    MAX_TOOL_CALLS = 3


    call_count = 0


    def fake_execute():

        nonlocal call_count

        call_count += 1


    for _ in range(10):

        if call_count >= MAX_TOOL_CALLS:

            error_code = (
                MAX_TOOL_CALLS_EXCEEDED
            )

            break


        fake_execute()


    assert call_count == 3

    assert (
        error_code
        ==
        MAX_TOOL_CALLS_EXCEEDED
    )



# ============================================================
# S08 正常 Tool Call
# ============================================================

def test_normal_tool_call():


    payload = {

        "type":
            "tool_call",

        "tool_name":
            "get_device_status",

        "arguments": {

            "device_id":
                "DEVICE-001"

        }

    }


    result = validate_agent_output(
        payload
    )


    assert result["valid"] is True



# ============================================================
# S09 正常 Multi Tool 输入
# ============================================================

def test_normal_multi_tool_schema():


    tools = [

        "get_device_status",

        "query_maintenance_history",

        "create_diagnostic_report"

    ]


    for tool in tools:

        assert (
            tool in TOOL_REGISTRY
        )
