"""
agent/validation.py

Day22 Stage 2

Tool Call Validation Layer

职责：
1. 验证 JSON 是否能够解析；
2. 验证 type；
3. 验证 tool_name；
4. 验证 arguments 是否为 object；
5. 根据 JSON Schema 验证参数；
6. 执行业务级参数验证；
7. 验证全部通过后，才允许执行 Tool。

核心原则：

LLM 只能提出 Tool Call 请求。

LLM 本身没有直接执行 Python 函数的权限。
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from agent.registry import TOOL_REGISTRY
from agent.tool_schemas import TOOL_SCHEMAS


# ============================================================
# Error Codes
# ============================================================

INVALID_JSON = "INVALID_JSON"
INVALID_TYPE = "INVALID_TYPE"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
VALIDATION_FAILED = "VALIDATION_FAILED"


# ============================================================
# Agent Output Types
# ============================================================

ALLOWED_OUTPUT_TYPES = {
    "tool_call",
    "final_answer",
}


# ============================================================
# Business Validation Rules
# ============================================================

# 与 Stage 1 的设备编号形式保持一致：
#
# DEVICE-001
# DEVICE-102
# DEVICE-999
#
# 不允许：
#
# abc
# DEVICE-
# DEVICE-1
# ../../DEVICE-001
# DEVICE-001; rm -rf /
#
DEVICE_ID_PATTERN = re.compile(r"^DEVICE-\d{3}$")


# ============================================================
# Envelope Schema
# ============================================================

TOOL_CALL_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "const": "tool_call",
        },
        "tool_name": {
            "type": "string",
            "minLength": 1,
        },
        "arguments": {
            "type": "object",
        },
    },
    "required": [
        "type",
        "tool_name",
        "arguments",
    ],
    "additionalProperties": False,
}


FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "const": "final_answer",
        },
        "answer": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": [
        "type",
        "answer",
    ],
    "additionalProperties": False,
}


# ============================================================
# Validation Result
# ============================================================

def _validation_success(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    生成统一验证成功结果。
    """
    return {
        "valid": True,
        "error_code": None,
        "message": None,
        "payload": payload,
    }


def _validation_failure(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """
    生成统一验证失败结果。
    """
    return {
        "valid": False,
        "error_code": error_code,
        "message": message,
        "payload": None,
    }


# ============================================================
# JSON Parsing
# ============================================================

def parse_json_output(
    raw_output: str,
) -> dict[str, Any]:
    """
    将 LLM 原始 JSON 字符串解析成 Python dict。

    Raises:
        json.JSONDecodeError
        ValueError
    """

    parsed = json.loads(raw_output)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Agent output must be a JSON object."
        )

    return parsed


# ============================================================
# Business Validation
# ============================================================

def validate_device_id(device_id: str) -> bool:
    """
    验证 device_id 是否符合业务规则。

    合法：
        DEVICE-001
        DEVICE-102
        DEVICE-999

    非法：
        abc
        ""
        DEVICE-1
        ../../DEVICE-001
        DEVICE-001; rm -rf /
    """

    if not isinstance(device_id, str):
        return False

    return DEVICE_ID_PATTERN.fullmatch(device_id) is not None


def _validate_non_blank_string(
    value: Any,
) -> bool:
    """
    验证字符串不是空字符串或纯空白。
    """
    return isinstance(value, str) and bool(value.strip())


def _validate_string_list(
    value: Any,
) -> bool:
    """
    验证：
    - 必须是 list；
    - 至少一个元素；
    - 所有元素必须是非空字符串。
    """

    if not isinstance(value, list):
        return False

    if not value:
        return False

    return all(
        _validate_non_blank_string(item)
        for item in value
    )


def validate_business_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Tool 参数通过 JSON Schema 后，
    再进行业务级验证。

    Schema 负责：
        数据结构是否合法。

    Business Validation 负责：
        数据的业务值是否合法。
    """

    # --------------------------------------------------------
    # 所有当前 Tool 都包含 device_id
    # --------------------------------------------------------

    device_id = arguments.get("device_id")

    if not validate_device_id(device_id):
        return _validation_failure(
            VALIDATION_FAILED,
            (
                "Invalid device_id. "
                "Expected format: DEVICE-XXX "
                "(for example DEVICE-001)."
            ),
        )

    # --------------------------------------------------------
    # create_diagnostic_report 的额外业务校验
    # --------------------------------------------------------

    if tool_name == "create_diagnostic_report":

        summary = arguments.get("summary")
        findings = arguments.get("findings")
        recommendations = arguments.get(
            "recommendations"
        )

        if not _validate_non_blank_string(summary):
            return _validation_failure(
                VALIDATION_FAILED,
                "summary must not be blank.",
            )

        if not _validate_string_list(findings):
            return _validation_failure(
                VALIDATION_FAILED,
                (
                    "findings must contain at least "
                    "one non-empty string."
                ),
            )

        if not _validate_string_list(
            recommendations
        ):
            return _validation_failure(
                VALIDATION_FAILED,
                (
                    "recommendations must contain at "
                    "least one non-empty string."
                ),
            )

    return {
        "valid": True,
        "error_code": None,
        "message": None,
        "payload": arguments,
    }


# ============================================================
# Tool Call Validation
# ============================================================

def validate_tool_call(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    验证一个 Tool Call。

    该函数关注 Tool Call 本身：

        tool_name
        arguments

    如果 payload 中存在 type，
    则 type 必须是 tool_call。

    Validation 顺序：

        tool_name
            ↓
        Registry
            ↓
        arguments
            ↓
        Schema
            ↓
        Business Validation

    注意：
    这里不执行任何 Tool。
    """

    if not isinstance(payload, dict):
        return _validation_failure(
            VALIDATION_FAILED,
            "Tool call payload must be an object.",
        )

    # --------------------------------------------------------
    # 如果显式提供 type，则必须是 tool_call
    # --------------------------------------------------------

    if "type" in payload:
        if payload["type"] != "tool_call":
            return _validation_failure(
                INVALID_TYPE,
                (
                    "validate_tool_call only accepts "
                    "type='tool_call'."
                ),
            )

    # --------------------------------------------------------
    # 1. tool_name
    # --------------------------------------------------------

    tool_name = payload.get("tool_name")

    if (
        not isinstance(tool_name, str)
        or not tool_name.strip()
    ):
        return _validation_failure(
            VALIDATION_FAILED,
            "tool_name must be a non-empty string.",
        )

    # --------------------------------------------------------
    # 2. Registry 白名单
    #
    # 故意放在 arguments 前检查。
    #
    # 因此：
    #
    # {"tool_name": "delete_database"}
    #
    # 可以直接得到 UNKNOWN_TOOL。
    # --------------------------------------------------------

    if tool_name not in TOOL_REGISTRY:
        return _validation_failure(
            UNKNOWN_TOOL,
            f"Unknown tool: {tool_name}",
        )

    if tool_name not in TOOL_SCHEMAS:
        return _validation_failure(
            UNKNOWN_TOOL,
            (
                f"Tool '{tool_name}' has no "
                "registered schema."
            ),
        )

    # --------------------------------------------------------
    # 3. arguments 必须存在并且是 object
    # --------------------------------------------------------

    arguments = payload.get("arguments")

    if not isinstance(arguments, dict):
        return _validation_failure(
            INVALID_ARGUMENTS,
            "arguments must be a JSON object.",
        )

    # --------------------------------------------------------
    # 4. JSON Schema Validation
    # --------------------------------------------------------

    parameter_schema = TOOL_SCHEMAS[
        tool_name
    ]["parameters"]

    try:
        jsonschema_validate(
            instance=arguments,
            schema=parameter_schema,
        )

    except ValidationError as exc:
        return _validation_failure(
            VALIDATION_FAILED,
            f"Schema validation failed: {exc.message}",
        )

    # --------------------------------------------------------
    # 5. Business Validation
    # --------------------------------------------------------

    business_result = validate_business_arguments(
        tool_name=tool_name,
        arguments=arguments,
    )

    if not business_result["valid"]:
        return business_result

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    return _validation_success(payload)


# ============================================================
# Final Answer Validation
# ============================================================

def validate_final_answer(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    验证 final_answer 类型输出。
    """

    try:
        jsonschema_validate(
            instance=payload,
            schema=FINAL_ANSWER_SCHEMA,
        )

    except ValidationError as exc:
        return _validation_failure(
            VALIDATION_FAILED,
            (
                "Final answer validation failed: "
                f"{exc.message}"
            ),
        )

    answer = payload["answer"]

    if not answer.strip():
        return _validation_failure(
            VALIDATION_FAILED,
            "answer must not be blank.",
        )

    return _validation_success(payload)


# ============================================================
# Agent Output Validation
# ============================================================

def validate_agent_output(
    raw_output: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Stage 2 的主要入口。

    支持：

        JSON string

    或：

        Python dict

    完整流程：

        JSON Parse
            ↓
        type
            ↓
        tool_call / final_answer
            ↓
        对应 Validator
    """

    # --------------------------------------------------------
    # 1. JSON Parsing
    # --------------------------------------------------------

    if isinstance(raw_output, str):

        try:
            payload = parse_json_output(raw_output)

        except (
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            return _validation_failure(
                INVALID_JSON,
                f"Invalid JSON: {exc}",
            )

    elif isinstance(raw_output, dict):

        payload = raw_output.copy()

    else:
        return _validation_failure(
            INVALID_JSON,
            (
                "Agent output must be a JSON string "
                "or Python dict."
            ),
        )

    # --------------------------------------------------------
    # 2. type
    # --------------------------------------------------------

    output_type = payload.get("type")

    if output_type not in ALLOWED_OUTPUT_TYPES:
        return _validation_failure(
            INVALID_TYPE,
            (
                "type must be one of: "
                "tool_call, final_answer."
            ),
        )

    # --------------------------------------------------------
    # 3. Tool Call
    # --------------------------------------------------------

    if output_type == "tool_call":

        # 首先验证外层 Envelope。
        try:
            jsonschema_validate(
                instance=payload,
                schema=TOOL_CALL_ENVELOPE_SCHEMA,
            )

        except ValidationError as exc:

            # 如果 tool_name 已经明确是未知工具，
            # 优先返回 UNKNOWN_TOOL。
            tool_name = payload.get("tool_name")

            if (
                isinstance(tool_name, str)
                and tool_name
                and tool_name not in TOOL_REGISTRY
            ):
                return _validation_failure(
                    UNKNOWN_TOOL,
                    f"Unknown tool: {tool_name}",
                )

            return _validation_failure(
                VALIDATION_FAILED,
                (
                    "Tool call envelope validation "
                    f"failed: {exc.message}"
                ),
            )

        return validate_tool_call(payload)

    # --------------------------------------------------------
    # 4. Final Answer
    # --------------------------------------------------------

    return validate_final_answer(payload)


# ============================================================
# Validated Execution
# ============================================================

def execute_tool_call(
    payload: str | dict[str, Any],
) -> dict[str, Any]:
    """
    安全执行 Tool Call。

    只有验证全部通过后，
    才从 TOOL_REGISTRY 中取得真实 Python Tool。

    注意：
    这不是 Agent Loop。

    Stage 2 只是证明：

        Validation
            ↓
        Registry
            ↓
        Tool

    这条链路安全可控。
    """

    validation_result = validate_agent_output(
        payload
    )

    if not validation_result["valid"]:

        tool_name = None

        if isinstance(payload, dict):
            tool_name = payload.get("tool_name")

        return {
            "success": False,
            "tool": tool_name,
            "data": None,
            "error": {
                "code":
                    validation_result["error_code"],
                "message":
                    validation_result["message"],
            },
        }

    validated_payload = validation_result[
        "payload"
    ]

    # execute_tool_call 只负责执行 tool_call。
    if validated_payload["type"] != "tool_call":
        return {
            "success": False,
            "tool": None,
            "data": None,
            "error": {
                "code": INVALID_TYPE,
                "message": (
                    "execute_tool_call requires "
                    "type='tool_call'."
                ),
            },
        }

    tool_name = validated_payload["tool_name"]
    arguments = validated_payload["arguments"]

    # 此时 tool_name 已经过 Registry Validation。
    tool = TOOL_REGISTRY[tool_name]

    # 只有这里才真正执行 Python Tool。
    try:

        return tool(**arguments)


    except Exception as exc:

        return {
            "success": False,
            "tool": tool_name,
            "data": None,
            "error": {
                "code": "TOOL_EXECUTION_ERROR",
                "message": str(exc)
            }
        }