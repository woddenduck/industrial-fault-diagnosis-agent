"""
agent/tool_schemas.py

Day22 Stage 2
定义 Agent 中所有 Python Tool 的 Schema。

职责：
1. 描述工具名称；
2. 描述工具用途；
3. 描述 arguments 的 JSON Schema；
4. 不执行任何真实业务逻辑。

注意：
Schema 是 LLM / Agent Runtime 与 Python Tool 之间的数据契约，
不是 Tool 本身。
"""

from typing import Any, Final


# ============================================================
# Tool 1: get_device_status
# ============================================================

GET_DEVICE_STATUS_SCHEMA: Final[dict[str, Any]] = {
    "name": "get_device_status",
    "description": (
        "查询指定工业设备的当前运行状态。"
        "适用于获取设备是否存在、当前状态、运行参数和告警信息。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": (
                    "设备唯一标识，例如 DEVICE-001、DEVICE-102。"
                ),
            }
        },
        "required": ["device_id"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool 2: query_maintenance_history
# ============================================================

QUERY_MAINTENANCE_HISTORY_SCHEMA: Final[dict[str, Any]] = {
    "name": "query_maintenance_history",
    "description": (
        "查询指定工业设备的历史维护记录。"
        "用于了解设备过去的维护、维修、故障和处理情况。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": (
                    "设备唯一标识，例如 DEVICE-001、DEVICE-102。"
                ),
            }
        },
        "required": ["device_id"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool 3: create_diagnostic_report
# ============================================================

CREATE_DIAGNOSTIC_REPORT_SCHEMA: Final[dict[str, Any]] = {
    "name": "create_diagnostic_report",
    "description": (
        "根据已经获得的设备状态、诊断发现和处理建议，"
        "生成结构化设备诊断报告。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": (
                    "设备唯一标识，例如 DEVICE-001。"
                ),
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "description": "本次诊断的简要总结。",
            },
            "severity": {
                "type": "string",
                "enum": [
                    "low",
                    "medium",
                    "high",
                    "critical",
                ],
                "description": "诊断严重程度。",
            },
            "findings": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
                "description": "诊断过程中得到的主要发现。",
            },
            "recommendations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
                "description": "针对当前问题给出的处理建议。",
            },
        },
        "required": [
            "device_id",
            "summary",
            "severity",
            "findings",
            "recommendations",
        ],
        "additionalProperties": False,
    },
}


# ============================================================
# Schema Registry
# ============================================================

TOOL_SCHEMAS: Final[dict[str, dict[str, Any]]] = {
    GET_DEVICE_STATUS_SCHEMA["name"]: GET_DEVICE_STATUS_SCHEMA,
    QUERY_MAINTENANCE_HISTORY_SCHEMA["name"]:
        QUERY_MAINTENANCE_HISTORY_SCHEMA,
    CREATE_DIAGNOSTIC_REPORT_SCHEMA["name"]:
        CREATE_DIAGNOSTIC_REPORT_SCHEMA,
}


# 以后 Stage 3 向 LLM 暴露工具时，可以直接遍历这个列表。
TOOL_SCHEMA_LIST: Final[list[dict[str, Any]]] = [
    GET_DEVICE_STATUS_SCHEMA,
    QUERY_MAINTENANCE_HISTORY_SCHEMA,
    CREATE_DIAGNOSTIC_REPORT_SCHEMA,
]


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """
    根据工具名称获取对应 Schema。

    注意：
   这里只查询 Schema，不执行工具。
    """
    return TOOL_SCHEMAS.get(tool_name)