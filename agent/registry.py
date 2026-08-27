"""
agent/registry.py

Day22 Stage 2
维护允许 Agent 调用的 Python Tool 白名单。

职责：
1. 明确允许调用哪些 Tool；
2. 将 tool_name 映射到真实 Python 函数；
3. 禁止通过 globals() 等动态方式任意寻找函数。

安全原则：

LLM 给出的 tool_name 不是可信输入。

只有出现在 TOOL_REGISTRY 中的函数，
才允许进入后续执行流程。
"""

from collections.abc import Callable
from typing import Any, Final

from agent.tools import (
    create_diagnostic_report,
    get_device_status,
    query_maintenance_history,
)


ToolCallable = Callable[..., dict[str, Any]]


# ============================================================
# Tool 白名单
# ============================================================

TOOL_REGISTRY: Final[dict[str, ToolCallable]] = {
    "get_device_status": get_device_status,
    "query_maintenance_history": query_maintenance_history,
    "create_diagnostic_report": create_diagnostic_report,
}


def is_registered_tool(tool_name: str) -> bool:
    """
    判断工具是否存在于白名单。
    """
    return tool_name in TOOL_REGISTRY


def get_registered_tool(tool_name: str) -> ToolCallable | None:
    """
    根据名称获取真实 Python Tool。

    不存在时返回 None。

    注意：
    本函数本身不负责参数验证。
    调用方应该先经过 Validation。
    """
    return TOOL_REGISTRY.get(tool_name)


def list_registered_tools() -> list[str]:
    """
    返回当前所有允许调用的工具名称。
    """
    return list(TOOL_REGISTRY.keys())