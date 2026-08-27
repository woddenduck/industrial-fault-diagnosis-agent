"""Industrial Fault Diagnosis Agent 的统一错误定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
VALIDATION_FAILED = "VALIDATION_FAILED"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
TOOL_INVALID_RESPONSE = "TOOL_INVALID_RESPONSE"
RAG_EXECUTION_ERROR = "RAG_EXECUTION_ERROR"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NODE_EXECUTION_ERROR = "NODE_EXECUTION_ERROR"
REPORT_GENERATION_ERROR = "REPORT_GENERATION_ERROR"
MAX_TOOL_CALLS_EXCEEDED = "MAX_TOOL_CALLS_EXCEEDED"


@dataclass
class AgentError(Exception):
    """可同时用于异常抛出和 AgentState 错误记录的基础异常。"""

    code: str
    message: str
    details: dict[str, Any] | None = None
    retryable: bool = False

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "retryable": self.retryable,
            },
        }

    def to_state_item(self, *, node: str) -> dict[str, Any]:
        """转换为 AgentState.errors 使用的扁平结构。"""

        item: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "node": node,
            "retryable": self.retryable,
        }
        if self.details:
            item["details"] = self.details
        return item


class JSONParseAgentError(AgentError):
    def __init__(self, message: str, details=None):
        super().__init__(JSON_PARSE_ERROR, message, details)


class ValidationAgentError(AgentError):
    def __init__(self, message: str, details=None):
        super().__init__(VALIDATION_FAILED, message, details)


class UnknownToolError(AgentError):
    def __init__(self, tool_name: str):
        super().__init__(
            UNKNOWN_TOOL,
            f"Unknown tool: {tool_name}",
            {"tool_name": tool_name},
        )


class ToolExecutionError(AgentError):
    def __init__(
        self,
        message: str,
        details=None,
        *,
        retryable: bool = False,
    ):
        super().__init__(
            TOOL_EXECUTION_ERROR,
            message,
            details,
            retryable,
        )


class MaxToolCallsExceededError(AgentError):
    def __init__(self, max_calls: int):
        super().__init__(
            MAX_TOOL_CALLS_EXCEEDED,
            f"Maximum tool calls exceeded: {max_calls}",
            {"max_calls": max_calls},
        )
