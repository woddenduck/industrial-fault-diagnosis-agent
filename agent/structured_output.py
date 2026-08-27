"""
Day22 Stage 3 - Structured Output

职责：
1. 将 Stage 2 的 Tool Schema 提供给 LLM
2. 调用真实 LLM
3. 要求 LLM 只返回结构化 JSON
4. JSON Parse
5. 判断 final_answer / tool_call
6. 检查 tool_name 是否已经注册
7. 校验 arguments 是否符合 Tool Schema

非常重要：
    本模块绝不执行 Tool。

Stage 3 数据流：

User
  ↓
LLM
  ↓
Structured JSON
  ↓
json.loads()
  ↓
type
  ├── final_answer
  │      └── content
  │
  └── tool_call
         ├── tool_name
         └── arguments
                ↓
             Registry
                ↓
            Validation
                ↓
               STOP
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import httpx
from jsonschema import Draft202012Validator


# ============================================================
# Exception
# ============================================================


class StructuredOutputError(Exception):
    """Structured Output 处理失败。"""


class LLMCallError(StructuredOutputError):
    """LLM 请求失败。"""


class JSONParseError(StructuredOutputError):
    """LLM 返回内容不是合法 JSON。"""


class DecisionValidationError(StructuredOutputError):
    """LLM Structured Output 不符合数据契约。"""


# ============================================================
# LLM Configuration
# ============================================================


@dataclass(frozen=True)
class LLMConfig:
    """
    Stage 3 使用的最小 LLM 配置。

    默认兼容当前 Qwen3 + vLLM OpenAI-compatible API：

        http://127.0.0.1:6006/v1/chat/completions
    """

    base_url: str
    model: str
    api_key: str
    timeout: float
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv(
                "LLM_BASE_URL",
                "http://127.0.0.1:6006/v1",
            ).rstrip("/"),
            model=os.getenv(
                "LLM_MODEL",
                "Qwen/Qwen3-8B",
            ),
            api_key=os.getenv(
                "LLM_API_KEY",
                "EMPTY",
            ),
            timeout=float(
                os.getenv(
                    "LLM_TIMEOUT",
                    "60",
                )
            ),
            temperature=float(
                os.getenv(
                    "LLM_TEMPERATURE",
                    "0",
                )
            ),
        )


# ============================================================
# Stage 2 Tool Schema Adapter
# ============================================================


def _load_tool_schemas() -> List[Dict[str, Any]]:
    """
    从 Stage 2 的 agent.tool_schemas 加载 Tool Schema。

    推荐 Stage 2 中使用：

        TOOL_SCHEMAS = [
            {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        ]

    同时兼容：
        TOOL_SCHEMAS = {
            "tool_name": {...}
        }

    以及 OpenAI function schema：

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
    """

    try:
        from agent import tool_schemas
    except ImportError as exc:
        raise StructuredOutputError(
            "无法导入 agent.tool_schemas。"
            "请先确认 Stage 2 已完成。"
        ) from exc

    raw_schemas: Any = None

    # 推荐形式
    if hasattr(tool_schemas, "TOOL_SCHEMAS"):
        raw_schemas = getattr(tool_schemas, "TOOL_SCHEMAS")

    # 兼容其它可能命名
    elif hasattr(tool_schemas, "TOOLS"):
        raw_schemas = getattr(tool_schemas, "TOOLS")

    elif hasattr(tool_schemas, "get_tool_schemas"):
        raw_schemas = tool_schemas.get_tool_schemas()

    else:
        raise StructuredOutputError(
            "agent.tool_schemas 中没有找到 TOOL_SCHEMAS "
            "或 get_tool_schemas()。"
        )

    schemas: List[Dict[str, Any]] = []

    if isinstance(raw_schemas, Mapping):

        for name, schema in raw_schemas.items():

            if not isinstance(schema, Mapping):
                raise StructuredOutputError(
                    f"Tool Schema {name!r} 必须是 dict。"
                )

            item = dict(schema)

            # 如果 dict value 中没有 name，
            # 使用 key 作为 tool name
            if (
                "name" not in item
                and "function" not in item
            ):
                item["name"] = name

            schemas.append(item)

    elif isinstance(raw_schemas, list):

        for schema in raw_schemas:

            if not isinstance(schema, Mapping):
                raise StructuredOutputError(
                    "TOOL_SCHEMAS 中每个元素必须是 dict。"
                )

            schemas.append(dict(schema))

    else:
        raise StructuredOutputError(
            "TOOL_SCHEMAS 必须是 list 或 dict。"
        )

    if not schemas:
        raise StructuredOutputError(
            "当前没有任何 Tool Schema。"
        )

    return schemas


def _schema_name(schema: Mapping[str, Any]) -> str:
    """取得 Tool Schema 中的工具名。"""

    # Stage 2 推荐格式
    if isinstance(schema.get("name"), str):
        return schema["name"]

    # OpenAI function 格式
    function = schema.get("function")

    if isinstance(function, Mapping):
        name = function.get("name")

        if isinstance(name, str):
            return name

    raise StructuredOutputError(
        f"无法从 Tool Schema 中取得 name：{schema}"
    )


def _schema_parameters(
    schema: Mapping[str, Any],
) -> Dict[str, Any]:
    """取得 Tool 的 parameters JSON Schema。"""

    parameters = schema.get("parameters")

    if isinstance(parameters, Mapping):
        return dict(parameters)

    function = schema.get("function")

    if isinstance(function, Mapping):

        parameters = function.get("parameters")

        if isinstance(parameters, Mapping):
            return dict(parameters)

    raise StructuredOutputError(
        f"Tool {_schema_name(schema)!r} 没有合法 parameters。"
    )


def _get_tool_schema(
    tool_name: str,
) -> Dict[str, Any]:
    """根据 tool_name 找到 Tool Schema。"""

    for schema in _load_tool_schemas():

        if _schema_name(schema) == tool_name:
            return schema

    raise DecisionValidationError(
        f"未知 Tool：{tool_name}"
    )


# ============================================================
# Registry Adapter
# ============================================================


def get_registered_tool_names() -> List[str]:
    """
    从 Stage 2 registry 获取已经注册的工具名。

    注意：
        这里只检查 Registry。
        绝对不会调用 Registry 中的 Python Tool。
    """

    try:
        from agent import registry
    except ImportError as exc:
        raise StructuredOutputError(
            "无法导入 agent.registry。"
        ) from exc

    raw_registry: Any = None

    if hasattr(registry, "TOOL_REGISTRY"):
        raw_registry = getattr(
            registry,
            "TOOL_REGISTRY",
        )

    elif hasattr(registry, "REGISTRY"):
        raw_registry = getattr(
            registry,
            "REGISTRY",
        )

    elif hasattr(registry, "list_tools"):

        result = registry.list_tools()

        if isinstance(result, Mapping):
            return list(result.keys())

        if isinstance(result, (list, tuple, set)):
            return [
                str(item)
                for item in result
            ]

    if isinstance(raw_registry, Mapping):
        return list(raw_registry.keys())

    raise StructuredOutputError(
        "无法从 agent.registry 中读取 Tool Registry。\n"
        "推荐 Stage 2 定义：\n\n"
        "TOOL_REGISTRY = {\n"
        '    "get_device_status": get_device_status,\n'
        '    "query_maintenance_history": '
        "query_maintenance_history,\n"
        '    "create_diagnostic_report": '
        "create_diagnostic_report,\n"
        "}"
    )


# ============================================================
# Prompt
# ============================================================


def build_system_prompt() -> str:
    """
    构建 Stage 3 System Prompt。

    Tool Schema 直接来自 Stage 2。
    因此不在 Stage 3 重复定义 Tool 参数。
    """

    tool_schemas = _load_tool_schemas()

    schemas_text = json.dumps(
        tool_schemas,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是一个工业设备诊断 Agent 的决策模块。

你的任务不是执行工具。

你的任务只有一个：

根据用户请求决定：

1. 直接回答用户；
2. 请求 Python 程序调用某个 Tool。

你当前可以使用以下 Tool Schema：

{schemas_text}

你必须只输出一个 JSON Object。

禁止输出：
- Markdown
- ```json
- 解释文字
- 思考过程
- JSON 前后的额外文字


============================================================
情况一：需要调用 Tool
============================================================

严格返回：

{{
  "type": "tool_call",
  "tool_name": "<tool name>",
  "arguments": {{
    "<parameter>": "<value>"
  }}
}}

tool_name 必须来自已经提供的 Tool Schema。

arguments 必须严格满足该 Tool 的 parameters JSON Schema。

不能自行创造不存在的 Tool。

不能自行创造不存在的参数。


============================================================
情况二：不需要调用 Tool
============================================================

严格返回：

{{
  "type": "final_answer",
  "content": "<直接给用户的回答>"
}}


============================================================
重要规则
============================================================

如果用户只是：
- 打招呼
- 闲聊
- 不需要查询业务数据的问题

使用：

"type": "final_answer"


如果用户要求：
- 查询设备当前状态
- 查询维护记录
- 创建诊断报告

应该选择相应 Tool，并返回：

"type": "tool_call"


特别注意：

你只是产生 Tool Call 请求。

你不能假装 Tool 已经执行。

你不能编造 Tool Result。

你不能回答不存在的实时设备状态。

只返回 JSON。
""".strip()


# ============================================================
# LLM API
# ============================================================


def _build_chat_completions_url(
    base_url: str,
) -> str:
    """根据 LLM_BASE_URL 构造 OpenAI-compatible endpoint。"""

    base_url = base_url.rstrip("/")

    if base_url.endswith(
        "/v1/chat/completions"
    ):
        return base_url

    if base_url.endswith("/v1"):
        return (
            base_url
            + "/chat/completions"
        )

    return (
        base_url
        + "/v1/chat/completions"
    )


def call_llm(
    user_message: str,
    config: Optional[LLMConfig] = None,
) -> str:
    """
    调用真实 LLM。

    返回：
        LLM message.content 原始字符串。

    注意：
        此函数只负责获取模型输出。
        不负责 Tool Execution。
    """

    if not isinstance(user_message, str):
        raise TypeError(
            "user_message 必须是 str"
        )

    user_message = user_message.strip()

    if not user_message:
        raise ValueError(
            "user_message 不能为空"
        )

    config = config or LLMConfig.from_env()

    url = _build_chat_completions_url(
        config.base_url
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": (
            f"Bearer {config.api_key}"
        ),
    }

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": config.temperature,

        # OpenAI-compatible JSON mode。
        # 目的是尽量约束模型只输出 JSON Object。
        "response_format": {
            "type": "json_object",
        },

        # Qwen3 Stage 3 不需要 thinking 输出。
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    try:

        with httpx.Client(
            timeout=config.timeout,
        ) as client:

            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

    except httpx.TimeoutException as exc:
        raise LLMCallError(
            f"LLM 请求超时：{url}"
        ) from exc

    except httpx.RequestError as exc:
        raise LLMCallError(
            f"无法连接 LLM：{url}\n"
            f"{exc}"
        ) from exc

    if response.status_code != 200:

        raise LLMCallError(
            "LLM API 返回错误。\n"
            f"URL: {url}\n"
            f"HTTP: {response.status_code}\n"
            f"Body: {response.text[:1000]}"
        )

    try:
        body = response.json()

        content = (
            body["choices"][0]
            ["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:

        raise LLMCallError(
            "LLM API Response 格式异常：\n"
            f"{response.text[:1000]}"
        ) from exc

    if not isinstance(content, str):

        raise LLMCallError(
            "LLM message.content 不是字符串。"
        )

    content = content.strip()

    if not content:
        raise LLMCallError(
            "LLM 返回了空 content。"
        )

    return content


# ============================================================
# JSON Parse
# ============================================================


def parse_structured_output(
    raw_response: str,
) -> Dict[str, Any]:
    """
    将 LLM 原始输出解析成 Python dict。

    Stage 3 最关键的一步：

        LLM Response
              ↓
          json.loads()
              ↓
             dict

    不允许使用：

        if "get_device_status" in response

    这种字符串猜测方式。
    """

    if not isinstance(raw_response, str):
        raise JSONParseError(
            "LLM Response 必须是字符串。"
        )

    raw_response = raw_response.strip()

    if not raw_response:
        raise JSONParseError(
            "LLM Response 为空。"
        )

    # Stage 3 故意采用严格 JSON。
    #
    # 不自动删除：
    # ```json
    # ...
    # ```
    #
    # 因为如果模型输出 Markdown，
    # 就应该让验收失败，
    # 而不是偷偷替模型修复。
    try:

        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:

        raise JSONParseError(
            "LLM 没有返回合法的纯 JSON。\n"
            f"Raw Response:\n{raw_response}"
        ) from exc

    if not isinstance(data, dict):
        raise JSONParseError(
            "Structured Output 顶层必须是 JSON Object。"
        )

    return data


# ============================================================
# Structured Decision Validation
# ============================================================


def _validate_arguments(
    tool_name: str,
    arguments: Dict[str, Any],
) -> None:
    """
    使用 Stage 2 Tool Schema 校验 arguments。

    这里只做 Validation。
    不调用 Tool。
    """

    schema = _get_tool_schema(
        tool_name
    )

    parameter_schema = _schema_parameters(
        schema
    )

    validator = Draft202012Validator(
        parameter_schema
    )

    errors = sorted(
        validator.iter_errors(
            arguments
        ),
        key=lambda error: list(
            error.path
        ),
    )

    if errors:

        messages = []

        for error in errors:
            path = ".".join(
                str(item)
                for item in error.path
            )

            if not path:
                path = "<root>"

            messages.append(
                f"{path}: {error.message}"
            )

        raise DecisionValidationError(
            f"Tool {tool_name!r} arguments "
            "校验失败：\n"
            + "\n".join(messages)
        )


def validate_decision(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    校验完整的 LLM Structured Output。

    两种合法结果：

    1.
        {
            "type": "final_answer",
            "content": "..."
        }

    2.
        {
            "type": "tool_call",
            "tool_name": "...",
            "arguments": {...}
        }
    """

    if not isinstance(data, dict):
        raise DecisionValidationError(
            "Structured Output 必须是 dict。"
        )

    decision_type = data.get("type")

    # ========================================================
    # final_answer
    # ========================================================

    if decision_type == "final_answer":

        allowed_keys = {
            "type",
            "content",
        }

        extra_keys = (
            set(data.keys())
            - allowed_keys
        )

        if extra_keys:
            raise DecisionValidationError(
                "final_answer 包含不允许的字段："
                f"{sorted(extra_keys)}"
            )

        content = data.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise DecisionValidationError(
                "final_answer.content "
                "必须是非空字符串。"
            )

        return data

    # ========================================================
    # tool_call
    # ========================================================

    if decision_type == "tool_call":

        allowed_keys = {
            "type",
            "tool_name",
            "arguments",
        }

        extra_keys = (
            set(data.keys())
            - allowed_keys
        )

        if extra_keys:
            raise DecisionValidationError(
                "tool_call 包含不允许的字段："
                f"{sorted(extra_keys)}"
            )

        tool_name = data.get(
            "tool_name"
        )

        arguments = data.get(
            "arguments"
        )

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise DecisionValidationError(
                "tool_call.tool_name "
                "必须是非空字符串。"
            )

        if not isinstance(
            arguments,
            dict,
        ):
            raise DecisionValidationError(
                "tool_call.arguments "
                "必须是 JSON Object。"
            )

        # ----------------------------------------------------
        # Stage 2 Registry Check
        # ----------------------------------------------------

        registered_names = (
            get_registered_tool_names()
        )

        if (
            tool_name
            not in registered_names
        ):
            raise DecisionValidationError(
                f"Tool {tool_name!r} "
                "没有注册到 Registry。"
            )

        # ----------------------------------------------------
        # Stage 2 Schema Validation
        # ----------------------------------------------------

        _validate_arguments(
            tool_name=tool_name,
            arguments=arguments,
        )

        # ----------------------------------------------------
        # STOP
        #
        # 这里绝对不能：
        #
        # tool = registry.get_tool(...)
        # tool(**arguments)
        #
        # 那属于 Stage 4。
        # ----------------------------------------------------

        return data

    raise DecisionValidationError(
        "type 必须是："
        "'final_answer' 或 'tool_call'。\n"
        f"实际值：{decision_type!r}"
    )


# ============================================================
# Main Stage 3 Interface
# ============================================================


def generate_decision(
    user_message: str,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """
    Stage 3 对外核心接口。

    User Message
         ↓
        LLM
         ↓
      Raw JSON
         ↓
     json.loads()
         ↓
    Validate type
         ↓
    Registry Check
         ↓
    Argument Validation
         ↓
        STOP

    返回示例：

    {
        "type": "tool_call",
        "tool_name": "get_device_status",
        "arguments": {
            "device_id": "DEVICE-001"
        }
    }

    或：

    {
        "type": "final_answer",
        "content": "你好，请问有什么可以帮助你的？"
    }
    """

    raw_response = call_llm(
        user_message=user_message,
        config=config,
    )

    data = parse_structured_output(
        raw_response
    )

    return validate_decision(
        data
    )


# ============================================================
# Manual Demo
# ============================================================


if __name__ == "__main__":

    print(
        "=" * 70
    )
    print(
        "Day22 Stage 3 | Structured Output Demo"
    )
    print(
        "=" * 70
    )

    questions = [
        "DEVICE-001现在是什么状态？",
        "你好。",
    ]

    for question in questions:

        print()
        print(
            f"User: {question}"
        )

        try:

            result = generate_decision(
                question
            )

            print(
                "Decision:"
            )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )