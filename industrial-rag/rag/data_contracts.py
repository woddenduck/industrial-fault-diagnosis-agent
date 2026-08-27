"""
Day 18 - Stage 1
固定 RAG 数据契约

定义：
1. QueryContext
2. ContextChunk
3. StructuredAnswer
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ============================================================
# 1. Query Context
# ============================================================

@dataclass
class QueryContext:
    """
    当前用户问题的结构化表示。

    原则：
    - 不确定的单值字段使用 None
    - 不确定的多值字段使用 []
    - 禁止为了补全结构而猜测信息
    """

    original_query: str

    rewritten_query: Optional[str] = None

    device_model: Optional[str] = None

    fault_code: Optional[str] = None

    symptoms: List[str] = field(default_factory=list)

    task: Optional[str] = None

    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 2. Controlled Context Chunk
# ============================================================

@dataclass
class ContextChunk:
    """
    经过 Context Builder 处理后，
    允许进入 Prompt 的统一 Chunk 结构。
    """

    source_id: int

    chunk_id: str

    content: str

    rerank_score: Optional[float] = None

    device_model: Optional[str] = None

    source: Optional[str] = None

    section: Optional[str] = None

    page_start: Optional[int] = None

    page_end: Optional[int] = None

    injection_risk: bool = False

    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 3. Structured Answer
# ============================================================

@dataclass
class StructuredAnswer:
    """
    Day 18 最终回答的数据契约。

    当前阶段仅固定结构，
    暂时不调用 LLM。
    """

    possible_causes: List[str] = field(default_factory=list)

    evidence_sources: List[str] = field(default_factory=list)

    troubleshooting_steps: List[str] = field(default_factory=list)

    risk_warnings: List[str] = field(default_factory=list)

    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """
        转换为固定的最终 Markdown 结构。
        """

        def render_items(items: List[str]) -> str:
            if not items:
                return "暂无。"

            return "\n".join(
                f"- {item}"
                for item in items
            )

        return f"""## 可能原因

{render_items(self.possible_causes)}

## 证据来源

{render_items(self.evidence_sources)}

## 排查步骤

{render_items(self.troubleshooting_steps)}

## 风险提示

{render_items(self.risk_warnings)}

## 仍需确认的信息

{render_items(self.missing_information)}
"""


# ============================================================
# 4. 基础验证函数
# ============================================================

def validate_query_context(
    query_context: QueryContext
) -> List[str]:
    """
    检查 QueryContext 是否满足基础数据契约。

    返回：
        errors = []

    空列表表示通过。
    """

    errors = []

    if not query_context.original_query.strip():
        errors.append(
            "original_query 不能为空"
        )

    if query_context.device_model == "":
        errors.append(
            "device_model 未知时必须使用 None，不能使用空字符串"
        )

    if query_context.fault_code == "":
        errors.append(
            "fault_code 未知时必须使用 None，不能使用空字符串"
        )

    if query_context.task == "":
        errors.append(
            "task 未知时必须使用 None，不能使用空字符串"
        )

    return errors


def validate_context_chunk(
    chunk: ContextChunk
) -> List[str]:
    """
    检查 ContextChunk 基础字段。
    """

    errors = []

    if chunk.source_id <= 0:
        errors.append(
            "source_id 必须从 1 开始"
        )

    if not chunk.chunk_id.strip():
        errors.append(
            "chunk_id 不能为空"
        )

    if not chunk.content.strip():
        errors.append(
            "content 不能为空"
        )

    if chunk.token_count < 0:
        errors.append(
            "token_count 不能小于 0"
        )

    return errors