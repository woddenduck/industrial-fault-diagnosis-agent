"""
Day 18 - Stage 2
Query Context Extraction & Query Rewrite

第一版目标：
1. 从当前 Query / 历史对话 / 历史摘要中提取设备型号
2. 提取故障码
3. 按优先级补全 Query
4. 不允许编造不存在的信息

当前版本只做简单规则匹配，不调用 LLM。
"""

import re
from typing import Any, Dict, List, Optional


# ============================================================
# 1. 基础实体提取
# ============================================================

def _extract_device_model(text: str) -> Optional[str]:
    """
    从文本中提取设备型号。

    示例：
        G120C
        G120X
    """
    if not text:
        return None

    pattern = r"(?<![A-Za-z0-9])G\d{2,4}[A-Z]?(?![A-Za-z0-9])"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(0).upper()


def _extract_fault_code(text: str) -> Optional[str]:
    """
    从文本中提取故障码。

    示例：
        F30021
        F30024
    """
    if not text:
        return None

    pattern = r"(?<![A-Za-z0-9])F\d{4,6}(?![A-Za-z0-9])"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(0).upper()


# ============================================================
# 2. Query Context Extraction
# ============================================================

def extract_query_context(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    history_summary: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    提取当前 Query 所需上下文。

    优先级：

        当前 Query
        >
        最近 user 历史
        >
        更早 user 历史
        >
        history_summary

    注意：
    第一版只读取 user 历史，
    不读取 assistant 消息中的设备型号和故障码，
    避免模型回答污染 Query Context。
    """

    history = history or []

    # --------------------------------------------------------
    # Step 1：优先从当前 Query 提取
    # --------------------------------------------------------

    device_model = _extract_device_model(query)
    fault_code = _extract_fault_code(query)

    # --------------------------------------------------------
    # Step 2：从历史 user 消息倒序查找
    # --------------------------------------------------------

    if device_model is None or fault_code is None:

        for message in reversed(history):

            if message.get("role") != "user":
                continue

            content = str(
                message.get("content", "")
            )

            if device_model is None:
                device_model = _extract_device_model(content)

            if fault_code is None:
                fault_code = _extract_fault_code(content)

            # 两个都已经找到，不需要继续搜索
            if (
                device_model is not None
                and fault_code is not None
            ):
                break

    # --------------------------------------------------------
    # Step 3：最后才查 history_summary
    # --------------------------------------------------------

    if history_summary:

        if device_model is None:
            device_model = _extract_device_model(
                history_summary
            )

        if fault_code is None:
            fault_code = _extract_fault_code(
                history_summary
            )

    return {
        "device_model": device_model,
        "fault_code": fault_code,
    }


# ============================================================
# 3. Query Rewrite
# ============================================================

def rewrite_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    history_summary: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    根据明确提取出的上下文进行 Query Rewrite。

    原则：
    1. 不编造设备型号
    2. 不编造故障码
    3. 如果信息不足，则保持原 Query
    """

    original_query = query.strip()

    context = extract_query_context(
        query=original_query,
        history=history,
        history_summary=history_summary,
    )

    device_model = context["device_model"]
    fault_code = context["fault_code"]

    # --------------------------------------------------------
    # Case 1：设备型号 + 故障码都有
    # --------------------------------------------------------

    if device_model and fault_code:

        rewritten_query = (
            f"{device_model}设备出现"
            f"{fault_code}故障时应该如何处理？"
        )

    # --------------------------------------------------------
    # Case 2：只有故障码
    # --------------------------------------------------------

    elif fault_code:

        rewritten_query = (
            f"{fault_code}故障应该如何处理？"
        )

    # --------------------------------------------------------
    # Case 3：没有足够上下文
    # --------------------------------------------------------

    else:

        rewritten_query = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "device_model": device_model,
        "fault_code": fault_code,
    }


# ============================================================
# 4. 简单手工测试
# ============================================================

if __name__ == "__main__":

    history = [
        {
            "role": "user",
            "content": "我的设备是G120C。",
        },
        {
            "role": "assistant",
            "content": "好的。",
        },
        {
            "role": "user",
            "content": "现在报F30025。",
        },
        {
            "role": "assistant",
            "content": "收到。",
        },
    ]

    result = rewrite_query(
        query="这个故障怎么处理？",
        history=history,
    )

    print("=" * 60)
    print("Query Rewrite Test")
    print("=" * 60)

    print(
        f"Original Query : "
        f"{result['original_query']}"
    )

    print(
        f"Device Model   : "
        f"{result['device_model']}"
    )

    print(
        f"Fault Code     : "
        f"{result['fault_code']}"
    )

    print(
        f"Rewritten Query: "
        f"{result['rewritten_query']}"
    )