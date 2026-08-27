"""
Agent Conditional Routing。

Router 只负责读取 State 并决定下一条路径，
不执行 Tool、RAG 或诊断逻辑。
"""

from agent.state import AgentState


# ============================================================
# 旧 Graph 临时兼容路由
# ============================================================

def route_after_information_check(
    state: AgentState,
) -> str:
    """
    当前旧 graph.py 仍使用：

        ask_user / tool

    下一阶段重建正式 Graph 后删除。
    """

    if state.get("missing_fields"):
        return "ask_user"

    if (
        state.get("next_action")
        == "ask_user"
    ):
        return "ask_user"

    if state.get("device_id"):
        return "tool"

    return "ask_user"


# ============================================================
# 正式意图路由
# ============================================================

def route_by_intent(
    state: AgentState,
) -> str:
    """
    信息检查完成后，根据意图决定业务路径。

    返回值是业务路径名称，
    下一阶段由 graph.py 映射到对应 Node。
    """

    if state.get("missing_fields"):
        return "ask_user"

    if (
        state.get("next_action")
        == "ask_user"
    ):
        return "ask_user"

    intent = state.get(
        "intent",
        "unknown",
    )

    allowed_routes = {
        "device_status",
        "maintenance_history",
        "knowledge_query",
        "diagnosis",
    }

    if intent in allowed_routes:
        return intent

    return "ask_user"



# ============================================================
# 业务节点结果路由
# ============================================================

def route_after_device_status(
    state: AgentState,
) -> str:
    """
    设备状态节点完成后的路由。

    单独状态查询：
        device_status -> final_answer

    综合诊断：
        device_status -> maintenance_history

    工具失败：
        device_status -> final_answer
    """

    if state.get("status") == "failed":
        return "final_answer"

    if state.get("intent") == "diagnosis":
        return "maintenance_history"

    return "final_answer"


def route_after_maintenance_history(
    state: AgentState,
) -> str:
    """
    维修历史节点完成后的路由。

    单独维修历史查询：
        maintenance_history -> final_answer

    综合诊断：
        maintenance_history -> retrieve

    工具失败：
        maintenance_history -> final_answer
    """

    if state.get("status") == "failed":
        return "final_answer"

    if state.get("intent") == "diagnosis":
        return "retrieve"

    return "final_answer"


def route_after_retrieve(
    state: AgentState,
) -> str:
    """
    RAG 节点完成后的路由。

    knowledge_query：
        answered -> final_answer

    diagnosis：
        answered -> diagnosis

    rejected / error：
        -> final_answer
    """

    if state.get("status") in {
        "failed",
        "insufficient_evidence",
    }:
        return "final_answer"

    if (
        state.get("intent") == "diagnosis"
        and state.get("rag_status") == "answered"
    ):
        return "diagnosis"

    return "final_answer"


def route_after_diagnosis(
    state: AgentState,
) -> str:
    """
    诊断节点完成后的路由。
    """

    if state.get("status") in {
        "failed",
        "insufficient_evidence",
    }:
        return "final_answer"

    return "risk_check"


# ============================================================
# 风险路由
# ============================================================

def route_after_risk_check(
    state: AgentState,
) -> str:
    """
    风险判断后的路由。

    low / medium：
        create_report=false -> final_answer
        create_report=true  -> report

    high / critical：
        create_report=false -> human_review
        create_report=true  -> report
    """

    if state.get("risk_level") in {
        "high",
        "critical",
    }:
        if state.get("create_report", False):
            return "report"

        return "human_review"

    if state.get("create_report", False):
        return "report"

    return "final_answer"



# ============================================================
# 在文件末尾增加报告后路由
# ============================================================
def route_after_report(
    state: AgentState,
) -> str:
    """
    报告生成后的路由。

    报告失败：
        -> final_answer

    高风险报告成功：
        -> human_review

    普通报告成功：
        -> final_answer
    """

    if state.get("status") == "failed":
        return "final_answer"

    if state.get("human_review_required"):
        return "human_review"

    return "final_answer"
