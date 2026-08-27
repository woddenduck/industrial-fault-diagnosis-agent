"""
Industrial Fault Diagnosis Agent
正式 LangGraph 工作流。

职责：

- 注册生产节点；
- 连接节点；
- 根据 AgentState 执行条件路由；
- 在失败、证据不足和高风险时安全收口。

本模块不负责：

- Tool 具体业务逻辑；
- RAG HTTP 调用；
- 诊断内容生成；
- 风险关键词判断；
- API 请求与响应。
"""

from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agent.nodes import (
    ask_user_node,
    check_information_node,
    device_status_node,
    diagnosis_node,
    final_answer_node,
    human_review_node,
    intent_node,
    maintenance_history_node,
    report_node,
    retrieve_node,
    risk_check_node,
)

from agent.routing import (
    route_after_device_status,
    route_after_diagnosis,
    route_after_maintenance_history,
    route_after_report,
    route_after_retrieve,
    route_after_risk_check,
    route_by_intent,
)

from agent.state import AgentState


# ============================================================
# Graph Builder
# ============================================================

def build_graph():
    """
    构建并编译工业设备故障诊断工作流。

    Graph 本身不保存具体请求状态。
    每次请求都必须传入独立的 AgentState。
    """

    builder = StateGraph(
        AgentState
    )

    # ========================================================
    # 1. 注册节点
    # ========================================================

    builder.add_node(
        "intent",
        intent_node,
    )

    builder.add_node(
        "check_information",
        check_information_node,
    )

    builder.add_node(
        "ask_user",
        ask_user_node,
    )

    builder.add_node(
        "device_status",
        device_status_node,
    )

    builder.add_node(
        "maintenance_history",
        maintenance_history_node,
    )

    builder.add_node(
        "retrieve",
        retrieve_node,
    )

    builder.add_node(
        "diagnosis",
        diagnosis_node,
    )

    builder.add_node(
        "risk_check",
        risk_check_node,
    )

    builder.add_node(
        "report",
        report_node,
    )

    builder.add_node(
        "final_answer",
        final_answer_node,
    )

    builder.add_node(
        "human_review",
        human_review_node,
    )

    # ========================================================
    # 2. Graph 入口
    # ========================================================

    builder.add_edge(
        START,
        "intent",
    )

    builder.add_edge(
        "intent",
        "check_information",
    )

    # ========================================================
    # 3. 意图分流
    # ========================================================

    builder.add_conditional_edges(
        "check_information",
        route_by_intent,
        {
            "ask_user":
                "ask_user",

            "device_status":
                "device_status",

            "maintenance_history":
                "maintenance_history",

            "knowledge_query":
                "retrieve",

            "diagnosis":
                "device_status",
        },
    )

    # ========================================================
    # 4. 设备状态节点后路由
    # ========================================================

    builder.add_conditional_edges(
        "device_status",
        route_after_device_status,
        {
            "maintenance_history":
                "maintenance_history",

            "final_answer":
                "final_answer",
        },
    )

    # ========================================================
    # 5. 维修历史节点后路由
    # ========================================================

    builder.add_conditional_edges(
        "maintenance_history",
        route_after_maintenance_history,
        {
            "retrieve":
                "retrieve",

            "final_answer":
                "final_answer",
        },
    )

    # ========================================================
    # 6. RAG 节点后路由
    # ========================================================

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "diagnosis":
                "diagnosis",

            "final_answer":
                "final_answer",
        },
    )

    # ========================================================
    # 7. 诊断节点后路由
    # ========================================================

    builder.add_conditional_edges(
        "diagnosis",
        route_after_diagnosis,
        {
            "risk_check":
                "risk_check",

            "final_answer":
                "final_answer",
        },
    )

    # ========================================================
    # 8. 风险判断后路由
    # ========================================================

    builder.add_conditional_edges(
        "risk_check",
        route_after_risk_check,
        {
            "report":
                "report",

            "final_answer":
                "final_answer",

            "human_review":
                "human_review",
        },
    )

    # ========================================================
    # 9. 报告节点后路由
    # ========================================================

    builder.add_conditional_edges(
        "report",
        route_after_report,
        {
            "final_answer":
                "final_answer",

            "human_review":
                "human_review",
        },
    )

    # ========================================================
    # 10. Graph 出口
    # ========================================================

    builder.add_edge(
        "ask_user",
        END,
    )

    builder.add_edge(
        "final_answer",
        END,
    )

    builder.add_edge(
        "human_review",
        END,
    )

    return builder.compile()


# ============================================================
# 默认 Graph 实例
# ============================================================

industrial_graph = build_graph()