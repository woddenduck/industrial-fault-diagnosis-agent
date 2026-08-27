"""正式 LangGraph 工作流的离线集成测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent.nodes as nodes
from agent.graph import build_graph
from agent.state import create_initial_state


def _answered_rag(*args, **kwargs) -> dict:
    return {
        "success": True,
        "status": "answered",
        "answer": "检查散热风道、环境温度和电机负载。",
        "sources": [
            {
                "document": "G120C_manual.pdf",
                "page": "1",
                "section": "温度保护",
                "chunk_id": "chunk-001",
            }
        ],
        "rag_request_id": "rag-graph-test",
        "decision": {"evidence_sufficient": True},
        "degraded": False,
    }


def _diagnosis_state(*, create_report: bool = False) -> dict:
    return create_initial_state(
        user_query=(
            "请综合诊断 DEVICE-001 当前运行状态，"
            "结合维修记录判断是否存在过热风险。"
        ),
        device_id="DEVICE-001",
        device_model="G120C",
        knowledge_base_id="kb_test",
        create_report=create_report,
    )


def test_missing_device_routes_to_ask_user() -> None:
    result = build_graph().invoke(
        create_initial_state(user_query="请检查设备当前运行状态")
    )

    assert result["status"] == "needs_input"
    assert result["next_action"] == "ask_user"
    assert result["rag_status"] == "not_called"
    assert [item["node"] for item in result["execution_trace"]] == [
        "intent_node",
        "check_information_node",
        "ask_user_node",
    ]


def test_normal_diagnosis_completes_without_review(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "rag_client",
        SimpleNamespace(retrieve=_answered_rag),
    )

    result = build_graph().invoke(_diagnosis_state())

    assert result["status"] == "completed"
    assert result["risk_level"] == "low"
    assert result["human_review_required"] is False
    assert result["sources"][0]["chunk_id"] == "chunk-001"


def test_high_risk_report_waits_for_review(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "rag_client",
        SimpleNamespace(retrieve=_answered_rag),
    )
    monkeypatch.setattr(
        nodes,
        "get_device_status",
        lambda device_id: {
            "success": True,
            "data": {
                "device_id": device_id,
                "status": "normal",
                "temperature": 85.0,
                "running": True,
            },
        },
    )

    result = build_graph().invoke(_diagnosis_state(create_report=True))

    assert result["status"] == "human_review_required"
    assert result["risk_level"] == "high"
    assert result["report"]["severity"] == "high"
    assert result["next_action"] == "awaiting_human_review"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
