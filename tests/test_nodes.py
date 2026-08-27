"""生产节点的错误边界与 RAG 业务状态测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent.nodes as nodes
from agent.state import create_initial_state


def _state() -> dict:
    return create_initial_state(
        user_query="查询 G120C 过热处理方法",
        device_model="G120C",
        knowledge_base_id="kb_test",
    )


def test_retrieve_node_accepts_answered_result(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "rag_client",
        SimpleNamespace(
            retrieve=lambda *args, **kwargs: {
                "success": True,
                "status": "answered",
                "answer": "检查散热风道。",
                "sources": [{"document": "manual.pdf", "chunk_id": "c1"}],
                "rag_request_id": "rag-test-001",
                "decision": {"evidence_sufficient": True},
                "degraded": False,
            }
        ),
    )

    result = nodes.retrieve_node(_state())

    assert result["rag_status"] == "answered"
    assert result["sources"][0]["chunk_id"] == "c1"
    assert result["status"] == "running"


def test_retrieve_node_preserves_evidence_rejection(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "rag_client",
        SimpleNamespace(
            retrieve=lambda *args, **kwargs: {
                "success": True,
                "status": "rejected",
                "answer": "知识库证据不足。",
                "sources": [],
                "rag_request_id": "rag-test-002",
                "decision": {"evidence_sufficient": False},
                "degraded": False,
            }
        ),
    )

    result = nodes.retrieve_node(_state())

    assert result["rag_status"] == "rejected"
    assert result["status"] == "insufficient_evidence"
    assert result["next_action"] == "final_answer"


def test_device_node_converts_invalid_tool_response(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "get_device_status", lambda _: {"data": {}})
    state = create_initial_state(
        user_query="检查 DEVICE-001 当前状态",
        device_id="DEVICE-001",
    )

    result = nodes.device_status_node(state)

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "TOOL_INVALID_RESPONSE"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
