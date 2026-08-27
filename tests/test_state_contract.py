from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import DiagnoseRequest
from agent.state import create_initial_state


def test_create_minimal_initial_state():
    state = create_initial_state(
        user_query="检查 DEVICE-001 当前状态",
    )

    assert state["request_id"]
    assert state["user_query"] == (
        "检查 DEVICE-001 当前状态"
    )

    assert state["intent"] == "unknown"
    assert state["status"] == "running"
    assert state["rag_status"] == "not_called"

    assert state["device_status"] == {}
    assert state["maintenance_history"] == {}
    assert state["sources"] == []
    assert state["errors"] == []
    assert state["execution_trace"] == []

    assert state["human_review_required"] is False
    assert state["report"] is None


def test_input_values_are_normalized():
    state = create_initial_state(
        user_query="  检查设备  ",
        device_id=" DEVICE-001 ",
        device_model=" g120c ",
        knowledge_base_id=" kb_ab50652fe3a4 ",
    )

    assert state["user_query"] == "检查设备"
    assert state["device_id"] == "DEVICE-001"
    assert state["device_model"] == "G120C"
    assert (
        state["knowledge_base_id"]
        == "kb_ab50652fe3a4"
    )


def test_mutable_defaults_are_isolated():
    first = create_initial_state(
        user_query="first",
    )
    second = create_initial_state(
        user_query="second",
    )

    first["errors"].append(
        {
            "code": "TEST_ERROR",
            "message": "test",
        }
    )

    first["sources"].append(
        {
            "document": "test.pdf",
        }
    )

    assert second["errors"] == []
    assert second["sources"] == []


def test_request_ids_are_unique():
    first = create_initial_state(
        user_query="first",
    )
    second = create_initial_state(
        user_query="second",
    )

    assert first["request_id"]
    assert second["request_id"]
    assert first["request_id"] != second["request_id"]


def test_diagnose_request_accepts_minimal_input():
    request = DiagnoseRequest(
        query="查询 DEVICE-001 当前状态",
    )

    assert request.query
    assert request.device_id is None
    assert request.create_report is False
    assert request.history == []


def test_diagnose_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        DiagnoseRequest(
            query="",
        )


def test_history_is_copied():
    history = [
        {
            "role": "user",
            "content": "上一轮问题",
        }
    ]

    state = create_initial_state(
        user_query="继续分析",
        history=history,
    )

    history[0]["content"] = "已经修改"

    assert (
        state["history"][0]["content"]
        == "上一轮问题"
    )