"""正式 LangGraph 条件路由测试。"""

from __future__ import annotations

import pytest

from agent.routing import (
    route_after_device_status,
    route_after_diagnosis,
    route_after_maintenance_history,
    route_after_report,
    route_after_retrieve,
    route_after_risk_check,
    route_by_intent,
)


@pytest.mark.parametrize(
    ("router", "state", "expected"),
    [
        (route_by_intent, {"missing_fields": ["device_id"]}, "ask_user"),
        (route_by_intent, {"intent": "diagnosis"}, "diagnosis"),
        (route_by_intent, {"intent": "unknown"}, "ask_user"),
        (
            route_after_device_status,
            {"status": "running", "intent": "diagnosis"},
            "maintenance_history",
        ),
        (
            route_after_device_status,
            {"status": "running", "intent": "device_status"},
            "final_answer",
        ),
        (
            route_after_maintenance_history,
            {"status": "running", "intent": "diagnosis"},
            "retrieve",
        ),
        (
            route_after_maintenance_history,
            {"status": "failed", "intent": "diagnosis"},
            "final_answer",
        ),
        (
            route_after_retrieve,
            {"status": "running", "intent": "diagnosis", "rag_status": "answered"},
            "diagnosis",
        ),
        (
            route_after_retrieve,
            {"status": "insufficient_evidence", "intent": "diagnosis"},
            "final_answer",
        ),
        (route_after_diagnosis, {"status": "running"}, "risk_check"),
        (route_after_diagnosis, {"status": "failed"}, "final_answer"),
        (
            route_after_risk_check,
            {"risk_level": "low", "create_report": True},
            "report",
        ),
        (
            route_after_risk_check,
            {"risk_level": "high", "create_report": False},
            "human_review",
        ),
        (
            route_after_report,
            {"status": "running", "human_review_required": True},
            "human_review",
        ),
        (route_after_report, {"status": "failed"}, "final_answer"),
    ],
)
def test_production_routes(router, state: dict, expected: str) -> None:
    assert router(state) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
