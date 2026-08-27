"""Pytest 公共配置。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRUCTURED_OUTPUT_E2E_TESTS = {
    "test_s01_device_status_tool_call",
    "test_s02_maintenance_history_tool_call",
    "test_s03_greeting_final_answer",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(
    items: list[pytest.Item],
) -> None:
    """将真实 HTTP 测试标记为 e2e，默认离线测试不会执行它。"""

    for item in items:
        is_http_e2e = (
            item.path.name
            == "test_agent_http_e2e.py"
        )
        is_structured_output_e2e = (
            item.path.name
            == "test_structured_output.py"
            and item.name
            in STRUCTURED_OUTPUT_E2E_TESTS
        )

        if is_http_e2e or is_structured_output_e2e:
            item.add_marker(pytest.mark.e2e)
