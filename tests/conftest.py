"""Pytest 公共配置。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(
    items: list[pytest.Item],
) -> None:
    """将真实 HTTP 测试标记为 e2e，默认离线测试不会执行它。"""

    for item in items:
        if item.path.name == "test_agent_http_e2e.py":
            item.add_marker(pytest.mark.e2e)
