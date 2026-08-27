"""生产工具的离线契约测试。"""

from __future__ import annotations

import pytest

from agent.tools import (
    create_diagnostic_report,
    get_device_status,
    query_maintenance_history,
)


def test_device_status_reads_known_device() -> None:
    result = get_device_status("DEVICE-001")

    assert result["success"] is True
    assert result["data"]["device_id"] == "DEVICE-001"
    assert result["data"]["temperature"] == 42.5


@pytest.mark.parametrize("device_id", ["", "device-1", "DEVICE-ABC"])
def test_device_status_rejects_invalid_id(device_id: str) -> None:
    result = get_device_status(device_id)

    assert result["success"] is False
    assert result["error"]["code"] in {
        "EMPTY_DEVICE_ID",
        "INVALID_DEVICE_ID",
    }


def test_maintenance_history_reads_known_device() -> None:
    result = query_maintenance_history("DEVICE-001")

    assert result["success"] is True
    assert result["data"]["record_count"] == 1
    assert result["data"]["has_maintenance_records"] is True


def test_report_uses_stable_contract() -> None:
    result = create_diagnostic_report(
        device_id="DEVICE-001",
        summary="设备运行正常。",
        severity="low",
        findings=["温度低于关注阈值。"],
        recommendations=["继续按计划巡检。"],
    )

    assert result["success"] is True
    assert result["data"]["report_id"].startswith("REPORT-")
    assert result["data"]["severity"] == "low"


def test_report_rejects_invalid_severity() -> None:
    result = create_diagnostic_report(
        device_id="DEVICE-001",
        summary="设备状态待确认。",
        severity="unknown",
        findings=["缺少数据。"],
        recommendations=["补充现场数据。"],
    )

    assert result["success"] is False
    assert result["error"]["code"] == "INVALID_SEVERITY"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
