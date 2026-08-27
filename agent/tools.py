from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import uuid
from datetime import datetime

from agent.config import (
    DEVICE_STATUS_FILE,
    MAINTENANCE_HISTORY_FILE,
)

DEVICE_ID_PATTERN = re.compile(r"^DEVICE-\d{3}$")

VALID_SEVERITIES = {
    "low",
    "medium",
    "high",
    "critical",
}


def _success(
    tool_name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "success": True,
        "tool": tool_name,
        "data": data,
        "error": None,
    }


def _error(
    tool_name: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "tool": tool_name,
        "data": None,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _validate_device_id(
    device_id: Any,
    tool_name: str,
) -> dict[str, Any] | None:

    if not isinstance(device_id, str):
        return _error(
            tool_name,
            "INVALID_DEVICE_ID_TYPE",
            "device_id must be a string.",
        )

    device_id = device_id.strip()

    if not device_id:
        return _error(
            tool_name,
            "EMPTY_DEVICE_ID",
            "device_id cannot be empty.",
        )

    if not DEVICE_ID_PATTERN.fullmatch(device_id):
        return _error(
            tool_name,
            "INVALID_DEVICE_ID",
            "device_id must match format DEVICE-XXX.",
        )

    return None


def get_device_status(
    device_id: str,
) -> dict[str, Any]:

    tool_name = "get_device_status"

    validation_error = _validate_device_id(
        device_id,
        tool_name,
    )

    if validation_error:
        return validation_error

    device_id = device_id.strip()

    try:
        devices = _load_json(DEVICE_STATUS_FILE)

    except FileNotFoundError:
        return _error(
            tool_name,
            "DATA_FILE_NOT_FOUND",
            "Device status data file was not found.",
        )

    except json.JSONDecodeError:
        return _error(
            tool_name,
            "INVALID_DATA_FILE",
            "Device status data file contains invalid JSON.",
        )

    except OSError as exc:
        return _error(
            tool_name,
            "DATA_READ_ERROR",
            f"Failed to read device status data: {exc}",
        )

    device = devices.get(device_id)

    if device is None:
        return _error(
            tool_name,
            "DEVICE_NOT_FOUND",
            f"Device {device_id} does not exist.",
        )

    return _success(
        tool_name,
        {
            "device_id": device_id,
            "exists": True,
            "status": device.get("status"),
            "device_name": device.get("device_name"),
            "temperature": device.get("temperature"),
            "vibration": device.get("vibration"),
            "running": device.get("running"),
            "last_updated": device.get("last_updated"),
        },
    )


def query_maintenance_history(
    device_id: str,
) -> dict[str, Any]:

    tool_name = "query_maintenance_history"

    validation_error = _validate_device_id(
        device_id,
        tool_name,
    )

    if validation_error:
        return validation_error

    device_id = device_id.strip()

    try:
        devices = _load_json(DEVICE_STATUS_FILE)
        histories = _load_json(MAINTENANCE_HISTORY_FILE)

    except FileNotFoundError:
        return _error(
            tool_name,
            "DATA_FILE_NOT_FOUND",
            "Required business data file was not found.",
        )

    except json.JSONDecodeError:
        return _error(
            tool_name,
            "INVALID_DATA_FILE",
            "Business data file contains invalid JSON.",
        )

    except OSError as exc:
        return _error(
            tool_name,
            "DATA_READ_ERROR",
            f"Failed to read maintenance data: {exc}",
        )

    if device_id not in devices:
        return _error(
            tool_name,
            "DEVICE_NOT_FOUND",
            f"Device {device_id} does not exist.",
        )

    records = histories.get(device_id, [])

    return _success(
        tool_name,
        {
            "device_id": device_id,
            "record_count": len(records),
            "records": records,
            "has_maintenance_records": len(records) > 0,
            "interpretation": (
                "No maintenance records are currently available. "
                "This does not prove that the device has never "
                "experienced a fault."
                if not records
                else
                "Maintenance records were found."
            ),
        },
    )


def create_diagnostic_report(
    device_id: str,
    summary: str,
    severity: str,
    findings: list[str],
    recommendations: list[str],
) -> dict[str, Any]:

    tool_name = "create_diagnostic_report"


    validation_error = _validate_device_id(
        device_id,
        tool_name,
    )

    if validation_error:
        return validation_error


    if not isinstance(summary, str) or not summary.strip():

        return _error(
            tool_name,
            "INVALID_SUMMARY",
            "summary cannot be empty.",
        )


    if (
        not isinstance(severity, str)
        or severity not in VALID_SEVERITIES
    ):

        return _error(
            tool_name,
            "INVALID_SEVERITY",
            "severity must be one of: low, medium, high, critical.",
        )


    if not isinstance(findings, list):

        return _error(
            tool_name,
            "INVALID_FINDINGS",
            "findings must be a list.",
        )


    if not isinstance(recommendations, list):

        return _error(
            tool_name,
            "INVALID_RECOMMENDATIONS",
            "recommendations must be a list.",
        )


    if not all(
        isinstance(item, str)
        and item.strip()
        for item in findings
    ):

        return _error(
            tool_name,
            "INVALID_FINDINGS",
            "Every finding must be a non-empty string.",
        )


    if not all(
        isinstance(item, str)
        and item.strip()
        for item in recommendations
    ):

        return _error(
            tool_name,
            "INVALID_RECOMMENDATIONS",
            "Every recommendation must be a non-empty string.",
        )


    #
    # Stage6 Action Tool Side Effect
    #

    report_id = (
        "REPORT-"
        +
        datetime.now()
        .strftime("%Y%m%d")
        +
        "-"
        +
        uuid.uuid4()
        .hex[:8]
    )


    report = {

        "report_id":
            report_id,

        "device_id":
            device_id.strip(),

        "severity":
            severity,

        "summary":
            summary.strip(),

        "findings":
            findings,

        "recommendations":
            recommendations,

        "created_at":
            datetime.now()
            .isoformat()

    }


    return _success(
        tool_name,
        report,
    )
