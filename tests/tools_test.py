from pprint import pprint

from agent.tools import (
    create_diagnostic_report,
    get_device_status,
    query_maintenance_history,
)


def run_test(
    title: str,
    result: dict,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    pprint(result)


def main():

    # ---------------------------------------------------------
    # Tool 1
    # ---------------------------------------------------------

    run_test(
        "T01 | Normal device",
        get_device_status("DEVICE-001"),
    )

    run_test(
        "T02 | Device not found",
        get_device_status("DEVICE-999"),
    )

    run_test(
        "T03 | Empty device id",
        get_device_status(""),
    )

    run_test(
        "T04 | Invalid device id format",
        get_device_status("ABC-001"),
    )

    # ---------------------------------------------------------
    # Tool 2
    # ---------------------------------------------------------

    run_test(
        "T05 | Device with maintenance history",
        query_maintenance_history("DEVICE-004"),
    )

    run_test(
        "T06 | Device without maintenance history",
        query_maintenance_history("DEVICE-005"),
    )

    run_test(
        "T07 | Maintenance query - unknown device",
        query_maintenance_history("DEVICE-999"),
    )

    # ---------------------------------------------------------
    # Tool 3
    # ---------------------------------------------------------

    run_test(
        "T08 | Valid report",
        create_diagnostic_report(
            device_id="DEVICE-004",
            summary="Device temperature and vibration are abnormal.",
            severity="critical",
            findings=[
                "Temperature is above normal range.",
                "Vibration level is excessive.",
            ],
            recommendations=[
                "Stop the device if safe to do so.",
                "Inspect cooling and bearing systems.",
            ],
        ),
    )

    run_test(
        "T09 | Invalid severity",
        create_diagnostic_report(
            device_id="DEVICE-004",
            summary="Abnormal condition detected.",
            severity="extreme",
            findings=[
                "Temperature is too high.",
            ],
            recommendations=[
                "Inspect the device.",
            ],
        ),
    )

    run_test(
        "T10 | Missing summary",
        create_diagnostic_report(
            device_id="DEVICE-004",
            summary="",
            severity="high",
            findings=[
                "High vibration.",
            ],
            recommendations=[
                "Inspect bearings.",
            ],
        ),
    )


if __name__ == "__main__":
    main()