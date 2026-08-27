"""第七关：诊断风险分级质量验收。

测试直接调用 risk_check_node，不使用 RAG 或设备工具替身，
重点验证结构化设备事实与风险等级之间的对应关系。
"""

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.nodes import risk_check_node


def _state(
    *,
    status: str = "normal",
    temperature: float | None = 42.5,
    query: str = "请判断设备是否存在过热风险",
    rag_answer: str = "变频器故障可能导致火灾或电击危险。",
    extra_status: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    device_status: dict[str, Any] = {
        "device_id": "DEVICE-001",
        "exists": True,
        "status": status,
        "running": True,
    }
    if temperature is not None:
        device_status["temperature"] = temperature
    if extra_status:
        device_status.update(extra_status)

    return {
        "user_query": query,
        "status": "running",
        "device_status": device_status,
        "maintenance_history": {
            "device_id": "DEVICE-001",
            "records": records
            if records is not None
            else [
                {
                    "type": "inspection",
                    "description": "Routine inspection completed.",
                    "result": "normal",
                }
            ],
        },
        "rag_answer": rag_answer,
        "diagnosis": "【知识库参考】\n" + rag_answer,
        "execution_trace": [],
    }


def _assert_case(
    name: str,
    state: dict[str, Any],
    expected_risk: str,
    expected_review: bool,
) -> dict[str, Any]:
    result = risk_check_node(state)
    actual_risk = result.get("risk_level")
    actual_review = result.get("human_review_required")

    print(f"\n{name}")
    print(f"期望风险：{expected_risk}")
    print(f"实际风险：{actual_risk}")
    print(f"人工复核：{actual_review}")
    print(f"判断依据：{result['execution_trace'][-1].get('detail', '')}")

    assert actual_risk == expected_risk
    assert actual_review is expected_review
    assert "【风险判断依据】" in result.get("diagnosis", "")
    return result


def test_normal_device_is_low_risk() -> None:
    """通用手册危险警告不能把正常设备提升为 critical。"""

    _assert_case(
        "Q01｜正常设备 + 通用火灾警告",
        _state(),
        "low",
        False,
    )


def test_missing_temperature_is_unknown() -> None:
    """询问过热但缺少温度时，不伪造低风险结论。"""

    result = _assert_case(
        "Q02｜正常状态但缺少温度",
        _state(temperature=None),
        "unknown",
        False,
    )
    assert result.get("status") == "insufficient_evidence"


def test_attention_temperature_is_medium() -> None:
    _assert_case(
        "Q03｜温度达到关注阈值",
        _state(temperature=70.0),
        "medium",
        False,
    )


def test_high_temperature_requires_review() -> None:
    _assert_case(
        "Q04｜温度达到高风险阈值",
        _state(temperature=85.0),
        "high",
        True,
    )


def test_critical_temperature_requires_review() -> None:
    _assert_case(
        "Q05｜温度达到严重风险阈值",
        _state(temperature=98.0),
        "critical",
        True,
    )


def test_current_fault_signal_is_high_risk() -> None:
    _assert_case(
        "Q06｜当前存在明确过热报警",
        _state(
            temperature=72.0,
            extra_status={"fault_code": "F30005", "alarm": "功率单元过热"},
        ),
        "high",
        True,
    )


def test_unsafe_control_request_requires_review() -> None:
    _assert_case(
        "Q07｜请求解除保护并写入 PLC",
        _state(query="请解除保护并写入PLC后强制启动设备"),
        "high",
        True,
    )


def test_unresolved_maintenance_issue_is_medium() -> None:
    _assert_case(
        "Q08｜存在未解决的历史异常",
        _state(
            records=[
                {
                    "description": "设备曾发生过热故障",
                    "result": "未解决",
                }
            ]
        ),
        "medium",
        False,
    )


def main() -> int:
    tests = [
        test_normal_device_is_low_risk,
        test_missing_temperature_is_unknown,
        test_attention_temperature_is_medium,
        test_high_temperature_requires_review,
        test_critical_temperature_requires_review,
        test_current_fault_signal_is_high_risk,
        test_unsafe_control_request_requires_review,
        test_unresolved_maintenance_issue_is_medium,
    ]

    print("第七关：诊断风险分级质量验收")
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print("结果：通过")
        except Exception as exc:
            print(f"结果：失败，{exc}")
            print(f"\n验收结果：{passed}/{len(tests)} 通过")
            return 1

    print(f"\n验收结果：{passed}/{len(tests)} 全部通过")
    print("第七关：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
