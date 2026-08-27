"""Industrial Fault Diagnosis Agent 的生产节点层。

节点只负责读取 AgentState、调用只读工具/RAG，并返回局部状态更新。
本模块不会向 PLC 或工业设备下发控制指令。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.config import DEFAULT_DEVICE_MODEL
from agent.errors import (
    REPORT_GENERATION_ERROR,
    RAG_EXECUTION_ERROR,
    TOOL_EXECUTION_ERROR,
    TOOL_INVALID_RESPONSE,
)
from agent.rag_client import RAGClient
from agent.state import AgentState
from agent.tools import (
    create_diagnostic_report,
    get_device_status,
    query_maintenance_history,
)


rag_client = RAGClient()

DEVICE_ID_PATTERN = re.compile(r"\bDEVICE-\d{3}\b", re.IGNORECASE)

STATUS_KEYWORDS = (
    "当前状态", "运行状态", "运行情况", "设备状态", "实时状态",
    "现在是什么状态", "是否运行", "是否在线", "当前是否", "当前温度",
    "当前振动", "检查状态",
)
MAINTENANCE_KEYWORDS = (
    "维修记录", "维护记录", "维修历史", "维护历史", "历史维修",
    "历史维护", "保养记录", "以前发生", "过去发生", "过去有没有",
)
KNOWLEDGE_KEYWORDS = (
    "说明书", "手册", "知识库", "故障码", "报警码", "含义", "怎么处理",
    "如何处理", "应该检查", "排查步骤", "可能原因", "技术资料",
)
DIAGNOSIS_KEYWORDS = (
    "诊断", "分析", "异常", "故障原因", "综合判断", "结合维修",
    "结合历史", "判断问题", "严重异常",
)

UNSAFE_CONTROL_KEYWORDS = (
    "立即停机", "紧急停机", "断电", "关闭电源", "带电作业", "拆机",
    "执行控制", "写入plc", "解除保护", "旁路保护", "屏蔽联锁",
    "修改安全参数", "远程启停", "强制启动",
)

# 当前项目的设备数据以摄氏度表示。这里是演示数据的风险分界线，
# 生产环境应进一步按设备型号和测点类型移入配置层。
TEMPERATURE_ATTENTION_C = 65.0
TEMPERATURE_HIGH_C = 80.0
TEMPERATURE_CRITICAL_C = 95.0

CRITICAL_DEVICE_STATUSES = {
    "critical", "emergency", "danger", "紧急", "危险",
}
HIGH_DEVICE_STATUSES = {
    "fault", "failed", "failure", "trip", "故障", "跳闸",
}
MEDIUM_DEVICE_STATUSES = {
    "warning", "alarm", "offline", "异常", "报警", "离线",
}
NORMAL_DEVICE_STATUSES = {
    "normal", "healthy", "running", "正常", "运行中",
}

CRITICAL_SIGNAL_KEYWORDS = (
    "起火", "冒烟", "着火", "燃烧", "爆炸", "电弧", "绝缘击穿",
    "critical", "emergency",
)
HIGH_SIGNAL_KEYWORDS = (
    "过热", "过温", "温度过高", "严重故障", "跳闸", "trip", "fault",
)
MAINTENANCE_ISSUE_KEYWORDS = (
    "异常", "故障", "未修复", "未解决", "失败", "过热", "过温", "跳闸",
    "abnormal", "fault", "failed", "unresolved", "overheat", "trip",
)
MAINTENANCE_RESOLVED_KEYWORDS = (
    "已修复", "已解决", "已完成", "恢复正常", "正常", "通过",
    "resolved", "completed", "normal", "passed",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_device_id(query: str) -> str:
    match = DEVICE_ID_PATTERN.search(query)
    return match.group(0).upper() if match else ""


def _to_float(value: Any) -> float | None:
    """安全读取数值，避免脏数据导致风险节点异常。"""

    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_signal_text(value: Any) -> list[str]:
    """将报警和故障字段整理成可检查的文本，不扫描整段 RAG 回答。"""

    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_flatten_signal_text(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_signal_text(item))
        return result
    return [str(value)]


def _device_signal_text(device_status: dict[str, Any]) -> str:
    """只读取能够代表当前设备异常的结构化字段。"""

    fields = (
        "alarm", "alarms", "alarm_code", "alarm_codes",
        "fault", "faults", "fault_code", "fault_codes",
        "error", "error_code", "warning", "warnings",
    )
    parts: list[str] = []
    for field in fields:
        if field in device_status:
            parts.extend(_flatten_signal_text(device_status.get(field)))
    return " ".join(parts).lower()


def _has_unresolved_maintenance_issue(data: dict[str, Any]) -> bool:
    """判断维修历史中是否存在尚未解决的异常记录。"""

    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return False

    for record in records:
        if not isinstance(record, dict):
            continue
        text = json.dumps(record, ensure_ascii=False).lower()
        has_issue = _contains_any(
            text,
            tuple(word.lower() for word in MAINTENANCE_ISSUE_KEYWORDS),
        )
        is_resolved = _contains_any(
            text,
            tuple(word.lower() for word in MAINTENANCE_RESOLVED_KEYWORDS),
        )
        if has_issue and not is_resolved:
            return True
    return False


def _append_risk_basis(diagnosis: str, basis: list[str]) -> str:
    """把风险依据加入诊断，保证报告和最终回答能够解释分级结果。"""

    content = diagnosis.strip()
    if "【风险判断依据】" in content:
        content = content.split("【风险判断依据】", 1)[0].rstrip()
    reason = "；".join(basis) if basis else "未获得足够的结构化风险证据"
    return f"{content}\n\n【风险判断依据】\n{reason}".strip()


def _append_trace(
    state: AgentState,
    *,
    node: str,
    status: str,
    detail: str,
) -> list[dict[str, Any]]:
    trace = list(state.get("execution_trace") or [])
    trace.append({"node": node, "status": status, "detail": detail})
    return trace


def _error_update(
    state: AgentState,
    *,
    node: str,
    code: str,
    message: str,
    retryable: bool = False,
    status: str = "failed",
    next_action: str = "final_answer",
    **extra: Any,
) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    errors.append(
        {
            "code": code,
            "message": message,
            "node": node,
            "retryable": retryable,
        }
    )
    return {
        "status": status,
        "next_action": next_action,
        "error": message,
        "errors": errors,
        "execution_trace": _append_trace(
            state,
            node=node,
            status="failed",
            detail=f"{code}: {message}",
        ),
        **extra,
    }


def _read_error(error: Any, default_code: str, default_message: str) -> tuple[str, str, bool]:
    if isinstance(error, dict):
        return (
            str(error.get("code") or default_code),
            str(error.get("message") or default_message),
            bool(error.get("retryable", False)),
        )
    if isinstance(error, str) and error.strip():
        return default_code, error.strip(), False
    return default_code, default_message, False


def _run_readonly_tool(
    state: AgentState,
    *,
    node: str,
    state_field: str,
    tool: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    device_id = str(state.get("device_id") or "").strip().upper()
    try:
        result = tool(device_id)
    except Exception as exc:  # 防止第三方/文件层异常逃出图运行时
        return _error_update(
            state,
            node=node,
            code=TOOL_EXECUTION_ERROR,
            message=f"{node} 调用异常：{exc}",
        )

    if not isinstance(result, dict) or "success" not in result:
        return _error_update(
            state,
            node=node,
            code=TOOL_INVALID_RESPONSE,
            message=f"{node} 返回格式不符合工具契约",
        )

    if result.get("success") is not True:
        code, message, retryable = _read_error(
            result.get("error"), TOOL_EXECUTION_ERROR, f"{node} 执行失败"
        )
        return _error_update(
            state,
            node=node,
            code=code,
            message=message,
            retryable=retryable,
        )

    data = result.get("data")
    if not isinstance(data, dict):
        return _error_update(
            state,
            node=node,
            code=TOOL_INVALID_RESPONSE,
            message=f"{node} 成功结果缺少 data 对象",
        )

    return {
        state_field: data,
        "status": "running",
        "error": "",
        "execution_trace": _append_trace(
            state, node=node, status="success", detail=f"读取 {device_id} 成功"
        ),
    }


def intent_node(state: AgentState) -> dict[str, Any]:
    """以可解释规则识别最小业务意图。"""

    query = str(state.get("user_query") or "").strip()
    explicit_device_id = str(state.get("device_id") or "").strip().upper()
    device_id = explicit_device_id or _extract_device_id(query)
    device_model = str(state.get("device_model") or "").strip().upper()
    device_model = device_model or DEFAULT_DEVICE_MODEL

    if not query:
        intent = "unknown"
    else:
        has_status = _contains_any(query, STATUS_KEYWORDS)
        has_maintenance = _contains_any(query, MAINTENANCE_KEYWORDS)
        has_knowledge = _contains_any(query, KNOWLEDGE_KEYWORDS)
        has_diagnosis = _contains_any(query, DIAGNOSIS_KEYWORDS)

        if has_status and has_maintenance:
            intent = "diagnosis"
        elif device_id and has_diagnosis:
            intent = "diagnosis"
        elif has_maintenance:
            intent = "maintenance_history"
        elif has_status:
            intent = "device_status"
        elif has_knowledge or has_diagnosis:
            intent = "knowledge_query"
        else:
            intent = "unknown"

    return {
        "intent": intent,
        "device_id": device_id,
        "device_model": device_model,
        "execution_trace": _append_trace(
            state, node="intent_node", status="success", detail=f"intent={intent}"
        ),
    }


def check_information_node(state: AgentState) -> dict[str, Any]:
    """按意图检查必要字段，不对所有请求强制要求设备编号。"""

    intent = state.get("intent", "unknown")
    device_id = str(state.get("device_id") or "").strip().upper()
    device_model = str(state.get("device_model") or "").strip().upper()
    device_model = device_model or DEFAULT_DEVICE_MODEL
    knowledge_base_id = str(state.get("knowledge_base_id") or "").strip()

    required_fields = {
        "device_status": ("device_id",),
        "maintenance_history": ("device_id",),
        "knowledge_query": ("device_model", "knowledge_base_id"),
        "diagnosis": ("device_id", "device_model", "knowledge_base_id"),
    }

    if intent == "unknown":
        message = (
            "暂时无法识别请求类型，请说明需要查询设备状态、维修记录、"
            "知识库，还是进行故障诊断。"
        )
        return {
            "device_id": device_id,
            "device_model": device_model,
            "knowledge_base_id": knowledge_base_id,
            "missing_fields": ["intent"],
            "next_action": "ask_user",
            "status": "needs_input",
            "error": message,
            "execution_trace": _append_trace(
                state, node="check_information_node", status="needs_input", detail=message
            ),
        }

    values = {
        "device_id": device_id,
        "device_model": device_model,
        "knowledge_base_id": knowledge_base_id,
    }
    missing_fields = [
        name for name in required_fields.get(intent, ()) if not values.get(name)
    ]
    if (
        "device_id" in required_fields.get(intent, ())
        and device_id
        and not DEVICE_ID_PATTERN.fullmatch(device_id)
        and "device_id" not in missing_fields
    ):
        missing_fields.append("device_id")

    if missing_fields:
        message = "缺少或无效的必要信息：" + ", ".join(missing_fields)
        return {
            **values,
            "missing_fields": missing_fields,
            "next_action": "ask_user",
            "status": "needs_input",
            "error": message,
            "execution_trace": _append_trace(
                state, node="check_information_node", status="needs_input", detail=message
            ),
        }

    return {
        **values,
        "missing_fields": [],
        "next_action": "continue",
        "status": "running",
        "error": "",
        "execution_trace": _append_trace(
            state, node="check_information_node", status="success", detail="必要信息完整"
        ),
    }


def device_status_node(state: AgentState) -> dict[str, Any]:
    return _run_readonly_tool(
        state,
        node="device_status_node",
        state_field="device_status",
        tool=get_device_status,
    )


def maintenance_history_node(state: AgentState) -> dict[str, Any]:
    return _run_readonly_tool(
        state,
        node="maintenance_history_node",
        state_field="maintenance_history",
        tool=query_maintenance_history,
    )


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """调用 RAG，并保留 answered/rejected/error 三种业务语义。"""

    try:
        result = rag_client.retrieve(
            str(state.get("user_query") or ""),
            str(state.get("knowledge_base_id") or ""),
            str(state.get("device_model") or ""),
            history=state.get("history") or [],
            history_summary=state.get("history_summary"),
        )
    except Exception as exc:
        return _error_update(
            state,
            node="retrieve_node",
            code=RAG_EXECUTION_ERROR,
            message=f"RAG Client 调用异常：{exc}",
            rag_status="error",
            rag_answer="",
            rag_request_id="",
            rag_decision={},
            sources=[],
            degraded=False,
        )

    if not isinstance(result, dict):
        return _error_update(
            state,
            node="retrieve_node",
            code=RAG_EXECUTION_ERROR,
            message="RAG Client 返回格式无效",
            rag_status="error",
            rag_answer="",
            rag_request_id="",
            rag_decision={},
            sources=[],
            degraded=False,
        )

    rag_request_id = str(result.get("rag_request_id") or "")
    if result.get("success") is not True:
        code, message, retryable = _read_error(
            result.get("error"), RAG_EXECUTION_ERROR, "Industrial RAG 调用失败"
        )
        return _error_update(
            state,
            node="retrieve_node",
            code=code,
            message=message,
            retryable=retryable,
            rag_status="error",
            rag_answer="",
            rag_request_id=rag_request_id,
            rag_decision={},
            sources=[],
            degraded=False,
        )

    business_status = result.get("status")
    sources = result.get("sources") if isinstance(result.get("sources"), list) else []
    common = {
        "rag_status": business_status,
        "rag_answer": str(result.get("answer") or "").strip(),
        "rag_request_id": rag_request_id,
        "rag_decision": result.get("decision")
        if isinstance(result.get("decision"), dict)
        else {},
        "sources": sources,
        "degraded": bool(result.get("degraded", False)),
        "error": "",
    }

    if business_status == "answered":
        return {
            **common,
            "status": "running",
            "execution_trace": _append_trace(
                state,
                node="retrieve_node",
                status="success",
                detail=f"RAG answered，sources={len(sources)}",
            ),
        }

    if business_status == "rejected":
        return {
            **common,
            "sources": [],
            "status": "insufficient_evidence",
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state,
                node="retrieve_node",
                status="rejected",
                detail="RAG 证据门禁拒绝生成",
            ),
        }

    return _error_update(
        state,
        node="retrieve_node",
        code=RAG_EXECUTION_ERROR,
        message=f"未知 RAG 业务状态：{business_status!r}",
        rag_status="error",
        rag_answer="",
        rag_request_id=rag_request_id,
        rag_decision={},
        sources=[],
        degraded=False,
    )


def _device_summary(data: dict[str, Any]) -> str:
    if not data:
        return "未取得实时设备状态。"
    parts = [f"设备 {data.get('device_id', '未知')} 当前状态：{data.get('status', '未知')}。"]
    if data.get("temperature") is not None:
        parts.append(f"温度：{data['temperature']}。")
    if data.get("vibration") is not None:
        parts.append(f"振动：{data['vibration']}。")
    if data.get("running") is not None:
        parts.append(f"运行标志：{data['running']}。")
    return "".join(parts)


def _maintenance_summary(data: dict[str, Any]) -> str:
    if not data:
        return "未取得维修历史。"
    count = data.get("record_count", 0)
    return f"共查询到 {count} 条维修记录。{data.get('interpretation', '')}".strip()


def diagnosis_node(state: AgentState) -> dict[str, Any]:
    """聚合实时状态、维修历史和有证据的 RAG 回答，不再生成固定模拟诊断。"""

    rag_status = state.get("rag_status", "not_called")
    rag_answer = str(state.get("rag_answer") or "").strip()

    if rag_status == "rejected":
        diagnosis = rag_answer or "知识库证据不足，当前不能给出可靠诊断。"
        return {
            "diagnosis": diagnosis,
            "status": "insufficient_evidence",
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state, node="diagnosis_node", status="rejected", detail="证据不足，停止综合诊断"
            ),
        }

    if rag_status == "error":
        diagnosis = "知识库服务调用失败，当前不能完成有依据的综合诊断。"
        return {
            "diagnosis": diagnosis,
            "status": "failed",
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state, node="diagnosis_node", status="failed", detail="RAG error"
            ),
        }

    sections = [
        "【实时状态】\n" + _device_summary(state.get("device_status") or {}),
        "【维修历史】\n" + _maintenance_summary(state.get("maintenance_history") or {}),
    ]
    if rag_status == "answered" and rag_answer:
        sections.append(
            "【知识库参考】\n"
            "以下内容用于解释潜在原因和安全排查要求，"
            "不代表当前设备已经发生对应故障。\n"
            + rag_answer
        )
    else:
        sections.append("【知识库参考】\n尚未取得知识库诊断结果。")

    diagnosis = "\n\n".join(sections)
    return {
        "diagnosis": diagnosis,
        "status": "running",
        "next_action": "risk_check",
        "error": "",
        "execution_trace": _append_trace(
            state, node="diagnosis_node", status="success", detail="完成三类证据聚合"
        ),
    }


def risk_check_node(state: AgentState) -> dict[str, Any]:
    """根据结构化事实分级，不使用手册通用危险词直接判定当前风险。"""

    if state.get("status") in {"failed", "insufficient_evidence", "needs_input"}:
        return {
            "risk_level": "unknown",
            "human_review_required": False,
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state, node="risk_check_node", status="skipped", detail="无完整诊断可供分级"
            ),
        }

    device_status = state.get("device_status") or {}
    maintenance = state.get("maintenance_history") or {}
    diagnosis = str(state.get("diagnosis") or "")
    query = str(state.get("user_query") or "").lower()

    if not isinstance(device_status, dict) or not device_status:
        basis = ["未取得实时设备状态，不能可靠判断当前风险"]
        return {
            "diagnosis": _append_risk_basis(diagnosis, basis),
            "risk_level": "unknown",
            "human_review_required": False,
            "status": "insufficient_evidence",
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state,
                node="risk_check_node",
                status="success",
                detail="risk_level=unknown; basis=缺少实时设备状态",
            ),
        }

    status_text = str(device_status.get("status") or "").strip().lower()
    temperature = _to_float(device_status.get("temperature"))
    signal_text = _device_signal_text(device_status)
    maintenance_issue = _has_unresolved_maintenance_issue(maintenance)
    unsafe_control_requested = _contains_any(
        query,
        tuple(word.lower() for word in UNSAFE_CONTROL_KEYWORDS),
    )
    critical_signal = _contains_any(
        signal_text,
        tuple(word.lower() for word in CRITICAL_SIGNAL_KEYWORDS),
    )
    high_signal = _contains_any(
        signal_text,
        tuple(word.lower() for word in HIGH_SIGNAL_KEYWORDS),
    )

    basis: list[str] = []
    if unsafe_control_requested:
        risk_level = "high"
        basis.append("请求涉及停机、断电、PLC 写入或保护参数变更等受控操作")
    elif (
        status_text in CRITICAL_DEVICE_STATUSES
        or critical_signal
        or (temperature is not None and temperature >= TEMPERATURE_CRITICAL_C)
    ):
        risk_level = "critical"
        if status_text in CRITICAL_DEVICE_STATUSES:
            basis.append(f"设备状态为 {status_text}")
        if critical_signal:
            basis.append("当前报警或故障字段包含紧急危险信号")
        if temperature is not None and temperature >= TEMPERATURE_CRITICAL_C:
            basis.append(
                f"当前温度 {temperature:g}℃达到严重风险阈值 "
                f"{TEMPERATURE_CRITICAL_C:g}℃"
            )
    elif (
        status_text in HIGH_DEVICE_STATUSES
        or high_signal
        or (temperature is not None and temperature >= TEMPERATURE_HIGH_C)
    ):
        risk_level = "high"
        if status_text in HIGH_DEVICE_STATUSES:
            basis.append(f"设备状态为 {status_text}")
        if high_signal:
            basis.append("当前报警或故障字段包含明确异常信号")
        if temperature is not None and temperature >= TEMPERATURE_HIGH_C:
            basis.append(
                f"当前温度 {temperature:g}℃达到高风险阈值 "
                f"{TEMPERATURE_HIGH_C:g}℃"
            )
    elif (
        status_text in MEDIUM_DEVICE_STATUSES
        or maintenance_issue
        or (temperature is not None and temperature >= TEMPERATURE_ATTENTION_C)
    ):
        risk_level = "medium"
        if status_text in MEDIUM_DEVICE_STATUSES:
            basis.append(f"设备状态为 {status_text}，需要进一步检查")
        if maintenance_issue:
            basis.append("维修历史中存在尚未解决的异常记录")
        if temperature is not None and temperature >= TEMPERATURE_ATTENTION_C:
            basis.append(
                f"当前温度 {temperature:g}℃达到关注阈值 "
                f"{TEMPERATURE_ATTENTION_C:g}℃"
            )
    elif status_text in NORMAL_DEVICE_STATUSES:
        if (
            temperature is None
            and _contains_any(query, ("温度", "过热", "过温"))
        ):
            risk_level = "unknown"
            basis.append("设备状态正常，但缺少温度数据，不能判断过热风险")
        else:
            risk_level = "low"
            basis.append(f"设备状态为 {status_text}")
            if temperature is not None:
                basis.append(
                    f"当前温度 {temperature:g}℃低于关注阈值 "
                    f"{TEMPERATURE_ATTENTION_C:g}℃"
                )
            if not signal_text:
                basis.append("未发现有效报警或故障码")
            if not maintenance_issue:
                basis.append("维修历史未发现尚未解决的异常")
    else:
        risk_level = "unknown"
        basis.append(f"设备状态 {status_text or '未知'} 缺少明确的分级规则")

    human_review_required = risk_level in {"high", "critical"}
    if human_review_required:
        result_status = "human_review_required"
    elif risk_level == "unknown":
        result_status = "insufficient_evidence"
    else:
        result_status = "running"
    diagnosis = _append_risk_basis(diagnosis, basis)

    return {
        "diagnosis": diagnosis,
        "risk_level": risk_level,
        "human_review_required": human_review_required,
        "status": result_status,
        "next_action": "human_review" if human_review_required else "final_answer",
        "execution_trace": _append_trace(
            state,
            node="risk_check_node",
            status="success",
            detail=f"risk_level={risk_level}; basis={'；'.join(basis)}",
        ),
    }


def report_node(state: AgentState) -> dict[str, Any]:
    """按请求生成结构化诊断报告；未请求时无副作用跳过。"""

    if not state.get("create_report", False):
        return {
            "next_action": "final_answer",
            "execution_trace": _append_trace(
                state, node="report_node", status="skipped", detail="create_report=false"
            ),
        }

    device_id = str(state.get("device_id") or "").strip().upper()
    diagnosis = str(state.get("diagnosis") or "").strip()
    if not device_id or not diagnosis:
        return _error_update(
            state,
            node="report_node",
            code=REPORT_GENERATION_ERROR,
            message="生成诊断报告需要 device_id 和 diagnosis",
        )

    findings = [diagnosis]
    recommendations = (
        ["保持设备隔离，等待具备权限的工程师现场复核；Agent 不执行控制操作。"]
        if state.get("human_review_required")
        else ["按照知识库建议完成检查，并记录复测数据。"]
    )
    severity = str(state.get("risk_level") or "low")
    if severity not in {"low", "medium", "high", "critical"}:
        severity = "low"
    try:
        result = create_diagnostic_report(
            device_id=device_id,
            summary=diagnosis,
            severity=severity,
            findings=findings,
            recommendations=recommendations,
        )
    except Exception as exc:
        return _error_update(
            state,
            node="report_node",
            code=REPORT_GENERATION_ERROR,
            message=f"诊断报告生成异常：{exc}",
        )

    if not isinstance(result, dict) or result.get("success") is not True:
        error = result.get("error") if isinstance(result, dict) else None
        code, message, retryable = _read_error(
            error, REPORT_GENERATION_ERROR, "诊断报告生成失败"
        )
        return _error_update(
            state,
            node="report_node",
            code=code,
            message=message,
            retryable=retryable,
        )

    report = result.get("data")
    if not isinstance(report, dict):
        return _error_update(
            state,
            node="report_node",
            code=TOOL_INVALID_RESPONSE,
            message="报告工具成功结果缺少 data 对象",
        )

    return {
        "report": report,
        "next_action": "final_answer",
        "error": "",
        "execution_trace": _append_trace(
            state, node="report_node", status="success", detail=f"report_id={report.get('report_id', '')}"
        ),
    }


def _source_text(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = []
    for index, source in enumerate(sources, start=1):
        location = "，".join(
            item
            for item in (
                f"页 {source.get('page')}" if source.get("page") else "",
                str(source.get("section") or ""),
            )
            if item
        )
        lines.append(f"[{index}] {source.get('document', '未知文档')}" + (f"（{location}）" if location else ""))
    return "\n".join(lines)


def _maintenance_detail(data: dict[str, Any]) -> str:
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list) or not records:
        return _maintenance_summary(data)
    shown = [json.dumps(item, ensure_ascii=False) for item in records[:5]]
    suffix = f"\n其余 {len(records) - 5} 条未展开。" if len(records) > 5 else ""
    return _maintenance_summary(data) + "\n" + "\n".join(shown) + suffix


def final_answer_node(state: AgentState) -> dict[str, Any]:
    """将节点结果转换成可直接由 API 返回给用户的最终文本。"""

    intent = state.get("intent", "unknown")
    sources = state.get("sources") or []

    if state.get("status") == "failed":
        answer = "本次请求未能完成：" + str(state.get("error") or "内部服务异常")
        final_status = "failed"
    elif intent == "device_status":
        answer = _device_summary(state.get("device_status") or {})
        final_status = "completed"
    elif intent == "maintenance_history":
        answer = _maintenance_detail(state.get("maintenance_history") or {})
        final_status = "completed"
    elif intent == "knowledge_query":
        answer = str(state.get("rag_answer") or "知识库未返回答案。")
        final_status = (
            "insufficient_evidence"
            if state.get("rag_status") == "rejected"
            else "completed"
        )
    elif intent == "diagnosis":
        answer = str(state.get("diagnosis") or "未生成诊断结果。")
        risk_level = state.get("risk_level", "unknown")
        answer += f"\n\n【风险等级】{risk_level}"
        final_status = (
            "insufficient_evidence"
            if state.get("status") == "insufficient_evidence"
            else "completed"
        )
    else:
        answer = str(state.get("error") or "请补充具体的查询或诊断需求。")
        final_status = "needs_input"

    citations = _source_text(sources)
    if citations:
        answer += "\n\n【证据来源】\n" + citations
    if state.get("degraded"):
        answer += "\n\n提示：本次检索发生降级，结果需结合现场数据复核。"

    if state.get("human_review_required"):
        final_status = "human_review_required"
        next_action = "human_review"
    else:
        next_action = "completed"

    return {
        "final_answer": answer,
        "status": final_status,
        "next_action": next_action,
        "execution_trace": _append_trace(
            state, node="final_answer_node", status="success", detail=f"status={final_status}"
        ),
    }


def human_review_node(state: AgentState) -> dict[str, Any]:
    base = str(state.get("final_answer") or state.get("diagnosis") or "检测到高风险内容。")
    warning = (
        "\n\n【人工复核要求】该结果涉及高风险工况或操作建议。"
        "Agent 不会自动执行停机、断电、PLC 写入或保护参数变更；"
        "请由具备权限的工程师依据现场安全规程确认。"
    )
    return {
        "final_answer": base + warning,
        "status": "human_review_required",
        "human_review_required": True,
        "next_action": "awaiting_human_review",
        "execution_trace": _append_trace(
            state, node="human_review_node", status="waiting", detail="等待授权人员复核"
        ),
    }


def ask_user_node(state: AgentState) -> dict[str, Any]:
    missing = list(state.get("missing_fields") or [])
    labels = {
        "intent": "具体需求（状态、维修记录、知识查询或故障诊断）",
        "device_id": "设备编号（格式 DEVICE-XXX）",
        "device_model": "设备型号",
        "knowledge_base_id": "知识库编号",
    }
    requested = "、".join(labels.get(item, item) for item in missing)
    message = f"请补充：{requested}。" if requested else "请补充完成请求所需的信息。"
    return {
        "final_answer": message,
        "status": "needs_input",
        "next_action": "ask_user",
        "error": str(state.get("error") or message),
        "execution_trace": _append_trace(
            state, node="ask_user_node", status="needs_input", detail=message
        ),
    }


NODE_REGISTRY = {
    "intent_node": intent_node,
    "check_information_node": check_information_node,
    "device_status_node": device_status_node,
    "maintenance_history_node": maintenance_history_node,
    "retrieve_node": retrieve_node,
    "diagnosis_node": diagnosis_node,
    "risk_check_node": risk_check_node,
    "report_node": report_node,
    "final_answer_node": final_answer_node,
    "human_review_node": human_review_node,
    "ask_user_node": ask_user_node,
}
