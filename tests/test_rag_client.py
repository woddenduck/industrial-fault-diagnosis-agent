"""
Industrial RAG Client 直接运行验收程序。

运行方式：

1. 模拟接口验收：

    python tests/test_rag_client.py

2. 同时调用真实 Industrial RAG：

    RAG_REAL_TEST=1 \
    RAG_KB_ID=kb_ab50652fe3a4 \
    python tests/test_rag_client.py

特点：

- 不依赖 pytest；
- 打印每个测试用例的输入和输出；
- 打印每一项检查结果；
- 任意测试失败时进程退出码为 1；
- 所有测试通过时进程退出码为 0。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


# ============================================================
# 添加项目根目录
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import httpx

from agent.rag_client import RAGClient


CHAT_URL = "http://127.0.0.1:8000/chat"
KB_ID = "kb_ab50652fe3a4"


# ============================================================
# 输出辅助
# ============================================================

def pretty(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def show(
    label: str,
    value: Any,
) -> None:
    print(f"\n[{label}]")
    print(pretty(value))


def check(
    condition: bool,
    description: str,
    *,
    actual: Any = None,
) -> None:
    if not condition:
        raise AssertionError(
            f"{description}\n"
            f"实际值：{actual!r}"
        )

    print(
        f"[CHECK PASS] {description}"
    )

    if actual is not None:
        print(
            f"             实际值：{actual!r}"
        )


def check_equal(
    actual: Any,
    expected: Any,
    description: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{description}\n"
            f"期望值：{expected!r}\n"
            f"实际值：{actual!r}"
        )

    print(
        f"[CHECK PASS] {description}"
    )
    print(
        f"             实际值：{actual!r}"
    )


# ============================================================
# 测试运行器
# ============================================================

class AcceptanceRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(
        self,
        case_id: str,
        title: str,
        case_func: Callable[[], None],
    ) -> None:
        print("\n")
        print("=" * 80)
        print(f"{case_id} | {title}")
        print("=" * 80)

        try:
            case_func()

        except Exception as exc:
            self.failed += 1

            print("\n[RESULT] FAIL")
            print(
                f"[ERROR] {type(exc).__name__}: "
                f"{exc}"
            )

            print("\n[TRACEBACK]")
            traceback.print_exc()

        else:
            self.passed += 1
            print("\n[RESULT] PASS")

    def finish(self) -> None:
        total = self.passed + self.failed

        print("\n")
        print("=" * 80)
        print("Industrial RAG Client Acceptance Summary")
        print("=" * 80)
        print(f"Total : {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print("=" * 80)

        if self.failed:
            print("FINAL RESULT: FAIL")
            raise SystemExit(1)

        print("FINAL RESULT: PASS")


# ============================================================
# HTTP Response 构造
# ============================================================

def make_response(
    status_code: int,
    data: dict[str, Any] | None = None,
    *,
    content: bytes | None = None,
) -> httpx.Response:
    request = httpx.Request(
        "POST",
        CHAT_URL,
    )

    if content is not None:
        return httpx.Response(
            status_code,
            content=content,
            request=request,
        )

    return httpx.Response(
        status_code,
        json=data or {},
        request=request,
    )


def make_answered_response() -> httpx.Response:
    return make_response(
        200,
        {
            "status": "answered",
            "answer": (
                "建议检查变频器风扇、"
                "环境温度和电机负载。"
            ),
            "llm_called": True,
            "sources": [
                {
                    "document": (
                        "G120C_op_instr_0226_zh-CHS.pdf"
                    ),
                    "page": "370-371",
                    "section": (
                        "8.23 通过温度监控实现的"
                        "变频器保护"
                    ),
                    "chunk_id": (
                        "G120C_section_0468_00"
                    ),
                }
            ],
            "knowledge_base_id": KB_ID,
            "device_model": "G120C",
            "degraded": False,
            "retrieval_mode": "rerank",
            "rerank_executed": True,
            "rerank_error": None,
            "decision": {
                "evidence_sufficient": True,
                "allow_llm": True,
                "context_status": "ANSWERABLE",
                "reason": "SAFE_CONTEXT_READY",
            },
            "latency": {
                "retrieval_ms": 6800.0,
                "llm_ms": 11600.0,
                "total_ms": 18400.0,
            },
            "request_id": "rag_request_001",
        },
    )


# ============================================================
# C01：正常回答
# ============================================================

def case_answered_response() -> None:
    client = RAGClient(
        max_retries=0,
    )

    captured_payload: dict[str, Any] = {}

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        captured_payload.update(payload)
        return make_answered_response()

    client._post = fake_post

    request_data = {
        "question": "变频器温度太高应该检查什么？",
        "knowledge_base_id": KB_ID,
        "device_model": "g120c",
        "history": [
            {
                "role": "user",
                "content": "设备温度异常",
            }
        ],
    }

    show("Test Input", request_data)

    result = client.retrieve(
        request_data["question"],
        request_data["knowledge_base_id"],
        request_data["device_model"],
        history=request_data["history"],
    )

    show("HTTP Payload", captured_payload)
    show("Client Result", result)

    check(
        result["success"] is True,
        "请求处理成功",
        actual=result["success"],
    )

    check_equal(
        result["status"],
        "answered",
        "业务状态为 answered",
    )

    check(
        result["evidence_sufficient"] is True,
        "检索证据充分",
        actual=result["evidence_sufficient"],
    )

    check(
        result["allow_llm"] is True,
        "安全策略允许 LLM 回答",
        actual=result["allow_llm"],
    )

    check(
        result["llm_called"] is True,
        "RAG 已调用 LLM",
        actual=result["llm_called"],
    )

    check_equal(
        result["rag_request_id"],
        "rag_request_001",
        "保留上游 RAG request_id",
    )

    check_equal(
        captured_payload["device_model"],
        "G120C",
        "设备型号被规范化为大写",
    )

    check_equal(
        len(result["sources"]),
        1,
        "成功保留一条证据来源",
    )

    source = result["sources"][0]

    check(
        bool(source["document"]),
        "Source 包含文档名",
        actual=source["document"],
    )

    check(
        bool(source["page"]),
        "Source 包含页码",
        actual=source["page"],
    )

    check(
        bool(source["section"]),
        "Source 包含章节",
        actual=source["section"],
    )

    check(
        bool(source["chunk_id"]),
        "Source 包含 chunk_id",
        actual=source["chunk_id"],
    )


# ============================================================
# C02：证据不足
# ============================================================

def case_rejected_response() -> None:
    client = RAGClient(
        max_retries=0,
    )

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        return make_response(
            200,
            {
                "status": "rejected",
                "answer": (
                    "当前知识库中没有找到"
                    "足够可靠的证据。"
                ),
                "llm_called": False,
                "sources": [],
                "degraded": False,
                "decision": {
                    "evidence_sufficient": False,
                    "allow_llm": False,
                    "reason": "LOW_RERANK_SCORE",
                },
                "request_id":
                    "rag_request_rejected",
            },
        )

    client._post = fake_post

    result = client.retrieve(
        "量子纠缠冷却反应堆故障",
        KB_ID,
        "G120C",
    )

    show("Client Result", result)

    check(
        result["success"] is True,
        "rejected 属于正常业务结果",
        actual=result["success"],
    )

    check_equal(
        result["status"],
        "rejected",
        "业务状态为 rejected",
    )

    check(
        result["evidence_sufficient"] is False,
        "证据不足",
        actual=result["evidence_sufficient"],
    )

    check(
        result["allow_llm"] is False,
        "禁止 LLM 无证据回答",
        actual=result["allow_llm"],
    )

    check(
        result["llm_called"] is False,
        "RAG 没有调用 LLM",
        actual=result["llm_called"],
    )

    check_equal(
        result["sources"],
        [],
        "证据不足时不返回弱引用",
    )

    check(
        result["error"] is None,
        "业务拒答不是系统错误",
        actual=result["error"],
    )


# ============================================================
# C03-C06：HTTP错误映射
# ============================================================

def case_http_error(
    *,
    status_code: int,
    body: dict[str, Any],
    expected_code: str,
) -> None:
    client = RAGClient(
        max_retries=0,
    )

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        return make_response(
            status_code,
            body,
        )

    client._post = fake_post

    result = client.retrieve(
        "测试问题",
        KB_ID,
        "G120C",
    )

    show(
        "Simulated HTTP Status",
        status_code,
    )
    show("Client Result", result)

    check(
        result["success"] is False,
        "HTTP错误转换为失败结果",
        actual=result["success"],
    )

    check_equal(
        result["status"],
        "error",
        "统一业务状态为 error",
    )

    check_equal(
        result["error"]["code"],
        expected_code,
        "错误码映射正确",
    )

    check_equal(
        result["http_status"],
        status_code,
        "保留原始 HTTP 状态码",
    )


# ============================================================
# C07：502重试成功
# ============================================================

def case_retry_then_success() -> None:
    client = RAGClient(
        max_retries=1,
    )

    responses = iter(
        [
            make_response(
                502,
                {
                    "status": "error",
                    "message": (
                        "上游服务暂时不可用"
                    ),
                },
            ),
            make_answered_response(),
        ]
    )

    call_count = 0

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        nonlocal call_count

        call_count += 1
        response = next(responses)

        print(
            f"[HTTP ATTEMPT {call_count}] "
            f"status={response.status_code}"
        )

        return response

    client._post = fake_post

    result = client.retrieve(
        "温度太高怎么办？",
        KB_ID,
        "G120C",
    )

    show("Client Result", result)

    check_equal(
        call_count,
        2,
        "502 后只重试一次",
    )

    check(
        result["success"] is True,
        "第二次请求成功",
        actual=result["success"],
    )

    check_equal(
        result["attempts"],
        2,
        "记录两次请求尝试",
    )


# ============================================================
# C08：连接失败
# ============================================================

def case_connection_failure() -> None:
    client = RAGClient(
        max_retries=1,
    )

    request = httpx.Request(
        "POST",
        CHAT_URL,
    )

    call_count = 0

    def broken_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        nonlocal call_count

        call_count += 1

        print(
            f"[HTTP ATTEMPT {call_count}] "
            "connection refused"
        )

        raise httpx.ConnectError(
            "connection refused",
            request=request,
        )

    client._post = broken_post

    result = client.retrieve(
        "测试问题",
        KB_ID,
        "G120C",
    )

    show("Client Result", result)

    check_equal(
        call_count,
        2,
        "连接失败后重试一次",
    )

    check(
        result["success"] is False,
        "最终返回失败",
        actual=result["success"],
    )

    check_equal(
        result["error"]["code"],
        "RAG_UNAVAILABLE",
        "返回 RAG_UNAVAILABLE",
    )

    check(
        result["error"]["retryable"] is True,
        "连接失败标记为可重试",
        actual=result["error"]["retryable"],
    )


# ============================================================
# C09：非法JSON
# ============================================================

def case_invalid_json() -> None:
    client = RAGClient(
        max_retries=0,
    )

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        return make_response(
            200,
            content=b"not-json",
        )

    client._post = fake_post

    result = client.retrieve(
        "测试问题",
        KB_ID,
        "G120C",
    )

    show("Client Result", result)

    check(
        result["success"] is False,
        "非法JSON被拒绝",
        actual=result["success"],
    )

    check_equal(
        result["error"]["code"],
        "RAG_INVALID_RESPONSE",
        "返回稳定的响应格式错误码",
    )


# ============================================================
# C10：请求字段缺失
# ============================================================

def case_invalid_request() -> None:
    client = RAGClient(
        max_retries=1,
    )

    call_count = 0

    def fake_post(
        payload: dict[str, Any],
    ) -> httpx.Response:
        nonlocal call_count

        call_count += 1
        return make_answered_response()

    client._post = fake_post

    result = client.retrieve(
        "",
        "",
        "",
    )

    show("Client Result", result)

    check_equal(
        call_count,
        0,
        "字段缺失时不发送HTTP请求",
    )

    check(
        result["success"] is False,
        "缺少字段返回失败",
        actual=result["success"],
    )

    check_equal(
        result["error"]["code"],
        "INVALID_RAG_REQUEST",
        "返回请求校验错误码",
    )

    check_equal(
        result["attempts"],
        0,
        "HTTP请求次数为零",
    )


# ============================================================
# R01：真实RAG调用
# ============================================================

def case_real_rag_call() -> None:
    knowledge_base_id = os.getenv(
        "RAG_KB_ID",
        "",
    ).strip()

    if not knowledge_base_id:
        raise ValueError(
            "真实测试缺少 RAG_KB_ID 环境变量"
        )

    question = os.getenv(
        "RAG_TEST_QUESTION",
        "变频器温度太高时应该检查哪些方面？",
    )

    client = RAGClient()

    request_data = {
        "question": question,
        "knowledge_base_id":
            knowledge_base_id,
        "device_model": "G120C",
    }

    show("Real RAG Request", request_data)

    result = client.retrieve(
        question,
        knowledge_base_id,
        "G120C",
    )

    show("Real RAG Result", result)

    check(
        result["success"] is True,
        "真实RAG正确处理请求",
        actual=result["success"],
    )

    check_equal(
        result["status"],
        "answered",
        "已知问题返回 answered",
    )

    check(
        bool(result["answer"]),
        "真实回答不为空",
        actual=result["answer"][:100],
    )

    check(
        bool(result["sources"]),
        "真实回答包含引用来源",
        actual=len(result["sources"]),
    )

    check(
        bool(result["rag_request_id"]),
        "真实回答包含 RAG request_id",
        actual=result["rag_request_id"],
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    runner = AcceptanceRunner()

    runner.run(
        "C01",
        "正常 answered 响应",
        case_answered_response,
    )

    runner.run(
        "C02",
        "证据不足 rejected 响应",
        case_rejected_response,
    )

    runner.run(
        "C03",
        "HTTP 400 请求错误",
        lambda: case_http_error(
            status_code=400,
            body={
                "detail": (
                    "Knowledge Base 与设备型号不匹配"
                )
            },
            expected_code="RAG_INVALID_REQUEST",
        ),
    )

    runner.run(
        "C04",
        "HTTP 404 Knowledge Base不存在",
        lambda: case_http_error(
            status_code=404,
            body={
                "detail": "Knowledge Base 不存在"
            },
            expected_code=(
                "KNOWLEDGE_BASE_NOT_FOUND"
            ),
        ),
    )

    runner.run(
        "C05",
        "HTTP 409 Knowledge Base未就绪",
        lambda: case_http_error(
            status_code=409,
            body={
                "detail": (
                    "Knowledge Base 当前不可用于检索"
                )
            },
            expected_code=(
                "KNOWLEDGE_BASE_NOT_READY"
            ),
        ),
    )

    runner.run(
        "C06",
        "HTTP 504 LLM超时",
        lambda: case_http_error(
            status_code=504,
            body={
                "status": "error",
                "error": "LLM_TIMEOUT",
                "message": "LLM 服务响应超时",
                "request_id": "rag_timeout_id",
            },
            expected_code=(
                "RAG_UPSTREAM_TIMEOUT"
            ),
        ),
    )

    runner.run(
        "C07",
        "HTTP 502后有限重试",
        case_retry_then_success,
    )

    runner.run(
        "C08",
        "RAG连接失败",
        case_connection_failure,
    )

    runner.run(
        "C09",
        "RAG返回非法JSON",
        case_invalid_json,
    )

    runner.run(
        "C10",
        "请求字段缺失",
        case_invalid_request,
    )

    real_test_enabled = (
        os.getenv(
            "RAG_REAL_TEST",
            "0",
        )
        == "1"
    )

    if real_test_enabled:
        runner.run(
            "R01",
            "真实 Industrial RAG 调用",
            case_real_rag_call,
        )
    else:
        print("\n")
        print("=" * 80)
        print("R01 | 真实 Industrial RAG 调用")
        print("=" * 80)
        print(
            "[SKIPPED] RAG_REAL_TEST != 1"
        )
        print(
            "如需运行真实测试，请设置："
        )
        print(
            "RAG_REAL_TEST=1 "
            "RAG_KB_ID=kb_ab50652fe3a4"
        )

    runner.finish()


if __name__ == "__main__":
    main()