"""
Day 20 - Stage 7
RAG V1 主动故障注入验收

严格只验证故障行为，不修改 RAG 算法。

Case:
F01  上传 JPG              -> HTTP 400 / Unsupported File
F02  重复 PDF              -> status=duplicate
F03  G120C KB 查询 E300    -> 拒绝（设备型号不匹配）
F04  知识库无答案           -> rejected + llm_called=false
F05  Reranker 关闭          -> degraded + hybrid_fallback + /chat 可用
F06  LLM Timeout           -> HTTP 504 + LLM_TIMEOUT + request_id

推荐执行：

1. Reranker 正常时：
   python tests/test_rag_v1_faults.py --case baseline

2. 关闭 Reranker 后：
   python tests/test_rag_v1_faults.py --case F05

3. F06 为进程内确定性故障注入，不需要真的等待 LLM 超时：
   python tests/test_rag_v1_faults.py --case F06
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_BASE_URL = os.getenv(
    "RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

KNOWLEDGE_BASE_FILE = BASE_DIR / "data" / "knowledge_base.json"
DOCUMENT_FILE = BASE_DIR / "data" / "documents.json"

HTTP_TIMEOUT = 180


# ============================================================
# Helpers
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def pretty(data: Any) -> None:
    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def response_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"响应不是合法 JSON：HTTP={response.status_code}, body={response.text[:500]!r}"
        ) from exc

    require(
        isinstance(data, dict),
        f"响应 JSON 必须是 dict，实际={type(data).__name__}",
    )
    return data


def load_ready_kb(
    knowledge_base_id: str | None,
    expected_device_model: str = "G120C",
) -> dict[str, Any]:
    """从 data/knowledge_base.json 中定位 READY Knowledge Base。"""
    if not KNOWLEDGE_BASE_FILE.exists():
        raise FileNotFoundError(
            f"未找到 Knowledge Base 元数据：{KNOWLEDGE_BASE_FILE}"
        )

    with KNOWLEDGE_BASE_FILE.open("r", encoding="utf-8") as f:
        items = json.load(f)

    require(
        isinstance(items, list),
        "data/knowledge_base.json 顶层必须是 list",
    )

    if knowledge_base_id:
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("knowledge_base_id") == knowledge_base_id
            ):
                require(
                    item.get("status") == "ready",
                    f"Knowledge Base 不是 ready：{item.get('status')}",
                )
                return item

        raise AssertionError(
            f"没有找到 Knowledge Base：{knowledge_base_id}"
        )

    expected = expected_device_model.strip().upper()

    # 优先选择最后一个 READY G120C KB。
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ready":
            continue
        if str(item.get("device_model", "")).strip().upper() != expected:
            continue
        return item

    raise AssertionError(
        f"没有找到 READY 的 {expected_device_model} Knowledge Base；"
        "请通过 --knowledge-base-id 显式指定。"
    )


def resolve_duplicate_pdf(
    kb: dict[str, Any],
    explicit_pdf: str | None,
) -> Path:
    """F02 优先使用 --pdf，否则从 documents.json 找 KB 已登记 PDF。"""
    if explicit_pdf:
        path = Path(explicit_pdf).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        require(path.is_file(), f"PDF 不存在：{path}")
        require(path.suffix.lower() == ".pdf", f"F02 必须使用 PDF：{path}")
        return path

    if not DOCUMENT_FILE.exists():
        raise FileNotFoundError(
            f"没有找到 {DOCUMENT_FILE}；请为 F02 提供 --pdf。"
        )

    with DOCUMENT_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    require(isinstance(documents, list), "data/documents.json 顶层必须是 list")

    wanted_ids = set(kb.get("document_ids") or [])
    path_keys = (
        "path",
        "file_path",
        "saved_path",
        "storage_path",
        "upload_path",
    )

    for doc in documents:
        if not isinstance(doc, dict):
            continue
        if wanted_ids and doc.get("document_id") not in wanted_ids:
            continue

        for key in path_keys:
            raw = doc.get(key)
            if not raw:
                continue

            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                candidate = (BASE_DIR / candidate).resolve()

            if candidate.is_file() and candidate.suffix.lower() == ".pdf":
                return candidate

    raise AssertionError(
        "无法从 data/documents.json 自动定位 KB 的 PDF；"
        "请使用 --pdf /absolute/path/manual.pdf"
    )


def chat_request(
    *,
    base_url: str,
    question: str,
    knowledge_base_id: str,
    device_model: str,
) -> requests.Response:
    return requests.post(
        f"{base_url}/chat",
        json={
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "device_model": device_model,
            "history": [],
            "history_summary": None,
        },
        timeout=HTTP_TIMEOUT,
    )


def upload_request(
    *,
    base_url: str,
    path: Path,
    device_model: str = "G120C",
    document_type: str = "manual",
    version: str = "stage7-fault-test",
) -> requests.Response:
    with path.open("rb") as f:
        return requests.post(
            f"{base_url}/documents/upload",
            files={
                "file": (
                    path.name,
                    f,
                    "application/pdf"
                    if path.suffix.lower() == ".pdf"
                    else "image/jpeg",
                )
            },
            data={
                "device_model": device_model,
                "document_type": document_type,
                "version": version,
            },
            timeout=60,
        )


# ============================================================
# F01 - Unsupported File
# ============================================================

def test_f01_upload_jpg(base_url: str) -> None:
    section("F01 | Upload JPG -> 400 / Unsupported File")

    with tempfile.TemporaryDirectory(prefix="day20_f01_") as tmp_dir:
        jpg = Path(tmp_dir) / "fault_injection.jpg"
        jpg.write_bytes(b"\xff\xd8\xff\xe0DAY20_FAULT_INJECTION")

        response = upload_request(
            base_url=base_url,
            path=jpg,
        )

    print("HTTP:", response.status_code)
    data = response_json(response)
    pretty(data)

    require(
        response.status_code == 400,
        f"F01 FAIL：JPG 应返回 HTTP 400，实际={response.status_code}",
    )

    detail = str(data.get("detail", "")).lower()
    require(
        "不支持" in detail
        or "unsupported" in detail
        or ".jpg" in detail,
        f"F01 FAIL：错误信息没有体现 Unsupported File：{detail!r}",
    )

    print("F01 PASS")


# ============================================================
# F02 - Duplicate PDF
# ============================================================

def test_f02_duplicate_pdf(
    base_url: str,
    kb: dict[str, Any],
    explicit_pdf: str | None,
) -> None:
    section("F02 | Duplicate PDF -> status=duplicate")

    pdf = resolve_duplicate_pdf(kb, explicit_pdf)
    print("PDF:", pdf)

    first = upload_request(
        base_url=base_url,
        path=pdf,
        device_model=str(kb.get("device_model") or "G120C"),
    )
    first_data = response_json(first)

    print("First Upload HTTP:", first.status_code)
    pretty(first_data)

    require(
        first.status_code == 200,
        f"F02 FAIL：第一次上传 HTTP={first.status_code}",
    )

    second = upload_request(
        base_url=base_url,
        path=pdf,
        device_model=str(kb.get("device_model") or "G120C"),
    )
    second_data = response_json(second)

    print("Second Upload HTTP:", second.status_code)
    pretty(second_data)

    require(
        second.status_code == 200,
        f"F02 FAIL：Duplicate 当前契约应返回 HTTP 200，实际={second.status_code}",
    )
    require(
        second_data.get("status") == "duplicate",
        f"F02 FAIL：第二次上传 status 应为 duplicate，实际={second_data.get('status')!r}",
    )
    require(
        bool(second_data.get("document_id")),
        "F02 FAIL：Duplicate 响应缺少 document_id",
    )

    # 若第一次本来就是 duplicate，也同样满足幂等性；若第一次 success，
    # 第二次必须返回第一次创建的 document_id。
    if first_data.get("document_id"):
        require(
            second_data.get("document_id") == first_data.get("document_id"),
            "F02 FAIL：重复上传返回了不同 document_id",
        )

    print("F02 PASS")


# ============================================================
# F03 - KB / Device Model Boundary
# ============================================================

def test_f03_wrong_device(
    base_url: str,
    kb: dict[str, Any],
) -> None:
    section("F03 | G120C KB Query E300 -> Reject")

    response = chat_request(
        base_url=base_url,
        question="E300 设备出现故障时应该怎么处理？",
        knowledge_base_id=str(kb["knowledge_base_id"]),
        device_model="E300",
    )

    print("HTTP:", response.status_code)
    data = response_json(response)
    pretty(data)

    require(
        response.status_code == 400,
        f"F03 FAIL：错误设备型号应返回 HTTP 400，实际={response.status_code}",
    )

    detail = str(data.get("detail", ""))
    require(
        "不匹配" in detail
        or "mismatch" in detail.lower(),
        f"F03 FAIL：没有明确指出设备型号不匹配：{detail!r}",
    )

    print("F03 PASS")


# ============================================================
# F04 - No Evidence / No LLM
# ============================================================

def test_f04_no_evidence(
    base_url: str,
    kb: dict[str, Any],
) -> None:
    section("F04 | No Evidence -> Reject + LLM Not Called")

    question = (
        "请依据 G120C 官方说明书解释量子纠缠冷却反应堆的 QX-99999 故障，"
        "并给出核燃料更换步骤。"
    )

    response = chat_request(
        base_url=base_url,
        question=question,
        knowledge_base_id=str(kb["knowledge_base_id"]),
        device_model=str(kb["device_model"]),
    )

    print("HTTP:", response.status_code)
    data = response_json(response)
    pretty(data)

    require(
        response.status_code == 200,
        "F04 FAIL：Evidence 不足属于正常业务拒答，预期 HTTP 200",
    )
    require(
        data.get("status") == "rejected",
        f"F04 FAIL：status 应为 rejected，实际={data.get('status')!r}",
    )
    require(
        data.get("llm_called") is False,
        "F04 FAIL：Evidence 不足时仍然调用了 LLM",
    )

    decision = data.get("decision") or {}
    require(
        decision.get("evidence_sufficient") is False,
        "F04 FAIL：decision.evidence_sufficient 应为 false",
    )
    require(
        decision.get("allow_llm") is False,
        "F04 FAIL：decision.allow_llm 应为 false",
    )

    print("F04 PASS")


# ============================================================
# F05 - Reranker Down / Hybrid Fallback
# ============================================================

def test_f05_reranker_fallback(
    base_url: str,
    kb: dict[str, Any],
) -> None:
    section("F05 | Reranker Down -> Hybrid Fallback")

    health_response = requests.get(
        f"{base_url}/health",
        timeout=15,
    )
    health = response_json(health_response)

    print("Health HTTP:", health_response.status_code)
    pretty(health)

    require(
        health_response.status_code == 200,
        f"F05 FAIL：业务 /health 不可用，HTTP={health_response.status_code}",
    )
    require(
        health.get("status") == "degraded",
        "F05 前置条件不满足：请先关闭 Reranker(6009)，"
        f"当前 health.status={health.get('status')!r}",
    )

    components = health.get("components") or {}
    reranker_health = components.get("reranker") or {}
    require(
        reranker_health.get("status") == "unhealthy",
        "F05 前置条件不满足：Reranker 仍然是 healthy，请先关闭 6009。",
    )

    response = chat_request(
        base_url=base_url,
        question="变频器温度太高时应该检查哪些方面？",
        knowledge_base_id=str(kb["knowledge_base_id"]),
        device_model=str(kb["device_model"]),
    )

    print("Chat HTTP:", response.status_code)
    data = response_json(response)
    pretty(data)

    require(
        response.status_code == 200,
        f"F05 FAIL：Reranker Down 后 /chat 仍应可用，实际={response.status_code}",
    )
    require(
        data.get("rerank_executed") is False,
        "F05 FAIL：rerank_executed 应为 false",
    )
    require(
        data.get("degraded") is True,
        "F05 FAIL：degraded 应为 true",
    )
    require(
        data.get("retrieval_mode") == "hybrid_fallback",
        "F05 FAIL：retrieval_mode 应为 hybrid_fallback，"
        f"实际={data.get('retrieval_mode')!r}",
    )
    require(
        bool(data.get("rerank_error")),
        "F05 FAIL：fallback 响应应保留 rerank_error，便于排障",
    )
    require(
        data.get("status") == "answered",
        "F05 FAIL：已选择已知可回答问题，Fallback 后应继续正常回答",
    )
    require(
        data.get("llm_called") is True,
        "F05 FAIL：Fallback 后证据充分时应继续调用 LLM",
    )
    require(
        bool(data.get("request_id")),
        "F05 FAIL：正常 /chat 响应应包含 request_id",
    )

    print("F05 PASS")
    print("Reranker unavailable -> Fallback to hybrid ranking -> /chat still available")


# ============================================================
# F06 - Deterministic LLM Timeout Injection
# ============================================================

def test_f06_llm_timeout() -> None:
    section("F06 | LLM Timeout -> 504 + request_id")

    # --------------------------------------------------------
    # Part A: requests.Timeout -> LLMTimeoutError
    # --------------------------------------------------------
    import app.services.qa_service as qa_module

    original_post = qa_module.requests.post

    def injected_timeout(*args, **kwargs):
        raise requests.Timeout("Stage7 injected LLM timeout")

    qa_module.requests.post = injected_timeout
    try:
        try:
            qa_module.call_llm(
                "Stage7 timeout injection",
                timeout=0.01,
            )
        except qa_module.LLMTimeoutError as exc:
            print("QA Timeout Mapping: PASS")
            print("Mapped Exception:", exc)
        else:
            raise AssertionError(
                "F06 FAIL：requests.Timeout 没有转换为 LLMTimeoutError"
            )
    finally:
        qa_module.requests.post = original_post

    # --------------------------------------------------------
    # Part B: LLMTimeoutError -> Business /chat 504 + request_id
    # 不调用真实 Retrieval / LLM，直接在业务边界注入超时。
    # --------------------------------------------------------
    import app.main as main_module
    from app.schemas import ChatRequest
    from starlette.requests import Request

    original_answer = main_module.qa_service.answer

    def injected_answer_timeout(*args, **kwargs):
        raise qa_module.LLMTimeoutError(
            "Stage7 injected Gateway / LLM timeout"
        )

    main_module.qa_service.answer = injected_answer_timeout

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat",
        "raw_path": b"/chat",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    http_request = Request(scope)

    try:
        response = asyncio.run(
            main_module.chat(
                ChatRequest(
                    question="F06 timeout injection",
                    knowledge_base_id="kb_fault_injection",
                    device_model="G120C",
                    history=[],
                    history_summary=None,
                ),
                http_request,
            )
        )
    finally:
        main_module.qa_service.answer = original_answer

    require(
        response.status_code == 504,
        f"F06 FAIL：LLM Timeout 应返回 HTTP 504，实际={response.status_code}",
    )

    payload = json.loads(response.body.decode("utf-8"))
    pretty(payload)

    require(
        payload.get("status") == "error",
        "F06 FAIL：status 应为 error",
    )
    require(
        payload.get("error") == "LLM_TIMEOUT",
        f"F06 FAIL：error 应为 LLM_TIMEOUT，实际={payload.get('error')!r}",
    )

    request_id = payload.get("request_id")
    require(
        isinstance(request_id, str) and bool(request_id.strip()),
        "F06 FAIL：错误响应缺少 request_id",
    )
    require(
        response.headers.get("X-Request-ID") == request_id,
        "F06 FAIL：X-Request-ID 与响应体 request_id 不一致",
    )

    print("F06 PASS")
    print("request_id:", request_id)


# ============================================================
# Runner
# ============================================================

def run_case(
    case: str,
    *,
    base_url: str,
    knowledge_base_id: str | None,
    pdf: str | None,
) -> None:
    kb: dict[str, Any] | None = None

    if case in {"F02", "F03", "F04", "F05", "baseline"}:
        kb = load_ready_kb(knowledge_base_id)
        print(
            "Using Knowledge Base:",
            kb["knowledge_base_id"],
            "| Device:",
            kb["device_model"],
        )

    if case == "F01":
        test_f01_upload_jpg(base_url)
    elif case == "F02":
        assert kb is not None
        test_f02_duplicate_pdf(base_url, kb, pdf)
    elif case == "F03":
        assert kb is not None
        test_f03_wrong_device(base_url, kb)
    elif case == "F04":
        assert kb is not None
        test_f04_no_evidence(base_url, kb)
    elif case == "F05":
        assert kb is not None
        test_f05_reranker_fallback(base_url, kb)
    elif case == "F06":
        test_f06_llm_timeout()
    elif case == "baseline":
        assert kb is not None
        test_f01_upload_jpg(base_url)
        test_f02_duplicate_pdf(base_url, kb, pdf)
        test_f03_wrong_device(base_url, kb)
        test_f04_no_evidence(base_url, kb)
    else:
        raise ValueError(f"未知 case：{case}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day20 Stage7 - RAG V1 Fault Injection Acceptance"
    )
    parser.add_argument(
        "--case",
        choices=(
            "baseline",
            "F01",
            "F02",
            "F03",
            "F04",
            "F05",
            "F06",
        ),
        default="baseline",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Industrial RAG API 地址，默认 {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=None,
        help=(
            "READY 的 G120C KB ID；不传时自动读取 "
            "data/knowledge_base.json 中最后一个 READY G120C KB"
        ),
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help=(
            "F02 使用的 PDF；不传时尝试根据 KB document_ids "
            "从 data/documents.json 自动定位"
        ),
    )

    args = parser.parse_args()

    section("Day20 Stage7 | Fault Injection")
    print("Case     :", args.case)
    print("Base URL :", args.base_url)

    run_case(
        args.case,
        base_url=args.base_url.rstrip("/"),
        knowledge_base_id=args.knowledge_base_id,
        pdf=args.pdf,
    )

    print()
    print("=" * 80)
    print(f"DAY20 STAGE7 {args.case}: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
