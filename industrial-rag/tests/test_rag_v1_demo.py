"""
Day 20 - Stage 9
Industrial RAG V1 End-to-End Demo Acceptance Test

目标：
1. 检查 industrial-rag API 健康状态
2. 检查运行中的模型信息
3. 上传真实 G120C PDF
4. 创建并构建 Knowledge Base
5. 使用真实 KB 查询 “F30021怎么处理？”
6. 验证结构化回答、LLM 调用、来源引用
7. 使用不存在的 KB 验证受控拒绝
8. 输出最终 PASS / FAIL 汇总

推荐运行：
    python tests/test_rag_v1_demo.py \
        --pdf /path/to/G120C_op_instr_0226_zh-CHS.pdf

也支持环境变量：
    export RAG_DEMO_PDF=/path/to/G120C_op_instr_0226_zh-CHS.pdf
    export RAG_API_BASE_URL=http://127.0.0.1:8000
    python tests/test_rag_v1_demo.py

说明：
- 这是“真实 HTTP 端到端验收”，不会 import app.main，也不会 mock Service。
- 测试前应已经启动：
    LLM -> Embedding -> Reranker -> Gateway -> industrial-rag API
- 如果上传接口把重复 PDF 返回为 duplicate，但同时返回已有 document_id，
  本脚本仍允许继续，这样 Demo 可以重复执行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


# ============================================================
# 1. Default Configuration
# ============================================================

DEFAULT_BASE_URL = os.getenv(
    "RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_PDF = os.getenv("RAG_DEMO_PDF")

DEFAULT_DEVICE_MODEL = "G120C"
DEFAULT_DOCUMENT_TYPE = "operation_manual"
DEFAULT_VERSION = "V1"
DEFAULT_KB_NAME = "G120C故障知识库"
DEFAULT_QUESTION = "F30021怎么处理？"

HTTP_TIMEOUT = 30
UPLOAD_TIMEOUT = 120
KB_BUILD_TIMEOUT = 3600
CHAT_TIMEOUT = 300


# ============================================================
# 2. Console Helpers
# ============================================================

LINE = "=" * 88
SUB_LINE = "-" * 88


def title(text: str) -> None:
    print()
    print(LINE)
    print(text)
    print(LINE)


def section(step: str, text: str) -> None:
    print()
    print(f"[{step}] {text}")
    print(SUB_LINE)


def pretty(data: Any) -> str:
    try:
        return json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except Exception:
        return repr(data)


def print_response(response: requests.Response) -> Any:
    print(f"HTTP Status : {response.status_code}")

    try:
        data = response.json()
        print(pretty(data))
        return data
    except ValueError:
        print(response.text)
        return response.text


# ============================================================
# 3. Acceptance Result
# ============================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    elapsed: float | None = None


RESULTS: list[CheckResult] = []


def record(
    name: str,
    passed: bool,
    detail: str,
    elapsed: float | None = None,
) -> None:
    RESULTS.append(
        CheckResult(
            name=name,
            passed=passed,
            detail=detail,
            elapsed=elapsed,
        )
    )

    marker = "PASS" if passed else "FAIL"
    suffix = ""
    if elapsed is not None:
        suffix = f" | {elapsed:.3f}s"

    print(f"\n[{marker}] {name}: {detail}{suffix}")


def require(
    condition: bool,
    name: str,
    detail_ok: str,
    detail_fail: str,
    elapsed: float | None = None,
) -> None:
    record(
        name=name,
        passed=condition,
        detail=detail_ok if condition else detail_fail,
        elapsed=elapsed,
    )

    if not condition:
        raise AssertionError(f"{name}: {detail_fail}")


# ============================================================
# 4. Generic JSON Helpers
# ============================================================

def recursive_values_by_key(
    data: Any,
    candidate_keys: Iterable[str],
) -> list[Any]:
    """
    在未知但合法的业务响应结构中递归查找指定 key。

    目的：
    - 不把测试脚本绑死在 Service 内部的额外包装层；
    - 但仍然严格检查 main.py 已经承诺的核心业务结果。
    """
    wanted = {str(key).lower() for key in candidate_keys}
    found: list[Any] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in wanted:
                found.append(value)

            found.extend(
                recursive_values_by_key(
                    value,
                    candidate_keys,
                )
            )

    elif isinstance(data, list):
        for item in data:
            found.extend(
                recursive_values_by_key(
                    item,
                    candidate_keys,
                )
            )

    return found


def first_nonempty_string(
    data: Any,
    candidate_keys: Iterable[str],
) -> str | None:
    for value in recursive_values_by_key(data, candidate_keys):
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def first_value(
    data: Any,
    candidate_keys: Iterable[str],
) -> Any:
    values = recursive_values_by_key(data, candidate_keys)
    return values[0] if values else None


def flatten_key_names(data: Any) -> set[str]:
    keys: set[str] = set()

    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(str(key).lower())
            keys.update(flatten_key_names(value))

    elif isinstance(data, list):
        for item in data:
            keys.update(flatten_key_names(item))

    return keys


def find_sources(data: Any) -> list[Any]:
    """
    优先读取标准 sources 字段。
    """
    values = recursive_values_by_key(data, ("sources",))

    for value in values:
        if isinstance(value, list):
            return value

    return []


def source_has_any_key(
    source: Any,
    candidate_keys: Iterable[str],
) -> bool:
    keys = flatten_key_names(source)
    return bool(keys.intersection({k.lower() for k in candidate_keys}))


# ============================================================
# 5. HTTP Client Helpers
# ============================================================

def get(
    base_url: str,
    path: str,
    timeout: int = HTTP_TIMEOUT,
) -> requests.Response:
    return requests.get(
        f"{base_url}{path}",
        timeout=timeout,
    )


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: int,
) -> requests.Response:
    return requests.post(
        f"{base_url}{path}",
        json=payload,
        timeout=timeout,
    )


# ============================================================
# 6. Step 0 - Local Input Validation
# ============================================================

def validate_pdf(pdf_path: Path) -> None:
    section("0", "检查 Demo 输入文件")

    require(
        pdf_path.exists(),
        "PDF exists",
        f"找到 PDF：{pdf_path}",
        f"PDF 不存在：{pdf_path}",
    )

    require(
        pdf_path.is_file(),
        "PDF is file",
        "输入路径是文件",
        f"输入路径不是文件：{pdf_path}",
    )

    require(
        pdf_path.suffix.lower() == ".pdf",
        "PDF extension",
        "扩展名为 .pdf",
        f"Demo 要求上传 PDF，当前文件：{pdf_path.name}",
    )

    size = pdf_path.stat().st_size

    require(
        size > 0,
        "PDF non-empty",
        f"PDF 大小：{size / 1024 / 1024:.2f} MB",
        "PDF 是空文件",
    )


# ============================================================
# 7. Step 1 - API Health
# ============================================================

def check_health(base_url: str) -> dict[str, Any]:
    section("1", "GET /health - 检查完整运行环境")

    started = time.perf_counter()

    try:
        response = get(base_url, "/health")
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started
        record(
            "API health",
            False,
            f"无法访问 {base_url}/health：{exc}",
            elapsed,
        )
        raise

    elapsed = time.perf_counter() - started
    data = print_response(response)

    require(
        response.status_code == 200,
        "GET /health",
        "HTTP 200",
        f"预期 HTTP 200，实际 {response.status_code}",
        elapsed,
    )

    require(
        isinstance(data, dict),
        "Health JSON",
        "返回 JSON object",
        "返回值不是 JSON object",
    )

    status = str(data.get("status", "")).lower()

    require(
        status == "healthy",
        "Overall health",
        "所有关键组件和 Reranker 均 healthy",
        (
            f"完整现场 Demo 要求 status=healthy，实际 status={status!r}。"
            "如果是 degraded，请先启动 Reranker。"
        ),
    )

    components = data.get("components")

    require(
        isinstance(components, dict),
        "Health components",
        "components 字段存在",
        "缺少 components",
    )

    expected = {
        "gateway",
        "llm",
        "embedding",
        "reranker",
        "vector_store",
    }

    missing = sorted(expected.difference(components.keys()))

    require(
        not missing,
        "Required components",
        "Gateway / LLM / Embedding / Reranker / Vector Store 均已报告",
        f"缺少组件：{missing}",
    )

    bad_components = {
        name: detail
        for name, detail in components.items()
        if isinstance(detail, dict)
        and detail.get("status") != "healthy"
    }

    require(
        not bad_components,
        "Component status",
        "所有组件均 healthy",
        f"存在非 healthy 组件：{pretty(bad_components)}",
    )

    return data


# ============================================================
# 8. Step 2 - Runtime Models
# ============================================================

def check_models(base_url: str) -> dict[str, Any]:
    section("2", "GET /models - 确认实际模型")

    started = time.perf_counter()
    response = get(base_url, "/models")
    elapsed = time.perf_counter() - started

    data = print_response(response)

    require(
        response.status_code == 200,
        "GET /models",
        "HTTP 200",
        f"预期 HTTP 200，实际 {response.status_code}",
        elapsed,
    )

    require(
        isinstance(data, dict),
        "Models JSON",
        "返回 JSON object",
        "返回值不是 JSON object",
    )

    for field in ("llm", "embedding", "reranker"):
        value = data.get(field)

        require(
            isinstance(value, str) and bool(value.strip()),
            f"Model: {field}",
            f"{field}={value}",
            f"{field} 模型名为空：{value!r}",
        )

    return data


# ============================================================
# 9. Step 3 - Upload PDF
# ============================================================

def upload_document(
    base_url: str,
    pdf_path: Path,
    device_model: str,
    document_type: str,
    version: str | None,
) -> tuple[str, dict[str, Any]]:
    section("3", "POST /documents/upload - 上传 G120C PDF")

    form_data = {
        "device_model": device_model,
        "document_type": document_type,
    }

    if version:
        form_data["version"] = version

    started = time.perf_counter()

    with pdf_path.open("rb") as file_obj:
        response = requests.post(
            f"{base_url}/documents/upload",
            files={
                "file": (
                    pdf_path.name,
                    file_obj,
                    "application/pdf",
                )
            },
            data=form_data,
            timeout=UPLOAD_TIMEOUT,
        )

    elapsed = time.perf_counter() - started
    data = print_response(response)

    require(
        response.status_code == 200,
        "Upload PDF",
        "PDF 上传接口 HTTP 200",
        (
            f"PDF 上传失败，HTTP {response.status_code}。"
            "若返回 duplicate，但 HTTP 400，请确认 Document Service "
            "是否会同时返回已有 document_id。"
        ),
        elapsed,
    )

    require(
        isinstance(data, dict),
        "Upload JSON",
        "上传响应为 JSON object",
        "上传响应不是 JSON object",
    )

    document_id = first_nonempty_string(
        data,
        ("document_id",),
    )

    require(
        bool(document_id),
        "document_id",
        f"获得 document_id={document_id}",
        (
            "上传响应中没有找到 document_id。"
            "端到端 Demo 无法进入 Knowledge Base 构建。"
        ),
    )

    return str(document_id), data


# ============================================================
# 10. Step 4 - Build Knowledge Base
# ============================================================

def build_knowledge_base(
    base_url: str,
    document_id: str,
    device_model: str,
    knowledge_name: str,
) -> tuple[str, dict[str, Any]]:
    section("4", "POST /knowledge-bases - 创建并构建知识库")

    payload = {
        "knowledge_name": knowledge_name,
        "device_model": device_model,
        "document_ids": [document_id],
    }

    print("Request:")
    print(pretty(payload))

    started = time.perf_counter()

    response = post_json(
        base_url,
        "/knowledge-bases",
        payload,
        timeout=KB_BUILD_TIMEOUT,
    )

    elapsed = time.perf_counter() - started
    data = print_response(response)

    require(
        response.status_code == 200,
        "Build Knowledge Base",
        "Knowledge Base 构建接口 HTTP 200",
        f"Knowledge Base 构建失败，HTTP {response.status_code}",
        elapsed,
    )

    require(
        isinstance(data, dict),
        "KB JSON",
        "KB 响应为 JSON object",
        "KB 响应不是 JSON object",
    )

    kb_id = first_nonempty_string(
        data,
        (
            "knowledge_base_id",
            "kb_id",
        ),
    )

    require(
        bool(kb_id),
        "knowledge_base_id",
        f"获得 knowledge_base_id={kb_id}",
        (
            "Knowledge Base 响应中没有找到 "
            "knowledge_base_id / kb_id"
        ),
    )

    status = first_nonempty_string(
        data,
        ("status",),
    )

    if status is not None:
        require(
            status.lower() == "ready",
            "KB ready",
            "Knowledge Base status=ready",
            f"Knowledge Base 尚未 ready：status={status!r}",
        )
    else:
        record(
            "KB ready",
            True,
            (
                "响应未暴露 status 字段；根据 /knowledge-bases "
                "HTTP 200 且已返回 KB ID 继续验收"
            ),
        )

    return str(kb_id), data


# ============================================================
# 11. Step 5 - Real Chat
# ============================================================

def chat_with_real_kb(
    base_url: str,
    kb_id: str,
    device_model: str,
    question: str,
) -> tuple[dict[str, Any], float]:
    section("5", "POST /chat - 真实故障诊断")

    payload = {
        "question": question,
        "knowledge_base_id": kb_id,
        "device_model": device_model,
        "history": [],
        "history_summary": None,
    }

    print("Request:")
    print(pretty(payload))

    started = time.perf_counter()

    response = post_json(
        base_url,
        "/chat",
        payload,
        timeout=CHAT_TIMEOUT,
    )

    elapsed = time.perf_counter() - started
    data = print_response(response)

    require(
        response.status_code == 200,
        "Real KB chat",
        "真实 KB 查询 HTTP 200",
        f"真实 KB 查询失败，HTTP {response.status_code}",
        elapsed,
    )

    require(
        isinstance(data, dict),
        "Chat JSON",
        "Chat 响应为 JSON object",
        "Chat 响应不是 JSON object",
    )

    request_id = data.get("request_id")

    require(
        isinstance(request_id, str) and bool(request_id.strip()),
        "request_id",
        f"request_id={request_id}",
        "正常 Chat 响应缺少 request_id",
    )

    answer = first_nonempty_string(
        data,
        ("answer",),
    )

    require(
        bool(answer),
        "Answer",
        f"获得非空答案，共 {len(answer or '')} 字符",
        "Chat 响应没有非空 answer",
    )

    llm_called = first_value(
        data,
        ("llm_called",),
    )

    require(
        llm_called is True,
        "LLM called",
        "证据充分，llm_called=true",
        f"预期 llm_called=true，实际 {llm_called!r}",
    )

    sources = find_sources(data)

    require(
        len(sources) > 0,
        "Sources",
        f"返回 {len(sources)} 个来源",
        "正常诊断回答没有 sources",
    )

    return data, elapsed


# ============================================================
# 12. Step 6 - Citation Acceptance
# ============================================================

def validate_citations(chat_data: dict[str, Any]) -> None:
    section("6", "来源引用验收 - PDF / 页码 / 章节 / Chunk ID")

    sources = find_sources(chat_data)

    document_keys = (
        "source",
        "document",
        "document_name",
        "document_title",
        "file",
        "filename",
        "source_file",
        "pdf",
    )
    page_keys = (
        "page",
        "page_no",
        "page_number",
        "page_start",
        "page_end",
        "pages",
    )
    section_keys = (
        "section",
        "section_title",
        "chapter",
        "chapter_title",
        "heading",
        "title",
    )
    chunk_keys = (
        "chunk_id",
    )

    has_document = any(
        source_has_any_key(source, document_keys)
        for source in sources
    )
    has_page = any(
        source_has_any_key(source, page_keys)
        for source in sources
    )
    has_section = any(
        source_has_any_key(source, section_keys)
        for source in sources
    )
    has_chunk = any(
        source_has_any_key(source, chunk_keys)
        for source in sources
    )

    require(
        has_document,
        "Citation document",
        "来源可追溯到 PDF / 文档",
        "sources 中未发现 PDF / 文档字段",
    )

    require(
        has_page,
        "Citation page",
        "来源包含页码信息",
        "sources 中未发现页码字段",
    )

    require(
        has_section,
        "Citation section",
        "来源包含章节信息",
        "sources 中未发现 section/chapter/heading/title 字段",
    )

    require(
        has_chunk,
        "Citation chunk_id",
        "来源包含 chunk_id",
        "sources 中未发现 chunk_id",
    )

    # 最理想情况：至少一个来源自身就能完整追溯。
    complete_source_indexes: list[int] = []

    for index, source in enumerate(sources, start=1):
        complete = (
            source_has_any_key(source, document_keys)
            and source_has_any_key(source, page_keys)
            and source_has_any_key(source, section_keys)
            and source_has_any_key(source, chunk_keys)
        )

        if complete:
            complete_source_indexes.append(index)

    require(
        bool(complete_source_indexes),
        "Complete citation",
        (
            "至少一个 Source 同时包含 "
            "PDF + Page + Section + Chunk ID；"
            f"完整来源序号={complete_source_indexes}"
        ),
        (
            "虽然不同来源中可能分别出现引用字段，"
            "但没有任何一个 Source 能独立完成 "
            "PDF -> Page -> Section -> Chunk ID 追溯"
        ),
    )


# ============================================================
# 13. Step 7 - Missing KB Rejection
# ============================================================

def validate_missing_kb_rejection(
    base_url: str,
    device_model: str,
) -> None:
    section("7", "POST /chat - 不存在的 Knowledge Base 必须拒绝")

    fake_kb_id = f"kb_demo_not_exist_{uuid.uuid4().hex[:12]}"

    payload = {
        "question": DEFAULT_QUESTION,
        "knowledge_base_id": fake_kb_id,
        "device_model": device_model,
        "history": [],
        "history_summary": None,
    }

    print("Request:")
    print(pretty(payload))

    started = time.perf_counter()

    response = post_json(
        base_url,
        "/chat",
        payload,
        timeout=CHAT_TIMEOUT,
    )

    elapsed = time.perf_counter() - started
    data = print_response(response)

    require(
        response.status_code == 404,
        "Missing KB rejection",
        (
            "不存在 KB 返回 HTTP 404，"
            "没有进入正常 LLM 回答链路"
        ),
        (
            "根据当前 main.py，缺失 KB 应由 KeyError 映射为 HTTP 404；"
            f"实际 HTTP {response.status_code}"
        ),
        elapsed,
    )

    require(
        isinstance(data, dict),
        "Missing KB error JSON",
        "错误响应为 JSON object",
        "错误响应不是 JSON object",
    )

    detail = data.get("detail")

    require(
        isinstance(detail, str) and bool(detail.strip()),
        "Missing KB error detail",
        f"返回明确错误 detail={detail!r}",
        "HTTP 404 响应缺少明确 detail",
    )


# ============================================================
# 14. Final Summary
# ============================================================

def print_summary(
    pdf_path: Path,
    document_id: str | None,
    kb_id: str | None,
    chat_elapsed: float | None,
) -> bool:
    title("Day 20 Stage 9 | RAG V1 Demo Acceptance Summary")

    print(f"PDF               : {pdf_path}")
    print(f"Document ID       : {document_id}")
    print(f"Knowledge Base ID : {kb_id}")

    if chat_elapsed is not None:
        print(f"Chat Response Time: {chat_elapsed:.3f}s")

    print()
    print(f"{'Result':<8} {'Check':<34} Detail")
    print(SUB_LINE)

    for item in RESULTS:
        marker = "PASS" if item.passed else "FAIL"
        elapsed_text = (
            f" ({item.elapsed:.3f}s)"
            if item.elapsed is not None
            else ""
        )

        print(
            f"{marker:<8} "
            f"{item.name:<34} "
            f"{item.detail}{elapsed_text}"
        )

    passed_count = sum(1 for item in RESULTS if item.passed)
    total_count = len(RESULTS)
    failed = [item for item in RESULTS if not item.passed]

    print(SUB_LINE)
    print(f"Passed: {passed_count}/{total_count}")

    if failed:
        print("\nFINAL RESULT: FAIL")
        print("失败项：")
        for item in failed:
            print(f"  - {item.name}: {item.detail}")
        return False

    print("\nFINAL RESULT: PASS")
    print(
        "V1 Demo 已完成："
        "Health -> Models -> Upload -> KB Build -> Chat -> Citation -> Reject"
    )
    return True


# ============================================================
# 15. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Day20 Stage9 Industrial RAG V1 "
            "真实 HTTP 端到端验收"
        )
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default=DEFAULT_PDF,
        help=(
            "G120C PDF 路径；也可使用环境变量 RAG_DEMO_PDF"
        ),
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=(
            "industrial-rag API 地址，默认 "
            "http://127.0.0.1:8000"
        ),
    )

    parser.add_argument(
        "--device-model",
        type=str,
        default=DEFAULT_DEVICE_MODEL,
    )

    parser.add_argument(
        "--document-type",
        type=str,
        default=DEFAULT_DOCUMENT_TYPE,
    )

    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
    )

    parser.add_argument(
        "--knowledge-name",
        type=str,
        default=DEFAULT_KB_NAME,
    )

    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    title("Day 20 Stage 9 | Industrial RAG V1 End-to-End Demo")

    if not args.pdf:
        print(
            "[FAIL] 未指定 G120C PDF。\n\n"
            "请使用：\n"
            "  python tests/test_rag_v1_demo.py "
            "--pdf /path/to/G120C.pdf\n\n"
            "或者设置：\n"
            "  export RAG_DEMO_PDF=/path/to/G120C.pdf"
        )
        return 2

    pdf_path = Path(args.pdf).expanduser().resolve()
    base_url = args.base_url.rstrip("/")

    print(f"API Base URL : {base_url}")
    print(f"PDF          : {pdf_path}")
    print(f"Device Model : {args.device_model}")
    print(f"Question     : {args.question}")

    document_id: str | None = None
    kb_id: str | None = None
    chat_elapsed: float | None = None

    try:
        validate_pdf(pdf_path)
        check_health(base_url)
        check_models(base_url)

        document_id, _ = upload_document(
            base_url=base_url,
            pdf_path=pdf_path,
            device_model=args.device_model,
            document_type=args.document_type,
            version=args.version,
        )

        kb_id, _ = build_knowledge_base(
            base_url=base_url,
            document_id=document_id,
            device_model=args.device_model,
            knowledge_name=args.knowledge_name,
        )

        chat_data, chat_elapsed = chat_with_real_kb(
            base_url=base_url,
            kb_id=kb_id,
            device_model=args.device_model,
            question=args.question,
        )

        validate_citations(chat_data)

        validate_missing_kb_rejection(
            base_url=base_url,
            device_model=args.device_model,
        )

    except (
        AssertionError,
        requests.RequestException,
        OSError,
    ) as exc:
        print()
        print(f"[DEMO STOPPED] {type(exc).__name__}: {exc}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 用户终止测试")
        return 130

    except Exception as exc:
        record(
            "Unexpected error",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        print()
        print(
            f"[UNEXPECTED ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

    success = print_summary(
        pdf_path=pdf_path,
        document_id=document_id,
        kb_id=kb_id,
        chat_elapsed=chat_elapsed,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
