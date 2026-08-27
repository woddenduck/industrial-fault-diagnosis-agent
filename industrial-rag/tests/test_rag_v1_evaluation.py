import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = PROJECT_ROOT / "tests" / "rag_v1_test_questions.json"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "rag_v1_evaluation.json"

CHAT_URL = "http://127.0.0.1:8000/chat"

TIMEOUT_SECONDS = 120


# ============================================================
# Utils
# ============================================================

def load_test_cases() -> List[Dict[str, Any]]:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def extract_source_chunk_ids(response: Dict[str, Any]) -> List[str]:
    """
    尽可能兼容不同 sources 数据结构。
    """

    chunk_ids = []

    sources = response.get("sources") or []

    for source in sources:
        if not isinstance(source, dict):
            continue

        chunk_id = (
            source.get("chunk_id")
            or source.get("id")
            or source.get("source_chunk_id")
        )

        if chunk_id:
            chunk_ids.append(str(chunk_id))

    return chunk_ids


# ============================================================
# Individual Evaluation
# ============================================================

def evaluate_retrieval_hit(
    expected: Dict[str, Any],
    response: Dict[str, Any],
) -> bool:
    """
    Ground Truth Chunk 是否进入最终 sources。

    如果当前测试用例还没有配置 expected_chunks，
    暂时不判失败。
    """

    expected_chunks = expected.get("expected_chunks") or []

    if not expected_chunks:
        return True

    actual_chunks = extract_source_chunk_ids(response)

    return any(
        chunk_id in actual_chunks
        for chunk_id in expected_chunks
    )


def evaluate_answer_correct(
    expected: Dict[str, Any],
    response: Dict[str, Any],
) -> bool:

    should_answer = expected.get("should_answer", True)

    if not should_answer:
        return True

    answer = safe_lower(response.get("answer"))

    expected_keywords = expected.get("expected_keywords") or []

    if not expected_keywords:
        return bool(answer.strip())

    return all(
        safe_lower(keyword) in answer
        for keyword in expected_keywords
    )


def evaluate_citation_correct(
    expected: Dict[str, Any],
    response: Dict[str, Any],
) -> bool:

    should_answer = expected.get("should_answer", True)

    if not should_answer:
        return True

    answer = str(response.get("answer") or "")
    sources = response.get("sources") or []

    if not sources:
        return False

    citation_tokens = [
        "[来源",
        "来源1",
        "来源 1",
        "证据来源",
    ]

    return any(token in answer for token in citation_tokens)


def evaluate_format_complete(
    expected: Dict[str, Any],
    response: Dict[str, Any],
) -> bool:

    if not expected.get("should_answer", True):
        return True

    answer = str(response.get("answer") or "")

    required_sections = [
        "可能原因",
        "证据来源",
        "排查步骤",
        "风险提示",
    ]

    return all(
        section in answer
        for section in required_sections
    )


def evaluate_refused(
    response: Dict[str, Any],
) -> bool:

    status = safe_lower(response.get("status"))

    if status in {
        "rejected",
        "refused",
        "insufficient_evidence",
    }:
        return True

    decision = response.get("decision") or {}

    if decision.get("allow_llm") is False:
        return True

    return False


def evaluate_device_filter(
    expected: Dict[str, Any],
    response: Dict[str, Any],
) -> bool:

    expected_model = expected.get("device_model")

    if not expected_model:
        return True

    actual_model = response.get("device_model")

    if actual_model is None:
        return True

    return (
        safe_lower(actual_model)
        == safe_lower(expected_model)
    )


# ============================================================
# Error Attribution
# ============================================================

def attribute_error(
    case: Dict[str, Any],
    response: Dict[str, Any],
    result: Dict[str, Any],
) -> List[str]:

    errors = []

    expected = case["expected"]

    # --------------------------------------------------------
    # Retrieval Error
    # --------------------------------------------------------

    if not result["retrieval_hit"]:
        errors.append("Retrieval Error")

    # --------------------------------------------------------
    # Business Logic Error
    # --------------------------------------------------------

    expected_refuse = expected.get("should_refuse", False)

    if expected_refuse != result["refused"]:
        errors.append("Business Logic Error")

    expected_llm = expected.get("should_call_llm")

    actual_llm = response.get("llm_called")

    if (
        expected_llm is not None
        and actual_llm is not None
        and bool(expected_llm) != bool(actual_llm)
    ):
        if "Business Logic Error" not in errors:
            errors.append("Business Logic Error")

    # --------------------------------------------------------
    # Context / Device Filter Error
    # --------------------------------------------------------

    if not result["device_filter_correct"]:
        errors.append("Context Error")

    # --------------------------------------------------------
    # LLM Error
    # --------------------------------------------------------

    if (
        result["retrieval_hit"]
        and not result["answer_correct"]
        and expected.get("should_answer")
    ):
        errors.append("LLM Error")

    # --------------------------------------------------------
    # Citation Error
    # --------------------------------------------------------

    if (
        expected.get("should_answer")
        and not result["citation_correct"]
    ):
        errors.append("LLM Error")

    # --------------------------------------------------------
    # Output Contract
    # --------------------------------------------------------

    if (
        expected.get("should_answer")
        and not result["format_complete"]
    ):
        errors.append("Business Logic Error")

    return list(dict.fromkeys(errors))


# ============================================================
# Run Case
# ============================================================

def run_case(case: Dict[str, Any]) -> Dict[str, Any]:

    question_id = case["question_id"]
    expected = case["expected"]

    payload = {
        "question": case["question"],
        "knowledge_base_id": case["knowledge_base_id"],
        "device_model": case.get("device_model"),
        "history": [],
    }

    print()
    print("=" * 80)
    print(f"{question_id} | {case['category']}")
    print("=" * 80)
    print("Question :", case["question"])
    print("Device   :", case.get("device_model"))

    start = time.perf_counter()

    try:
        http_response = requests.post(
            CHAT_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        http_status = http_response.status_code

        try:
            response = http_response.json()
        except Exception:
            response = {
                "status": "http_error",
                "answer": http_response.text,
            }

    except Exception as exc:

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        response = {
            "status": "request_error",
            "answer": str(exc),
            "sources": [],
            "llm_called": False,
        }

        http_status = None

    retrieval_hit = evaluate_retrieval_hit(
        expected,
        response,
    )

    answer_correct = evaluate_answer_correct(
        expected,
        response,
    )

    citation_correct = evaluate_citation_correct(
        expected,
        response,
    )

    format_complete = evaluate_format_complete(
        expected,
        response,
    )

    refused = evaluate_refused(response)

    device_filter_correct = evaluate_device_filter(
        expected,
        response,
    )

    result = {
        "retrieval_hit": retrieval_hit,
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "format_complete": format_complete,
        "refused": refused,
        "device_filter_correct": device_filter_correct,
        "latency_ms": latency_ms,
    }

    errors = attribute_error(
        case,
        response,
        result,
    )

    expected_refused = expected.get(
        "should_refuse",
        False,
    )

    passed = (
        retrieval_hit
        and answer_correct
        and citation_correct
        and format_complete
        and device_filter_correct
        and refused == expected_refused
    )

    print("HTTP      :", http_status)
    print("Status    :", response.get("status"))
    print("LLM Called:", response.get("llm_called"))
    print("Retrieval :", retrieval_hit)
    print("Answer    :", answer_correct)
    print("Citation  :", citation_correct)
    print("Format    :", format_complete)
    print("Refused   :", refused)
    print("Dev Filter:", device_filter_correct)
    print("Latency   :", latency_ms, "ms")
    print("PASS      :", passed)

    if errors:
        print("Attribution:", ", ".join(errors))
    else:
        print("Attribution: None")

    return {
        "question_id": question_id,
        "category": case["category"],
        "question": case["question"],
        "expected": expected,
        "result": result,
        "passed": passed,
        "error_attribution": errors,
        "response_snapshot": {
            "status": response.get("status"),
            "answer": response.get("answer"),
            "llm_called": response.get("llm_called"),
            "sources": response.get("sources"),
            "device_model": response.get("device_model"),
            "decision": response.get("decision"),
        },
    }


# ============================================================
# Summary
# ============================================================

def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:

    total = len(results)

    passed = sum(
        1
        for item in results
        if item["passed"]
    )

    answered_cases = [
        item
        for item in results
        if item["expected"].get("should_answer")
    ]

    retrieval_hits = sum(
        1
        for item in answered_cases
        if item["result"]["retrieval_hit"]
    )

    answer_correct = sum(
        1
        for item in answered_cases
        if item["result"]["answer_correct"]
    )

    citation_correct = sum(
        1
        for item in answered_cases
        if item["result"]["citation_correct"]
    )

    format_complete = sum(
        1
        for item in answered_cases
        if item["result"]["format_complete"]
    )

    refusal_cases = [
        item
        for item in results
        if item["expected"].get("should_refuse")
    ]

    refusal_correct = sum(
        1
        for item in refusal_cases
        if item["result"]["refused"]
    )

    latencies = [
        item["result"]["latency_ms"]
        for item in results
    ]

    error_counts = {}

    for item in results:
        for error in item["error_attribution"]:
            error_counts[error] = (
                error_counts.get(error, 0) + 1
            )

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round(
            passed / total,
            4,
        ) if total else 0,
        "retrieval_hit_rate": round(
            retrieval_hits / len(answered_cases),
            4,
        ) if answered_cases else 0,
        "answer_accuracy": round(
            answer_correct / len(answered_cases),
            4,
        ) if answered_cases else 0,
        "citation_accuracy": round(
            citation_correct / len(answered_cases),
            4,
        ) if answered_cases else 0,
        "format_complete_rate": round(
            format_complete / len(answered_cases),
            4,
        ) if answered_cases else 0,
        "refusal_accuracy": round(
            refusal_correct / len(refusal_cases),
            4,
        ) if refusal_cases else 0,
        "average_latency_ms": round(
            sum(latencies) / len(latencies),
            2,
        ) if latencies else 0,
        "error_counts": error_counts,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("Day20 Stage8 | RAG V1 Basic Evaluation")
    print("=" * 80)

    test_cases = load_test_cases()

    results = []

    for case in test_cases:
        result = run_case(case)
        results.append(result)

    summary = build_summary(results)

    report = {
        "evaluation": "RAG V1 Basic Evaluation",
        "summary": summary,
        "cases": results,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    print(
        f"Passed              : "
        f"{summary['passed_cases']}/{summary['total_cases']}"
    )

    print(
        f"Pass Rate           : "
        f"{summary['pass_rate']:.2%}"
    )

    print(
        f"Retrieval Hit Rate  : "
        f"{summary['retrieval_hit_rate']:.2%}"
    )

    print(
        f"Answer Accuracy     : "
        f"{summary['answer_accuracy']:.2%}"
    )

    print(
        f"Citation Accuracy   : "
        f"{summary['citation_accuracy']:.2%}"
    )

    print(
        f"Format Complete     : "
        f"{summary['format_complete_rate']:.2%}"
    )

    print(
        f"Refusal Accuracy    : "
        f"{summary['refusal_accuracy']:.2%}"
    )

    print(
        f"Average Latency     : "
        f"{summary['average_latency_ms']} ms"
    )

    print()
    print("Error Attribution:")

    if summary["error_counts"]:
        for name, count in summary["error_counts"].items():
            print(f"  {name}: {count}")
    else:
        print("  None")

    print()
    print("Report:")
    print(REPORT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()