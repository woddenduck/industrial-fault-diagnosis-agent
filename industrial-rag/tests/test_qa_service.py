"""
Day 20 - Stage 5
test_qa_service.py

运行：
    python test_qa_service.py

验收目标：
1. 证据充分 -> LLM 必须调用一次；
2. 证据不足 -> LLM 必须 0 次调用；
3. allow_llm=False -> 即使 evidence_sufficient=True，也必须拒答；
4. allow_llm=True 但 context_text 为空 -> fail-closed，不调用 LLM；
5. Sources 必须由业务层独立返回；
6. Latency 必须包含规定字段。
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# 0. Project Path
# ============================================================
# 当前文件位于：
#     industrial-rag/tests/test_qa_service.py
#
# QA Service 位于：
#     industrial-rag/app/services/qa_service.py
#
# 直接执行：
#     python tests/test_qa_service.py
#
# 时，Python 默认首先把 tests/ 放入 sys.path，
# 因此这里显式把项目根目录 industrial-rag/ 加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.services.qa_service import QAService, REJECT_MESSAGE


# ============================================================
# 1. Fake Components
# ============================================================

class FakeRetrievalService:
    def __init__(self, result: dict):
        self.result = result
        self.call_count = 0
        self.last_kwargs = None

    def retrieve(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.result


class FakeLLM:
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt

        return """## 可能原因
根据当前证据，故障与功率单元相关。

## 证据来源
请以业务接口 sources 字段为准。

## 排查步骤
1. 按照官方文档检查相关部件。
2. 在断电并满足安全要求后继续排查。

## 风险提示
涉及电气设备维护时应遵守安全规范。

## 仍需确认的信息
仍需结合现场状态进一步确认。"""


def fake_build_prompt(
    *,
    question: str,
    context: str,
) -> str:
    assert question
    assert context

    return (
        "SYSTEM: 只能依据上下文回答。\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "请固定输出五个章节。"
    )


# ============================================================
# 2. Test Data
# ============================================================

def make_allowed_result() -> dict:
    return {
        "knowledge_base_id": "kb_g120c",
        "device_model": "G120C",

        "query_context": {
            "original_query": "G120C出现F30021怎么办？",
            "rewritten_query": (
                "G120C设备出现F30021故障时应该如何处理？"
            ),
            "device_model": "G120C",
            "fault_code": "F30021",
        },

        "retrieval_query": (
            "G120C设备出现F30021故障时应该如何处理？"
        ),

        "final_context": [
            {
                "source_id": 1,
                "chunk_id": "g120c_chunk_001",
                "content": "F30021相关故障处理说明。",
                "rerank_score": 0.99,
            }
        ],

        "context_text": (
            "[来源1]\n"
            "文档：G120C操作说明.pdf\n"
            "章节：故障和报警\n"
            "页码：456-457\n"
            "设备型号：G120C\n"
            "Chunk ID：g120c_chunk_001\n"
            "内容：F30021相关故障处理说明。"
        ),

        "sources": [
            {
                "source_id": 1,
                "document": "G120C操作说明.pdf",
                "source_file": "G120C_manual.pdf",
                "page": "456-457",
                "page_start": 456,
                "page_end": 457,
                "section": "故障和报警",
                "device_model": "G120C",
                "chunk_id": "g120c_chunk_001",
            }
        ],

        "evidence_sufficient": True,
        "context_status": "ok",
        "context_reason": "evidence_sufficient",
        "allow_llm": True,

        # 模拟未来 Retrieval Service 若增加细分 latency，
        # QA Service 可以自动读取。
        "latency": {
            "rewrite_ms": 10.0,
            "rerank_ms": 120.0,
        },
    }


def make_rejected_result() -> dict:
    return {
        "knowledge_base_id": "kb_g120c",
        "device_model": "G120C",
        "final_context": [],
        "context_text": "",
        "sources": [],
        "evidence_sufficient": False,
        "context_status": "insufficient_evidence",
        "context_reason": "no_reliable_evidence",
        "allow_llm": False,
    }


# ============================================================
# 3. Assert Helpers
# ============================================================

def assert_latency_shape(result: dict) -> None:
    latency = result.get("latency")

    assert isinstance(latency, dict)

    expected_keys = {
        "rewrite_ms",
        "retrieval_ms",
        "rerank_ms",
        "llm_ms",
        "total_ms",
    }

    assert expected_keys.issubset(
        latency.keys()
    ), (
        "latency 缺少字段："
        f"{expected_keys - set(latency.keys())}"
    )

    assert isinstance(
        latency["retrieval_ms"],
        (int, float),
    )
    assert isinstance(
        latency["llm_ms"],
        (int, float),
    )
    assert isinstance(
        latency["total_ms"],
        (int, float),
    )

    assert latency["retrieval_ms"] >= 0
    assert latency["llm_ms"] >= 0
    assert latency["total_ms"] >= 0


# ============================================================
# 4. Case A - Evidence sufficient -> LLM called
# ============================================================

def test_answer_when_evidence_is_sufficient():
    retrieval = FakeRetrievalService(
        make_allowed_result()
    )
    llm = FakeLLM()

    service = QAService(
        retrieval_service=retrieval,
        prompt_builder=fake_build_prompt,
        llm_caller=llm,
    )

    result = service.answer(
        query="G120C出现F30021怎么办？",
        knowledge_base_id="kb_g120c",
        device_model="G120C",
    )

    assert result["status"] == "answered"
    assert result["llm_called"] is True
    assert llm.call_count == 1
    assert retrieval.call_count == 1

    # Prompt Builder 必须收到真实 Final Context。
    assert "F30021相关故障处理说明" in llm.last_prompt
    assert "G120C出现F30021怎么办？" in llm.last_prompt

    # 五段式回答验收。
    answer = result["answer"]
    for heading in (
        "## 可能原因",
        "## 证据来源",
        "## 排查步骤",
        "## 风险提示",
        "## 仍需确认的信息",
    ):
        assert heading in answer, (
            f"LLM Answer 缺少固定章节：{heading}"
        )

    # Source 必须由 QA 业务返回，而不是只写在自然语言答案里。
    assert result["sources"] == [
        {
            "document": "G120C操作说明.pdf",
            "page": "456-457",
            "section": "故障和报警",
            "chunk_id": "g120c_chunk_001",
        }
    ]

    assert result["latency"]["rewrite_ms"] == 10.0
    assert result["latency"]["rerank_ms"] == 120.0
    assert result["latency"]["llm_ms"] >= 0

    assert_latency_shape(result)


# ============================================================
# 5. Case B - Evidence insufficient -> LLM NOT called
# ============================================================

def test_reject_when_evidence_is_insufficient():
    retrieval = FakeRetrievalService(
        make_rejected_result()
    )
    llm = FakeLLM()

    service = QAService(
        retrieval_service=retrieval,
        prompt_builder=fake_build_prompt,
        llm_caller=llm,
    )

    result = service.answer(
        query="G120C出现XYZ999怎么办？",
        knowledge_base_id="kb_g120c",
        device_model="G120C",
    )

    assert result["status"] == "rejected"
    assert result["answer"] == REJECT_MESSAGE

    # 最关键验收：
    assert result["llm_called"] is False
    assert llm.call_count == 0

    assert result["sources"] == []
    assert result["latency"]["llm_ms"] == 0.0

    assert_latency_shape(result)


# ============================================================
# 6. Case C - Security gate has final authority
# ============================================================

def test_reject_when_allow_llm_is_false():
    data = make_allowed_result()

    # 故意制造：
    # evidence 足够，但安全层不允许调用 LLM。
    data["evidence_sufficient"] = True
    data["allow_llm"] = False
    data["context_status"] = "blocked"
    data["context_reason"] = "prompt_injection_blocked"

    retrieval = FakeRetrievalService(data)
    llm = FakeLLM()

    service = QAService(
        retrieval_service=retrieval,
        prompt_builder=fake_build_prompt,
        llm_caller=llm,
    )

    result = service.answer(
        query="G120C出现F30021怎么办？",
        knowledge_base_id="kb_g120c",
        device_model="G120C",
    )

    assert result["status"] == "rejected"
    assert result["llm_called"] is False
    assert llm.call_count == 0

    assert (
        result["decision"]["reason"]
        == "prompt_injection_blocked"
    )


# ============================================================
# 7. Case D - Fail closed when context is unexpectedly empty
# ============================================================

def test_reject_when_context_is_empty():
    data = make_allowed_result()
    data["context_text"] = ""

    retrieval = FakeRetrievalService(data)
    llm = FakeLLM()

    service = QAService(
        retrieval_service=retrieval,
        prompt_builder=fake_build_prompt,
        llm_caller=llm,
    )

    result = service.answer(
        query="G120C出现F30021怎么办？",
        knowledge_base_id="kb_g120c",
        device_model="G120C",
    )

    assert result["status"] == "rejected"
    assert result["llm_called"] is False
    assert llm.call_count == 0

    assert (
        result["decision"]["reason"]
        == "allow_llm_but_context_text_empty"
    )


# ============================================================
# 8. Runner
# ============================================================

def run_test(
    name: str,
    test_fn,
) -> None:
    print("=" * 80)
    print(name)
    print("=" * 80)

    test_fn()

    print("PASS")


def main():
    tests = [
        (
            "Case A | Evidence sufficient -> LLM called",
            test_answer_when_evidence_is_sufficient,
        ),
        (
            "Case B | Evidence insufficient -> Reject without LLM",
            test_reject_when_evidence_is_insufficient,
        ),
        (
            "Case C | allow_llm=False -> Security gate blocks LLM",
            test_reject_when_allow_llm_is_false,
        ),
        (
            "Case D | Empty context -> Fail closed",
            test_reject_when_context_is_empty,
        ),
    ]

    print()
    print("=" * 80)
    print("Day20 Stage5 | QA Service Acceptance Test")
    print("=" * 80)

    passed = 0

    for name, test_fn in tests:
        run_test(name, test_fn)
        passed += 1

    print()
    print("=" * 80)
    print(
        f"Final Result: {passed}/{len(tests)} PASSED"
    )
    print("Day20 Stage5 QA Service: ACCEPTED")
    print("=" * 80)


if __name__ == "__main__":
    main()