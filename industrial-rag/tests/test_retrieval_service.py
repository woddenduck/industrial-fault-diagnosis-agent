"""
Day20 Stage4 - Retrieval Service Acceptance Test

验收目标：
1. 从 data/knowledge_base.json 自动读取已经 READY 的 Knowledge Base；
2. 验证 KB / device_model 约束；
3. 打印 Query Rewrite；
4. 打印 BM25 / Vector / Hybrid；
5. 打印 Rerank 结果与 rerank_executed；
6. 打印 Final Context / Evidence / Sources / Security；
7. 明确确认本阶段没有 LLM Answer；
8. 验证错误设备型号会在检索前被拒绝。

运行：
    python tests/test_retrieval_service.py

前置条件：
- Stage3 Knowledge Base 已构建成功，status=ready；
- Embedding / Gateway 服务可用；
- 如希望验收正常 Rerank，请启动 Reranker 服务。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# 1. Project Path
# ============================================================

THIS_FILE = Path(__file__).resolve()

# 正常位置：<project>/tests/test_retrieval_service.py
BASE_DIR = THIS_FILE.parents[1]
RAG_DIR = BASE_DIR / "rag"
SERVICES_DIR = BASE_DIR / "app" / "services"

for path in (
    BASE_DIR,
    RAG_DIR,
    SERVICES_DIR,
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# 2. Imports
# ============================================================

import knowledge_service
import query_rewriter
import hybrid_retriever
import rerank_pipeline
import context_builder
from retrieval_service import RetrievalService


# ============================================================
# 3. Test Configuration
# ============================================================

KNOWLEDGE_BASE_FILE = (
    BASE_DIR
    / "data"
    / "knowledge_base.json"
)

TEST_QUESTION_TEMPLATE = (
    "{device_model} 变频器温度过高时应该检查什么？"
)

RECALL_K = 10
RERANK_CANDIDATE_K = 8
FINAL_TOP_K = 3
VECTOR_THRESHOLD = None
RRF_K = 60
CONTENT_PREVIEW = 260


# ============================================================
# 4. Helpers
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 24, title, "=" * 24)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_ready_knowledge_base() -> dict:
    """直接根据 Stage3 生成的 knowledge_base.json 选择验收 KB。"""
    if not KNOWLEDGE_BASE_FILE.exists():
        raise FileNotFoundError(
            f"knowledge_base.json 不存在：{KNOWLEDGE_BASE_FILE}"
        )

    with KNOWLEDGE_BASE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "knowledge_base.json 顶层必须是 list"
        )

    ready_items = [
        item
        for item in data
        if isinstance(item, dict)
        and item.get("status") == "ready"
    ]

    if not ready_items:
        raise RuntimeError(
            "knowledge_base.json 中没有 status=ready 的 Knowledge Base。\n"
            "请先完成 Day20 Stage3 Knowledge Base Build。"
        )

    # 优先选择 G120C，符合当前工业故障诊断主线。
    for item in ready_items:
        if str(item.get("device_model", "")).strip().upper() == "G120C":
            return item

    # 如果以后更换设备，也可以自动用第一个 READY KB 验收。
    return ready_items[0]


def print_query_context(result: dict) -> None:
    section("Query Rewrite")
    query_context = result.get("query_context") or {}
    print(
        json.dumps(
            query_context,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "Retrieval Query:",
        result.get("retrieval_query")
    )


def _preview(text: Any) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) > CONTENT_PREVIEW:
        return value[:CONTENT_PREVIEW] + "..."
    return value


def print_bm25(results: list[dict]) -> None:
    section("BM25")
    print("Count:", len(results))

    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        print()
        print(f"Rank          : {rank}")
        print(f"Chunk ID      : {item.get('chunk_id')}")
        print(f"BM25 Rank     : {item.get('bm25_rank')}")
        print(f"BM25 Score    : {item.get('bm25_score')}")
        print(f"KB ID         : {metadata.get('knowledge_base_id')}")
        print(f"Device Model  : {metadata.get('device_model')}")
        print(f"Source        : {metadata.get('source')}")
        print(
            "Page          :",
            f"{metadata.get('page_start')}-{metadata.get('page_end')}"
        )
        print("Content       :", _preview(item.get("content")))


def print_vector(results: list[dict]) -> None:
    section("Vector")
    print("Count:", len(results))

    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        print()
        print(f"Rank          : {rank}")
        print(f"Chunk ID      : {item.get('chunk_id')}")
        print(f"Vector Rank   : {item.get('vector_rank', item.get('rank'))}")
        print(f"Distance      : {item.get('vector_distance', item.get('distance'))}")
        print(f"KB ID         : {metadata.get('knowledge_base_id')}")
        print(f"Device Model  : {metadata.get('device_model')}")
        print(f"Source        : {metadata.get('source')}")
        print("Content       :", _preview(item.get("content")))


def print_hybrid(results: list[dict]) -> None:
    section("Hybrid")
    print("Count:", len(results))

    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        print()
        print(f"Hybrid Rank   : {rank}")
        print(f"Chunk ID      : {item.get('chunk_id')}")
        print(f"BM25 Rank     : {item.get('bm25_rank')}")
        print(f"Vector Rank   : {item.get('vector_rank')}")
        print(f"Fusion Score  : {item.get('fusion_score')}")
        print(f"KB ID         : {metadata.get('knowledge_base_id')}")
        print(f"Device Model  : {metadata.get('device_model')}")
        print("Content       :", _preview(item.get("content")))


def print_rerank(result: dict) -> None:
    section("Rerank")
    print("Rerank Executed:", result.get("rerank_executed"))
    print("Rerank Error   :", result.get("rerank_error"))

    results = result.get("rerank_results") or []
    print("Final Top-K    :", len(results))

    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        print()
        print(f"Final Rank     : {rank}")
        print(f"Chunk ID       : {item.get('chunk_id')}")
        print(f"Hybrid Rank    : {item.get('hybrid_rank')}")
        print(f"Rerank Rank    : {item.get('rerank_rank')}")
        print(f"Rerank Score   : {item.get('rerank_score')}")
        print(f"KB ID          : {metadata.get('knowledge_base_id')}")
        print(f"Device Model   : {metadata.get('device_model')}")
        print(f"Source         : {metadata.get('source')}")
        print("Content        :", _preview(item.get("content")))


def print_final_context(result: dict) -> None:
    section("Final Context")
    chunks = result.get("final_context") or []

    print("Context Status      :", result.get("context_status"))
    print("Context Reason      :", result.get("context_reason"))
    print("Evidence Sufficient :", result.get("evidence_sufficient"))
    print("Allow LLM           :", result.get("allow_llm"))
    print("Context Chunk Count :", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        print()
        print(f"[Context {index}]")
        print("Chunk ID      :", chunk.get("chunk_id"))
        print("Rerank Score  :", chunk.get("rerank_score"))
        print("Source        :", metadata.get("source"))
        print(
            "Page          :",
            f"{metadata.get('page_start')}-{metadata.get('page_end')}"
        )
        print("Device Model  :", metadata.get("device_model"))
        print("Content       :", _preview(chunk.get("content")))

    section("Sources")
    print(
        json.dumps(
            result.get("sources") or [],
            ensure_ascii=False,
            indent=2,
        )
    )

    section("Security")
    print(
        json.dumps(
            result.get("security") or {},
            ensure_ascii=False,
            indent=2,
        )
    )


def assert_kb_scope(
    results: list[dict],
    knowledge_base_id: str,
    stage_name: str,
) -> None:
    for item in results:
        metadata = item.get("metadata") or {}
        require(
            metadata.get("knowledge_base_id") == knowledge_base_id,
            (
                f"{stage_name} 出现非目标 KB Chunk："
                f"chunk_id={item.get('chunk_id')}, "
                f"actual={metadata.get('knowledge_base_id')}, "
                f"expected={knowledge_base_id}"
            ),
        )


# ============================================================
# 5. Acceptance Cases
# ============================================================

def run_happy_path(
    service: RetrievalService,
    kb: dict,
) -> dict:
    knowledge_base_id = str(
        kb["knowledge_base_id"]
    )
    device_model = str(
        kb["device_model"]
    )
    query = TEST_QUESTION_TEMPLATE.format(
        device_model=device_model
    )

    print("=" * 88)
    print("Day20 Stage4 | Retrieval Service Acceptance Test")
    print("=" * 88)
    print("Knowledge Base ID :", knowledge_base_id)
    print("Knowledge Name    :", kb.get("name"))
    print("KB Status         :", kb.get("status"))
    print("Device Model      :", device_model)
    print("Question          :", query)
    print("Chunk Count       :", kb.get("chunk_count"))
    print("Vector Count      :", kb.get("vector_count"))
    print("BM25 Count        :", kb.get("bm25_document_count"))
    print("=" * 88)

    result = service.retrieve(
        query=query,
        knowledge_base_id=knowledge_base_id,
        device_model=device_model,
        recall_k=RECALL_K,
        rerank_candidate_k=RERANK_CANDIDATE_K,
        final_top_k=FINAL_TOP_K,
        vector_threshold=VECTOR_THRESHOLD,
        rrf_k=RRF_K,
        token_budget=True,
        block_high_risk=True,
    )

    print_query_context(result)
    print_bm25(result["bm25_results"])
    print_vector(result["vector_results"])
    print_hybrid(result["hybrid_results"])
    print_rerank(result)
    print_final_context(result)

    # -------------------------
    # 核心数据契约验收
    # -------------------------
    require(
        result["knowledge_base_id"] == knowledge_base_id,
        "返回的 knowledge_base_id 错误"
    )
    require(
        str(result["device_model"]).upper()
        == device_model.upper(),
        "返回的 device_model 错误"
    )

    query_context = result.get("query_context") or {}
    require(
        bool(query_context.get("original_query")),
        "Query Rewrite 缺少 original_query"
    )
    require(
        bool(query_context.get("rewritten_query")),
        "Query Rewrite 缺少 rewritten_query"
    )
    require(
        str(query_context.get("device_model", "")).upper()
        == device_model.upper(),
        "Query Context device_model 与 KB 不一致"
    )

    require(
        isinstance(result.get("bm25_results"), list),
        "bm25_results 必须是 list"
    )
    require(
        isinstance(result.get("vector_results"), list),
        "vector_results 必须是 list"
    )
    require(
        isinstance(result.get("hybrid_results"), list),
        "hybrid_results 必须是 list"
    )
    require(
        isinstance(result.get("rerank_results"), list),
        "rerank_results 必须是 list"
    )
    require(
        isinstance(result.get("rerank_executed"), bool),
        "rerank_executed 必须是 bool"
    )
    require(
        isinstance(result.get("final_context"), list),
        "final_context 必须是 list"
    )
    require(
        isinstance(result.get("sources"), list),
        "sources 必须是 list"
    )
    require(
        isinstance(result.get("evidence_sufficient"), bool),
        "evidence_sufficient 必须是 bool"
    )

    assert_kb_scope(
        result["bm25_results"],
        knowledge_base_id,
        "BM25",
    )
    assert_kb_scope(
        result["vector_results"],
        knowledge_base_id,
        "Vector",
    )
    assert_kb_scope(
        result["hybrid_results"],
        knowledge_base_id,
        "Hybrid",
    )
    assert_kb_scope(
        result["rerank_results"],
        knowledge_base_id,
        "Rerank",
    )

    # Stage4 强制要求：不能生成 LLM Answer。
    forbidden_answer_keys = {
        "answer",
        "llm_answer",
        "generated_answer",
    }
    require(
        not forbidden_answer_keys.intersection(result.keys()),
        "Stage4 Retrieval Service 不允许返回 LLM Answer"
    )

    section("No LLM Answer")
    print("PASS - Retrieval Service 未调用/未返回 LLM Answer。")

    return result


def run_wrong_request_device_case(
    service: RetrievalService,
    kb: dict,
) -> None:
    """验证 KB=G120C，而 API 参数 device_model=S120 时立即拒绝。"""
    section("Negative Case - Request Device Mismatch")

    kb_model = str(kb["device_model"]).strip().upper()
    wrong_model = "S120"
    if wrong_model == kb_model:
        wrong_model = "G999"

    try:
        service.retrieve(
            query=f"{wrong_model} 温度过高怎么检查？",
            knowledge_base_id=str(kb["knowledge_base_id"]),
            device_model=wrong_model,
        )
    except ValueError as exc:
        print("PASS - 已在检索前拒绝错误型号：")
        print(exc)
        return

    raise AssertionError(
        "错误 device_model 没有被 Knowledge Base 校验拒绝"
    )


def run_query_device_conflict_case(
    service: RetrievalService,
    kb: dict,
) -> None:
    """
    API 参数仍是正确 KB 型号，但 Query 文本明确写另一个 G 系列型号，
    应在 Query Rewrite 后、真正检索前拒绝。
    """
    section("Negative Case - Query Device Conflict")

    kb_model = str(kb["device_model"]).strip().upper()

    # query_rewriter 当前设备型号规则识别 Gxx/Gxxx/Gxxxx + 可选字母。
    wrong_query_model = "G999"
    if wrong_query_model == kb_model:
        wrong_query_model = "G998"

    try:
        service.retrieve(
            query=f"{wrong_query_model} 变频器温度过高怎么检查？",
            knowledge_base_id=str(kb["knowledge_base_id"]),
            device_model=str(kb["device_model"]),
        )
    except ValueError as exc:
        print("PASS - Query Rewrite 后识别到型号冲突并拒绝：")
        print(exc)
        return

    # 如果当前项目以后更换了 Query Rewrite 的型号抽取规则，
    # 这里失败正好提醒你重新检查阶段4的 Query/KB 约束。
    raise AssertionError(
        "Query 中明确的错误设备型号没有被拒绝"
    )


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    kb = load_ready_knowledge_base()

    service = RetrievalService(
        kb_manager=knowledge_service,
        query_rewriter=query_rewriter,
        hybrid_retriever=hybrid_retriever,
        reranker=rerank_pipeline,
        context_builder=context_builder,
    )

    result = run_happy_path(
        service,
        kb,
    )

    run_wrong_request_device_case(
        service,
        kb,
    )

    run_query_device_conflict_case(
        service,
        kb,
    )

    section("Stage4 Acceptance Summary")
    print("Knowledge Base Filter : PASS")
    print("Query Rewrite         : PASS")
    print("BM25 Retrieval        : PASS")
    print("Vector Retrieval      : PASS")
    print("Hybrid / RRF          : PASS")
    print(
        "Reranker             :",
        "PASS" if result["rerank_executed"] else "PASS (Fallback)",
    )
    print(
        "Context Builder       :",
        result.get("context_status"),
    )
    print(
        "Evidence Sufficient   :",
        result.get("evidence_sufficient"),
    )
    print("No LLM Answer         : PASS")
    print()
    print("DAY20 STAGE4 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
