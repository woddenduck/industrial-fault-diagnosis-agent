"""
Day 17 - Stage 6
Hybrid Retrieval -> Reranker -> Fallback

职责：
1. 复用 hybrid_retriever.py 获取 RRF 后候选结果；
2. 只把少量 Hybrid Candidate 的 content 发送给 Reranker；
3. 使用 Reranker 返回的 index 映射回原始 Candidate；
4. 保留 chunk_id / metadata / BM25 / Vector / RRF 信息；
5. Reranker 超时、连接失败、HTTP 错误或响应异常时自动降级；
6. 最终返回 Top-K 结果。

注意：
- 本文件不重新实现 BM25；
- 本文件不重新实现 Vector Search；
- 本文件不重新实现 RRF。
"""

from __future__ import annotations

import os
from typing import Any

import requests

from app.config import GATEWAY_BASE_URL, HTTP_TIMEOUT

from hybrid_retriever import retrieve as hybrid_retrieve


# ============================================================
# 1. 默认配置
# ============================================================

# 可以通过环境变量覆盖：
# export RERANK_URL=http://127.0.0.1:6009/rerank
RERANK_URL = os.getenv(
    "RERANK_URL",
    f"{GATEWAY_BASE_URL}/rerank",
)

RERANK_TIMEOUT = float(
    os.getenv("RERANK_TIMEOUT", str(HTTP_TIMEOUT))
)

# ------------------------------------------------------------
# 三个 K 的含义必须区分：
#
# recall_k:
#     传给 hybrid_retriever.retrieve(candidate_k=...)
#     表示 BM25 / Vector 各自召回多少条。
#
# rerank_candidate_k:
#     传给 hybrid_retriever.retrieve(top_k=...)
#     表示 RRF 融合后保留多少条，送给 Reranker。
#
# final_top_k:
#     Reranker 最终返回多少条。
# ------------------------------------------------------------

DEFAULT_RECALL_K = 10
DEFAULT_RERANK_CANDIDATE_K = 8
DEFAULT_FINAL_TOP_K = 3
DEFAULT_RRF_K = 60


# ============================================================
# 2. 基础参数校验
# ============================================================

def _validate_positive_int(
    name: str,
    value: int,
) -> None:
    """
    校验正整数参数。
    """

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"{name} 必须是大于 0 的整数"
        )


# ============================================================
# 3. Reranker 响应解析
# ============================================================

def _extract_reranker_results(
    data: Any,
) -> list[dict]:
    """
    解析 Reranker 返回结果。

    当前优先支持：

    1.
    {
        "results": [
            {"index": 2, "score": 0.91}
        ]
    }

    2.
    [
        {"index": 2, "score": 0.91}
    ]

    同时兼容部分服务将结果放在 data 中：

    {
        "data": [...]
    }

    每条结果必须能够提供候选下标。
    """

    if isinstance(data, list):
        results = data

    elif isinstance(data, dict):

        if isinstance(
            data.get("results"),
            list,
        ):
            results = data["results"]

        elif isinstance(
            data.get("data"),
            list,
        ):
            results = data["data"]

        else:
            raise ValueError(
                "Reranker 响应中未找到 list 类型的 "
                "'results' 或 'data'"
            )

    else:
        raise ValueError(
            "Reranker 响应必须是 dict 或 list"
        )

    normalized: list[dict] = []

    for position, item in enumerate(
        results
    ):
        if not isinstance(item, dict):
            raise ValueError(
                f"Reranker results[{position}] 不是 dict"
            )

        # 阶段 6 标准字段为 index。
        # 额外兼容少量常见命名。
        rerank_index = item.get(
            "index"
        )

        if rerank_index is None:
            rerank_index = item.get(
                "document_index"
            )

        if rerank_index is None:
            rerank_index = item.get(
                "corpus_id"
            )

        if (
            not isinstance(rerank_index, int)
            or isinstance(rerank_index, bool)
        ):
            raise ValueError(
                f"Reranker results[{position}] "
                "缺少有效整数 index"
            )

        score = item.get(
            "score"
        )

        if score is None:
            score = item.get(
                "relevance_score"
            )

        if score is not None:
            try:
                score = float(
                    score
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Reranker results[{position}] "
                    "score 不是有效数字"
                ) from exc

        normalized.append(
            {
                "index": rerank_index,
                "score": score,
            }
        )

    if not normalized:
        raise ValueError(
            "Reranker 返回结果为空"
        )

    return normalized


# ============================================================
# 4. 调用 Reranker Service
# ============================================================

def call_reranker(
    query: str,
    documents: list[str],
    top_k: int,
    timeout: float = RERANK_TIMEOUT,
) -> list[dict]:
    """
    调用 Reranker Service。

    输入：
        query
        documents
        top_k

    典型请求：
    {
        "query": "...",
        "documents": ["...", "..."],
        "top_k": 3
    }

    本函数不做 fallback。
    HTTP / JSON / Schema 异常继续向上抛，
    由 run_pipeline() 统一降级。
    """

    if (
        not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError(
            "query 不能为空"
        )

    if not isinstance(
        documents,
        list,
    ) or not documents:
        raise ValueError(
            "documents 不能为空"
        )

    _validate_positive_int(
        "top_k",
        top_k,
    )

    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout 必须是大于 0 的数字"
        )

    payload = {
        "query": query.strip(),
        "documents": documents,
        "top_k": min(
            top_k,
            len(documents),
        ),
    }

    response = requests.post(
        RERANK_URL,
        json=payload,
        timeout=float(timeout),
    )

    # 4xx / 5xx -> HTTPError
    response.raise_for_status()

    try:
        data = response.json()

    except ValueError as exc:
        raise ValueError(
            "Reranker 服务返回的不是合法 JSON"
        ) from exc

    return _extract_reranker_results(
        data
    )


# ============================================================
# 5. Hybrid Candidate 准备
# ============================================================

def _prepare_hybrid_candidates(
    hybrid_result: dict,
) -> list[dict]:
    """
    从 hybrid_retriever.retrieve() 的返回字典中取出：

        hybrid_result["hybrid_results"]

    并加入 hybrid_rank。

    注意：
    hybrid_retriever.retrieve() 返回的是 dict，
    不是 list。
    """

    if not isinstance(
        hybrid_result,
        dict,
    ):
        raise TypeError(
            "hybrid_retrieve() 返回值必须是 dict"
        )

    raw_candidates = hybrid_result.get(
        "hybrid_results"
    )

    if not isinstance(
        raw_candidates,
        list,
    ):
        raise ValueError(
            "Hybrid 返回结构缺少 "
            "'hybrid_results' list"
        )

    candidates: list[dict] = []

    for hybrid_rank, item in enumerate(
        raw_candidates,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        candidate = item.copy()

        candidate[
            "hybrid_rank"
        ] = hybrid_rank

        candidates.append(
            candidate
        )

    return candidates


# ============================================================
# 6. Rerank Index -> 原始 Candidate
# ============================================================

def map_rerank_results(
    candidates: list[dict],
    reranker_results: list[dict],
    final_top_k: int,
) -> list[dict]:
    """
    根据 Reranker 返回的 index，
    映射回原始 candidates[index]。

    这是阶段 6 最关键的数据一致性逻辑：

        documents[index]
             ↕
        candidates[index]
             ↓
        chunk_id / metadata

    从而避免：
    文本属于 Chunk A，
    Metadata 却来自 Chunk B。
    """

    _validate_positive_int(
        "final_top_k",
        final_top_k,
    )

    if not candidates:
        return []

    final_results: list[dict] = []

    seen_indices: set[int] = set()

    for rerank_rank, item in enumerate(
        reranker_results,
        start=1,
    ):
        rerank_index = item[
            "index"
        ]

        if (
            rerank_index < 0
            or rerank_index >= len(candidates)
        ):
            raise IndexError(
                "Reranker 返回越界 index："
                f"{rerank_index}，"
                f"Candidate 数量={len(candidates)}"
            )

        # 同一个 candidate 不重复返回。
        if rerank_index in seen_indices:
            continue

        seen_indices.add(
            rerank_index
        )

        # ----------------------------------------------------
        # 核心映射：
        # index -> 原始 Candidate
        # ----------------------------------------------------

        result = candidates[
            rerank_index
        ].copy()

        result[
            "rerank_index"
        ] = rerank_index

        result[
            "rerank_rank"
        ] = rerank_rank

        result[
            "rerank_score"
        ] = item.get(
            "score"
        )

        result[
            "rerank_executed"
        ] = True

        final_results.append(
            result
        )

        if len(final_results) >= final_top_k:
            break

    if not final_results:
        raise ValueError(
            "Reranker 结果无法映射到任何 Candidate"
        )

    return final_results


# ============================================================
# 7. Fallback
# ============================================================

def build_fallback_results(
    candidates: list[dict],
    final_top_k: int,
) -> list[dict]:
    """
    Reranker 不可用时，
    直接保持 Hybrid 原排名并返回 Top-K。
    """

    _validate_positive_int(
        "final_top_k",
        final_top_k,
    )

    fallback_results: list[dict] = []

    for candidate in candidates[
        :final_top_k
    ]:
        result = candidate.copy()

        result[
            "rerank_index"
        ] = None

        result[
            "rerank_rank"
        ] = None

        result[
            "rerank_score"
        ] = None

        result[
            "rerank_executed"
        ] = False

        fallback_results.append(
            result
        )

    return fallback_results


# ============================================================
# 8. 完整 Stage 6 Pipeline
# ============================================================

def run_pipeline(
    query: str,
    recall_k: int = DEFAULT_RECALL_K,
    rerank_candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    where: dict | None = None,
    vector_threshold: float | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_timeout: float = RERANK_TIMEOUT,
) -> dict:
    """
    Stage 6 统一入口。

    数据流：

        Query
          │
          ├─ BM25 Top-recall_k
          │
          └─ Vector Top-recall_k
                  ↓
                 RRF
                  ↓
        Hybrid Top-rerank_candidate_k
                  ↓
               Reranker
                  ↓
          Final Top-final_top_k

    参数映射到现有 hybrid_retriever.py：

        recall_k
            -> hybrid_retrieve(candidate_k=recall_k)

        rerank_candidate_k
            -> hybrid_retrieve(top_k=rerank_candidate_k)

    这是本文件与现有 Hybrid 函数签名最关键的对应关系。
    """

    if (
        not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError(
            "query 不能为空"
        )

    _validate_positive_int(
        "recall_k",
        recall_k,
    )

    _validate_positive_int(
        "rerank_candidate_k",
        rerank_candidate_k,
    )

    _validate_positive_int(
        "final_top_k",
        final_top_k,
    )

    if final_top_k > rerank_candidate_k:
        raise ValueError(
            "final_top_k 不能大于 "
            "rerank_candidate_k"
        )

    if (
        not isinstance(rrf_k, int)
        or isinstance(rrf_k, bool)
        or rrf_k < 0
    ):
        raise ValueError(
            "rrf_k 必须是大于等于 0 的整数"
        )

    query = query.strip()

    # ========================================================
    # Step 1
    # 调用现有 Hybrid Retriever
    # ========================================================

    hybrid_result = hybrid_retrieve(
        query=query,

        # RRF 后最终保留多少条，
        # 也就是即将送给 Reranker 的候选数。
        top_k=rerank_candidate_k,

        # BM25 和 Vector 各自召回多少条。
        candidate_k=recall_k,

        where=where,

        vector_threshold=vector_threshold,

        rrf_k=rrf_k,
    )

    # ========================================================
    # Step 2
    # 从 dict 中取 hybrid_results
    # ========================================================

    candidates = _prepare_hybrid_candidates(
        hybrid_result
    )

    if not candidates:
        return {
            "query": query,

            "recall_k": recall_k,
            "rerank_candidate_k":
                rerank_candidate_k,
            "final_top_k":
                final_top_k,
            "rrf_k": rrf_k,

            "rerank_executed": False,
            "rerank_error": None,

            "hybrid_result":
                hybrid_result,
            "hybrid_candidates": [],
            "final_results": [],
        }

    # ========================================================
    # Step 3
    # Candidate content -> Reranker documents
    #
    # 下标必须完全一致：
    #
    # documents[0] <-> candidates[0]
    # documents[1] <-> candidates[1]
    # ...
    # ========================================================

    documents = [
        str(
            candidate.get(
                "content",
                "",
            )
            or ""
        )
        for candidate in candidates
    ]

    # ========================================================
    # Step 4
    # Reranker
    # ========================================================

    try:
        reranker_results = call_reranker(
            query=query,
            documents=documents,
            top_k=final_top_k,
            timeout=rerank_timeout,
        )

        final_results = map_rerank_results(
            candidates=candidates,
            reranker_results=reranker_results,
            final_top_k=final_top_k,
        )

        print()
        print(
            "[INFO] Reranker executed successfully."
        )

        rerank_executed = True
        rerank_error = None

    # ========================================================
    # Step 5
    # Graceful Degradation
    # ========================================================

    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:

        rerank_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "[WARNING] Reranker unavailable."
        )

        print(
            f"[WARNING] Reason: {rerank_error}"
        )

        print(
            "[WARNING] "
            "Fallback to hybrid retrieval results."
        )

        final_results = build_fallback_results(
            candidates=candidates,
            final_top_k=final_top_k,
        )

        rerank_executed = False

    # ========================================================
    # Step 6
    # 返回统一结构
    # ========================================================

    return {
        "query": query,

        "recall_k": recall_k,

        "rerank_candidate_k":
            rerank_candidate_k,

        "final_top_k":
            final_top_k,

        "rrf_k": rrf_k,

        "rerank_executed":
            rerank_executed,

        "rerank_error":
            rerank_error,

        # 保留完整 Stage 5 结果，
        # 后续调试仍能查看 BM25 / Vector / Hybrid。
        "hybrid_result":
            hybrid_result,

        # RRF 后送进 Reranker 的 Candidate。
        "hybrid_candidates":
            candidates,

        # Stage 6 最终结果。
        "final_results":
            final_results,
    }


# ============================================================
# 9. 打印辅助
# ============================================================

def _format_optional_float(
    value: Any,
    digits: int = 6,
) -> str:
    """
    格式化可能为空的数字。
    """

    if value is None:
        return "None"

    try:
        return (
            f"{float(value):.{digits}f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def print_hybrid_candidates(
    candidates: list[dict],
    content_preview: int = 220,
) -> None:
    """
    打印 Rerank 前的 Hybrid 排名，
    用于 Stage 6 比较排名变化。
    """

    print()
    print(
        "=" * 27,
        "Hybrid Candidates",
        "=" * 27,
    )

    if not candidates:
        print(
            "没有 Hybrid Candidate。"
        )
        return

    for item in candidates:

        print()
        print(
            f'Hybrid Rank: '
            f'{item.get("hybrid_rank")}'
        )

        print(
            f'Chunk ID: '
            f'{item.get("chunk_id")}'
        )

        print(
            f'BM25 Rank: '
            f'{item.get("bm25_rank")}'
        )

        print(
            f'Vector Rank: '
            f'{item.get("vector_rank")}'
        )

        print(
            f'Fusion Score: '
            f'{_format_optional_float(item.get("fusion_score"), 8)}'
        )

        print(
            "Content:"
        )

        print(
            (
                item.get(
                    "content",
                    "",
                )
                or ""
            )[:content_preview]
        )


def print_final_results(
    result: dict,
    content_preview: int = 300,
) -> None:
    """
    打印最终 Rerank / Fallback 结果。
    """

    final_results = result.get(
        "final_results",
        [],
    )

    print()
    print(
        "=" * 29,
        "Final Results",
        "=" * 29,
    )

    print(
        "Rerank Executed:",
        result.get(
            "rerank_executed"
        ),
    )

    if result.get(
        "rerank_error"
    ):
        print(
            "Rerank Error:",
            result[
                "rerank_error"
            ],
        )

    if not final_results:
        print(
            "没有最终检索结果。"
        )
        return

    for final_rank, item in enumerate(
        final_results,
        start=1,
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"Final Rank: {final_rank}"
        )

        print(
            f'Chunk ID: '
            f'{item.get("chunk_id")}'
        )

        print(
            f'Hybrid Rank: '
            f'{item.get("hybrid_rank")}'
        )

        print(
            f'BM25 Rank: '
            f'{item.get("bm25_rank")}'
        )

        print(
            f'BM25 Score: '
            f'{_format_optional_float(item.get("bm25_score"))}'
        )

        print(
            f'Vector Rank: '
            f'{item.get("vector_rank")}'
        )

        print(
            f'Vector Distance: '
            f'{_format_optional_float(item.get("vector_distance"))}'
        )

        print(
            f'Fusion Score: '
            f'{_format_optional_float(item.get("fusion_score"), 8)}'
        )

        print(
            f'Rerank Index: '
            f'{item.get("rerank_index")}'
        )

        print(
            f'Rerank Rank: '
            f'{item.get("rerank_rank")}'
        )

        print(
            f'Rerank Score: '
            f'{_format_optional_float(item.get("rerank_score"))}'
        )

        print(
            f'Rerank Executed: '
            f'{item.get("rerank_executed")}'
        )

        metadata = item.get(
            "metadata",
            {},
        ) or {}

        print(
            f'Source: '
            f'{metadata.get("source", "unknown")}'
        )

        print(
            f'Page: '
            f'{metadata.get("page_start", "unknown")}'
            f'-'
            f'{metadata.get("page_end", "unknown")}'
        )

        print(
            "Content:"
        )

        print(
            (
                item.get(
                    "content",
                    "",
                )
                or ""
            )[:content_preview]
        )


# ============================================================
# 10. Stage 6 最小验收
# ============================================================

def validate_pipeline(
    result: dict,
) -> dict:
    """
    自动检查 Stage 6 最核心结构。

    注意：
    这里不强制要求“排名一定变化”，
    因为 Hybrid 原排名本身可能已经正确。
    """

    candidates = result.get(
        "hybrid_candidates",
        [],
    )

    final_results = result.get(
        "final_results",
        [],
    )

    final_top_k = result.get(
        "final_top_k",
        DEFAULT_FINAL_TOP_K,
    )

    top_k_valid = (
        len(final_results)
        <= final_top_k
    )

    no_duplicate = (
        len(
            [
                item.get("chunk_id")
                for item in final_results
            ]
        )
        ==
        len(
            set(
                item.get("chunk_id")
                for item in final_results
            )
        )
    )

    rerank_executed = bool(
        result.get(
            "rerank_executed"
        )
    )

    if rerank_executed:

        mode_valid = all(
            item.get(
                "rerank_executed"
            ) is True
            and item.get(
                "rerank_index"
            ) is not None
            for item in final_results
        )

    else:

        # Fallback 时必须保持 Hybrid 原顺序。
        expected_ids = [
            item.get(
                "chunk_id"
            )
            for item in candidates[
                :final_top_k
            ]
        ]

        actual_ids = [
            item.get(
                "chunk_id"
            )
            for item in final_results
        ]

        mode_valid = (
            actual_ids == expected_ids
            and all(
                item.get(
                    "rerank_executed"
                ) is False
                for item in final_results
            )
        )

    passed = all(
        (
            top_k_valid,
            no_duplicate,
            mode_valid,
        )
    )

    return {
        "passed": passed,
        "top_k_valid":
            top_k_valid,
        "no_duplicate_chunk_id":
            no_duplicate,
        "mode_valid":
            mode_valid,
    }


# ============================================================
# 11. Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "Day 17 - Stage 6 | "
        "Hybrid -> Reranker -> Fallback"
    )

    print(
        "=" * 80
    )

    print(
        f"Reranker URL: {RERANK_URL}"
    )

    query = input(
        "请输入测试问题："
    ).strip()

    if not query:
        print(
            "问题不能为空。"
        )
        return

    result = run_pipeline(
        query=query,

        # BM25 / Vector 各召回 10 条。
        recall_k=10,

        # RRF 后保留 8 条送给 Reranker。
        rerank_candidate_k=8,

        # Reranker 最终 Top 3。
        final_top_k=3,

        where=None,
        vector_threshold=None,
        rrf_k=60,
        rerank_timeout=10.0,
    )

    # --------------------------------------------------------
    # 先打印 Rerank 前结果，
    # 再打印最终结果，方便确认排名是否变化。
    # --------------------------------------------------------

    print_hybrid_candidates(
        result.get(
            "hybrid_candidates",
            [],
        )
    )

    print_final_results(
        result
    )

    # --------------------------------------------------------
    # 自动验收
    # --------------------------------------------------------

    validation = validate_pipeline(
        result
    )

    print()
    print(
        "=" * 30,
        "Validation",
        "=" * 30,
    )

    print(
        "PASS"
        if validation[
            "top_k_valid"
        ]
        else "FAIL",
        "| Final Top-K 合法",
    )

    print(
        "PASS"
        if validation[
            "no_duplicate_chunk_id"
        ]
        else "FAIL",
        "| Final chunk_id 无重复",
    )

    print(
        "PASS"
        if validation[
            "mode_valid"
        ]
        else "FAIL",
        "| Rerank / Fallback 模式正确",
    )

    print()

    print(
        "Stage 6:",
        (
            "PASS"
            if validation[
                "passed"
            ]
            else "FAIL"
        ),
    )


if __name__ == "__main__":
    main()
