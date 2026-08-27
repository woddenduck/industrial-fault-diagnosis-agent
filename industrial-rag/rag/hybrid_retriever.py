"""
Hybrid Retriever

Day 17 - Stage 5

职责：

1. 复用 bm25_retriever.py 执行 BM25 Search
2. 复用 vector_store.py 执行 Vector Search
3. 根据 chunk_id 合并、去重
4. 使用 RRF（Reciprocal Rank Fusion）融合排名
5. 返回统一 Hybrid Top-K 结果
6. 保留 BM25 / Vector 两侧原始排名与分数信息

当前阶段暂不负责：

- Reranker
- LLM
- Query Rewrite
- Score Normalize
- Weighted RRF
"""

from __future__ import annotations

from typing import Any

from bm25_retriever import (
    INDEX_PATH,
    build_index,
    load_chunks,
    load_index,
    save_index,
    search as bm25_retrieve,
)

from vector_store import (
    search as vector_retrieve,
)


# ============================================================
# 1. 默认配置
# ============================================================

DEFAULT_CANDIDATE_K = 10
DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60


# ============================================================
# 2. BM25 Index 缓存
# ============================================================

_BM25_INDEX_DATA: dict | None = None


def get_bm25_index() -> dict:
    """
    获取 BM25 Index。

    优先：
        已加载内存缓存
            ↓
        本地已有 bm25_index.pkl
            ↓
        load_index()

    如果 Index 不存在：
        load_chunks()
            ↓
        build_index()
            ↓
        save_index()

    注意：
    Hybrid Retriever 只负责编排，
    不重新实现 BM25 构建逻辑。
    """

    global _BM25_INDEX_DATA

    if _BM25_INDEX_DATA is not None:
        return _BM25_INDEX_DATA

    if INDEX_PATH.exists():

        print(
            f"[Hybrid] 加载 BM25 Index："
            f"{INDEX_PATH}"
        )

        _BM25_INDEX_DATA = load_index()

    else:

        print(
            "[Hybrid] 未发现 BM25 Index，"
            "调用 bm25_retriever 构建..."
        )

        chunks = load_chunks()

        _BM25_INDEX_DATA = build_index(
            chunks
        )

        save_index(
            _BM25_INDEX_DATA
        )

    return _BM25_INDEX_DATA


# ============================================================
# 3. BM25 Search
# ============================================================

def bm25_search(
    query: str,
    top_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict]:
    """
    调用已有 bm25_retriever.search()。

    原始返回结构：

    {
        "chunk_id": ...,
        "content": ...,
        "bm25_score": ...,
        "bm25_rank": ...,
        "metadata": {...}
    }
    """

    if (
        not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError(
            "query 不能为空"
        )

    if (
        not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k 必须是大于 0 的整数"
        )

    index_data = get_bm25_index()

    return bm25_retrieve(
        query=query.strip(),
        index_data=index_data,
        top_k=top_k,
    )


# ============================================================
# 4. Vector Search
# ============================================================

def vector_search(
    query: str,
    top_k: int = DEFAULT_CANDIDATE_K,
    where: dict | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """
    调用已有 vector_store.search()。

    vector_store 当前原始返回结构：

    {
        "rank": ...,
        "chunk_id": ...,
        "distance": ...,
        "content": ...,
        "metadata": {...}
    }

    Hybrid 层不修改 Vector Store 底层实现。
    """

    if (
        not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError(
            "query 不能为空"
        )

    if (
        not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k 必须是大于 0 的整数"
        )

    return vector_retrieve(
        query=query.strip(),
        top_k=top_k,
        where=where,
        threshold=threshold,
    )


# ============================================================
# 5. Metadata 辅助
# ============================================================

def _is_empty_metadata_value(
    value: Any,
) -> bool:
    """
    判断 metadata 值是否属于空值 / unknown。

    仅用于两个 Retriever 命中同一个 Chunk 时，
    尽量补全 metadata。
    """

    return value in (
        None,
        "",
        "unknown",
        -1,
    ) or value == [] or value == {}


def _merge_metadata(
    base: dict | None,
    incoming: dict | None,
) -> dict:
    """
    合并两侧 metadata。

    原则：
    - 已有有效值优先保留；
    - 已有值为空 / unknown 时，用另一侧补充；
    - 不覆盖正常有效字段。
    """

    merged = dict(
        base or {}
    )

    for key, value in (
        incoming or {}
    ).items():

        if (
            key not in merged
            or _is_empty_metadata_value(
                merged.get(key)
            )
        ):
            merged[key] = value

    return merged


# ============================================================
# 6. 创建统一融合记录
# ============================================================

def _new_fused_item(
    chunk_id: str,
    content: str = "",
    metadata: dict | None = None,
) -> dict:
    """
    创建 Hybrid 统一结果结构。
    """

    return {
        "chunk_id": chunk_id,

        "bm25_rank": None,
        "bm25_score": None,

        "vector_rank": None,
        "vector_distance": None,

        "fusion_score": 0.0,

        "content": content or "",
        "metadata": dict(
            metadata or {}
        ),
    }


# ============================================================
# 7. RRF Fusion
# ============================================================

def fuse_results(
    bm25_results: list[dict],
    vector_results: list[dict],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    使用 Reciprocal Rank Fusion 合并两路检索结果。

    RRF：

        fusion_score += 1 / (rrf_k + rank)

    规则：

    1. 不比较 BM25 Score 与 Vector Distance；
    2. 只使用两个 Retriever 的 Rank；
    3. 使用 chunk_id 去重；
    4. 同一个 Chunk 同时命中两路时累计两次 RRF 分数；
    5. 同一路 Retriever 中即使意外出现重复 chunk_id，
       也只使用第一次（最高排名）进行 RRF 计分。
    """

    if not isinstance(
        bm25_results,
        list,
    ):
        raise TypeError(
            "bm25_results 必须是 list"
        )

    if not isinstance(
        vector_results,
        list,
    ):
        raise TypeError(
            "vector_results 必须是 list"
        )

    if (
        not isinstance(rrf_k, int)
        or rrf_k < 0
    ):
        raise ValueError(
            "rrf_k 必须是大于等于 0 的整数"
        )

    fused: dict[str, dict] = {}

    seen_bm25: set[str] = set()
    seen_vector: set[str] = set()


    # --------------------------------------------------------
    # 7.1 BM25
    # --------------------------------------------------------

    for fallback_rank, item in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = item.get(
            "chunk_id"
        )

        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
        ):
            continue

        chunk_id = chunk_id.strip()

        # 同一路中重复 Chunk 只记最高排名一次。
        if chunk_id in seen_bm25:
            continue

        seen_bm25.add(
            chunk_id
        )

        rank = item.get(
            "bm25_rank"
        )

        if (
            not isinstance(rank, int)
            or rank <= 0
        ):
            rank = fallback_rank

        if chunk_id not in fused:

            fused[chunk_id] = _new_fused_item(
                chunk_id=chunk_id,
                content=item.get(
                    "content",
                    "",
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )

        fused_item = fused[
            chunk_id
        ]

        fused_item[
            "bm25_rank"
        ] = rank

        bm25_score = item.get(
            "bm25_score"
        )

        fused_item[
            "bm25_score"
        ] = (
            float(bm25_score)
            if bm25_score is not None
            else None
        )

        fused_item[
            "fusion_score"
        ] += (
            1.0
            / (rrf_k + rank)
        )

        if (
            not fused_item["content"]
            and item.get("content")
        ):
            fused_item[
                "content"
            ] = item["content"]

        fused_item[
            "metadata"
        ] = _merge_metadata(
            fused_item.get(
                "metadata"
            ),
            item.get(
                "metadata"
            ),
        )


    # --------------------------------------------------------
    # 7.2 Vector
    # --------------------------------------------------------

    for fallback_rank, item in enumerate(
        vector_results,
        start=1,
    ):

        chunk_id = item.get(
            "chunk_id"
        )

        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
        ):
            continue

        chunk_id = chunk_id.strip()

        # 同一路中重复 Chunk 只记最高排名一次。
        if chunk_id in seen_vector:
            continue

        seen_vector.add(
            chunk_id
        )

        # vector_store.py 当前字段名称是 rank。
        rank = item.get(
            "rank"
        )

        if (
            not isinstance(rank, int)
            or rank <= 0
        ):
            rank = fallback_rank

        if chunk_id not in fused:

            fused[chunk_id] = _new_fused_item(
                chunk_id=chunk_id,
                content=item.get(
                    "content",
                    "",
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )

        fused_item = fused[
            chunk_id
        ]

        fused_item[
            "vector_rank"
        ] = rank

        # vector_store.py 当前字段名称是 distance。
        vector_distance = item.get(
            "distance"
        )

        fused_item[
            "vector_distance"
        ] = (
            float(vector_distance)
            if vector_distance is not None
            else None
        )

        fused_item[
            "fusion_score"
        ] += (
            1.0
            / (rrf_k + rank)
        )

        if (
            not fused_item["content"]
            and item.get("content")
        ):
            fused_item[
                "content"
            ] = item["content"]

        fused_item[
            "metadata"
        ] = _merge_metadata(
            fused_item.get(
                "metadata"
            ),
            item.get(
                "metadata"
            ),
        )


    # --------------------------------------------------------
    # 7.3 Fusion Score 降序
    # --------------------------------------------------------

    results = list(
        fused.values()
    )

    # fusion_score 相同时：
    # 优先选择在任一 Retriever 中排名更靠前的 Chunk，
    # 再使用 chunk_id 保证排序结果稳定。
    def sort_key(
        item: dict,
    ):
        best_rank = min(
            rank
            for rank in (
                item.get(
                    "bm25_rank"
                ),
                item.get(
                    "vector_rank"
                ),
            )
            if rank is not None
        )

        return (
            -item[
                "fusion_score"
            ],
            best_rank,
            item[
                "chunk_id"
            ],
        )

    results.sort(
        key=sort_key
    )

    return results


# ============================================================
# 8. Hybrid Retrieve
# ============================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    where: dict | None = None,
    vector_threshold: float | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict:
    """
    Hybrid Retriever 统一入口。

    流程：

        Query
          │
          ├──────────────┐
          ↓              ↓
        BM25           Vector
          │              │
        Top-N          Top-N
          │              │
          └──────┬───────┘
                 ↓
         chunk_id 去重
                 ↓
              RRF
                 ↓
         Fusion Score 排序
                 ↓
          Hybrid Top-K

    参数：

    candidate_k:
        BM25 和 Vector 各自召回的候选数量。

    top_k:
        RRF 融合后最终返回的数量。

    where:
        仅传递给 Vector Search 的 Chroma Metadata Filter。

    vector_threshold:
        仅传递给 Vector Search 的 Distance Threshold。

    rrf_k:
        RRF 平滑常数，默认 60。
        注意：它不是 Retrieval Top-K。
    """

    if (
        not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError(
            "query 不能为空"
        )

    if (
        not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k 必须是大于 0 的整数"
        )

    if (
        not isinstance(candidate_k, int)
        or candidate_k <= 0
    ):
        raise ValueError(
            "candidate_k 必须是大于 0 的整数"
        )

    query = query.strip()


    # --------------------------------------------------------
    # 8.1 BM25 Recall
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query=query,
        top_k=candidate_k,
    )


    # --------------------------------------------------------
    # 8.2 Vector Recall
    # --------------------------------------------------------

    vector_results = vector_search(
        query=query,
        top_k=candidate_k,
        where=where,
        threshold=vector_threshold,
    )


    # --------------------------------------------------------
    # 8.3 RRF
    # --------------------------------------------------------

    fused_results = fuse_results(
        bm25_results=bm25_results,
        vector_results=vector_results,
        rrf_k=rrf_k,
    )


    # --------------------------------------------------------
    # 8.4 Final Top-K
    # --------------------------------------------------------

    hybrid_results = fused_results[
        :top_k
    ]


    return {
        "query": query,

        "candidate_k": candidate_k,
        "top_k": top_k,
        "rrf_k": rrf_k,

        "bm25_results": bm25_results,
        "vector_results": vector_results,

        "hybrid_results": hybrid_results,

        # 便于后续验收“合并后共有多少唯一 Chunk”。
        "fused_candidate_count": len(
            fused_results
        ),
    }


# ============================================================
# 9. 打印函数
# ============================================================

def _format_optional_float(
    value: float | None,
    digits: int = 6,
) -> str:
    """
    格式化可为空的浮点数。
    """

    if value is None:
        return "None"

    return f"{value:.{digits}f}"


def print_retrieval_results(
    result: dict,
    content_preview: int = 300,
) -> None:
    """
    按 Stage 5 验收格式打印：

    ================= BM25 =================
    ================= Vector ===============
    ================= Hybrid RRF ===========
    """

    bm25_results = result.get(
        "bm25_results",
        [],
    )

    vector_results = result.get(
        "vector_results",
        [],
    )

    hybrid_results = result.get(
        "hybrid_results",
        [],
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print()
    print(
        "=" * 32,
        "BM25",
        "=" * 32,
    )

    if not bm25_results:
        print(
            "没有 BM25 检索结果。"
        )

    for item in bm25_results:

        print()
        print(
            f'Rank '
            f'{item.get("bm25_rank")}'
        )

        print(
            f'Chunk ID: '
            f'{item.get("chunk_id")}'
        )

        print(
            f'BM25 Score: '
            f'{_format_optional_float(item.get("bm25_score"))}'
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


    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    print()
    print(
        "=" * 31,
        "Vector",
        "=" * 31,
    )

    if not vector_results:
        print(
            "没有 Vector 检索结果。"
        )

    for item in vector_results:

        print()
        print(
            f'Rank '
            f'{item.get("rank")}'
        )

        print(
            f'Chunk ID: '
            f'{item.get("chunk_id")}'
        )

        print(
            f'Vector Distance: '
            f'{_format_optional_float(item.get("distance"))}'
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


    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    print()
    print(
        "=" * 28,
        "Hybrid RRF",
        "=" * 28,
    )

    if not hybrid_results:
        print(
            "没有 Hybrid 检索结果。"
        )

    for rank, item in enumerate(
        hybrid_results,
        start=1,
    ):

        print()
        print(
            f"Rank {rank}"
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
# 10. 最小自检
# ============================================================

def validate_hybrid_results(
    result: dict,
) -> dict:
    """
    对 Hybrid 最核心的结构做快速验收。

    检查：

    1. Hybrid 中 chunk_id 是否重复；
    2. Fusion Score 是否按降序；
    3. 每条结果是否至少来自一个 Retriever；
    4. Hybrid 数量是否不超过 top_k。
    """

    hybrid_results = result.get(
        "hybrid_results",
        [],
    )

    chunk_ids = [
        item.get(
            "chunk_id"
        )
        for item in hybrid_results
    ]

    no_duplicate = (
        len(chunk_ids)
        == len(set(chunk_ids))
    )

    scores = [
        item.get(
            "fusion_score",
            0.0,
        )
        for item in hybrid_results
    ]

    fusion_sorted = (
        scores
        == sorted(
            scores,
            reverse=True,
        )
    )

    has_source = all(
        (
            item.get(
                "bm25_rank"
            )
            is not None
        )
        or
        (
            item.get(
                "vector_rank"
            )
            is not None
        )
        for item in hybrid_results
    )

    top_k_valid = (
        len(hybrid_results)
        <= result.get(
            "top_k",
            len(hybrid_results),
        )
    )

    passed = all(
        (
            no_duplicate,
            fusion_sorted,
            has_source,
            top_k_valid,
        )
    )

    return {
        "passed": passed,
        "no_duplicate_chunk_id":
            no_duplicate,
        "fusion_score_descending":
            fusion_sorted,
        "all_results_have_source":
            has_source,
        "top_k_valid":
            top_k_valid,
    }


# ============================================================
# 11. Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Day 17 - Stage 5 | "
        "Hybrid + RRF"
    )

    print(
        "=" * 80
    )

    query = input(
        "请输入测试问题："
    ).strip()

    if not query:
        print(
            "问题不能为空。"
        )
        return

    result = retrieve(
        query=query,
        top_k=5,
        candidate_k=10,
        where=None,
        vector_threshold=None,
        rrf_k=60,
    )

    print_retrieval_results(
        result
    )


    # --------------------------------------------------------
    # 自动验收
    # --------------------------------------------------------

    validation = validate_hybrid_results(
        result
    )

    print()
    print(
        "=" * 28,
        "Validation",
        "=" * 28,
    )

    print(
        "PASS"
        if validation[
            "no_duplicate_chunk_id"
        ]
        else "FAIL",
        "| chunk_id 无重复",
    )

    print(
        "PASS"
        if validation[
            "fusion_score_descending"
        ]
        else "FAIL",
        "| Fusion Score 降序",
    )

    print(
        "PASS"
        if validation[
            "all_results_have_source"
        ]
        else "FAIL",
        "| 每条结果至少来自一个 Retriever",
    )

    print(
        "PASS"
        if validation[
            "top_k_valid"
        ]
        else "FAIL",
        "| Hybrid 结果数量符合 Top-K",
    )

    print()

    print(
        "Stage 5:",
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
