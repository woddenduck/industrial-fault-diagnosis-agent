"""
Day 20 - Stage 4
Retrieval Service

职责：
1. 校验 Knowledge Base 是否存在、是否 READY、设备型号是否匹配；
2. 调用 Day18 query_rewriter；
3. 将检索严格限定在目标 Knowledge Base；
4. 调用 Day17 Hybrid Retrieval + Reranker；
5. 保留 BM25 / Vector / Hybrid / Rerank 调试结果；
6. 调用 Day18 Context Builder；
7. 返回 final_context / evidence_sufficient / sources；
8. 本阶段绝不调用 LLM。

注意：
- 当前 hybrid_retriever.py 的 Vector 支持 where filter；
- 当前 hybrid_retriever.py 的 BM25 默认使用全局缓存；
  因此本 Service 会在每次请求前显式加载目标 KB 专属 BM25 Index，
  并切换 Hybrid Retriever 的 BM25 cache，保证 BM25 与 Vector 都被 KB 隔离。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class RetrievalService:
    """Day20 Stage4 检索业务编排层。"""

    def __init__(
        self,
        kb_manager,
        query_rewriter,
        hybrid_retriever,
        reranker,
        context_builder,
    ):
        self.kb_manager = kb_manager
        self.query_rewriter = query_rewriter
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder

    # ========================================================
    # 1. 基础校验
    # ========================================================

    @staticmethod
    def _require_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()

    @staticmethod
    def _normalize_device_model(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    # ========================================================
    # 2. Knowledge Base 校验
    # ========================================================

    def _get_knowledge_base(self, knowledge_base_id: str) -> dict:
        """复用 Stage3 knowledge_service.get_knowledge_base()。"""
        getter = getattr(
            self.kb_manager,
            "get_knowledge_base",
            None,
        )

        if not callable(getter):
            raise TypeError(
                "kb_manager 必须提供 get_knowledge_base(knowledge_base_id)"
            )

        # Knowledge Service 对不存在的 KB 使用 KeyError 语义。
        # 这里必须原样透传，不能转换成 ValueError：
        #   KeyError   -> FastAPI main.py -> HTTP 404
        #   ValueError -> FastAPI main.py -> HTTP 400
        #
        # Stage9 端到端验收依赖这个异常边界。
        kb = getter(knowledge_base_id)

        if not isinstance(kb, dict):
            raise TypeError("get_knowledge_base() 返回值必须是 dict")

        return kb

    def _validate_knowledge_base(
        self,
        knowledge_base_id: str,
        device_model: str,
    ) -> dict:
        """
        关卡 A：检索之前先做业务约束。

        规则：
        1. KB 必须存在；
        2. 已有 status 字段时必须为 ready；
        3. 请求 device_model 必须与 KB device_model 一致。
        """
        kb = self._get_knowledge_base(
            knowledge_base_id
        )

        status = kb.get("status")

        # 兼容更早版本没有 status 字段的 JSON；
        # Day20 Stage3 新版本有 status 时，必须 ready 才允许检索。
        if status not in (None, "ready"):
            raise RuntimeError(
                "Knowledge Base 当前不可用于检索："
                f"id={knowledge_base_id}, status={status}"
            )

        kb_device_model = self._normalize_device_model(
            kb.get("device_model")
        )
        request_device_model = self._normalize_device_model(
            device_model
        )

        if not kb_device_model:
            raise ValueError(
                f"Knowledge Base 缺少 device_model：{knowledge_base_id}"
            )

        if request_device_model != kb_device_model:
            raise ValueError(
                "Knowledge Base 与请求设备型号不匹配："
                f"KB={kb_device_model}, Query={request_device_model}"
            )

        return kb

    # ========================================================
    # 3. KB 专属 BM25 Index 激活
    # ========================================================

    def _activate_kb_bm25_index(
        self,
        knowledge_base_id: str,
    ) -> dict:
        """
        当前 Hybrid Retriever 的 BM25 使用模块级 _BM25_INDEX_DATA cache。

        Stage3 已经为每个 KB 构建：
            data/bm25/{knowledge_base_id}.pkl

        因此这里在检索前：
            load_index(knowledge_base_id=...)
                -> 校验 Index 属于目标 KB
                -> 写入 Hybrid cache

        这样 BM25 不会跨 Knowledge Base 搜索。
        """
        load_index = getattr(
            self.hybrid_retriever,
            "load_index",
            None,
        )

        if not callable(load_index):
            raise TypeError(
                "hybrid_retriever 必须暴露 bm25_retriever.load_index()"
            )

        index_data = load_index(
            knowledge_base_id=knowledge_base_id
        )

        if not isinstance(index_data, dict):
            raise TypeError("BM25 load_index() 返回值必须是 dict")

        stored_kb_id = index_data.get(
            "knowledge_base_id"
        )

        # Stage3 新索引应明确记录 KB ID。
        if stored_kb_id not in (None, knowledge_base_id):
            raise RuntimeError(
                "BM25 Index 与 Knowledge Base 不一致："
                f"stored={stored_kb_id}, requested={knowledge_base_id}"
            )

        # 再做一次 Chunk 级防御校验。
        mismatched_chunks = []
        for chunk in index_data.get("chunks") or []:
            chunk_kb_id = chunk.get("knowledge_base_id")
            if chunk_kb_id not in (None, knowledge_base_id):
                mismatched_chunks.append(
                    str(chunk.get("chunk_id", "unknown"))
                )

        if mismatched_chunks:
            raise RuntimeError(
                "BM25 Index 中存在跨 Knowledge Base Chunk："
                + ", ".join(mismatched_chunks[:10])
            )

        # 1) 激活传入的 hybrid_retriever 模块 cache。
        setattr(
            self.hybrid_retriever,
            "_BM25_INDEX_DATA",
            index_data,
        )

        # 2) 某些项目启动方式可能让 rerank_pipeline.hybrid_retrieve
        #    指向另一个导入名下的 hybrid_retriever 模块。
        #    尽量同步它，避免 Python 包导入方式差异造成 cache 不一致。
        rerank_hybrid_fn = getattr(
            self.reranker,
            "hybrid_retrieve",
            None,
        )

        if callable(rerank_hybrid_fn):
            module_name = getattr(
                rerank_hybrid_fn,
                "__module__",
                None,
            )
            module = sys.modules.get(module_name)
            if module is not None and hasattr(
                module,
                "_BM25_INDEX_DATA",
            ):
                setattr(
                    module,
                    "_BM25_INDEX_DATA",
                    index_data,
                )

        return index_data

    # ========================================================
    # 4. Query Rewrite
    # ========================================================

    def _rewrite_query(
        self,
        query: str,
        history: list[dict] | None,
        history_summary: str | None,
        kb_device_model: str,
    ) -> dict:
        rewrite_fn = getattr(
            self.query_rewriter,
            "rewrite_query",
            None,
        )

        if not callable(rewrite_fn):
            raise TypeError(
                "query_rewriter 必须提供 rewrite_query()"
            )

        result = rewrite_fn(
            query=query,
            history=history,
            history_summary=history_summary,
        )

        if not isinstance(result, dict):
            raise TypeError("rewrite_query() 返回值必须是 dict")

        original_query = result.get("original_query")
        rewritten_query = result.get("rewritten_query")

        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("Query Rewrite 缺少 original_query")

        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise ValueError("Query Rewrite 缺少 rewritten_query")

        extracted_model = self._normalize_device_model(
            result.get("device_model")
        )
        kb_model = self._normalize_device_model(
            kb_device_model
        )

        # 用户 Query 自己明确写了另一个型号时，立即拒绝。
        if extracted_model and extracted_model != kb_model:
            raise ValueError(
                "Query 中的设备型号与 Knowledge Base 不匹配："
                f"KB={kb_model}, Query={extracted_model}"
            )

        # 如果 Query 本身没写型号，但 API 参数已经明确指定型号，
        # 将可信的业务层型号补到 Query Context，供 Context Builder 使用。
        if not extracted_model:
            result["device_model"] = kb_model
        else:
            result["device_model"] = extracted_model

        return result

    # ========================================================
    # 5. 检索结果 KB Scope 防御检查
    # ========================================================

    @staticmethod
    def _assert_results_scoped(
        results: list[dict],
        knowledge_base_id: str,
        stage_name: str,
    ) -> None:
        """防止任何一层意外混入其他 KB 的 Chunk。"""
        for index, item in enumerate(results or []):
            metadata = item.get("metadata") or {}
            item_kb_id = metadata.get("knowledge_base_id")

            # Day20 Stage3 新数据应该都存在该字段。
            if item_kb_id != knowledge_base_id:
                raise RuntimeError(
                    f"{stage_name} 出现越界 Chunk："
                    f"index={index}, "
                    f"chunk_id={item.get('chunk_id')}, "
                    f"expected_kb={knowledge_base_id}, "
                    f"actual_kb={item_kb_id}"
                )

    # ========================================================
    # 6. Context Builder
    # ========================================================

    def _build_context(
        self,
        query_context: dict,
        final_results: list[dict],
        *,
        token_budget: bool,
        evidence_threshold: float | None,
        block_high_risk: bool,
    ) -> dict:
        build_fn = getattr(
            self.context_builder,
            "build_llm_context_decision",
            None,
        )

        if not callable(build_fn):
            raise TypeError(
                "context_builder 必须提供 build_llm_context_decision()"
            )

        kwargs = {
            "query_context": query_context,
            "chunks": final_results,
            "token_budget": token_budget,
            "block_high_risk": block_high_risk,
        }

        # None 表示沿用 context_builder.py 自己的默认阈值，
        # 避免 Retrieval Service 重复定义 Day18 参数。
        if evidence_threshold is not None:
            kwargs["threshold"] = evidence_threshold

        result = build_fn(**kwargs)

        if not isinstance(result, dict):
            raise TypeError(
                "build_llm_context_decision() 返回值必须是 dict"
            )

        return result

    # ========================================================
    # 7. Retrieval Service 主入口
    # ========================================================

    def retrieve(
        self,
        query: str,
        knowledge_base_id: str,
        device_model: str,
        *,
        history: list[dict] | None = None,
        history_summary: str | None = None,
        recall_k: int = 10,
        rerank_candidate_k: int = 8,
        final_top_k: int = 3,
        vector_threshold: float | None = None,
        rrf_k: int = 60,
        rerank_timeout: float | None = None,
        token_budget: bool = True,
        evidence_threshold: float | None = None,
        block_high_risk: bool = True,
    ) -> dict:
        """
        完整 Stage4 Retrieval Pipeline。

        Query
          -> KB Validation
          -> Query Rewrite
          -> KB-scoped BM25 + Vector
          -> RRF
          -> Reranker / Fallback
          -> Context Builder
          -> Structured Retrieval Result

        本函数不调用 LLM。
        """
        query = self._require_text(query, "query")
        knowledge_base_id = self._require_text(
            knowledge_base_id,
            "knowledge_base_id",
        )
        device_model = self._require_text(
            device_model,
            "device_model",
        )

        # ----------------------------------------------------
        # Step 1: KB Validation
        # ----------------------------------------------------
        kb = self._validate_knowledge_base(
            knowledge_base_id=knowledge_base_id,
            device_model=device_model,
        )

        kb_device_model = str(
            kb.get("device_model")
        ).strip()

        # ----------------------------------------------------
        # Step 2: Query Rewrite
        # ----------------------------------------------------
        query_context = self._rewrite_query(
            query=query,
            history=history,
            history_summary=history_summary,
            kb_device_model=kb_device_model,
        )

        retrieval_query = str(
            query_context["rewritten_query"]
        ).strip()

        # ----------------------------------------------------
        # Step 3: Knowledge Base Scope
        # BM25 -> 专属 KB index
        # Vector -> Chroma where filter
        # ----------------------------------------------------
        self._activate_kb_bm25_index(
            knowledge_base_id
        )

        vector_where = {
            "$and": [
                {"knowledge_base_id": knowledge_base_id},
                {"device_model": kb_device_model},
            ]
        }

        # ----------------------------------------------------
        # Step 4: Hybrid + Reranker
        # run_pipeline 已经内部完成 Hybrid，不重复搜索。
        # ----------------------------------------------------
        run_pipeline = getattr(
            self.reranker,
            "run_pipeline",
            None,
        )

        if not callable(run_pipeline):
            raise TypeError(
                "reranker 必须提供 run_pipeline()"
            )

        rerank_kwargs = {
            "query": retrieval_query,
            "recall_k": recall_k,
            "rerank_candidate_k": rerank_candidate_k,
            "final_top_k": final_top_k,
            "where": vector_where,
            "vector_threshold": vector_threshold,
            "rrf_k": rrf_k,
        }

        if rerank_timeout is not None:
            rerank_kwargs["rerank_timeout"] = rerank_timeout

        rerank_result = run_pipeline(
            **rerank_kwargs
        )

        if not isinstance(rerank_result, dict):
            raise TypeError("run_pipeline() 返回值必须是 dict")

        hybrid_result = rerank_result.get(
            "hybrid_result"
        ) or {}

        bm25_results = hybrid_result.get(
            "bm25_results"
        ) or []
        vector_results = hybrid_result.get(
            "vector_results"
        ) or []
        hybrid_results = hybrid_result.get(
            "hybrid_results"
        ) or []
        final_results = rerank_result.get(
            "final_results"
        ) or []

        rerank_executed = bool(
            rerank_result.get("rerank_executed", False)
        )
        retrieval_mode = (
            "rerank"
            if rerank_executed
            else "hybrid_fallback"
        )
        degraded = not rerank_executed

        # ----------------------------------------------------
        # Step 5: KB Scope 防御检查
        # ----------------------------------------------------
        self._assert_results_scoped(
            bm25_results,
            knowledge_base_id,
            "BM25",
        )
        self._assert_results_scoped(
            vector_results,
            knowledge_base_id,
            "Vector",
        )
        self._assert_results_scoped(
            hybrid_results,
            knowledge_base_id,
            "Hybrid",
        )
        self._assert_results_scoped(
            final_results,
            knowledge_base_id,
            (
                "Rerank"
                if rerank_executed
                else "HybridFallback"
            ),
        )

        # ----------------------------------------------------
        # Step 6: Context Builder
        # ----------------------------------------------------
        context_result = self._build_context(
            query_context=query_context,
            final_results=final_results,
            token_budget=token_budget,
            evidence_threshold=evidence_threshold,
            block_high_risk=block_high_risk,
        )

        evidence_check = context_result.get(
            "evidence_check"
        )

        if isinstance(evidence_check, dict):
            evidence_sufficient = bool(
                evidence_check.get("sufficient", False)
            )
        else:
            evidence_sufficient = False

        # ----------------------------------------------------
        # Step 7: 统一返回
        # 注意：这里只返回检索结果，不生成 LLM Answer。
        # ----------------------------------------------------
        return {
            "knowledge_base": kb,
            "knowledge_base_id": knowledge_base_id,
            "device_model": kb_device_model,

            "query_context": query_context,
            "retrieval_query": retrieval_query,
            "vector_where": vector_where,

            "bm25_results": bm25_results,
            "vector_results": vector_results,
            "hybrid_results": hybrid_results,

            "rerank_results": final_results,
            "rerank_executed": rerank_executed,
            "rerank_error": rerank_result.get(
                "rerank_error"
            ),
            "retrieval_mode": retrieval_mode,
            "degraded": degraded,

            "final_context": context_result.get(
                "context_chunks",
                [],
            ),
            "context_text": context_result.get(
                "context_text",
                "",
            ),
            "sources": context_result.get(
                "sources",
                [],
            ),
            "evidence_sufficient": evidence_sufficient,
            "evidence_mode": (
                evidence_check.get("evidence_mode")
                if isinstance(evidence_check, dict)
                else None
            ),

            "context_status": context_result.get(
                "status"
            ),
            "context_reason": context_result.get(
                "reason"
            ),
            "allow_llm": bool(
                context_result.get("allow_llm", False)
            ),
            "security": context_result.get(
                "security",
                {},
            ),
            "blocked_context_chunks": context_result.get(
                "blocked_context_chunks",
                [],
            ),

            # 调试时保留完整下层结果，后续 Stage5/QA Service 可选择不对外暴露。
            "debug": {
                "hybrid_result": hybrid_result,
                "rerank_result": rerank_result,
                "context_result": context_result,
            },
        }