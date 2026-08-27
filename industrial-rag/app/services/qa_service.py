"""
Day 20 - Stage 7
QA Service - Fault Tolerance Hardened

职责：
1. 调用 Retrieval Service；
2. 只使用 Retrieval Service / Context Builder 已经给出的 Evidence Decision；
3. 证据不足或安全策略不允许时，在业务层直接拒答，绝不调用 LLM；
4. 证据充分时调用 Day18 Prompt Builder；
5. 调用 Gateway /chat；
6. 独立返回 answer / llm_called / sources / latency。

注意：
- QA Service 不重新定义 rerank_score 阈值；
- Retrieval Service 已经完成 Query Rewrite、Hybrid、Rerank、Context Builder；
- allow_llm=False 时，无论其它字段如何，都不得调用 LLM。
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

from app.config import (
    CHAT_MODEL as CONFIG_CHAT_MODEL,
    GATEWAY_BASE_URL,
    LLM_TIMEOUT,
)


# ============================================================
# 1. 配置
# ============================================================

CHAT_URL = os.getenv(
    "CHAT_URL",
    f"{GATEWAY_BASE_URL}/chat",
)
CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    CONFIG_CHAT_MODEL or "",
).strip()
CHAT_TIMEOUT = float(
    os.getenv("CHAT_TIMEOUT", str(LLM_TIMEOUT))
)

REJECT_MESSAGE = (
    "当前知识库中没有找到足够可靠的证据，\n"
    "暂时无法确认该故障的原因和处理步骤。\n\n"
    "请确认设备型号、故障码和文档版本。"
)


class LLMTimeoutError(RuntimeError):
    """Gateway / LLM 调用发生超时。"""




# Day18 的实际 Prompt Builder 接口：
#     from prompt_builder import build_prompt
#     build_prompt(question=..., context=...)
#
# 为了让 test_qa_service.py 可以脱离真实 prompt_builder 做单元验收，
# 这里允许测试时注入一个假的 prompt_builder。
try:
    from prompt_builder import build_prompt as _default_build_prompt
except ImportError:
    _default_build_prompt = None


# ============================================================
# 2. Gateway LLM 调用
# ============================================================

def _extract_answer(data: Any) -> str:
    """
    兼容 Gateway 常见返回结构。

    支持：
    - {"answer": "..."}
    - {"content": "..."}
    - {"response": "..."}
    - {"text": "..."}
    - {"message": {"content": "..."}}
    - OpenAI style:
      {"choices": [{"message": {"content": "..."}}]}
    """
    if not isinstance(data, dict):
        raise ValueError("Gateway 返回 JSON 必须是 dict")

    for key in ("answer", "content", "response", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            first_message = first.get("message")
            if isinstance(first_message, dict):
                content = first_message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    raise ValueError(
        "Gateway 返回中没有找到可识别的回答字段："
        "answer/content/response/text/message.content/choices"
    )


def call_llm(
    prompt: str,
    *,
    chat_url: str = CHAT_URL,
    chat_model: str = CHAT_MODEL,
    timeout: float = CHAT_TIMEOUT,
) -> str:
    """调用现有 Gateway /chat。"""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 不能为空")

    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
        "stream": False,
    }

    if chat_model:
        payload["model"] = chat_model

    try:
        response = requests.post(
            chat_url,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise LLMTimeoutError(
            f"调用 Gateway /chat 超时：timeout={timeout}s"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"调用 Gateway /chat 失败：{exc}"
        ) from exc

    # Gateway 已经把上游 LLM 超时转换为 HTTP 504 时，
    # QA Service 必须保留“超时”语义，不能再次压成普通 502。
    if response.status_code == 504:
        upstream_request_id = None
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                upstream_request_id = error_data.get("request_id")
        except ValueError:
            pass

        request_suffix = (
            f"，upstream_request_id={upstream_request_id}"
            if upstream_request_id
            else ""
        )
        raise LLMTimeoutError(
            "Gateway /chat 返回 504：LLM 服务响应超时"
            + request_suffix
        )

    if response.status_code == 422:
        raise RuntimeError(
            "Gateway /chat 返回 422，通常表示请求体与 Gateway Schema 不一致。"
        )

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"调用 Gateway /chat 失败：{exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Gateway /chat 返回的不是合法 JSON"
        ) from exc

    return _extract_answer(data)


# ============================================================
# 3. Source 构造
# ============================================================

def build_sources(raw_sources: list[dict] | None) -> list[dict]:
    """
    将 Day18 Context Builder 的 sources 转成 QA 业务接口需要的稳定结构。

    Context Builder 当前 source 可包含：
        document
        source_file
        page
        page_start
        page_end
        section
        device_model
        chunk_id

    QA Service 对外只保留最关键的：
        document
        page
        section
        chunk_id
    """
    sources: list[dict] = []
    seen: set[tuple] = set()

    for source in raw_sources or []:
        if not isinstance(source, dict):
            continue

        document = (
            source.get("document")
            or source.get("source_file")
            or "未知文档"
        )
        page = source.get("page")
        section = source.get("section") or "未知"
        chunk_id = source.get("chunk_id")

        # 如果上游没有格式化 page，则 QA 层做一个非常轻量的兜底。
        if page in (None, ""):
            page_start = source.get("page_start")
            page_end = source.get("page_end")

            if page_start is not None and page_end is not None:
                page = (
                    str(page_start)
                    if page_start == page_end
                    else f"{page_start}-{page_end}"
                )
            elif page_start is not None:
                page = str(page_start)
            elif page_end is not None:
                page = str(page_end)
            else:
                page = "未知"

        item = {
            "document": document,
            "page": str(page),
            "section": section,
            "chunk_id": chunk_id,
        }

        dedup_key = (
            item["document"],
            item["page"],
            item["section"],
            item["chunk_id"],
        )
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        sources.append(item)

    return sources


# ============================================================
# 4. Latency 辅助
# ============================================================

def _round_ms(value: float) -> float:
    return round(max(float(value), 0.0), 2)


def _find_latency_ms(
    retrieval_result: dict,
    key: str,
) -> float | None:
    """
    Retrieval Service 当前没有强制输出 rewrite_ms / rerank_ms。

    如果以后下层增加 latency，本函数会自动读取；
    当前没有时返回 None，避免伪造耗时数据。
    """
    candidates: list[Any] = [
        retrieval_result.get("latency"),
    ]

    debug = retrieval_result.get("debug")
    if isinstance(debug, dict):
        candidates.append(debug.get("latency"))

        hybrid_result = debug.get("hybrid_result")
        if isinstance(hybrid_result, dict):
            candidates.append(hybrid_result.get("latency"))

        rerank_result = debug.get("rerank_result")
        if isinstance(rerank_result, dict):
            candidates.append(rerank_result.get("latency"))

        context_result = debug.get("context_result")
        if isinstance(context_result, dict):
            candidates.append(context_result.get("latency"))

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return _round_ms(value)

    return None


# ============================================================
# 5. QA Service
# ============================================================

class QAService:
    """Day20 Stage5 最终问答业务编排层。"""

    def __init__(
        self,
        retrieval_service,
        *,
        prompt_builder: Callable[..., str] | None = None,
        llm_caller: Callable[[str], str] | None = None,
    ):
        self.retrieval_service = retrieval_service

        if prompt_builder is None:
            prompt_builder = _default_build_prompt

        if not callable(prompt_builder):
            raise TypeError(
                "prompt_builder 必须是可调用对象；"
                "正式运行时请确保 prompt_builder.py 提供 build_prompt()"
            )

        self.prompt_builder = prompt_builder
        self.llm_caller = llm_caller or call_llm

        if not callable(self.llm_caller):
            raise TypeError("llm_caller 必须是可调用对象")

    # --------------------------------------------------------
    # 基础校验
    # --------------------------------------------------------

    @staticmethod
    def _require_text(
        value: Any,
        field_name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()

    # --------------------------------------------------------
    # 业务拒答
    # --------------------------------------------------------

    @staticmethod
    def _build_reject_response(
        *,
        knowledge_base_id: str,
        device_model: str,
        retrieval_result: dict,
        retrieval_ms: float,
        total_start: float,
        reason: str,
    ) -> dict:
        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "status": "rejected",
            "answer": REJECT_MESSAGE,
            "llm_called": False,

            # 证据未达到可回答标准时不把弱证据当作正式引用返回。
            "sources": [],

            "knowledge_base_id": knowledge_base_id,
            "device_model": device_model,

            "degraded": bool(
                retrieval_result.get("degraded", False)
            ),
            "retrieval_mode": retrieval_result.get(
                "retrieval_mode"
            ),
            "rerank_executed": bool(
                retrieval_result.get("rerank_executed", False)
            ),
            "rerank_error": retrieval_result.get(
                "rerank_error"
            ),

            "decision": {
                "evidence_sufficient": bool(
                    retrieval_result.get(
                        "evidence_sufficient",
                        False,
                    )
                ),
                "allow_llm": bool(
                    retrieval_result.get(
                        "allow_llm",
                        False,
                    )
                ),
                "context_status": retrieval_result.get(
                    "context_status"
                ),
                "evidence_mode": retrieval_result.get(
                    "evidence_mode"
                ),
                "reason": reason,
            },

            "latency": {
                "rewrite_ms": _find_latency_ms(
                    retrieval_result,
                    "rewrite_ms",
                ),
                "retrieval_ms": _round_ms(retrieval_ms),
                "rerank_ms": _find_latency_ms(
                    retrieval_result,
                    "rerank_ms",
                ),

                # 拒答路径根本没有调用 LLM，因此这里可以准确记为 0。
                "llm_ms": 0.0,
                "total_ms": _round_ms(total_ms),
            },
        }

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def answer(
        self,
        query: str,
        knowledge_base_id: str,
        device_model: str,
        *,
        history: list[dict] | None = None,
        history_summary: str | None = None,
        **retrieval_kwargs,
    ) -> dict:
        """
        完整 QA Pipeline：

        Query
          -> Retrieval Service
          -> Evidence Decision
              -> reject: 业务层直接拒答，不调用 LLM
              -> allow:
                   Final Context
                   -> Day18 Prompt Builder
                   -> Gateway /chat
          -> answer + sources + latency
        """
        total_start = time.perf_counter()

        query = self._require_text(query, "query")
        knowledge_base_id = self._require_text(
            knowledge_base_id,
            "knowledge_base_id",
        )
        device_model = self._require_text(
            device_model,
            "device_model",
        )

        retrieve_fn = getattr(
            self.retrieval_service,
            "retrieve",
            None,
        )
        if not callable(retrieve_fn):
            raise TypeError(
                "retrieval_service 必须提供 retrieve()"
            )

        # ----------------------------------------------------
        # Step 1: Retrieval
        # ----------------------------------------------------
        retrieval_start = time.perf_counter()

        retrieval_result = retrieve_fn(
            query=query,
            knowledge_base_id=knowledge_base_id,
            device_model=device_model,
            history=history,
            history_summary=history_summary,
            **retrieval_kwargs,
        )

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        if not isinstance(retrieval_result, dict):
            raise TypeError(
                "RetrievalService.retrieve() 返回值必须是 dict"
            )

        # ----------------------------------------------------
        # Step 2: Evidence Decision
        # ----------------------------------------------------
        # 不在 QA Service 中重新使用 rerank_score 设阈值。
        # Day18 Context Builder 已经综合判断：
        #   evidence_sufficient
        #   冲突
        #   Prompt Injection
        #   Token Budget
        #
        # allow_llm 是最终安全闸门。
        evidence_sufficient = bool(
            retrieval_result.get(
                "evidence_sufficient",
                False,
            )
        )
        allow_llm = bool(
            retrieval_result.get(
                "allow_llm",
                False,
            )
        )

        if not evidence_sufficient or not allow_llm:
            reason = (
                retrieval_result.get("context_reason")
                or "evidence_not_sufficient_or_llm_not_allowed"
            )

            return self._build_reject_response(
                knowledge_base_id=knowledge_base_id,
                device_model=device_model,
                retrieval_result=retrieval_result,
                retrieval_ms=retrieval_ms,
                total_start=total_start,
                reason=str(reason),
            )

        # ----------------------------------------------------
        # Step 3: Final Context
        # ----------------------------------------------------
        context_text = retrieval_result.get(
            "context_text",
            "",
        )

        if not isinstance(context_text, str) or not context_text.strip():
            # 上游若说 allow_llm=True，却没有真正可进入 Prompt 的 Context，
            # QA 层采用 fail-closed，绝不让 LLM 空证据回答。
            return self._build_reject_response(
                knowledge_base_id=knowledge_base_id,
                device_model=device_model,
                retrieval_result=retrieval_result,
                retrieval_ms=retrieval_ms,
                total_start=total_start,
                reason="allow_llm_but_context_text_empty",
            )

        # ----------------------------------------------------
        # Step 4: Day18 Prompt Builder
        # ----------------------------------------------------
        prompt = self.prompt_builder(
            question=query,
            context=context_text,
        )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "build_prompt() 必须返回非空 str"
            )

        # ----------------------------------------------------
        # Step 5: LLM
        # ----------------------------------------------------
        llm_start = time.perf_counter()

        answer = self.llm_caller(prompt)

        llm_ms = (
            time.perf_counter() - llm_start
        ) * 1000

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "LLM 返回内容必须是非空 str"
            )

        # ----------------------------------------------------
        # Step 6: Sources
        # ----------------------------------------------------
        sources = build_sources(
            retrieval_result.get("sources")
        )

        # ----------------------------------------------------
        # Step 7: Latency
        # ----------------------------------------------------
        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        return {
            "status": "answered",
            "answer": answer.strip(),
            "llm_called": True,
            "sources": sources,

            "knowledge_base_id": knowledge_base_id,
            "device_model": device_model,

            "degraded": bool(
                retrieval_result.get("degraded", False)
            ),
            "retrieval_mode": retrieval_result.get(
                "retrieval_mode"
            ),
            "rerank_executed": bool(
                retrieval_result.get("rerank_executed", False)
            ),
            "rerank_error": retrieval_result.get(
                "rerank_error"
            ),

            "decision": {
                "evidence_sufficient": evidence_sufficient,
                "allow_llm": allow_llm,
                "context_status": retrieval_result.get(
                    "context_status"
                ),
                "evidence_mode": retrieval_result.get(
                    "evidence_mode"
                ),
                "reason": retrieval_result.get(
                    "context_reason"
                ),
            },

            "latency": {
                "rewrite_ms": _find_latency_ms(
                    retrieval_result,
                    "rewrite_ms",
                ),
                "retrieval_ms": _round_ms(retrieval_ms),
                "rerank_ms": _find_latency_ms(
                    retrieval_result,
                    "rerank_ms",
                ),
                "llm_ms": _round_ms(llm_ms),
                "total_ms": _round_ms(total_ms),
            },
        }
