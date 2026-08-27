"""
Day18 Context Builder

Stage3:
1. chunk_id 去重
2. normalized content 去重
3. source 合并

Stage4:
4. Token Budget 控制
5. 基于真实 Tokenizer 选择 Context

Stage5:
6. 设备型号冲突检测
7. 无证据 / 低证据判定
8. 回答状态决策
9. 决定是否允许调用 LLM

Stage6:
10. Prompt Injection 基础检测
11. 高风险 Chunk 隔离，不直接删除整份文档
12. Context 统一来源编号
13. 构造独立的结构化 sources 列表
14. 生成可直接交给 Prompt Builder 的安全 Context

说明：
- build_context() 保持 Stage3/4 兼容，仍返回 List[chunk]
- build_context_decision() 保持 Stage5 兼容，返回决策字典
- prepare_llm_context() 处理已经通过 Stage5 的 context_chunks
- build_llm_context_decision() 为 Stage6 推荐总入口

Stage6 只负责“上下文安全与来源组织”。
系统提示的信任边界、固定回答结构等规则应由 prompt_builder.py 负责。
"""

import os
import re
from copy import deepcopy

from tokenizer import count_tokens
from token_budget import CONTEXT_BUDGET


# ============================================================
# Stage5 状态定义
# ============================================================

ANSWERABLE = "ANSWERABLE"
ANSWERABLE_DEGRADED = "ANSWERABLE_DEGRADED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
DEVICE_CONFLICT = "DEVICE_CONFLICT"
MISSING_DEVICE_MODEL = "MISSING_DEVICE_MODEL"
BLOCKED_CONTEXT = "BLOCKED_CONTEXT"  # Stage6：高风险 Context 被安全策略阻断


# ============================================================
# Stage5 证据阈值配置
# ============================================================

def _get_env_float(name, default):
    """
    从环境变量读取 float。
    环境变量不存在或格式错误时使用 default。
    """
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Day18 只建立“可配置入口”。
# 0.50 只是当前测试默认值，不代表行业标准。
# Day19 应通过评测数据重新确定。
RERANK_MIN_SCORE = _get_env_float(
    "RERANK_MIN_SCORE",
    0.50
)


# ============================================================
# 内容标准化
# ============================================================

def normalize_content(content):
    """
    轻量文本规范化。

    处理：
    - 去除首尾空格
    - 统一换行
    - 合并连续空格
    """

    if not content:
        return ""

    text = content.strip()
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = " ".join(text.split())

    return text


# ============================================================
# 来源格式化
# ============================================================

def extract_source(chunk):
    metadata = chunk.get(
        "metadata",
        {}
    )

    return {
        "source": metadata.get("source"),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
    }


# ============================================================
# Chunk ID 去重
# ============================================================

def deduplicate_chunks(chunks):
    print(
        "\n========== Chunk ID Deduplication =========="
    )

    before = len(chunks)
    unique = {}
    removed = []

    for chunk in chunks:
        chunk_id = chunk.get(
            "chunk_id"
        )

        if chunk_id not in unique:
            unique[chunk_id] = deepcopy(
                chunk
            )
            continue

        old = unique[chunk_id]

        old_score = old.get(
            "rerank_score",
            0
        )

        new_score = chunk.get(
            "rerank_score",
            0
        )

        # None / 非数字时避免比较报错
        try:
            old_score = float(old_score)
        except (TypeError, ValueError):
            old_score = 0.0

        try:
            new_score = float(new_score)
        except (TypeError, ValueError):
            new_score = 0.0

        if new_score > old_score:
            removed.append(
                {
                    "chunk_id": chunk_id,
                    "reason": "replace by higher rerank_score"
                }
            )

            unique[chunk_id] = deepcopy(
                chunk
            )

        else:
            removed.append(
                {
                    "chunk_id": chunk_id,
                    "reason": "duplicate chunk_id"
                }
            )

    result = list(
        unique.values()
    )

    print(
        "Before:",
        before
    )

    print(
        "After:",
        len(result)
    )

    for item in removed:
        print(
            "Removed:",
            item
        )

    return result


# ============================================================
# 正文重复 + 来源合并
# ============================================================

def merge_sources(chunks):
    print(
        "\n========== Content Merge =========="
    )

    merged = {}
    removed = []

    for chunk in chunks:
        content = normalize_content(
            chunk.get(
                "content",
                ""
            )
        )

        if content not in merged:
            new_chunk = deepcopy(
                chunk
            )

            new_chunk["content"] = content

            new_chunk["sources"] = [
                extract_source(
                    chunk
                )
            ]

            merged[content] = new_chunk
            continue

        existing = merged[content]

        source = extract_source(
            chunk
        )

        if source not in existing["sources"]:
            existing["sources"].append(
                source
            )

        # 如果相同正文来自多个 chunk，保留最高 rerank_score
        old_score = existing.get(
            "rerank_score",
            0
        )

        new_score = chunk.get(
            "rerank_score",
            0
        )

        try:
            old_score = float(old_score)
        except (TypeError, ValueError):
            old_score = 0.0

        try:
            new_score = float(new_score)
        except (TypeError, ValueError):
            new_score = 0.0

        if new_score > old_score:
            existing["rerank_score"] = new_score

        removed.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "reason": "duplicate normalized content"
            }
        )

    print(
        "Merged Count:",
        len(merged)
    )

    for item in removed:
        print(
            "Removed:",
            item
        )

    return list(
        merged.values()
    )


# ============================================================
# Token Budget
# ============================================================

def select_chunks_by_budget(
    chunks,
    budget=CONTEXT_BUDGET
):
    """
    根据 Token 预算选择 Chunk。

    当前优先级：
    1. rerank_score 高的优先
    """

    print(
        "\n========== Token Budget Selection =========="
    )

    def score_value(chunk):
        """
        正常模式优先使用 rerank_score。

        Reranker fallback 时 rerank_score=None，
        此时使用 RRF fusion_score；如果 fusion_score 也不存在，
        再使用 hybrid_rank 保持 Hybrid 原始顺序。
        """
        rerank_value = chunk.get("rerank_score")
        try:
            if rerank_value is not None:
                return (2, float(rerank_value))
        except (TypeError, ValueError):
            pass

        fusion_value = chunk.get("fusion_score")
        try:
            if fusion_value is not None:
                return (1, float(fusion_value))
        except (TypeError, ValueError):
            pass

        hybrid_rank = chunk.get("hybrid_rank")
        try:
            # rank 越小越好，取负数后 reverse=True 可保持正确优先级。
            return (0, -float(hybrid_rank))
        except (TypeError, ValueError):
            return (0, 0.0)

    chunks = sorted(
        chunks,
        key=score_value,
        reverse=True
    )

    selected = []
    dropped = []
    used_tokens = 0

    for chunk in chunks:
        content = chunk.get(
            "content",
            ""
        )

        tokens = count_tokens(
            content
        )

        if (
            used_tokens
            +
            tokens
            <=
            budget
        ):
            selected.append(
                chunk
            )

            used_tokens += tokens

        else:
            dropped.append(
                chunk.get(
                    "chunk_id"
                )
            )

    print(
        "Context Budget:",
        budget
    )

    print(
        "Used Tokens:",
        used_tokens
    )

    print(
        "Selected Chunks:",
        len(selected)
    )

    print(
        "Dropped Chunks:",
        len(dropped)
    )

    return selected


# ============================================================
# Stage5 - 设备型号标准化
# ============================================================

def _extract_device_codes(device_model):
    """
    从设备型号文本中提取用于比较的型号代码。

    示例：
        "G120C"            -> ["G120C"]
        "SINAMICS G120C"   -> ["G120C"]
        "S120"             -> ["S120"]
        "V20"              -> ["V20"]

    这里只做轻量规则，不调用 LLM。
    """

    if device_model is None:
        return []

    if isinstance(device_model, (list, tuple, set)):
        result = []

        for item in device_model:
            result.extend(
                _extract_device_codes(
                    item
                )
            )

        return list(
            dict.fromkeys(
                result
            )
        )

    text = str(
        device_model
    ).strip().upper()

    if not text:
        return []

    # 优先提取“包含数字的型号 token”
    # 可识别 G120C / S120 / V20 等。
    codes = re.findall(
        r"\b[A-Z][A-Z0-9_-]*\d[A-Z0-9_-]*\b",
        text
    )

    if codes:
        return list(
            dict.fromkeys(
                codes
            )
        )

    # 如果无法提取，则退化为标准化字符串
    fallback = re.sub(
        r"\s+",
        "",
        text
    )

    if fallback:
        return [
            fallback
        ]

    return []


def _get_chunk_device_models(chunk):
    """
    获取一个 chunk 声明的设备型号。

    Stage5 主要读取：
        chunk["metadata"]["device_model"]

    同时兼容未来 sources 中带 device_model 的情况。
    """

    models = []

    metadata = chunk.get(
        "metadata",
        {}
    ) or {}

    raw_model = metadata.get(
        "device_model"
    )

    models.extend(
        _extract_device_codes(
            raw_model
        )
    )

    # 兼容未来来源合并后的结构
    for source in chunk.get(
        "sources",
        []
    ) or []:

        if not isinstance(
            source,
            dict
        ):
            continue

        models.extend(
            _extract_device_codes(
                source.get(
                    "device_model"
                )
            )
        )

    return list(
        dict.fromkeys(
            models
        )
    )


# ============================================================
# Stage5 - 关卡 A：设备型号冲突检测
# ============================================================

def detect_device_conflict(
    user_device_model,
    chunks
):
    """
    检测用户设备型号与检索资料设备型号是否冲突。

    返回：
    {
        "user_device_models": [...],
        "document_models": [...],
        "matched": [...],
        "mismatch": [...],
        "unknown": [...],
        "has_conflict": bool,
        "multiple_document_models": bool
    }

    规则：
    1. 用户明确型号时：
       - 同型号 -> matched
       - 不同型号 -> mismatch
       - chunk 没有型号 -> unknown

    2. 用户没有型号时：
       - 不猜型号
       - 只统计检索结果中出现了哪些型号
       - 如果出现多个不同型号，由 decide_answer_status()
         进入 MISSING_DEVICE_MODEL
    """

    print(
        "\n========== Device Conflict Detection =========="
    )

    user_models = _extract_device_codes(
        user_device_model
    )

    matched = []
    mismatch = []
    unknown = []

    document_models = []

    for chunk in chunks:
        chunk_models = _get_chunk_device_models(
            chunk
        )

        for model in chunk_models:
            if model not in document_models:
                document_models.append(
                    model
                )

        if not user_models:
            # 用户未给型号，此处不做 matched/mismatch 猜测
            if not chunk_models:
                unknown.append(
                    chunk
                )

            continue

        if not chunk_models:
            unknown.append(
                chunk
            )
            continue

        # 用户型号与 chunk 型号只要有交集，就视为匹配
        if set(user_models) & set(chunk_models):
            matched.append(
                chunk
            )
        else:
            mismatch.append(
                chunk
            )

    has_conflict = bool(
        user_models
        and
        mismatch
    )

    multiple_document_models = (
        len(document_models)
        >
        1
    )

    print(
        "User Device Model:",
        user_device_model
    )

    print(
        "Normalized User Models:",
        user_models
    )

    print(
        "Document Models:",
        document_models
    )

    print(
        "Matched:",
        len(matched)
    )

    print(
        "Mismatch:",
        len(mismatch)
    )

    print(
        "Unknown:",
        len(unknown)
    )

    print(
        "Has Conflict:",
        has_conflict
    )

    return {
        "user_device_models": user_models,
        "document_models": document_models,
        "matched": matched,
        "mismatch": mismatch,
        "unknown": unknown,
        "has_conflict": has_conflict,
        "multiple_document_models": multiple_document_models,
    }


# ============================================================
# Stage5 - 关卡 D：证据充分性检查
# ============================================================

def _safe_float(value):
    """将值安全转换为 float；失败时返回 None。"""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_explicit_hybrid_fallback(chunks):
    """
    判断当前候选是否明确来自 Reranker fallback。

    rerank_pipeline.py 的 fallback 结果会写入：
        rerank_executed = False
        rerank_score = None

    只有明确看到 rerank_executed=False 才进入 fallback policy，
    避免把“字段缺失”误认为“已执行降级”。
    """
    flags = [
        chunk.get("rerank_executed")
        for chunk in (chunks or [])
        if isinstance(chunk, dict)
        and "rerank_executed" in chunk
    ]

    return bool(flags) and all(flag is False for flag in flags)


def _check_hybrid_fallback_evidence(chunks):
    """
    Reranker 不可用时的保守 Hybrid Evidence Policy。

    V1 不拿 fusion_score 冒充 rerank_score，也不设置伪造的
    rerank 阈值。只验证 Hybrid/RRF 结果是否具备完整、可解释的
    检索信号：

    1. 至少存在一个非空 Chunk；
    2. Chunk 有有效 hybrid_rank；
    3. fusion_score 为正数；
    4. BM25 或 Vector 至少一个召回通道命中。

    设备型号冲突、Prompt Injection、Token Budget 仍由原流程继续检查。
    """
    signal_chunks = []
    fusion_scores = []

    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue

        content = str(chunk.get("content") or "").strip()
        if not content:
            continue

        fusion_score = _safe_float(chunk.get("fusion_score"))
        hybrid_rank = _safe_float(chunk.get("hybrid_rank"))

        has_branch_signal = (
            chunk.get("bm25_rank") is not None
            or chunk.get("vector_rank") is not None
        )

        if (
            fusion_score is not None
            and fusion_score > 0
            and hybrid_rank is not None
            and hybrid_rank > 0
            and has_branch_signal
        ):
            signal_chunks.append(chunk)
            fusion_scores.append(fusion_score)

    if not signal_chunks:
        return {
            "sufficient": False,
            "reason": "NO_VALID_HYBRID_SIGNAL",
            "evidence_mode": "hybrid_fallback",
            "degraded": True,
            "max_rerank_score": None,
            "max_fusion_score": None,
            "threshold": None,
            "scored_chunks": 0,
            "signal_chunks": 0,
        }

    return {
        "sufficient": True,
        "reason": "HYBRID_FALLBACK_ACCEPTED",
        "evidence_mode": "hybrid_fallback",
        "degraded": True,
        "max_rerank_score": None,
        "max_fusion_score": max(fusion_scores),
        "threshold": None,
        "scored_chunks": 0,
        "signal_chunks": len(signal_chunks),
    }


def check_evidence_sufficiency(
    chunks,
    threshold=RERANK_MIN_SCORE
):
    """
    检查当前候选证据是否足够。

    两种模式：

    1. 正常 Reranker 模式
       - 有效 rerank_score >= threshold -> sufficient=True
       - 无 rerank_score / 分数过低 -> 拒答

    2. Hybrid Fallback 模式
       - 仅当 Chunk 明确标记 rerank_executed=False 时进入；
       - 不使用 rerank_score 阈值；
       - 使用 Hybrid/RRF 检索信号做保守判定；
       - 通过后标记 evidence_mode=hybrid_fallback、degraded=True。
    """
    print(
        "\n========== Evidence Sufficiency =========="
    )

    if not chunks:
        result = {
            "sufficient": False,
            "reason": "NO_RESULTS",
            "evidence_mode": "none",
            "degraded": False,
            "max_rerank_score": None,
            "max_fusion_score": None,
            "threshold": threshold,
            "scored_chunks": 0,
            "signal_chunks": 0,
        }

        print("Evidence:", result)
        return result

    # --------------------------------------------------------
    # A. Reranker 明确失败：进入 Hybrid fallback policy
    # --------------------------------------------------------
    if _is_explicit_hybrid_fallback(chunks):
        result = _check_hybrid_fallback_evidence(chunks)
        print("Evidence:", result)
        return result

    # --------------------------------------------------------
    # B. 正常模式：继续使用 rerank_score
    # --------------------------------------------------------
    scores = []

    for chunk in chunks:
        score = _safe_float(chunk.get("rerank_score"))
        if score is not None:
            scores.append(score)

    if not scores:
        result = {
            "sufficient": False,
            "reason": "NO_RERANK_SCORE",
            "evidence_mode": "rerank",
            "degraded": False,
            "max_rerank_score": None,
            "max_fusion_score": None,
            "threshold": threshold,
            "scored_chunks": 0,
            "signal_chunks": 0,
        }

        print("Evidence:", result)
        return result

    max_score = max(scores)

    if max_score < threshold:
        result = {
            "sufficient": False,
            "reason": "LOW_RERANK_SCORE",
            "evidence_mode": "rerank",
            "degraded": False,
            "max_rerank_score": max_score,
            "max_fusion_score": None,
            "threshold": threshold,
            "scored_chunks": len(scores),
            "signal_chunks": 0,
        }

        print("Evidence:", result)
        return result

    result = {
        "sufficient": True,
        "reason": "SUFFICIENT",
        "evidence_mode": "rerank",
        "degraded": False,
        "max_rerank_score": max_score,
        "max_fusion_score": None,
        "threshold": threshold,
        "scored_chunks": len(scores),
        "signal_chunks": 0,
    }

    print("Evidence:", result)
    return result


# ============================================================
# Stage5 - 关卡 E：回答状态决策
# ============================================================

def decide_answer_status(
    query_context,
    chunks,
    threshold=RERANK_MIN_SCORE
):
    """
    根据 Query Context + Retrieval Chunks 决定是否允许进入 LLM。

    Day18 使用保守策略：
    - DEVICE_CONFLICT -> 不调用 LLM
    - MISSING_DEVICE_MODEL -> 不调用 LLM
    - INSUFFICIENT_EVIDENCE -> 不调用 LLM
    - 只有 ANSWERABLE -> allow_llm=True

    返回结构：
    {
        "status": "...",
        "allow_llm": bool,
        "reason": "...",
        "message": "...",
        "valid_chunks": [...],
        "rejected_chunks": [...],
        "unknown_model_chunks": [...],
        "device_check": {...},
        "evidence_check": {...}
    }
    """

    print(
        "\n========== Answer Status Decision =========="
    )

    query_context = query_context or {}
    chunks = chunks or []

    # --------------------------------------------------------
    # 1. 完全没有检索结果
    # --------------------------------------------------------

    if not chunks:
        evidence = check_evidence_sufficiency(
            chunks,
            threshold=threshold
        )

        return {
            "status": INSUFFICIENT_EVIDENCE,
            "allow_llm": False,
            "reason": evidence["reason"],
            "message": (
                "当前没有检索到可用于回答该问题的资料，"
                "因此不调用 LLM 生成答案。"
            ),
            "valid_chunks": [],
            "rejected_chunks": [],
            "unknown_model_chunks": [],
            "device_check": None,
            "evidence_check": evidence,
        }

    user_device_model = query_context.get(
        "device_model"
    )

    device_check = detect_device_conflict(
        user_device_model,
        chunks
    )

    # --------------------------------------------------------
    # 2. 用户没给型号，但检索到了多个不同设备型号
    # --------------------------------------------------------

    if (
        not device_check["user_device_models"]
        and
        device_check["multiple_document_models"]
    ):
        return {
            "status": MISSING_DEVICE_MODEL,
            "allow_llm": False,
            "reason": "MULTIPLE_DEVICE_MODELS",
            "message": (
                "当前检索到多个设备型号的资料，"
                "暂时无法确定应该采用哪一份说明。"
                "请先确认设备型号。"
            ),
            "valid_chunks": [],
            "rejected_chunks": [],
            "unknown_model_chunks": device_check["unknown"],
            "device_check": device_check,
            "evidence_check": None,
        }

    # --------------------------------------------------------
    # 3. 用户明确型号，但出现其他型号资料
    # --------------------------------------------------------

    if device_check["has_conflict"]:
        # matched 保留下来用于诊断与后续策略调整，
        # 但 Day18 当前采用保守策略，不继续调用 LLM。
        evidence_target = device_check["matched"]

        evidence = check_evidence_sufficiency(
            evidence_target,
            threshold=threshold
        )

        return {
            "status": DEVICE_CONFLICT,
            "allow_llm": False,
            "reason": "DEVICE_MODEL_MISMATCH",
            "message": (
                "检索结果中同时出现了与用户设备型号不一致的资料。"
                "已将不同型号资料标记为 mismatch，"
                "当前不允许把这些资料混合发送给 LLM。"
            ),
            "valid_chunks": device_check["matched"],
            "rejected_chunks": device_check["mismatch"],
            "unknown_model_chunks": device_check["unknown"],
            "device_check": device_check,
            "evidence_check": evidence,
        }

    # --------------------------------------------------------
    # 4. 确定真正用于证据检查的 chunks
    # --------------------------------------------------------

    if device_check["user_device_models"]:
        # 用户有型号：
        # - 有明确 matched -> 只信任 matched
        # - 所有文档都没有 device_model -> 暂时使用原 chunks，
        #   因为此时系统“无法验证型号”，但也没有明确冲突证据
        if device_check["matched"]:
            candidate_chunks = device_check["matched"]
        else:
            candidate_chunks = chunks
    else:
        # 用户没型号，但只有 0/1 个文档型号，不形成多型号歧义
        candidate_chunks = chunks

    # --------------------------------------------------------
    # 5. 证据分数检查
    # --------------------------------------------------------

    evidence = check_evidence_sufficiency(
        candidate_chunks,
        threshold=threshold
    )

    if not evidence["sufficient"]:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "allow_llm": False,
            "reason": evidence["reason"],
            "message": (
                "当前检索结果的证据不足，"
                "因此不调用 LLM 生成确定性答案。"
            ),
            "valid_chunks": candidate_chunks,
            "rejected_chunks": [],
            "unknown_model_chunks": device_check["unknown"],
            "device_check": device_check,
            "evidence_check": evidence,
        }

    # --------------------------------------------------------
    # 6. 允许进入 LLM
    # --------------------------------------------------------

    degraded = bool(evidence.get("degraded", False))
    evidence_mode = evidence.get("evidence_mode", "rerank")

    return {
        "status": (
            ANSWERABLE_DEGRADED
            if degraded
            else ANSWERABLE
        ),
        "allow_llm": True,
        "reason": (
            "HYBRID_FALLBACK_ACCEPTED"
            if degraded
            else "EVIDENCE_ACCEPTED"
        ),
        "message": (
            "Reranker 当前不可用，已使用 Hybrid/RRF 结果完成降级证据检查，"
            "允许进入 LLM 生成阶段。"
            if degraded
            else
            "设备型号与证据检查通过，可以进入 LLM 生成阶段。"
        ),
        "degraded": degraded,
        "retrieval_mode": evidence_mode,
        "valid_chunks": candidate_chunks,
        "rejected_chunks": [],
        "unknown_model_chunks": device_check["unknown"],
        "device_check": device_check,
        "evidence_check": evidence,
    }


# ============================================================
# Stage3/4 兼容入口
# ============================================================

def build_context(
    chunks,
    token_budget=True
):
    """
    原 Stage3/4 Context Builder 入口。

    Pipeline:
    1. chunk_id dedup
    2. content merge
    3. token budget

    注意：
    该函数继续返回 List[chunk]，
    防止破坏之前已经接好的代码。
    """

    print(
        "\n========== Context Builder =========="
    )

    print(
        "Input chunks:",
        len(chunks)
    )

    chunks = deduplicate_chunks(
        chunks
    )

    chunks = merge_sources(
        chunks
    )

    if token_budget:
        chunks = select_chunks_by_budget(
            chunks
        )

    print(
        "\nFinal Context:",
        len(chunks)
    )

    return chunks


# ============================================================
# Stage5 新入口
# ============================================================

def build_context_decision(
    query_context,
    chunks,
    token_budget=True,
    threshold=RERANK_MIN_SCORE
):
    """
    Stage5 推荐入口。

    Pipeline:

    Reranker Output
          |
          v
    chunk_id dedup
          |
          v
    Device Conflict / Missing Model
          |
          v
    Evidence Sufficiency
          |
          +---- 不通过 ----> status != ANSWERABLE
          |                 allow_llm=False
          |
          v
    content merge
          |
          v
    Token Budget
          |
          v
    ANSWERABLE
    allow_llm=True
          |
          v
    LLM

    返回：
    {
        "status": ...,
        "allow_llm": ...,
        "reason": ...,
        "message": ...,
        "context_chunks": [...],     # 真正允许送入 LLM 的 chunks
        "valid_chunks": [...],       # 设备过滤后仍有效的 chunks
        "rejected_chunks": [...],    # mismatch
        ...
    }
    """

    print(
        "\n============================================"
    )
    print(
        "Stage5 Context Decision"
    )
    print(
        "============================================"
    )

    chunks = chunks or []

    # 无结果直接决策，不做无意义的后续处理
    if not chunks:
        decision = decide_answer_status(
            query_context,
            [],
            threshold=threshold
        )

        decision["context_chunks"] = []

        return decision

    # 先按 chunk_id 去重。
    # 型号冲突必须在 content merge 之前检测，
    # 防止两个不同型号但正文相同的 chunk 被提前合并。
    deduped_chunks = deduplicate_chunks(
        chunks
    )

    decision = decide_answer_status(
        query_context,
        deduped_chunks,
        threshold=threshold
    )

    # 非 ANSWERABLE：禁止构造 LLM Context
    if not decision["allow_llm"]:
        decision["context_chunks"] = []

        return decision

    # 只有通过决策的 valid_chunks 才继续做正文合并
    final_chunks = merge_sources(
        decision["valid_chunks"]
    )

    # 再做 Token Budget
    if token_budget:
        final_chunks = select_chunks_by_budget(
            final_chunks
        )

    # 极端情况：
    # 有证据，但所有 chunk 都因为 Token Budget 被丢弃
    if not final_chunks:
        decision.update(
            {
                "status": INSUFFICIENT_EVIDENCE,
                "allow_llm": False,
                "reason": "TOKEN_BUDGET_EMPTY",
                "message": (
                    "候选证据存在，但在当前 Token Budget 下"
                    "没有任何 Chunk 能进入上下文，因此不调用 LLM。"
                ),
                "context_chunks": [],
            }
        )

        return decision

    decision["context_chunks"] = final_chunks

    return decision



# ============================================================
# Stage6 - Prompt Injection 基础检测
# ============================================================

# 第一版只做“可解释的规则检测”，不调用 LLM。
# 规则分为 HIGH / MEDIUM：
# - HIGH：建议阻断当前 Chunk，不送入 LLM
# - MEDIUM：仅标记风险，仍可作为事实证据使用
#
# 注意：这里阻断的是 Chunk，不是整份文档。
INJECTION_RULES = (
    {
        "name": "IGNORE_PREVIOUS_INSTRUCTIONS_ZH",
        "display": "忽略之前的指令",
        "severity": "HIGH",
        "pattern": r"忽略\s*(?:之前|以上|前面)?\s*(?:的)?\s*(?:指令|要求|规则)",
    },
    {
        "name": "IGNORE_SYSTEM_RULES_ZH",
        "display": "忽略系统要求",
        "severity": "HIGH",
        "pattern": r"忽略\s*(?:系统|开发者|应用)\s*(?:指令|要求|规则|提示)",
    },
    {
        "name": "MODIFY_SYSTEM_PROMPT_ZH",
        "display": "修改系统提示",
        "severity": "HIGH",
        "pattern": r"(?:修改|覆盖|替换|重写)\s*(?:系统提示|系统指令|system\s*prompt)",
    },
    {
        "name": "DO_NOT_FOLLOW_RULES_ZH",
        "display": "不要遵守规则",
        "severity": "HIGH",
        "pattern": r"不要\s*(?:遵守|服从|执行)\s*(?:规则|指令|系统要求|系统提示)",
    },
    {
        "name": "REVEAL_SECRET_ZH",
        "display": "输出密码",
        "severity": "HIGH",
        "pattern": r"(?:输出|显示|泄露|返回|告诉我).{0,12}(?:密码|口令|密钥|API\s*KEY|TOKEN)",
    },
    {
        "name": "ADMIN_PASSWORD_ZH",
        "display": "管理员密码",
        "severity": "HIGH",
        "pattern": r"(?:管理员|root|admin).{0,8}(?:密码|口令|密钥)",
    },
    {
        "name": "EXECUTE_COMMAND_ZH",
        "display": "执行以下命令",
        "severity": "HIGH",
        "pattern": r"(?:执行|运行)\s*(?:以下|下面|下列)\s*(?:命令|指令|代码|脚本)",
    },
    {
        "name": "ROLE_OVERRIDE_ZH",
        "display": "你现在是",
        "severity": "MEDIUM",
        "pattern": r"你现在是\s*[^。！？\n]{1,40}",
    },
    {
        "name": "IGNORE_PREVIOUS_INSTRUCTIONS_EN",
        "display": "ignore previous instructions",
        "severity": "HIGH",
        "pattern": r"\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions?|rules?|prompts?)\b",
    },
    {
        "name": "REVEAL_SYSTEM_PROMPT_EN",
        "display": "reveal system prompt",
        "severity": "HIGH",
        "pattern": r"\b(?:reveal|show|print|output|leak)\b.{0,20}\b(?:system\s+prompt|developer\s+message|secret|password|api\s*key|token)\b",
    },
    {
        "name": "ROLE_OVERRIDE_EN",
        "display": "you are now / act as",
        "severity": "MEDIUM",
        "pattern": r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b",
    },
)


_RISK_LEVEL_ORDER = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def _max_risk_level(current, new):
    """返回两个风险级别中更高的一个。"""
    current = str(current or "NONE").upper()
    new = str(new or "NONE").upper()

    if _RISK_LEVEL_ORDER.get(new, 0) > _RISK_LEVEL_ORDER.get(current, 0):
        return new

    return current


def detect_prompt_injection(text):
    """
    检测单段文本是否存在基础 Prompt Injection 风险。

    第一版原则：
    1. 不改写原始内容；
    2. 不因为出现一个关键词就删除整份文档；
    3. HIGH 风险建议阻断当前 Chunk；
    4. MEDIUM 风险只标记，由上层继续使用系统信任边界约束。

    返回：
    {
        "injection_risk": bool,
        "risk_level": "NONE" | "MEDIUM" | "HIGH",
        "matched_patterns": [...],
        "matches": [
            {
                "name": "...",
                "pattern": "...",
                "severity": "HIGH",
                "matched_text": "..."
            }
        ],
        "block_recommended": bool
    }
    """
    if text is None:
        text = ""

    text = str(text)
    matches = []
    matched_patterns = []
    risk_level = "NONE"

    for rule in INJECTION_RULES:
        found = re.search(
            rule["pattern"],
            text,
            flags=re.IGNORECASE,
        )

        if not found:
            continue

        display = rule["display"]

        if display not in matched_patterns:
            matched_patterns.append(display)

        matches.append(
            {
                "name": rule["name"],
                "pattern": display,
                "severity": rule["severity"],
                "matched_text": found.group(0),
            }
        )

        risk_level = _max_risk_level(
            risk_level,
            rule["severity"],
        )

    return {
        "injection_risk": bool(matches),
        "risk_level": risk_level,
        "matched_patterns": matched_patterns,
        "matches": matches,
        "block_recommended": risk_level == "HIGH",
    }


# ============================================================
# Stage6 - Chunk 级安全扫描
# ============================================================

def scan_chunks_for_prompt_injection(
    chunks,
    block_high_risk=True,
):
    """
    对 Context Chunks 做 Prompt Injection 扫描。

    HIGH 风险时：
    - block_high_risk=True  -> 仅隔离该 Chunk
    - block_high_risk=False -> 保留该 Chunk，但继续标记风险

    返回：
    {
        "safe_chunks": [...],
        "blocked_chunks": [...],
        "flagged_chunks": [...],
        "injection_risk": bool,
        "highest_risk_level": "NONE" | "MEDIUM" | "HIGH",
        "matched_patterns": [...],
        "total_chunks": int,
        "safe_count": int,
        "blocked_count": int,
        "flagged_count": int
    }
    """
    chunks = chunks or []

    safe_chunks = []
    blocked_chunks = []
    flagged_chunks = []
    matched_patterns = []
    highest_risk_level = "NONE"

    for chunk in chunks:
        item = deepcopy(chunk)
        content = item.get("content", "")

        security = detect_prompt_injection(content)
        item["security"] = security

        if security["injection_risk"]:
            flagged_chunks.append(item)

        highest_risk_level = _max_risk_level(
            highest_risk_level,
            security["risk_level"],
        )

        for pattern in security["matched_patterns"]:
            if pattern not in matched_patterns:
                matched_patterns.append(pattern)

        should_block = (
            block_high_risk
            and security["block_recommended"]
        )

        if should_block:
            blocked_chunks.append(item)
        else:
            safe_chunks.append(item)

    return {
        "safe_chunks": safe_chunks,
        "blocked_chunks": blocked_chunks,
        "flagged_chunks": flagged_chunks,
        "injection_risk": bool(flagged_chunks),
        "highest_risk_level": highest_risk_level,
        "matched_patterns": matched_patterns,
        "total_chunks": len(chunks),
        "safe_count": len(safe_chunks),
        "blocked_count": len(blocked_chunks),
        "flagged_count": len(flagged_chunks),
    }


# ============================================================
# Stage6 - 来源字段标准化
# ============================================================

def _first_nonempty(*values):
    """返回第一个非 None、非空字符串的值。"""
    for value in values:
        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

            if not value:
                continue

        return value

    return None


def _format_page(page_start, page_end=None):
    """把 page_start/page_end 统一成人类可读页码。"""
    if page_start is None and page_end is None:
        return "未知"

    if page_start is None:
        return str(page_end)

    if page_end is None:
        return str(page_start)

    if str(page_start) == str(page_end):
        return str(page_start)

    return f"{page_start}-{page_end}"


def build_source_metadata(
    chunk,
    source_id,
):
    """
    把单个 Context Chunk 转换为前端可使用的结构化来源信息。

    字段兼容：
    document:
        document / document_title / title / source

    section:
        section / section_title / chapter

    page:
        page / page_start / page_end
    """
    chunk = chunk or {}
    metadata = chunk.get("metadata", {}) or {}

    document = _first_nonempty(
        metadata.get("document"),
        metadata.get("document_title"),
        metadata.get("title"),
        metadata.get("source"),
    )

    source_file = _first_nonempty(
        metadata.get("source"),
        metadata.get("file_name"),
        metadata.get("filename"),
    )

    section = _first_nonempty(
        metadata.get("section"),
        metadata.get("section_title"),
        metadata.get("chapter"),
    )

    page_start = _first_nonempty(
        metadata.get("page_start"),
        metadata.get("page"),
    )

    page_end = metadata.get("page_end")

    device_model = _first_nonempty(
        metadata.get("device_model"),
        metadata.get("model"),
    )

    return {
        "source_id": source_id,
        "document": document or "未知文档",
        "source_file": source_file,
        "page": _format_page(
            page_start,
            page_end,
        ),
        "page_start": page_start,
        "page_end": page_end,
        "section": section or "未知",
        "device_model": device_model or "未知",
        "chunk_id": chunk.get("chunk_id"),
    }


def extract_sources(chunks):
    """
    独立构造结构化 sources 列表。

    该列表由程序生成，而不是依赖 LLM 自己生成，
    方便后续前端制作来源引用卡片。
    """
    chunks = chunks or []

    return [
        build_source_metadata(
            chunk,
            source_id=index,
        )
        for index, chunk in enumerate(chunks, 1)
    ]


# ============================================================
# Stage6 - 统一来源格式
# ============================================================

def format_context_with_sources(chunks):
    """
    将 Context Chunks 格式化成带 [来源X] 编号的 LLM Context。

    返回：
    {
        "context_text": "...",
        "sources": [...]
    }

    注意：
    - source_id 与 context_text 中的 [来源X] 一一对应；
    - 这里只组织事实证据，不在此处写 System Prompt；
    - Prompt 的信任边界由 prompt_builder.py 声明。
    """
    chunks = chunks or []
    sources = extract_sources(chunks)
    context_parts = []

    for chunk, source in zip(chunks, sources):
        content = chunk.get("content", "")

        block = (
            f"[来源{source['source_id']}]\n"
            f"文档：{source['document']}\n"
            f"章节：{source['section']}\n"
            f"页码：{source['page']}\n"
            f"设备型号：{source['device_model']}\n"
            f"Chunk ID：{source['chunk_id']}\n"
            f"内容：\n{content}"
        )

        context_parts.append(block)

    return {
        "context_text": "\n\n".join(context_parts),
        "sources": sources,
    }


# ============================================================
# Stage6 - 已通过 Stage5 的 Context -> LLM Context
# ============================================================

def prepare_llm_context(
    chunks,
    block_high_risk=True,
):
    """
    对已经通过 Stage5 的 context_chunks 做 Stage6 处理。

    Pipeline:
        context_chunks
             |
             v
        Prompt Injection Scan
             |
             +---- HIGH ----> 隔离当前 Chunk
             |
             v
        Safe Chunks
             |
             v
        [来源X] 格式化
             |
             +----> context_text
             |
             +----> sources[]

    返回结果可以直接交给 prompt_builder.py。
    """
    chunks = chunks or []

    if not chunks:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "allow_llm": False,
            "reason": "NO_CONTEXT_CHUNKS",
            "message": "没有可用于构造 LLM Prompt 的 Context Chunk。",
            "context_text": "",
            "context": "",
            "context_chunks": [],
            "sources": [],
            "blocked_chunks": [],
            "security": {
                "injection_risk": False,
                "highest_risk_level": "NONE",
                "matched_patterns": [],
                "total_chunks": 0,
                "safe_count": 0,
                "blocked_count": 0,
                "flagged_count": 0,
            },
        }

    security = scan_chunks_for_prompt_injection(
        chunks,
        block_high_risk=block_high_risk,
    )

    safe_chunks = security["safe_chunks"]

    # 高风险隔离后一个安全 Chunk 都不剩：禁止调用 LLM。
    if not safe_chunks:
        return {
            "status": BLOCKED_CONTEXT,
            "allow_llm": False,
            "reason": "ALL_CONTEXT_BLOCKED_BY_INJECTION",
            "message": (
                "候选 Context 均命中高风险 Prompt Injection 规则，"
                "已阻止这些 Chunk 进入 LLM。"
            ),
            "context_text": "",
            "context": "",
            "context_chunks": [],
            "sources": [],
            "blocked_chunks": security["blocked_chunks"],
            "security": {
                key: value
                for key, value in security.items()
                if key not in {
                    "safe_chunks",
                    "blocked_chunks",
                    "flagged_chunks",
                }
            },
        }

    formatted = format_context_with_sources(
        safe_chunks
    )

    security_summary = {
        key: value
        for key, value in security.items()
        if key not in {
            "safe_chunks",
            "blocked_chunks",
            "flagged_chunks",
        }
    }

    return {
        "status": ANSWERABLE,
        "allow_llm": True,
        "reason": "SAFE_CONTEXT_READY",
        "message": (
            "Context 已完成来源编号和 Prompt Injection 检查，"
            "可以交给 Prompt Builder。"
        ),
        "context_text": formatted["context_text"],
        # 兼容前面讨论过的 context 字段命名。
        "context": formatted["context_text"],
        "context_chunks": safe_chunks,
        "sources": formatted["sources"],
        "blocked_chunks": security["blocked_chunks"],
        "security": security_summary,
    }


# ============================================================
# Stage6 推荐总入口
# ============================================================

def build_llm_context_decision(
    query_context,
    chunks,
    token_budget=True,
    threshold=RERANK_MIN_SCORE,
    block_high_risk=True,
):
    """
    Day18 Stage6 推荐总入口。

    完整 Pipeline：

        Reranker Output
              |
              v
        Stage5 Context Decision
        - chunk_id 去重
        - 设备型号冲突
        - 证据充分性
        - 正文去重 / 来源合并
        - Token Budget
              |
              +---- Stage5 不通过 ----> allow_llm=False
              |
              v
        Stage6 Security
        - Prompt Injection 检测
        - HIGH 风险 Chunk 隔离
              |
              +---- 全部被阻断 -------> BLOCKED_CONTEXT
              |
              v
        Source Formatter
        - [来源1] / [来源2]
        - 独立 sources[]
              |
              v
        Prompt Builder

    返回：
    {
        "status": ...,
        "allow_llm": ...,
        "context_chunks": [...],
        "context_text": "...",
        "sources": [...],
        "security": {...},
        "blocked_context_chunks": [...],
        ... Stage5 原始决策字段
    }
    """
    decision = build_context_decision(
        query_context=query_context,
        chunks=chunks,
        token_budget=token_budget,
        threshold=threshold,
    )

    # Stage5 已拒绝：Stage6 不应绕过前面的决策。
    if not decision["allow_llm"]:
        decision["context_text"] = ""
        decision["context"] = ""
        decision["sources"] = []
        decision["security"] = {
            "injection_risk": False,
            "highest_risk_level": "NONE",
            "matched_patterns": [],
            "total_chunks": 0,
            "safe_count": 0,
            "blocked_count": 0,
            "flagged_count": 0,
            "stage6_skipped": True,
        }
        decision["blocked_context_chunks"] = []

        return decision

    stage6 = prepare_llm_context(
        decision["context_chunks"],
        block_high_risk=block_high_risk,
    )

    decision["context_text"] = stage6["context_text"]
    decision["context"] = stage6["context"]
    decision["sources"] = stage6["sources"]
    decision["security"] = stage6["security"]
    decision["blocked_context_chunks"] = stage6["blocked_chunks"]

    # Stage6 安全检查可能把 Stage5 的 ANSWERABLE 改成 BLOCKED_CONTEXT。
    if not stage6["allow_llm"]:
        decision["status"] = stage6["status"]
        decision["allow_llm"] = False
        decision["reason"] = stage6["reason"]
        decision["message"] = stage6["message"]
        decision["context_chunks"] = []

        return decision

    # 只允许安全 Chunk 继续进入 Prompt Builder。
    decision["context_chunks"] = stage6["context_chunks"]
    decision["allow_llm"] = True

    evidence_check = decision.get("evidence_check") or {}
    degraded = bool(evidence_check.get("degraded", False))
    evidence_mode = evidence_check.get("evidence_mode", "rerank")

    decision["degraded"] = degraded
    decision["retrieval_mode"] = evidence_mode

    if degraded:
        decision["status"] = ANSWERABLE_DEGRADED
        decision["reason"] = "SAFE_CONTEXT_READY_DEGRADED"
        decision["message"] = (
            "Reranker 不可用，系统已使用 Hybrid/RRF fallback；"
            "设备、降级证据、Token Budget、Prompt Injection 和来源格式检查均通过，"
            "可以进入 Prompt Builder / LLM 阶段。"
        )
    else:
        decision["status"] = ANSWERABLE
        decision["reason"] = "SAFE_CONTEXT_READY"
        decision["message"] = (
            "设备、证据、Token Budget、Prompt Injection 和来源格式检查均通过，"
            "可以进入 Prompt Builder / LLM 阶段。"
        )

    return decision


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":
    test_query_context = {
        "original_query": "这个怎么处理？",
        "rewritten_query": "G120C 出现 F30021 故障时应该如何处理？",
        "device_model": "G120C",
        "fault_code": "F30021",
    }

    test_chunks = [
        {
            "chunk_id": "001",
            "content": "G120C：F30021 表示过电流故障。",
            "rerank_score": 0.95,
            "metadata": {
                "source": "g120c_manual.pdf",
                "page_start": 120,
                "device_model": "G120C",
            },
        },
        {
            "chunk_id": "002",
            "content": "S120：F30021 的说明。",
            "rerank_score": 0.90,
            "metadata": {
                "source": "s120_manual.pdf",
                "page_start": 36,
                "device_model": "S120",
            },
        },
    ]

    result = build_context_decision(
        query_context=test_query_context,
        chunks=test_chunks,
        token_budget=False,
    )

    print(
        "\n========== Decision Result =========="
    )
    print(
        "Status:",
        result["status"]
    )
    print(
        "Allow LLM:",
        result["allow_llm"]
    )
    print(
        "Reason:",
        result["reason"]
    )
    print(
        "Valid:",
        len(result["valid_chunks"])
    )
    print(
        "Rejected:",
        len(result["rejected_chunks"])
    )
    print(
        "Context:",
        len(result["context_chunks"])
    )
