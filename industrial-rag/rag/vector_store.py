"""
Vector Store 核心模块

职责：

1. 连接 / 创建 Chroma Collection
2. 调用 Embedding Service
3. Embedding 请求分批、413 自动降批、失败重试
4. 新增 / 更新 Chunk
5. Chroma 分批 upsert
6. 删除 Document
7. 数据统计与文档查询
8. Vector Search
9. Metadata Filter
10. Distance Threshold

正式 RAG 主链路：

Chunk[]
    ↓
add_documents()
    ↓
分批处理 Chunk
    ↓
get_embeddings()
    ↓
Embedding 小批次请求
    ↓
413 时自动二分降批
    ↓
Chroma 分批 upsert

Query
    ↓
search()
    ↓
Embedding Service
    ↓
Chroma
    ↓
Metadata Filter
    ↓
Top-K
    ↓
Threshold
    ↓
标准化检索结果
"""

from __future__ import annotations

import time
from typing import Any

import chromadb
import requests

try:
    # 作为 rag.vector_store 导入时使用。
    from .config import (
        COLLECTION_NAME,
        DISTANCE_THRESHOLD,
        EMBEDDING_TIMEOUT,
        EMBEDDING_URL,
        PERSIST_DIRECTORY,
        TOP_K,
    )
except ImportError:
    # 兼容原来的 `python vector_store.py` / 从 rag 目录运行方式。
    from config import (
        COLLECTION_NAME,
        DISTANCE_THRESHOLD,
        EMBEDDING_TIMEOUT,
        EMBEDDING_URL,
        PERSIST_DIRECTORY,
        TOP_K,
    )


UNKNOWN_VALUE = "unknown"

# ============================================================
# 批处理配置
# ============================================================

# 每次向 Embedding 服务发送多少条文本。
# 当前服务曾出现 HTTP 413，因此默认从较保守的 4 开始。
DEFAULT_EMBEDDING_BATCH_SIZE = 4

# 每次向 Chroma upsert 的 Chunk 数量。
DEFAULT_CHROMA_UPSERT_BATCH_SIZE = 64

# 仅对临时性 HTTP 错误进行重试。
DEFAULT_EMBEDDING_MAX_RETRIES = 3
DEFAULT_EMBEDDING_RETRY_BACKOFF = 1.5

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# 1. Collection
# ============================================================

def get_collection():
    """
    获取或创建持久化 Chroma Collection。

    Embedding 由自己的服务生成，
    Chroma 只负责向量存储和检索。
    """

    PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


# 兼容之前代码中的命名。
create_collection = get_collection


# ============================================================
# 2. Embedding Service
# ============================================================

def _normalize_texts(
    texts: list[str],
) -> list[str]:
    """
    校验并清理待 Embedding 文本。
    """

    if not texts:
        raise ValueError(
            "texts 不能为空"
        )

    normalized_texts: list[str] = []

    for index, text in enumerate(texts):
        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(
                f"texts[{index}] 必须是非空字符串"
            )

        normalized_texts.append(
            text.strip()
        )

    return normalized_texts


def _validate_embedding_response(
    data: dict,
    expected_count: int,
) -> list[list[float]]:
    """
    校验 Embedding 服务响应。
    """

    embeddings = data.get(
        "embeddings"
    )

    if not isinstance(
        embeddings,
        list,
    ):
        raise ValueError(
            "Embedding 服务返回格式错误："
            "缺少 embeddings"
        )

    if len(embeddings) != expected_count:
        raise ValueError(
            "Embedding 数量与输入文本数量不一致："
            f"输入={expected_count}，"
            f"返回={len(embeddings)}"
        )

    if embeddings:
        first_dimension = len(
            embeddings[0]
        )

        if first_dimension <= 0:
            raise ValueError(
                "Embedding 维度异常"
            )

        for index, vector in enumerate(
            embeddings
        ):
            if not isinstance(
                vector,
                list,
            ):
                raise ValueError(
                    f"embeddings[{index}] 不是 list"
                )

            if len(vector) != first_dimension:
                raise ValueError(
                    "同一批次 Embedding 维度不一致"
                )

    return embeddings


def _request_embedding_batch(
    texts: list[str],
    max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES,
    retry_backoff: float = DEFAULT_EMBEDDING_RETRY_BACKOFF,
) -> list[list[float]]:
    """
    向 Embedding 服务发送一个请求批次。

    特殊处理：
    - 413：交给上层自动拆小批次；
    - 429/5xx：有限次数重试；
    - 422：输出服务端返回内容，便于定位请求 Schema。
    """

    payload = {
        "texts": texts,
        "normalize": True,
    }

    for attempt in range(
        max_retries + 1
    ):
        try:
            response = requests.post(
                EMBEDDING_URL,
                json=payload,
                timeout=EMBEDDING_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    "调用 Embedding 服务失败："
                    f"{exc}"
                ) from exc

            wait_seconds = (
                retry_backoff
                * (2 ** attempt)
            )

            time.sleep(
                wait_seconds
            )
            continue

        # 413 不在这里直接 raise，
        # 由 _embed_batch_adaptive() 自动拆分。
        if response.status_code == 413:
            raise requests.HTTPError(
                "413 Request Entity Too Large",
                response=response,
            )

        if (
            response.status_code
            in RETRYABLE_STATUS_CODES
        ):
            if attempt >= max_retries:
                response.raise_for_status()

            wait_seconds = (
                retry_backoff
                * (2 ** attempt)
            )

            time.sleep(
                wait_seconds
            )
            continue

        if response.status_code == 422:
            detail = response.text[:2000]

            raise RuntimeError(
                "Embedding 服务返回 422。"
                "请求已到达服务端，但参数 Schema 校验失败。\n"
                f"URL: {EMBEDDING_URL}\n"
                f"Response: {detail}"
            )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                "Embedding 服务响应不是合法 JSON"
            ) from exc

        return _validate_embedding_response(
            data=data,
            expected_count=len(texts),
        )

    raise RuntimeError(
        "Embedding 请求重试结束但没有获得结果"
    )


def _embed_batch_adaptive(
    texts: list[str],
    max_retries: int,
    retry_backoff: float,
) -> list[list[float]]:
    """
    对一个 Embedding 批次执行自适应请求。

    如果收到 HTTP 413：

        batch
          ↓
        一分为二
          ↓
        分别请求
          ↓
        合并结果

    如果单条文本仍然 413，说明不是 batch 数量问题，
    而是这个 Chunk 本身太大，需要重新切片或提高服务端限制。
    """

    try:
        return _request_embedding_batch(
            texts=texts,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    except requests.HTTPError as exc:
        response = exc.response

        if (
            response is None
            or response.status_code != 413
        ):
            raise

        if len(texts) == 1:
            text = texts[0]
            byte_size = len(
                text.encode("utf-8")
            )

            raise RuntimeError(
                "单个 Chunk 调用 Embedding 时仍返回 HTTP 413。\n"
                "这说明该 Chunk 本身已经超过服务端请求体限制，"
                "继续减小 batch_size 也无法解决。\n"
                f"Chunk 字符数: {len(text)}\n"
                f"Chunk UTF-8 字节数: {byte_size}\n"
                "建议：检查该 Chunk 是否异常过长，"
                "重新切片，或提高 Embedding 服务/反向代理的请求体限制。"
            ) from exc

        middle = len(texts) // 2

        left = texts[:middle]
        right = texts[middle:]

        print(
            "  [Embedding] HTTP 413，"
            f"自动拆分批次：{len(texts)} "
            f"-> {len(left)} + {len(right)}"
        )

        left_embeddings = _embed_batch_adaptive(
            texts=left,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        right_embeddings = _embed_batch_adaptive(
            texts=right,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        return (
            left_embeddings
            + right_embeddings
        )


def get_embeddings(
    texts: list[str],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES,
    retry_backoff: float = DEFAULT_EMBEDDING_RETRY_BACKOFF,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    批量调用 Embedding Service。

    与旧版本不同：
    不再把所有 texts 一次性 POST。

    流程：
        texts
          ↓
        按 batch_size 切分
          ↓
        请求 Embedding API
          ↓
        413 时自动继续二分
          ↓
        合并所有向量

    默认 batch_size=4，优先保证稳定性。
    """

    if (
        not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size 必须是大于 0 的整数"
        )

    normalized_texts = _normalize_texts(
        texts
    )

    all_embeddings: list[list[float]] = []
    expected_dimension: int | None = None

    total_batches = (
        len(normalized_texts)
        + batch_size
        - 1
    ) // batch_size

    for start in range(
        0,
        len(normalized_texts),
        batch_size,
    ):
        batch = normalized_texts[
            start:start + batch_size
        ]

        batch_index = (
            start // batch_size
        ) + 1

        if show_progress:
            total_chars = sum(
                len(text)
                for text in batch
            )

            total_bytes = sum(
                len(text.encode("utf-8"))
                for text in batch
            )

            print(
                f"  [Embedding] "
                f"Batch {batch_index}/{total_batches} | "
                f"文本={len(batch)} | "
                f"字符={total_chars} | "
                f"字节={total_bytes}"
            )

        batch_embeddings = _embed_batch_adaptive(
            texts=batch,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        if batch_embeddings:
            current_dimension = len(
                batch_embeddings[0]
            )

            if expected_dimension is None:
                expected_dimension = (
                    current_dimension
                )

            elif (
                current_dimension
                != expected_dimension
            ):
                raise ValueError(
                    "不同 Embedding 批次返回的向量维度不一致："
                    f"期望={expected_dimension}，"
                    f"实际={current_dimension}"
                )

        all_embeddings.extend(
            batch_embeddings
        )

    if len(all_embeddings) != len(
        normalized_texts
    ):
        raise ValueError(
            "最终 Embedding 数量与输入文本数量不一致"
        )

    return all_embeddings


def get_embedding(
    text: str,
) -> list[float]:
    """
    单文本 Embedding。

    查询场景只发送一条文本，不打印批处理日志。
    """

    return get_embeddings(
        [text],
        batch_size=1,
        show_progress=False,
    )[0]


# ============================================================
# 3. Chroma Metadata
# ============================================================

def _metadata_value(
    value: Any,
    default: Any = UNKNOWN_VALUE,
):
    """
    Chroma metadata 只保存简单标量。

    None / 空字符串统一转成默认值。
    """

    if value is None:
        return default

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        return (
            value
            if value
            else default
        )

    return value


def build_chroma_metadata(
    chunk: dict,
) -> dict:
    """
    从最终 Chunk 中提取适合写入 Chroma 的 Metadata。

    heading_path / page_numbers 是 list，
    不直接写入 Chroma，转换成字符串。
    """

    heading_path = chunk.get(
        "heading_path",
        [],
    )

    if isinstance(
        heading_path,
        list,
    ):
        heading_path_text = " > ".join(
            str(item).strip()
            for item in heading_path
            if str(item).strip()
        )
    else:
        heading_path_text = str(
            heading_path or ""
        ).strip()

    page_numbers = chunk.get(
        "page_numbers",
        [],
    )

    if isinstance(
        page_numbers,
        list,
    ):
        page_numbers_text = ",".join(
            str(item)
            for item in page_numbers
        )
    else:
        page_numbers_text = str(
            page_numbers or ""
        )

    return {
        "knowledge_base_id": _metadata_value(
            chunk.get("knowledge_base_id")
        ),
        "document_id": _metadata_value(
            chunk.get("document_id")
        ),
        "source": _metadata_value(
            chunk.get("source")
        ),
        "file_type": _metadata_value(
            chunk.get("file_type")
        ),
        "document_type": _metadata_value(
            chunk.get("document_type")
        ),
        "device_model": _metadata_value(
            chunk.get("device_model")
        ),
        "version": _metadata_value(
            chunk.get("version")
        ),
        "strategy": _metadata_value(
            chunk.get("strategy")
        ),
        "section": _metadata_value(
            chunk.get("section")
        ),
        "content_type": _metadata_value(
            chunk.get("content_type")
        ),
        "heading_path": _metadata_value(
            heading_path_text
        ),
        "page_start": _metadata_value(
            chunk.get("page_start"),
            -1,
        ),
        "page_end": _metadata_value(
            chunk.get("page_end"),
            -1,
        ),
        "page_numbers": _metadata_value(
            page_numbers_text
        ),
        "section_index": _metadata_value(
            chunk.get("section_index"),
            -1,
        ),
        "section_part": _metadata_value(
            chunk.get("section_part"),
            -1,
        ),
    }


# ============================================================
# 4. 新增 / 更新 Chunk
# ============================================================

def _prepare_chunks(
    chunks: list[dict],
) -> tuple[
    list[str],
    list[str],
    list[dict],
]:
    """
    在真正写入前，统一校验全部 Chunk。

    这样可以尽量避免导入一半以后才发现：
    - chunk_id 缺失；
    - content 缺失；
    - 本次 JSON 中 chunk_id 重复。
    """

    if not chunks:
        raise ValueError(
            "chunks 不能为空"
        )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    seen_ids: set[str] = set()

    for index, chunk in enumerate(
        chunks
    ):
        if not isinstance(
            chunk,
            dict,
        ):
            raise TypeError(
                f"chunks[{index}] 必须是 dict"
            )

        chunk_id = chunk.get(
            "chunk_id"
        )

        content = chunk.get(
            "content"
        )

        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
        ):
            raise ValueError(
                f"chunks[{index}] 缺少 chunk_id"
            )

        chunk_id = chunk_id.strip()

        if chunk_id in seen_ids:
            raise ValueError(
                "本次导入数据中 chunk_id 重复："
                f"{chunk_id}"
            )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                f"{chunk_id} 缺少 content"
            )

        seen_ids.add(
            chunk_id
        )

        ids.append(
            chunk_id
        )

        texts.append(
            content.strip()
        )

        metadatas.append(
            build_chroma_metadata(
                chunk
            )
        )

    return (
        ids,
        texts,
        metadatas,
    )


def prepare_chunks(
    chunks: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """
    对外暴露的 Chunk 预处理入口。

    Stage 3 的 knowledge_builder 需要先独立完成 Embedding，
    再执行 Vector Store 写入，因此不能只依赖 add_documents()。
    """
    return _prepare_chunks(chunks)


def upsert_documents(
    chunks: list[dict],
    embeddings: list[list[float]],
    upsert_batch_size: int = DEFAULT_CHROMA_UPSERT_BATCH_SIZE,
) -> int:
    """
    使用已经计算好的 embeddings 写入 Chroma。

    这样 Stage 3 可以严格打印：
        [3/5] Embedding 完成
        [4/5] Vector Store 写入完成

    不会在 Vector Store 阶段再次重复计算 Embedding。
    """
    if (
        not isinstance(upsert_batch_size, int)
        or upsert_batch_size <= 0
    ):
        raise ValueError(
            "upsert_batch_size 必须是大于 0 的整数"
        )

    ids, texts, metadatas = _prepare_chunks(chunks)

    if not isinstance(embeddings, list):
        raise TypeError("embeddings 必须是 list")

    if len(embeddings) != len(ids):
        raise ValueError(
            "Embedding 数量与 Chunk 数量不一致："
            f"chunks={len(ids)}，embeddings={len(embeddings)}"
        )

    if embeddings:
        dimension = len(embeddings[0])
        if dimension <= 0:
            raise ValueError("Embedding 维度必须大于 0")
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or len(vector) != dimension:
                raise ValueError(
                    f"embeddings[{index}] 维度不一致或格式错误"
                )

    collection = get_collection()
    total = len(ids)
    written = 0

    for start in range(0, total, upsert_batch_size):
        end = min(start + upsert_batch_size, total)
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        written += end - start

    return written


def add_documents(
    chunks: list[dict],
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    upsert_batch_size: int = DEFAULT_CHROMA_UPSERT_BATCH_SIZE,
) -> int:
    """
    添加或更新 Chunk。

    两层批处理：

    第一层：Chroma 写入批次
        例如每次处理 64 个 Chunk。

    第二层：Embedding 请求批次
        例如 64 个 Chunk 内部再按 4 个文本调用 /embed。
        如果 4 个仍返回 413，会自动拆成 2，再拆成 1。

    使用 upsert：
    - chunk_id 不存在 -> 新增
    - chunk_id 已存在 -> 更新

    因此脚本中途失败后可直接重新执行，
    已经成功写入的相同 ID 会被更新，而不是制造重复记录。
    """

    if (
        not isinstance(embedding_batch_size, int)
        or embedding_batch_size <= 0
    ):
        raise ValueError(
            "embedding_batch_size 必须是大于 0 的整数"
        )

    if (
        not isinstance(upsert_batch_size, int)
        or upsert_batch_size <= 0
    ):
        raise ValueError(
            "upsert_batch_size 必须是大于 0 的整数"
        )

    (
        ids,
        texts,
        metadatas,
    ) = _prepare_chunks(
        chunks
    )

    collection = get_collection()

    total = len(ids)
    total_batches = (
        total
        + upsert_batch_size
        - 1
    ) // upsert_batch_size

    written = 0

    print(
        f"开始写入向量库：总 Chunk={total}，"
        f"Embedding Batch={embedding_batch_size}，"
        f"Chroma Upsert Batch={upsert_batch_size}"
    )

    for start in range(
        0,
        total,
        upsert_batch_size,
    ):
        end = min(
            start + upsert_batch_size,
            total,
        )

        write_batch_index = (
            start // upsert_batch_size
        ) + 1

        batch_ids = ids[
            start:end
        ]
        batch_texts = texts[
            start:end
        ]
        batch_metadatas = metadatas[
            start:end
        ]

        print()
        print(
            "=" * 60
        )
        print(
            f"写入批次 "
            f"{write_batch_index}/{total_batches}"
        )
        print(
            f"Chunk 范围: {start + 1}-{end}"
        )
        print(
            f"Chunk 数量: {len(batch_ids)}"
        )

        embeddings = get_embeddings(
            batch_texts,
            batch_size=embedding_batch_size,
            show_progress=True,
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )

        written += len(
            batch_ids
        )

        print(
            f"Chroma upsert 完成："
            f"本批={len(batch_ids)}，"
            f"累计={written}/{total}"
        )

    return written


# ============================================================
# 5. 数据管理
# ============================================================

def count_chunks(
    where: dict | None = None,
) -> int:
    """
    返回 Chunk 数量。

    - where=None：整个 Collection 的数量；
    - where!=None：只统计符合 Metadata Filter 的 Chunk。
    """
    collection = get_collection()

    if not where:
        return collection.count()

    result = collection.get(
        where=where,
        include=[],
    )
    return len(result.get("ids") or [])


def _require_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


def build_metadata_filter(
    *,
    knowledge_base_id: str | None = None,
    device_model: str | None = None,
    document_id: str | None = None,
    extra: dict | None = None,
) -> dict | None:
    """
    构造 Chroma where 条件。

    多个条件时使用 $and，避免调用方手工拼装过滤语法。
    """
    conditions: list[dict] = []

    for field, value in (
        ("knowledge_base_id", knowledge_base_id),
        ("device_model", device_model),
        ("document_id", document_id),
    ):
        if value is not None:
            value = _require_non_empty_string(value, field)
            conditions.append({field: value})

    if extra:
        if not isinstance(extra, dict):
            raise TypeError("extra 必须是 dict 或 None")
        conditions.append(extra)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def count_knowledge_base_chunks(
    knowledge_base_id: str,
) -> int:
    knowledge_base_id = _require_non_empty_string(
        knowledge_base_id,
        "knowledge_base_id",
    )
    return count_chunks(
        where={"knowledge_base_id": knowledge_base_id}
    )


def get_knowledge_base_records(
    knowledge_base_id: str,
) -> list[dict]:
    """
    读取某个 KB 在 Chroma 中的全部记录，供 Stage 3 验收使用。
    """
    knowledge_base_id = _require_non_empty_string(
        knowledge_base_id,
        "knowledge_base_id",
    )
    result = get_collection().get(
        where={"knowledge_base_id": knowledge_base_id},
        include=["documents", "metadatas"],
    )

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    return [
        {
            "chunk_id": chunk_id,
            "content": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) else {},
        }
        for index, chunk_id in enumerate(ids)
    ]


def delete_knowledge_base(
    knowledge_base_id: str,
) -> int:
    """按 knowledge_base_id 删除该 KB 的全部向量记录。"""
    knowledge_base_id = _require_non_empty_string(
        knowledge_base_id,
        "knowledge_base_id",
    )
    collection = get_collection()
    result = collection.get(
        where={"knowledge_base_id": knowledge_base_id},
        include=[],
    )
    ids = result.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[str]:
    """
    返回当前数据库中的 document_id 列表。
    """

    result = get_collection().get(
        include=["metadatas"]
    )

    document_ids = set()

    for metadata in (
        result.get("metadatas")
        or []
    ):
        if not metadata:
            continue

        document_id = metadata.get(
            "document_id"
        )

        if document_id:
            document_ids.add(
                str(document_id)
            )

    return sorted(
        document_ids
    )


def count_documents() -> int:
    """
    返回逻辑文档数量，而不是 Chunk 数量。
    """

    return len(
        list_documents()
    )


def count_document_chunks(
    document_id: str,
) -> int:
    """
    统计某个文档包含多少 Chunk。
    """

    if (
        not isinstance(document_id, str)
        or not document_id.strip()
    ):
        raise ValueError(
            "document_id 不能为空"
        )

    result = get_collection().get(
        where={
            "document_id":
                document_id.strip()
        },
        include=[],
    )

    return len(
        result.get("ids")
        or []
    )


def delete_documents(
    document_id: str,
) -> int:
    """
    按 document_id 删除整份文档的所有 Chunk。

    返回删除的 Chunk 数量。
    """

    if (
        not isinstance(document_id, str)
        or not document_id.strip()
    ):
        raise ValueError(
            "document_id 不能为空"
        )

    document_id = document_id.strip()

    collection = get_collection()

    result = collection.get(
        where={
            "document_id":
                document_id
        },
        include=[],
    )

    ids = (
        result.get("ids")
        or []
    )

    if ids:
        collection.delete(
            ids=ids
        )

    return len(ids)


# ============================================================
# 6. Vector Search
# ============================================================

def search(
    query: str,
    top_k: int = TOP_K,
    where: dict | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """
    向量检索统一入口。

    流程：

    Query
        ↓
    Embedding
        ↓
    Chroma
        ↓
    Metadata Filter
        ↓
    Top-K
        ↓
    Distance Threshold
        ↓
    标准化结果
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
        threshold is not None
        and (
            not isinstance(
                threshold,
                (int, float),
            )
            or threshold < 0
        )
    ):
        raise ValueError(
            "threshold 必须是大于等于 0 的数字或 None"
        )

    query_vector = get_embedding(
        query.strip()
    )

    collection = get_collection()

    query_kwargs = {
        "query_embeddings": [
            query_vector
        ],
        "n_results": top_k,
        "include": [
            "documents",
            "distances",
            "metadatas",
        ],
    }

    if where:
        query_kwargs[
            "where"
        ] = where

    raw = collection.query(
        **query_kwargs
    )

    ids = (
        raw.get("ids")
        or [[]]
    )[0]

    documents = (
        raw.get("documents")
        or [[]]
    )[0]

    distances = (
        raw.get("distances")
        or [[]]
    )[0]

    metadatas = (
        raw.get("metadatas")
        or [[]]
    )[0]

    results = []

    for (
        chunk_id,
        document,
        distance,
        metadata,
    ) in zip(
        ids,
        documents,
        distances,
        metadatas,
    ):
        distance = float(
            distance
        )

        if (
            threshold is not None
            and distance > threshold
        ):
            continue

        results.append(
            {
                "rank":
                    len(results) + 1,

                "chunk_id":
                    chunk_id,

                "distance":
                    distance,

                "content":
                    document or "",

                "metadata":
                    metadata or {},
            }
        )

    return results


# ============================================================
# 7. 调试输出
# ============================================================

def print_search_results(
    results: list[dict],
) -> None:
    """
    打印检索结果。
    """

    if not results:
        print(
            "没有找到足够相关的参考资料"
        )
        return

    for item in results:
        metadata = item.get(
            "metadata",
            {},
        )

        print()
        print("=" * 70)
        print(
            f"Top-{item['rank']}"
        )

        print("\nChunk ID:")
        print(
            item["chunk_id"]
        )

        print("\nDistance:")
        print(
            f"{item['distance']:.4f}"
        )

        print("\nSource:")
        print(
            metadata.get(
                "source",
                UNKNOWN_VALUE,
            )
        )

        print("\nPage:")
        print(
            f"{metadata.get('page_start', UNKNOWN_VALUE)}"
            f"-"
            f"{metadata.get('page_end', UNKNOWN_VALUE)}"
        )

        print("\nKnowledge Base ID:")
        print(
            metadata.get(
                "knowledge_base_id",
                UNKNOWN_VALUE,
            )
        )

        print("\nDevice Model:")
        print(
            metadata.get(
                "device_model",
                UNKNOWN_VALUE,
            )
        )

        print("\nDocument Type:")
        print(
            metadata.get(
                "document_type",
                UNKNOWN_VALUE,
            )
        )

        print("\nSection:")
        print(
            metadata.get(
                "section",
                UNKNOWN_VALUE,
            )
        )

        print("\nContent:")
        print(
            item["content"]
        )


# ============================================================
# 8. 最小自检入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Vector Store 状态")
    print("=" * 60)

    print(
        "Chunk 数量:",
        count_chunks(),
    )

    print(
        "Document 数量:",
        count_documents(),
    )

    print(
        "Document 列表:",
        list_documents(),
    )
