"""
Day 20 - Stage 3
Knowledge Base Builder

完整闭环：

knowledge_base_id + document_ids
        ↓
定位 data/uploads 中的文档
        ↓
PDF Parse
        ↓
Section Chunk
        ↓
Embedding
        ↓
Chroma Vector Store
        ↓
KB 专属 BM25
        ↓
一致性验收

说明：
- 复用 rag/chunking.py；
- 复用 rag/vector_store.py；
- 复用 rag/bm25_retriever.py；
- 不复制已有 RAG 能力。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
RAG_DIR = BASE_DIR / "rag"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHUNKS_DIR = DATA_DIR / "chunks"

# 兼容你当前 rag/*.py 内部仍使用 `from config import ...` 的运行方式。
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from chunking import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_REPEAT_THRESHOLD,
    apply_metadata_override,
    parse_pdf,
    save_chunks,
    section_chunk,
    validate_chunks_contract,
)
from vector_store import (  # noqa: E402
    DEFAULT_CHROMA_UPSERT_BATCH_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    count_knowledge_base_chunks,
    delete_knowledge_base,
    get_embeddings,
    get_knowledge_base_records,
    upsert_documents,
)
from bm25_retriever import (  # noqa: E402
    build_and_save_knowledge_base_index,
    count_index_chunks,
    load_index,
)

from app.services.knowledge_service import (  # noqa: E402
    KB_STATUS_BUILDING,
    KB_STATUS_FAILED,
    KB_STATUS_READY,
    create_knowledge_base,
    get_knowledge_base,
    set_knowledge_base_status,
    update_knowledge_base,
)


UNKNOWN_VALUE = "unknown"

# 如果你之后已有正式 Document Service，最终只需要替换
# resolve_document_info()，下面 Parser/Chunk/Vector/BM25 都不用改。
DOCUMENT_REGISTRY_CANDIDATES = (
    DATA_DIR / "documents.json",
    DATA_DIR / "document_metadata.json",
    UPLOAD_DIR / "documents.json",
    UPLOAD_DIR / "metadata.json",
)

_FILE_PATH_KEYS = (
    "file_path",
    "path",
    "saved_path",
    "storage_path",
    "upload_path",
)

_FILE_NAME_KEYS = (
    "stored_filename",
    "saved_filename",
    "filename",
    "file_name",
)

_SOURCE_KEYS = (
    "original_filename",
    "original_name",
    "source",
    "filename",
    "file_name",
)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_document_records(data: Any):
    """兼容常见的文档注册表 JSON 结构。"""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(data, dict):
        return

    # {"documents": [...]}
    documents = data.get("documents")
    if isinstance(documents, list):
        for item in documents:
            if isinstance(item, dict):
                yield item

    # {"doc_xxx": {...}}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        record = dict(value)
        record.setdefault("document_id", key)
        yield record


def _find_registry_record(document_id: str) -> dict | None:
    for registry_path in DOCUMENT_REGISTRY_CANDIDATES:
        if not registry_path.exists():
            continue

        try:
            data = _read_json_file(registry_path)
        except (json.JSONDecodeError, OSError):
            continue

        for record in _iter_document_records(data):
            if str(record.get("document_id", "")).strip() == document_id:
                result = dict(record)
                result["_registry_path"] = str(registry_path)
                return result

    return None


def _path_from_record(record: dict) -> Path | None:
    for key in _FILE_PATH_KEYS:
        value = record.get(key)
        if not value:
            continue

        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            # 优先按项目根目录解释；不存在则按 uploads 解释。
            project_candidate = BASE_DIR / path
            upload_candidate = UPLOAD_DIR / path
            if project_candidate.exists():
                path = project_candidate
            elif upload_candidate.exists():
                path = upload_candidate
            else:
                path = project_candidate

        if path.exists() and path.is_file():
            return path.resolve()

    for key in _FILE_NAME_KEYS:
        value = record.get(key)
        if not value:
            continue
        path = (UPLOAD_DIR / str(value)).resolve()
        if path.exists() and path.is_file():
            return path

    return None


def _pick_first_text(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def resolve_document_info(
    document_id: str,
    explicit_path: str | Path | None = None,
) -> dict:
    """
    document_id -> 上传文件信息。

    查找优先级：
    1. 调用者显式传入 document_paths；
    2. 常见 document registry JSON；
    3. data/uploads/{document_id}.*。

    如果出现多个候选文件，直接报错而不是猜。
    """
    document_id = _require_text(document_id, "document_id")

    record = _find_registry_record(document_id) or {}

    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"document_id={document_id} 指定文件不存在：{path}"
            )
    else:
        path = _path_from_record(record)

        if path is None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            candidates = sorted(
                item.resolve()
                for item in UPLOAD_DIR.glob(f"{document_id}.*")
                if item.is_file()
                and item.name not in {"documents.json", "metadata.json"}
            )

            if not candidates:
                raise FileNotFoundError(
                    f"无法通过 document_id={document_id} 定位上传文件。\n"
                    f"已检查目录：{UPLOAD_DIR}\n"
                    "支持：data/uploads/{document_id}.pdf，或在 documents.json "
                    "中提供 file_path / stored_filename。"
                )

            if len(candidates) > 1:
                raise RuntimeError(
                    f"document_id={document_id} 找到多个候选文件：\n"
                    + "\n".join(str(item) for item in candidates)
                )

            path = candidates[0]

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Stage 3 当前 Parser 只支持 PDF：{path}"
        )

    source = _pick_first_text(record, _SOURCE_KEYS) or path.name
    version = _pick_first_text(
        record,
        ("version", "document_version", "revision"),
    ) or UNKNOWN_VALUE
    document_type = _pick_first_text(
        record,
        ("document_type", "manual_type"),
    )
    device_model = _pick_first_text(
        record,
        ("device_model", "model", "product_model"),
    )

    return {
        "document_id": document_id,
        "path": path,
        "source": source,
        "version": version,
        "document_type": document_type,
        "device_model": device_model,
        "registry": record,
    }


def _validate_chunk_identity(
    chunks: list[dict],
    *,
    knowledge_base_id: str,
    device_model: str,
    document_ids: list[str],
) -> None:
    """在写入外部存储前做一次严格身份校验。"""
    if not chunks:
        raise ValueError("Chunk 结果为空")

    seen_ids: set[str] = set()
    allowed_document_ids = set(document_ids)

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")

        if chunk_id in seen_ids:
            raise ValueError(f"Chunk ID 重复：{chunk_id}")
        seen_ids.add(chunk_id)

        if chunk.get("knowledge_base_id") != knowledge_base_id:
            raise ValueError(
                f"Chunk #{index} knowledge_base_id 不一致：{chunk_id}"
            )
        if document_id not in allowed_document_ids:
            raise ValueError(
                f"Chunk #{index} document_id 不属于当前 KB：{document_id}"
            )
        if chunk.get("device_model") != device_model:
            raise ValueError(
                f"Chunk #{index} device_model 不一致："
                f"{chunk.get('device_model')} != {device_model}"
            )

        required = (
            "knowledge_base_id",
            "document_id",
            "source",
            "device_model",
            "document_type",
            "version",
            "page_start",
            "page_end",
            "section",
        )
        for field in required:
            if field not in chunk or chunk[field] is None:
                raise ValueError(
                    f"Chunk {chunk_id} 缺少字段：{field}"
                )


def validate_knowledge_base_build(
    knowledge_base_id: str,
    chunks: list[dict],
    bm25_index_data: dict,
) -> dict:
    """
    Stage 3 最终验收：

    Chunk数量 == Vector数量 == BM25文档数量
    并检查关键 Metadata 一致。
    """
    kb = get_knowledge_base(knowledge_base_id)
    expected_count = len(chunks)
    vector_count = count_knowledge_base_chunks(knowledge_base_id)
    bm25_count = count_index_chunks(bm25_index_data)

    errors: list[str] = []

    if expected_count != vector_count:
        errors.append(
            f"Chunk数量({expected_count}) != Vector数量({vector_count})"
        )
    if expected_count != bm25_count:
        errors.append(
            f"Chunk数量({expected_count}) != BM25文档数量({bm25_count})"
        )

    try:
        _validate_chunk_identity(
            chunks,
            knowledge_base_id=knowledge_base_id,
            device_model=kb["device_model"],
            document_ids=kb["document_ids"],
        )
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))

    # Vector Metadata 验收
    vector_records = get_knowledge_base_records(knowledge_base_id)
    vector_by_id = {
        item["chunk_id"]: item
        for item in vector_records
    }

    required_vector_fields = (
        "knowledge_base_id",
        "document_id",
        "device_model",
        "source",
        "page_start",
        "page_end",
        "section",
        "version",
    )

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        record = vector_by_id.get(chunk_id)
        if record is None:
            errors.append(f"Vector Store 缺少 Chunk：{chunk_id}")
            continue

        metadata = record.get("metadata") or {}
        for field in required_vector_fields:
            if field not in metadata:
                errors.append(
                    f"Vector Metadata 缺少 {field}: {chunk_id}"
                )
                continue

            expected = chunk.get(field)
            actual = metadata.get(field)
            if actual != expected:
                errors.append(
                    f"Vector Metadata 不一致 {field}: {chunk_id} | "
                    f"chunk={expected!r}, vector={actual!r}"
                )

    # BM25 identity 验收
    bm25_chunks = bm25_index_data.get("chunks") or []
    bm25_by_id = {
        item.get("chunk_id"): item
        for item in bm25_chunks
    }
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        bm25_chunk = bm25_by_id.get(chunk_id)
        if bm25_chunk is None:
            errors.append(f"BM25 Index 缺少 Chunk：{chunk_id}")
            continue

        for field in (
            "knowledge_base_id",
            "document_id",
            "device_model",
            "page_start",
            "page_end",
            "section",
            "version",
        ):
            if bm25_chunk.get(field) != chunk.get(field):
                errors.append(
                    f"BM25 Metadata 不一致 {field}: {chunk_id}"
                )

    return {
        "passed": not errors,
        "knowledge_base_id": knowledge_base_id,
        "chunk_count": expected_count,
        "vector_count": vector_count,
        "bm25_document_count": bm25_count,
        "document_ids": kb["document_ids"],
        "device_model": kb["device_model"],
        "errors": errors,
    }


def _print_acceptance(report: dict) -> None:
    print("\n" + "=" * 72)
    print("Stage 3 验收")
    print("=" * 72)
    print(f"Knowledge Base ID : {report['knowledge_base_id']}")
    print(f"Chunk 数量        : {report['chunk_count']}")
    print(f"Vector 数量       : {report['vector_count']}")
    print(f"BM25 文档数量     : {report['bm25_document_count']}")
    print(f"Document IDs      : {report['document_ids']}")
    print(f"Device Model      : {report['device_model']}")
    print(f"Result            : {'PASS' if report['passed'] else 'FAIL'}")

    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"][:30]:
            print(f"- {error}")

    print("=" * 72)


def build_knowledge_base(
    knowledge_base_id: str,
    *,
    document_paths: dict[str, str | Path] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    clean_threshold: float = DEFAULT_REPEAT_THRESHOLD,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    upsert_batch_size: int = DEFAULT_CHROMA_UPSERT_BATCH_SIZE,
    replace_existing_vectors: bool = True,
) -> dict:
    """
    构建一个已经创建好元数据的 Knowledge Base。

    document_paths 是可选兜底：
        {
            "doc_xxx": "/path/to/manual.pdf"
        }

    正常业务流程中不需要传，系统会通过 data/uploads/ 定位。
    """
    knowledge_base_id = _require_text(
        knowledge_base_id,
        "knowledge_base_id",
    )
    kb = get_knowledge_base(knowledge_base_id)

    set_knowledge_base_status(
        knowledge_base_id,
        KB_STATUS_BUILDING,
        error=None,
    )

    document_paths = document_paths or {}

    try:
        # ----------------------------------------------------
        # [1/5] Document Parse
        # ----------------------------------------------------
        parsed_documents: list[dict] = []

        for document_id in kb["document_ids"]:
            info = resolve_document_info(
                document_id,
                explicit_path=document_paths.get(document_id),
            )

            document = parse_pdf(
                info["path"],
                clean_threshold=clean_threshold,
            )

            # KB 的 device_model 是业务上明确给定的信息，优先级高于
            # Parser 对 PDF 文本的自动猜测。
            metadata_override = {
                "knowledge_base_id": knowledge_base_id,
                "document_id": document_id,
                "source": info["source"],
                "device_model": kb["device_model"],
                "version": info["version"],
            }

            if info.get("document_type"):
                metadata_override["document_type"] = info["document_type"]

            apply_metadata_override(
                document,
                metadata_override,
            )
            parsed_documents.append(document)

        print(
            f"[1/5] 文档解析完成：{len(parsed_documents)} 个 Document"
        )

        # ----------------------------------------------------
        # [2/5] Chunk
        # ----------------------------------------------------
        chunks: list[dict] = []

        for document in parsed_documents:
            document_chunks = section_chunk(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            validate_chunks_contract(
                document_chunks,
                raise_on_error=True,
            )
            chunks.extend(document_chunks)

        # 多文档合并后再做一次全局 chunk_id 唯一性校验。
        _validate_chunk_identity(
            chunks,
            knowledge_base_id=knowledge_base_id,
            device_model=kb["device_model"],
            document_ids=kb["document_ids"],
        )

        chunk_file = CHUNKS_DIR / f"{knowledge_base_id}.json"
        save_chunks(chunks, chunk_file)

        print(
            f"[2/5] Chunk生成完成：{len(chunks)}"
        )

        # ----------------------------------------------------
        # [3/5] Embedding
        # ----------------------------------------------------
        texts = [chunk["content"] for chunk in chunks]
        embeddings = get_embeddings(
            texts,
            batch_size=embedding_batch_size,
            show_progress=True,
        )

        print(
            f"[3/5] Embedding完成：{len(embeddings)}"
        )

        # ----------------------------------------------------
        # [4/5] Vector Store
        # ----------------------------------------------------
        if replace_existing_vectors:
            deleted = delete_knowledge_base(knowledge_base_id)
            if deleted:
                print(
                    f"      已清理旧 KB 向量：{deleted}"
                )

        written = upsert_documents(
            chunks,
            embeddings,
            upsert_batch_size=upsert_batch_size,
        )

        print(
            f"[4/5] Vector Store写入完成：{written}"
        )

        # ----------------------------------------------------
        # [5/5] BM25
        # ----------------------------------------------------
        bm25_index_data, bm25_path = (
            build_and_save_knowledge_base_index(
                knowledge_base_id,
                chunks,
            )
        )

        print(
            f"[5/5] BM25索引建立完成：{count_index_chunks(bm25_index_data)}"
        )

        # ----------------------------------------------------
        # Stage 3 验收
        # ----------------------------------------------------
        report = validate_knowledge_base_build(
            knowledge_base_id,
            chunks,
            bm25_index_data,
        )
        _print_acceptance(report)

        if not report["passed"]:
            raise RuntimeError(
                "Stage 3 验收失败：\n"
                + "\n".join(report["errors"][:30])
            )

        kb = update_knowledge_base(
            knowledge_base_id,
            status=KB_STATUS_READY,
            chunk_count=report["chunk_count"],
            vector_count=report["vector_count"],
            bm25_document_count=report["bm25_document_count"],
            chunk_file=str(chunk_file.relative_to(BASE_DIR)),
            bm25_index_file=str(bm25_path.relative_to(BASE_DIR)),
            last_error=None,
        )

        return {
            "knowledge_base": kb,
            "report": report,
            "chunk_file": chunk_file,
            "bm25_index_file": bm25_path,
        }

    except Exception as exc:
        # 构建失败仍然保留 KB 元数据，便于页面/日志展示失败状态。
        set_knowledge_base_status(
            knowledge_base_id,
            KB_STATUS_FAILED,
            error=str(exc),
        )
        raise


def create_and_build_knowledge_base(
    *,
    name: str,
    device_model: str,
    document_ids: list[str],
    document_paths: dict[str, str | Path] | None = None,
    **build_kwargs: Any,
) -> dict:
    """Stage 3 的一键入口：创建 KB 元数据，然后立即构建。"""
    kb = create_knowledge_base(
        name=name,
        device_model=device_model,
        document_ids=document_ids,
    )

    return build_knowledge_base(
        kb["knowledge_base_id"],
        document_paths=document_paths,
        **build_kwargs,
    )


if __name__ == "__main__":
    print(
        "请使用 tests/test_knowledge_build.py 做 Stage 3 完整验收，"
        "或在业务代码中调用 create_and_build_knowledge_base()."
    )
