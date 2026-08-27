"""
Day 20 - Stage 3
Knowledge Base Metadata Service

职责：
1. 创建 knowledge_base_id；
2. 持久化 Knowledge Base 元数据；
3. 查询 / 更新 Knowledge Base 状态；
4. 不负责 PDF 解析、Embedding、Vector Store、BM25。

持久化文件：
    data/knowledge_base.json
"""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.json"

KB_STATUS_CREATED = "created"
KB_STATUS_BUILDING = "building"
KB_STATUS_READY = "ready"
KB_STATUS_FAILED = "failed"

_ALLOWED_STATUS = {
    KB_STATUS_CREATED,
    KB_STATUS_BUILDING,
    KB_STATUS_READY,
    KB_STATUS_FAILED,
}


class KnowledgeBaseNotFoundError(KeyError):
    """指定的 Knowledge Base 不存在。

    继承 KeyError，保持与现有 FastAPI Controller 的
    `except KeyError -> HTTP 404` 契约兼容，同时让业务语义更明确。
    """

    def __str__(self) -> str:
        # KeyError 默认 str() 会额外包一层引号；业务错误响应不需要。
        if self.args:
            return str(self.args[0])
        return self.__class__.__name__


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


def _normalize_document_ids(document_ids: list[str]) -> list[str]:
    if not isinstance(document_ids, list) or not document_ids:
        raise ValueError("document_ids 必须是非空 list")

    result: list[str] = []
    seen: set[str] = set()

    for index, document_id in enumerate(document_ids):
        document_id = _require_text(
            document_id,
            f"document_ids[{index}]",
        )
        if document_id not in seen:
            result.append(document_id)
            seen.add(document_id)

    return result


def _load_all() -> list[dict]:
    if not KNOWLEDGE_BASE_FILE.exists():
        return []

    with KNOWLEDGE_BASE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Knowledge Base 元数据文件格式错误：{KNOWLEDGE_BASE_FILE}"
        )

    return data


def _atomic_save_all(items: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = KNOWLEDGE_BASE_FILE.with_suffix(
        KNOWLEDGE_BASE_FILE.suffix + ".tmp"
    )

    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            items,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(KNOWLEDGE_BASE_FILE)


def _generate_kb_id(existing_ids: set[str]) -> str:
    for _ in range(20):
        kb_id = f"kb_{uuid.uuid4().hex[:12]}"
        if kb_id not in existing_ids:
            return kb_id
    raise RuntimeError("无法生成唯一 knowledge_base_id")


def create_knowledge_base(
    name: str,
    device_model: str,
    document_ids: list[str],
) -> dict:
    """
    创建 Knowledge Base 元数据。

    输入示例：
        create_knowledge_base(
            name="G120C知识库",
            device_model="G120C",
            document_ids=["doc_a81c9f"],
        )

    返回示例：
        {
            "knowledge_base_id": "kb_xxx",
            ...
        }
    """
    name = _require_text(name, "name")
    device_model = _require_text(device_model, "device_model")
    document_ids = _normalize_document_ids(document_ids)

    items = _load_all()
    existing_ids = {
        str(item.get("knowledge_base_id"))
        for item in items
        if item.get("knowledge_base_id")
    }

    kb_id = _generate_kb_id(existing_ids)
    now = _now_iso()

    kb = {
        "knowledge_base_id": kb_id,
        "name": name,
        "device_model": device_model,
        "document_ids": document_ids,
        "status": KB_STATUS_CREATED,
        "chunk_count": 0,
        "vector_count": 0,
        "bm25_document_count": 0,
        "chunk_file": None,
        "bm25_index_file": None,
        "last_error": None,
        "created_at": now,
        "updated_at": now,
    }

    items.append(kb)
    _atomic_save_all(items)

    return deepcopy(kb)


def list_knowledge_bases() -> list[dict]:
    return deepcopy(_load_all())


def get_knowledge_base(knowledge_base_id: str) -> dict:
    knowledge_base_id = _require_text(
        knowledge_base_id,
        "knowledge_base_id",
    )

    for item in _load_all():
        if item.get("knowledge_base_id") == knowledge_base_id:
            return deepcopy(item)

    raise KnowledgeBaseNotFoundError(
        f"Knowledge Base 不存在：{knowledge_base_id}"
    )


def update_knowledge_base(
    knowledge_base_id: str,
    **updates: Any,
) -> dict:
    """
    更新 KB 的运行状态 / 构建统计。

    knowledge_base_id / created_at 不允许被覆盖。
    """
    knowledge_base_id = _require_text(
        knowledge_base_id,
        "knowledge_base_id",
    )

    forbidden = {"knowledge_base_id", "created_at"}
    illegal = forbidden.intersection(updates)
    if illegal:
        raise ValueError(
            f"不允许修改字段：{sorted(illegal)}"
        )

    if "status" in updates:
        status = updates["status"]
        if status not in _ALLOWED_STATUS:
            raise ValueError(
                f"不支持的 status：{status}"
            )

    items = _load_all()

    for index, item in enumerate(items):
        if item.get("knowledge_base_id") != knowledge_base_id:
            continue

        new_item = dict(item)
        new_item.update(updates)
        new_item["updated_at"] = _now_iso()
        items[index] = new_item

        _atomic_save_all(items)
        return deepcopy(new_item)

    raise KnowledgeBaseNotFoundError(
        f"Knowledge Base 不存在：{knowledge_base_id}"
    )


def set_knowledge_base_status(
    knowledge_base_id: str,
    status: str,
    *,
    error: str | None = None,
) -> dict:
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"不支持的 status：{status}")

    return update_knowledge_base(
        knowledge_base_id,
        status=status,
        last_error=error,
    )


if __name__ == "__main__":
    # 最小手工测试。正式创建建议由 test_knowledge_build.py 调用。
    print(f"Knowledge Base Metadata File: {KNOWLEDGE_BASE_FILE}")
    print(f"Knowledge Base Count: {len(list_knowledge_bases())}")
