"""文档登记服务的隔离测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import document_service


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch) -> Path:
    upload_dir = tmp_path / "uploads"
    database_file = tmp_path / "documents.json"
    monkeypatch.setattr(document_service, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(document_service, "DB_FILE", str(database_file))
    return tmp_path


def test_save_document_and_detect_duplicate(isolated_storage: Path) -> None:
    source = isolated_storage / "manual.txt"
    source.write_text("G120C 温度保护说明", encoding="utf-8")

    created = document_service.save_document(
        str(source),
        "G120C",
        "manual",
        "v1",
    )
    duplicate = document_service.save_document(
        str(source),
        "G120C",
        "manual",
        "v1",
    )

    assert created["status"] == "success"
    assert created["document_id"].startswith("doc_")
    assert duplicate == {
        "status": "duplicate",
        "document_id": created["document_id"],
    }


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("image.jpg", b"not-an-image", "不支持文件类型"),
        ("empty.txt", b"", "空文件"),
    ],
)
def test_rejects_invalid_document(
    isolated_storage: Path,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    source = isolated_storage / filename
    source.write_bytes(content)

    with pytest.raises(ValueError, match=message):
        document_service.save_document(
            str(source),
            "G120C",
            "manual",
        )


def test_rejects_missing_file(isolated_storage: Path) -> None:
    missing = isolated_storage / "missing.pdf"

    with pytest.raises(ValueError, match="文件不存在"):
        document_service.save_document(
            str(missing),
            "G120C",
            "manual",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
