from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz as pymupdf


# =========================================================
# 0. 默认配置
# =========================================================

UNKNOWN_VALUE = "unknow"

DEFAULT_INPUT_PATH = Path("../data/raw/test_manual.pdf")
DEFAULT_OUTPUT_PATH = Path("../data/chunks/pdf_chunks_section.json")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80
DEFAULT_REPEAT_THRESHOLD = 0.6

# 无法从文档中可靠识别时，可在这里按 document_id 人工覆盖。
# 例如：
# DOCUMENT_METADATA_OVERRIDES = {
#     "E300_manual": {
#         "document_type": "maintenance_manual",
#         "device_model": "E300",
#     }
# }
DOCUMENT_METADATA_OVERRIDES: dict[str, dict[str, str]] = {}

_FILE_TYPE_VALUES = {
    "pdf",
    "txt",
    "text",
    "md",
    "markdown",
    "doc",
    "docx",
}

CHINESE_NUMBER_PATTERN = r"[一二三四五六七八九十百千零〇两]+"


# =========================================================
# 1. 通用辅助函数
# =========================================================


def _clean_string(value: Any, default: str = UNKNOWN_VALUE) -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default



def _normalize_heading_path(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]



def _nested_metadata(document: dict) -> dict:
    metadata = document.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}



def _lookup_document_value(document: dict, *keys: str):
    metadata = _nested_metadata(document)

    for key in keys:
        value = document.get(key)
        if value is not None and str(value).strip():
            return value

        value = metadata.get(key)
        if value is not None and str(value).strip():
            return value

    return None


# =========================================================
# 2. PDF 文本抽取
# =========================================================


def _extract_pdf_pages(file_path: str | Path) -> dict:
    """
    只负责：
        PDF -> pages[]

    此时不做结构识别，避免后续清洗 page["text"] 后，
    blocks 与 text 不一致。
    """

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")

    if not path.is_file():
        raise ValueError(f"输入路径不是文件：{path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"当前 chunking.py 只接收 PDF：{path}")

    pages: list[dict] = []
    pdf = pymupdf.open(path)

    try:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text", sort=True)

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text.strip(),
                }
            )
    finally:
        pdf.close()

    return {
        "document_id": path.stem,
        "source": path.name,
        "file_type": "pdf",
        "pages": pages,
    }



def validate_page_numbers(document: dict) -> None:
    pages = document.get("pages", [])

    for index, page in enumerate(pages):
        expected = index + 1
        actual = page.get("page_number")

        if actual != expected:
            raise ValueError(
                f"页码映射错误：pages[{index}] page_number={actual}，"
                f"预期={expected}"
            )


# =========================================================
# 3. 文本清洗
# =========================================================


def clean_text(text: str) -> str:
    """
    轻量清洗：
    - 统一换行符；
    - 清理异常空格；
    - 压缩连续普通空格 / Tab；
    - 压缩过多空行。

    不主动删除标题编号、步骤编号、故障码、警告等业务信息。
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def _get_non_empty_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]



def _find_repeated_edge_lines(
    pages: list[dict],
    *,
    first: bool,
    threshold: float,
) -> set[str]:
    candidates: list[str] = []

    for page in pages:
        text = page.get("text", "")

        if text is None:
            text = ""

        if not isinstance(text, str):
            raise TypeError("page['text'] 必须是字符串")

        lines = _get_non_empty_lines(text)

        if lines:
            candidates.append(lines[0] if first else lines[-1])

    if len(candidates) < 2:
        return set()

    counter = Counter(candidates)
    min_count = max(2, math.ceil(len(candidates) * threshold))

    return {
        text
        for text, count in counter.items()
        if count >= min_count
    }



def _remove_first_matching_line(text: str, targets: set[str]) -> str:
    if not text or not targets:
        return text

    lines = text.split("\n")

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        if line.strip() in targets:
            lines.pop(index)

        break

    return "\n".join(lines)



def _remove_last_matching_line(text: str, targets: set[str]) -> str:
    if not text or not targets:
        return text

    lines = text.split("\n")

    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip():
            continue

        if lines[index].strip() in targets:
            lines.pop(index)

        break

    return "\n".join(lines)



def clean_document(document: dict, threshold: float = 0.6) -> dict:
    """
    对 PDF 页面文本进行跨页清洗。

    顺序：
        原始 page text
        -> 删除高频重复页眉/页脚
        -> 基础 clean_text
    """

    if not 0 < threshold <= 1:
        raise ValueError("threshold 必须在 0 到 1 之间")

    pages = document.get("pages", [])

    if not isinstance(pages, list):
        raise TypeError("document['pages'] 必须是 list")

    repeated_headers = _find_repeated_edge_lines(
        pages,
        first=True,
        threshold=threshold,
    )
    repeated_footers = _find_repeated_edge_lines(
        pages,
        first=False,
        threshold=threshold,
    )

    for page in pages:
        text = page.get("text", "")

        if text is None:
            text = ""

        if not isinstance(text, str):
            raise TypeError(
                f"page_number={page.get('page_number')} 的 text 必须是字符串"
            )

        text = _remove_first_matching_line(text, repeated_headers)
        text = _remove_last_matching_line(text, repeated_footers)
        page["text"] = clean_text(text)

    return document


# =========================================================
# 4. PDF 结构识别
# =========================================================


def detect_heading(text: str) -> tuple[int, str] | None:
    """
    支持：
    - 一、产品简介
    - 第三章 设备维护
    - 第2节 安全说明
    - 3.1 启动检查
    - 3.2.1 温度异常
    """

    text = text.strip()

    if not text:
        return None

    # 第三章 / 第2章
    match = re.match(
        rf"^第(?:{CHINESE_NUMBER_PATTERN}|\d+)(章|节|部分)\s*[：:]?\s*(.+)$",
        text,
    )
    if match:
        unit = match.group(1)
        level = 1 if unit in {"章", "部分"} else 2
        return level, text

    # 一、产品简介
    match = re.match(
        rf"^({CHINESE_NUMBER_PATTERN})、\s*(.+)$",
        text,
    )
    if match:
        return 1, text

    # 3.1 / 3.2.1
    match = re.match(
        r"^(\d+(?:\.\d+)+)\s+(.+)$",
        text,
    )
    if match:
        number = match.group(1)
        level = number.count(".") + 1
        return level, text

    return None



def is_list_item(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    return bool(
        re.match(
            r"^(?:\d+[.、)]|[（(]\d+[)）]|[①②③④⑤⑥⑦⑧⑨⑩])\s*.+$",
            text,
        )
    )



def is_page_marker(text: str) -> bool:
    """
    对“说明书 | 第 X 页”这类残留页码行做兜底识别。
    跨页重复清洗已先执行，此规则只是第二道防线。
    """

    text = text.strip()

    if not text:
        return False

    return bool(
        "说明书" in text
        and re.search(r"第\s*\d+\s*页", text)
    )



def update_heading_path(
    heading_stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> list[tuple[int, str]]:
    heading_stack = [
        item
        for item in heading_stack
        if item[0] < level
    ]
    heading_stack.append((level, title))
    return heading_stack



def recognize_pdf_structure(
    text: str,
    heading_stack: list[tuple[int, str]] | None = None,
) -> tuple[list[dict], list[tuple[int, str]]]:
    if heading_stack is None:
        heading_stack = []

    blocks: list[dict] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if is_page_marker(line):
            blocks.append(
                {
                    "type": "footer",
                    "text": line,
                    "heading_path": [title for _, title in heading_stack],
                }
            )
            continue

        heading_result = detect_heading(line)

        if heading_result is not None:
            level, title = heading_result
            heading_stack = update_heading_path(
                heading_stack,
                level,
                title,
            )
            heading_path = [title for _, title in heading_stack]

            blocks.append(
                {
                    "type": "heading",
                    "level": level,
                    "text": title,
                    "heading_path": heading_path,
                }
            )
            continue

        block_type = "list_item" if is_list_item(line) else "paragraph"

        blocks.append(
            {
                "type": block_type,
                "text": line,
                "heading_path": [title for _, title in heading_stack],
            }
        )

    return blocks, heading_stack



def recognize_document_structure(document: dict) -> dict:
    """
    必须在 clean_document() 之后执行，保证 blocks 来自清洗后的文本。
    """

    heading_stack: list[tuple[int, str]] = []

    for page in document.get("pages", []):
        blocks, heading_stack = recognize_pdf_structure(
            text=page.get("text", ""),
            heading_stack=heading_stack,
        )
        page["blocks"] = blocks

    return document



def parse_pdf(
    file_path: str | Path,
    clean_threshold: float = DEFAULT_REPEAT_THRESHOLD,
) -> dict:
    """
    工程化 PDF 解析入口：

        PDF
        -> 文本抽取
        -> 页码校验
        -> 文本清洗
        -> 结构识别
        -> document
    """

    document = _extract_pdf_pages(file_path)
    validate_page_numbers(document)
    clean_document(document, threshold=clean_threshold)
    recognize_document_structure(document)
    return document


# =========================================================
# 5. 文档级 Metadata 归一化
# =========================================================


def _document_preview_text(document: dict, max_pages: int = 2) -> str:
    pieces: list[str] = []

    for key in ("title", "name", "document_name"):
        value = document.get(key)
        if value:
            pieces.append(str(value))

    pages = document.get("pages", [])

    if not isinstance(pages, list):
        pages = []

    for page in pages[:max_pages]:
        if not isinstance(page, dict):
            continue

        page_text = page.get("text", "")
        if page_text:
            pieces.append(str(page_text))

        blocks = page.get("blocks", [])
        if isinstance(blocks, list):
            for block in blocks[:20]:
                if not isinstance(block, dict):
                    continue
                block_text = block.get("text", "")
                if block_text:
                    pieces.append(str(block_text))

    return "\n".join(pieces)



def _infer_document_type(document: dict) -> str:
    text = _document_preview_text(document)

    rules = (
        (
            "maintenance_manual",
            ("维护手册", "维修手册", "保养手册", "维护说明", "维修说明"),
        ),
        (
            "installation_manual",
            ("安装手册", "安装说明", "安装指南"),
        ),
        (
            "fault_manual",
            ("故障手册", "故障代码手册", "故障码手册", "报警代码手册"),
        ),
        (
            "operation_manual",
            ("使用说明书", "操作手册", "用户手册", "使用手册", "操作说明"),
        ),
        (
            "product_specification",
            ("产品规格书", "技术规格书", "产品规格", "技术规格"),
        ),
    )

    for document_type, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return document_type

    return UNKNOWN_VALUE



def _infer_device_model(document: dict) -> str:
    preview = _document_preview_text(document)
    source = _clean_string(document.get("source"), "")
    search_text = f"{source}\n{preview}"

    explicit_pattern = re.compile(
        r"(?:设备型号|产品型号|型号|机型)\s*[:：]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9._/-]{0,30})",
        re.IGNORECASE,
    )

    match = explicit_pattern.search(search_text)
    if match:
        return match.group(1)

    context_keywords = (
        "空调",
        "设备",
        "产品",
        "变频器",
        "控制器",
        "PLC",
        "驱动器",
        "机器人",
    )

    model_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"([A-Z][A-Z0-9_-]{0,12}\d[A-Z0-9_-]{0,12})"
        r"(?![A-Za-z0-9])"
    )

    for line in search_text.splitlines():
        if not any(keyword in line for keyword in context_keywords):
            continue

        match = model_pattern.search(line)
        if match:
            return match.group(1)

    return UNKNOWN_VALUE



def normalize_document_metadata(document: dict) -> dict:
    document_id = _clean_string(
        _lookup_document_value(document, "document_id")
    )
    source = _clean_string(
        _lookup_document_value(document, "source", "filename", "file_name")
    )

    raw_file_type = _lookup_document_value(document, "file_type")
    raw_document_type = _lookup_document_value(
        document,
        "document_type",
        "manual_type",
    )

    if (
        not raw_file_type
        and raw_document_type
        and str(raw_document_type).strip().lower() in _FILE_TYPE_VALUES
    ):
        raw_file_type = raw_document_type

    if not raw_file_type:
        suffix = source.rsplit(".", 1)[-1] if "." in source else ""
        if suffix:
            raw_file_type = "markdown" if suffix.lower() == "md" else suffix.lower()

    file_type = _clean_string(raw_file_type)
    document_type = _clean_string(raw_document_type)

    if document_type.lower() in _FILE_TYPE_VALUES:
        document_type = _infer_document_type(document)

    if document_type == UNKNOWN_VALUE:
        document_type = _infer_document_type(document)

    device_model = _clean_string(
        _lookup_document_value(
            document,
            "device_model",
            "model",
            "product_model",
            "machine_model",
        )
    )

    if device_model == UNKNOWN_VALUE:
        device_model = _infer_device_model(document)

    knowledge_base_id = _clean_string(
        _lookup_document_value(document, "knowledge_base_id")
    )
    version = _clean_string(
        _lookup_document_value(document, "version", "document_version")
    )

    return {
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "source": source,
        "file_type": file_type,
        "document_type": document_type,
        "device_model": device_model,
        "version": version,
    }



def apply_metadata_override(
    document: dict,
    override: dict[str, str] | None = None,
) -> dict:
    """
    可在自动识别不可靠时显式覆盖 document_type / device_model。
    """

    if override is None:
        document_id = _clean_string(document.get("document_id"))
        override = DOCUMENT_METADATA_OVERRIDES.get(document_id, {})

    if not isinstance(override, dict):
        return document

    for key in (
        "knowledge_base_id",
        "document_id",
        "source",
        "file_type",
        "document_type",
        "device_model",
        "version",
    ):
        value = override.get(key)
        if value is not None and str(value).strip():
            document[key] = str(value).strip()

    return document


# =========================================================
# 6. Token 粗略估算
# =========================================================


def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9_]+", text)
    punctuation = re.findall(r"[^\w\s\u4e00-\u9fff]", text)

    return len(chinese_chars) + len(english_words) + len(punctuation)



def _hard_split_text(text: str, max_tokens: int) -> list[str]:
    text = text.strip()

    if not text:
        return []

    if estimate_tokens(text) <= max_tokens:
        return [text]

    pieces: list[str] = []
    start = 0
    window_size = max(1, max_tokens)

    while start < len(text):
        end = min(start + window_size, len(text))
        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        start = end

    return pieces


# =========================================================
# 7. 章节收集
# =========================================================


def _collect_sections(document: dict) -> list[dict]:
    """
    document -> pages -> blocks -> sections

    原则：
    - heading 建立语义边界；
    - header/footer 不进入 RAG；
    - 正文保留真实 page_number；
    - 不同 heading_path 不随意混合。
    """

    sections: list[dict] = []
    current_section: dict | None = None

    for page in document.get("pages", []):
        page_number = page.get("page_number")
        blocks = page.get("blocks", [])

        for block in blocks:
            block_type = block.get("type", "paragraph")
            text = block.get("text", "").strip()

            if not text:
                continue

            heading_path = _normalize_heading_path(
                block.get("heading_path", [])
            )

            if block_type in {"header", "footer"}:
                continue

            if block_type == "heading":
                if current_section is not None and current_section["parts"]:
                    sections.append(current_section)

                current_section = {
                    "heading_path": list(heading_path),
                    "parts": [],
                }
                continue

            if current_section is None:
                current_section = {
                    "heading_path": list(heading_path),
                    "parts": [],
                }

            current_section["parts"].append(
                {
                    "page_number": page_number,
                    "text": text,
                    "block_type": block_type,
                }
            )

    if current_section is not None and current_section["parts"]:
        sections.append(current_section)

    return sections



def _prepare_section_fragments(
    parts: list[dict],
    max_tokens: int,
) -> list[dict]:
    fragments: list[dict] = []

    for part in parts:
        page_number = part.get("page_number")
        block_type = part.get("block_type", "paragraph")
        text = part.get("text", "").strip()

        if not text:
            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:
            pieces = _hard_split_text(line, max_tokens=max_tokens)

            for piece in pieces:
                fragments.append(
                    {
                        "page_number": page_number,
                        "text": piece,
                        "block_type": block_type,
                    }
                )

    return fragments


# =========================================================
# 8. Chunk 数据契约
# =========================================================


def validate_chunk_contract(chunk: dict) -> list[str]:
    errors: list[str] = []

    string_fields = (
        "chunk_id",
        "knowledge_base_id",
        "document_id",
        "source",
        "file_type",
        "document_type",
        "device_model",
        "version",
        "strategy",
        "section",
        "content",
    )

    for field in string_fields:
        value = chunk.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} 必须是非空字符串")

    integer_fields = (
        "section_index",
        "section_part",
        "page_start",
        "page_end",
        "estimated_tokens",
    )

    for field in integer_fields:
        if not isinstance(chunk.get(field), int):
            errors.append(f"{field} 必须是 int")

    heading_path = chunk.get("heading_path")
    if not isinstance(heading_path, list):
        errors.append("heading_path 必须是 list")
    elif not all(isinstance(item, str) for item in heading_path):
        errors.append("heading_path 元素必须全部是 str")

    page_numbers = chunk.get("page_numbers")
    if not isinstance(page_numbers, list):
        errors.append("page_numbers 必须是 list")
    elif not all(isinstance(item, int) for item in page_numbers):
        errors.append("page_numbers 元素必须全部是 int")

    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")

    if (
        isinstance(page_start, int)
        and isinstance(page_end, int)
        and page_start >= 0
        and page_end >= 0
        and page_start > page_end
    ):
        errors.append("page_start 不能大于 page_end")

    return errors



def validate_chunks_contract(
    chunks: list[dict],
    raise_on_error: bool = False,
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, chunk in enumerate(chunks, start=1):
        for error in validate_chunk_contract(chunk):
            errors.append(f"Chunk #{index}: {error}")

        chunk_id = chunk.get("chunk_id")

        if isinstance(chunk_id, str) and chunk_id:
            if chunk_id in seen_ids:
                errors.append(
                    f"Chunk #{index}: chunk_id 重复：{chunk_id}"
                )
            seen_ids.add(chunk_id)

    if errors and raise_on_error:
        raise ValueError(
            "Day 16 Chunk 数据契约校验失败：\n"
            + "\n".join(errors)
        )

    return errors



def _build_section_chunk(
    document: dict,
    heading_path: list[str],
    fragments: list[dict],
    section_index: int,
    section_part: int,
    strategy: str,
) -> dict:
    metadata = normalize_document_metadata(document)

    document_id = metadata["document_id"]
    heading_path = _normalize_heading_path(heading_path)
    section = heading_path[-1] if heading_path else UNKNOWN_VALUE

    heading_text = " > ".join(heading_path)
    body_text = "\n".join(
        fragment.get("text", "")
        for fragment in fragments
        if fragment.get("text", "").strip()
    ).strip()

    final_text = (
        f"{heading_text}\n\n{body_text}".strip()
        if heading_text
        else body_text
    )

    page_numbers = sorted(
        {
            fragment["page_number"]
            for fragment in fragments
            if isinstance(fragment.get("page_number"), int)
        }
    )

    page_start = min(page_numbers) if page_numbers else -1
    page_end = max(page_numbers) if page_numbers else -1

    chunk_id = (
        f"{document_id}_section_"
        f"{section_index:04d}_{section_part:02d}"
    )

    chunk = {
        "chunk_id": chunk_id,
        "knowledge_base_id": metadata["knowledge_base_id"],
        "document_id": document_id,
        "source": metadata["source"],
        "file_type": metadata["file_type"],
        "document_type": metadata["document_type"],
        "device_model": metadata["device_model"],
        "version": metadata["version"],
        "strategy": _clean_string(strategy),
        "section_index": section_index,
        "section_part": section_part,
        "section": section,
        "heading_path": heading_path,
        "page_start": page_start,
        "page_end": page_end,
        "page_numbers": page_numbers,
        "estimated_tokens": estimate_tokens(final_text),
        "content": final_text,
    }

    errors = validate_chunk_contract(chunk)

    if errors:
        raise ValueError(
            f"Chunk {chunk_id} 不符合 Day 16 数据契约："
            + "; ".join(errors)
        )

    return chunk


# =========================================================
# 9. 章节切片：结构优先，长度兜底
# =========================================================


def section_chunk(
    document: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    sections = _collect_sections(document)
    chunks: list[dict] = []

    for section_index, section in enumerate(sections, start=1):
        heading_path = _normalize_heading_path(
            section.get("heading_path", [])
        )
        heading_text = " > ".join(heading_path)
        heading_tokens = estimate_tokens(heading_text)

        # 给标题本身预留空间，避免“正文刚好500 + 标题”导致最终超限。
        max_body_tokens = max(1, chunk_size - heading_tokens - 2)

        fragments = _prepare_section_fragments(
            section.get("parts", []),
            max_tokens=max_body_tokens,
        )

        if not fragments:
            continue

        total_body = "\n".join(fragment["text"] for fragment in fragments)
        full_text = (
            f"{heading_text}\n\n{total_body}"
            if heading_text
            else total_body
        )

        if estimate_tokens(full_text) <= chunk_size:
            chunks.append(
                _build_section_chunk(
                    document=document,
                    heading_path=heading_path,
                    fragments=fragments,
                    section_index=section_index,
                    section_part=0,
                    strategy="section",
                )
            )
            continue

        current_fragments: list[dict] = []
        section_part = 0

        for fragment in fragments:
            candidate_fragments = current_fragments + [fragment]
            body = "\n".join(item["text"] for item in candidate_fragments)
            candidate_text = (
                f"{heading_text}\n\n{body}"
                if heading_text
                else body
            )

            if (
                current_fragments
                and estimate_tokens(candidate_text) > chunk_size
            ):
                chunks.append(
                    _build_section_chunk(
                        document=document,
                        heading_path=heading_path,
                        fragments=current_fragments,
                        section_index=section_index,
                        section_part=section_part,
                        strategy="section_fallback",
                    )
                )
                section_part += 1

                overlap_fragments: list[dict] = []
                overlap_tokens = 0

                for previous in reversed(current_fragments):
                    previous_tokens = estimate_tokens(previous["text"])

                    if (
                        overlap_fragments
                        and overlap_tokens + previous_tokens > overlap
                    ):
                        break

                    overlap_fragments.insert(0, previous)
                    overlap_tokens += previous_tokens

                    if overlap_tokens >= overlap:
                        break

                current_fragments = overlap_fragments

                # overlap 本身可能导致“旧尾部 + 新片段”再次超过 chunk_size。
                # 从 overlap 最前面逐步裁掉，直到新片段能放进去。
                while current_fragments:
                    retry_body = "\n".join(
                        item["text"]
                        for item in current_fragments + [fragment]
                    )
                    retry_text = (
                        f"{heading_text}\n\n{retry_body}"
                        if heading_text
                        else retry_body
                    )

                    if estimate_tokens(retry_text) <= chunk_size:
                        break

                    current_fragments.pop(0)

            current_fragments.append(fragment)

        if current_fragments:
            chunks.append(
                _build_section_chunk(
                    document=document,
                    heading_path=heading_path,
                    fragments=current_fragments,
                    section_index=section_index,
                    section_part=section_part,
                    strategy="section_fallback",
                )
            )

    return chunks


# =========================================================
# 10. 可选：Content Type / 层级 Chunk
# =========================================================

_FAULT_CODE_PATTERN = re.compile(
    r"\b(?:ERR[-_]?\d{1,5}|ALM[-_]?\d{1,5}|[EFP]\d{1,5})\b",
    re.IGNORECASE,
)



def _looks_like_table(text: str) -> bool:
    if not text:
        return False

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        return False

    markdown_rows = [
        line
        for line in lines
        if line.startswith("|") and line.endswith("|")
    ]

    if len(markdown_rows) >= 2:
        return True

    table_header_keywords = {
        "项目",
        "参数",
        "代码",
        "测试含义",
        "现象 / 代码",
        "可能原因",
        "建议处理",
        "检查项",
        "正常状态",
        "异常示例",
        "维护对象",
        "建议频率",
        "处理方法",
        "术语",
        "简要说明",
        "部件",
        "主要作用",
        "测试关键词",
        "室内情况",
        "建议测试模式",
        "备注",
    }

    header_hits = sum(
        1
        for line in lines[:12]
        if line in table_header_keywords
    )
    short_lines = sum(1 for line in lines if len(line) <= 30)
    short_ratio = short_lines / len(lines)

    return header_hits >= 2 and short_ratio >= 0.6



def detect_content_type(
    text: str,
    heading_path: list[str] | None = None,
) -> str:
    if not text:
        return "text"

    heading_path = heading_path or []
    heading_text = " > ".join(heading_path)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    warning_heading_keywords = (
        "警告",
        "安全",
        "注意事项",
        "危险",
        "使用边界",
    )

    if any(keyword in heading_text for keyword in warning_heading_keywords):
        return "warning"

    warning_pattern = re.compile(
        r"^(?:警告|注意|危险|WARNING|CAUTION|DANGER)\s*[:：]",
        re.IGNORECASE,
    )

    if any(warning_pattern.search(line) for line in lines[:3]):
        return "warning"

    fault_heading_keywords = (
        "故障代码",
        "故障码",
        "错误码",
        "报警码",
        "常见故障",
    )

    if any(keyword in heading_text for keyword in fault_heading_keywords):
        return "fault_code"

    if _FAULT_CODE_PATTERN.search(text):
        return "fault_code"

    procedure_heading_keywords = (
        "基本操作",
        "操作步骤",
        "处理步骤",
        "检查步骤",
        "维护步骤",
        "安装步骤",
        "排查顺序",
        "操作流程",
    )

    step_pattern = re.compile(
        r"^\s*(?:\d+[\.、)]|步骤\s*\d+|第[一二三四五六七八九十百]+步)"
    )

    step_count = sum(1 for line in lines if step_pattern.match(line))

    if (
        step_count >= 2
        and any(
            keyword in heading_text
            for keyword in procedure_heading_keywords
        )
    ):
        return "procedure"

    if _looks_like_table(text):
        return "table"

    return "text"



def _extract_chunk_body(chunk: dict) -> str:
    text = (chunk.get("content") or chunk.get("text", "")).strip()
    heading_path = _normalize_heading_path(chunk.get("heading_path", []))
    heading_text = " > ".join(heading_path)

    if not heading_text:
        return text

    prefix = f"{heading_text}\n\n"

    if text.startswith(prefix):
        return text[len(prefix):].strip()

    return text



def hierarchical_chunk(
    document: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    在 section_chunk 基础上增加 content_type 和 metadata。
    默认主流程仍使用 section 模式；需要时可切换为 hierarchical。
    """

    base_chunks = section_chunk(
        document=document,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    results: list[dict] = []

    for chunk in base_chunks:
        heading_path = _normalize_heading_path(
            chunk.get("heading_path", [])
        )
        body = _extract_chunk_body(chunk)
        content_type = detect_content_type(
            body,
            heading_path=heading_path,
        )

        old_strategy = chunk.get("strategy", "section")
        strategy = (
            "hierarchical"
            if old_strategy == "section"
            else "hierarchical_fallback"
            if old_strategy == "section_fallback"
            else f"hierarchical_{old_strategy}"
        )

        new_chunk = dict(chunk)
        new_chunk["strategy"] = strategy
        new_chunk["content_type"] = content_type
        new_chunk["metadata"] = {
            "knowledge_base_id": chunk.get("knowledge_base_id", UNKNOWN_VALUE),
            "document_id": chunk.get("document_id", UNKNOWN_VALUE),
            "source": chunk.get("source", UNKNOWN_VALUE),
            "file_type": chunk.get("file_type", UNKNOWN_VALUE),
            "document_type": chunk.get("document_type", UNKNOWN_VALUE),
            "device_model": chunk.get("device_model", UNKNOWN_VALUE),
            "version": chunk.get("version", UNKNOWN_VALUE),
            "section": chunk.get("section", UNKNOWN_VALUE),
            "heading_path": list(heading_path),
            "page_start": chunk.get("page_start", -1),
            "page_end": chunk.get("page_end", -1),
            "page_numbers": chunk.get("page_numbers", []),
            "content_type": content_type,
            "section_index": chunk.get("section_index", -1),
            "section_part": chunk.get("section_part", -1),
            "strategy": strategy,
        }

        errors = validate_chunk_contract(new_chunk)
        if errors:
            raise ValueError(
                f"Chunk {new_chunk.get('chunk_id', UNKNOWN_VALUE)} "
                "不符合 Day 16 数据契约："
                + "; ".join(errors)
            )

        results.append(new_chunk)

    return results


# =========================================================
# 11. 最终对外入口：PDF -> Chunk
# =========================================================


def chunk_pdf(
    file_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    clean_threshold: float = DEFAULT_REPEAT_THRESHOLD,
    metadata_override: dict[str, str] | None = None,
    document_id: str | None = None,
    knowledge_base_id: str | None = None,
    version: str | None = None,
    mode: str = "section",
) -> list[dict]:
    """
    Day 16 最终工程入口。

    外部只需要关心：

        PDF
        -> chunk_pdf(...)
        -> list[Chunk]

    内部自动完成：

        PDF Parse
        -> Clean
        -> Structure Recognition
        -> Metadata Normalization
        -> Section Chunk
        -> Contract Validation

    mode:
        "section"       : 当前 Day16 默认，输出稳定的 section chunks
        "hierarchical"  : 额外加入 content_type + metadata
    """

    document = parse_pdf(
        file_path=file_path,
        clean_threshold=clean_threshold,
    )

    merged_override = dict(metadata_override or {})
    if document_id is not None and str(document_id).strip():
        merged_override["document_id"] = str(document_id).strip()
    if knowledge_base_id is not None and str(knowledge_base_id).strip():
        merged_override["knowledge_base_id"] = str(knowledge_base_id).strip()
    if version is not None and str(version).strip():
        merged_override["version"] = str(version).strip()

    apply_metadata_override(
        document,
        override=merged_override or None,
    )

    if mode == "section":
        chunks = section_chunk(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif mode == "hierarchical":
        chunks = hierarchical_chunk(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    else:
        raise ValueError(
            "mode 只能是 'section' 或 'hierarchical'"
        )

    validate_chunks_contract(
        chunks,
        raise_on_error=True,
    )

    return chunks


# 语义更直观的别名：文档 -> chunks
chunk_document = chunk_pdf


# =========================================================
# 12. 保存最终 Chunk JSON
# =========================================================


def save_chunks(
    chunks: list[dict],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


# =========================================================
# 13. 调试输出
# =========================================================


def print_chunk_summary(chunks: list[dict], max_preview: int = 160) -> None:
    print("=" * 70)
    print(f"Chunk 总数：{len(chunks)}")
    print("=" * 70)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nTop/Chunk #{index}")
        print(f"Chunk ID      : {chunk.get('chunk_id')}")
        print(f"Source        : {chunk.get('source')}")
        print(
            f"Page          : {chunk.get('page_start')}"
            f"-{chunk.get('page_end')}"
        )
        print(f"KB ID         : {chunk.get('knowledge_base_id')}")
        print(f"Device Model  : {chunk.get('device_model')}")
        print(f"Document Type : {chunk.get('document_type')}")
        print(f"Version       : {chunk.get('version')}")
        print(f"Section       : {chunk.get('section')}")
        print(f"Tokens(approx): {chunk.get('estimated_tokens')}")
        print(
            "Content       : "
            + chunk.get("content", "")[:max_preview].replace("\n", " ")
        )


# =========================================================
# 14. CLI：直接 python chunking.py
# =========================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF -> 清洗 -> 结构识别 -> Section Chunk"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="原始 PDF 路径",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="最终 Chunk JSON 输出路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk 最大近似 Token 数，默认 500",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="超长章节二次切分 overlap，默认 80",
    )
    parser.add_argument(
        "--clean-threshold",
        type=float,
        default=DEFAULT_REPEAT_THRESHOLD,
        help="重复页眉/页脚识别阈值，默认 0.6",
    )
    parser.add_argument(
        "--mode",
        choices=("section", "hierarchical"),
        default="section",
        help="默认 section；需要 content_type 时可选 hierarchical",
    )
    parser.add_argument(
        "--device-model",
        default=None,
        help="可选：人工覆盖 device_model",
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="可选：人工覆盖 document_type",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="可选：人工覆盖 document_id（知识库创建时建议使用 doc_xxx）",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=None,
        help="可选：写入 knowledge_base_id（例如 kb_xxx）",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="可选：文档版本；未知时可不填",
    )

    args = parser.parse_args()

    metadata_override = {
        key: value
        for key, value in {
            "device_model": args.device_model,
            "document_type": args.document_type,
            "document_id": args.document_id,
            "knowledge_base_id": args.knowledge_base_id,
            "version": args.version,
        }.items()
        if value is not None and str(value).strip()
    }

    try:
        chunks = chunk_pdf(
            file_path=args.input,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            clean_threshold=args.clean_threshold,
            metadata_override=metadata_override or None,
            mode=args.mode,
        )

        output_path = save_chunks(
            chunks,
            args.output,
        )

        print_chunk_summary(chunks)
        print("\n" + "=" * 70)
        print("PDF -> Chunk 完成")
        print(f"输入 PDF：{Path(args.input).expanduser().resolve()}")
        print(f"输出 JSON：{output_path}")
        print(f"Chunk 数量：{len(chunks)}")
        print("数据契约：通过")
        print("=" * 70)

    except (FileNotFoundError, ValueError, TypeError) as exc:
        print("\nChunking 失败：")
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
