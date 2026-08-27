"""
BM25 Retriever

Day 17 - Stage 1

负责：

1. 从统一 Chunk JSON 构建 BM25 索引
2. 中文 + 工业编码分词
3. BM25 搜索
4. BM25 索引持久化
5. BM25 索引加载

暂时不负责：

- Hybrid Search
- RRF
- Reranker
- LLM
"""

import json
import pickle
import re

from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "chunks"
    / "G120C_0226_chunks_section.json"
)

BM25_DIR = (
    BASE_DIR
    / "data"
    / "bm25"
)

# 兼容 Day17 / 旧 Hybrid Retriever 的默认全局索引。
INDEX_PATH = BM25_DIR / "bm25_index.pkl"


def get_index_path(
    knowledge_base_id: str,
) -> Path:
    """
    返回某个 Knowledge Base 专属 BM25 索引路径：

        data/bm25/kb_xxx.pkl
    """
    if (
        not isinstance(knowledge_base_id, str)
        or not knowledge_base_id.strip()
    ):
        raise ValueError("knowledge_base_id 不能为空")

    knowledge_base_id = knowledge_base_id.strip()

    # 防止把 KB ID 当成任意文件路径使用。
    if not re.fullmatch(r"[A-Za-z0-9_-]+", knowledge_base_id):
        raise ValueError(
            "knowledge_base_id 只能包含字母、数字、下划线和连字符"
        )

    return BM25_DIR / f"{knowledge_base_id}.pkl"


# ============================================================
# 2. 工业编码正则
# ============================================================

# 目标保护：
#
# F30002
# E102
# G120C
# p0210
# QF-35GW/X1
#
# 至少要求编码中包含数字，
# 避免把普通英文全部判断成设备编码。

INDUSTRIAL_CODE_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (
        (?=[A-Za-z0-9/_-]*\d)
        [A-Za-z]
        [A-Za-z0-9]*
        (?:[-_/][A-Za-z0-9]+)*
    )
    (?![A-Za-z0-9])
    """,
    re.VERBOSE,
)


# ============================================================
# 3. 普通 Token 清洗
# ============================================================

VALID_TOKEN_PATTERN = re.compile(
    r"[\u4e00-\u9fff]+|[A-Za-z0-9]+"
)


def _tokenize_normal_text(text: str) -> list[str]:
    """
    对普通文本使用 jieba 分词。
    """

    tokens = []

    for word in jieba.lcut(text):

        word = word.strip().lower()

        if not word:
            continue

        # 去除纯标点
        if not VALID_TOKEN_PATTERN.fullmatch(word):
            continue

        tokens.append(word)

    return tokens


# ============================================================
# 4. 工业文本 Tokenizer
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    中文 + 工业编码 tokenizer。

    思路：

    普通中文
        ↓
    jieba

    工业编码
        ↓
    正则直接保护

    示例：

    设备型号 G120C 出现故障 F30002

    →

    [
        "设备",
        "型号",
        "g120c",
        "出现",
        "故障",
        "f30002"
    ]
    """

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    tokens = []

    last_end = 0

    for match in INDUSTRIAL_CODE_PATTERN.finditer(text):

        # --------------------------------
        # 处理工业编码之前的普通文本
        # --------------------------------

        normal_text = text[
            last_end:match.start()
        ]

        tokens.extend(
            _tokenize_normal_text(
                normal_text
            )
        )

        # --------------------------------
        # 工业编码整体保留
        # --------------------------------

        code = match.group(1).lower()

        tokens.append(code)

        last_end = match.end()

    # ------------------------------------
    # 处理最后剩余文本
    # ------------------------------------

    remaining_text = text[last_end:]

    tokens.extend(
        _tokenize_normal_text(
            remaining_text
        )
    )

    return tokens


# ============================================================
# 5. 加载 Chunk JSON
# ============================================================

def load_chunks(
    path: Path = CHUNKS_PATH,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Chunk 文件不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # ------------------------------------
    # 情况1：
    #
    # [
    #   chunk1,
    #   chunk2
    # ]
    # ------------------------------------

    if isinstance(data, list):

        chunks = data

    # ------------------------------------
    # 情况2：
    #
    # {
    #   "chunks": [...]
    # }
    # ------------------------------------

    elif (
        isinstance(data, dict)
        and isinstance(
            data.get("chunks"),
            list,
        )
    ):

        chunks = data["chunks"]

    else:

        raise ValueError(
            "无法识别 Chunk JSON 数据结构。"
        )

    # ------------------------------------
    # 基础校验
    # ------------------------------------

    valid_chunks = []

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get(
            "chunk_id"
        )

        content = chunk.get(
            "content"
        )

        if not chunk_id:
            continue

        if not isinstance(
            content,
            str,
        ):
            continue

        if not content.strip():
            continue

        valid_chunks.append(
            chunk
        )

    if not valid_chunks:

        raise ValueError(
            "没有找到有效 Chunk。"
        )

    return valid_chunks


# ============================================================
# 6. 构建 BM25 Index
# ============================================================

def build_index(
    chunks: list[dict],
    knowledge_base_id: str | None = None,
) -> dict:
    """
    构建 BM25 Index。

    knowledge_base_id=None 时保持 Day17 旧行为；
    传入 kb_xxx 时，会校验所有 Chunk 都属于同一个 KB。
    """

    if not chunks:

        raise ValueError(
            "chunks 不能为空。"
        )

    if knowledge_base_id is not None:
        if (
            not isinstance(knowledge_base_id, str)
            or not knowledge_base_id.strip()
        ):
            raise ValueError("knowledge_base_id 不能为空")
        knowledge_base_id = knowledge_base_id.strip()

        mismatched = [
            chunk.get("chunk_id", "unknown")
            for chunk in chunks
            if chunk.get("knowledge_base_id") != knowledge_base_id
        ]
        if mismatched:
            raise ValueError(
                "存在不属于目标 Knowledge Base 的 Chunk："
                + ", ".join(mismatched[:10])
            )

    tokenized_corpus = []

    valid_chunks = []

    print()
    print("开始构建 BM25 Index")
    print("=" * 60)

    for chunk in chunks:

        content = chunk.get(
            "content",
            "",
        )

        tokens = tokenize(
            content
        )

        if not tokens:
            continue

        valid_chunks.append(
            chunk
        )

        tokenized_corpus.append(
            tokens
        )

    if not tokenized_corpus:

        raise ValueError(
            "分词结果为空，无法构建 BM25。"
        )

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    document_ids = sorted(
        {
            str(chunk.get("document_id"))
            for chunk in valid_chunks
            if chunk.get("document_id")
        }
    )

    index_data = {

        "bm25": bm25,

        "chunks": valid_chunks,

        "tokenized_corpus": (
            tokenized_corpus
        ),

        "knowledge_base_id": knowledge_base_id,
        "chunk_count": len(valid_chunks),
        "document_ids": document_ids,
        "document_count": len(document_ids),
    }

    print(
        f"有效 Chunk 数量："
        f"{len(valid_chunks)}"
    )

    print(
        f"BM25 Corpus 数量："
        f"{len(tokenized_corpus)}"
    )

    print(
        "BM25 Index 构建完成"
    )

    return index_data


# ============================================================
# 7. 保存 BM25 Index
# ============================================================

def save_index(
    index_data: dict,
    path: Path | None = None,
    knowledge_base_id: str | None = None,
) -> Path:
    """
    保存 BM25 Index。

    优先级：
    1. 显式 path；
    2. knowledge_base_id -> data/bm25/{kb_id}.pkl；
    3. 兼容旧代码 -> data/bm25/bm25_index.pkl。
    """

    if path is None:
        path = (
            get_index_path(knowledge_base_id)
            if knowledge_base_id
            else INDEX_PATH
        )
    else:
        path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "wb"
    ) as f:

        pickle.dump(
            index_data,
            f,
        )

    print(
        f"BM25 Index 已保存：{path}"
    )

    return path


# ============================================================
# 8. 加载 BM25 Index
# ============================================================

def load_index(
    path: Path | None = None,
    knowledge_base_id: str | None = None,
) -> dict:
    """
    加载已经构建好的 BM25 Index。

    注意：
    pickle 只加载自己生成、可信的本地文件。
    """

    if path is None:
        path = (
            get_index_path(knowledge_base_id)
            if knowledge_base_id
            else INDEX_PATH
        )
    else:
        path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"BM25 Index 不存在：{path}"
        )

    with path.open(
        "rb"
    ) as f:

        index_data = pickle.load(
            f
        )

    required_keys = {
        "bm25",
        "chunks",
        "tokenized_corpus",
    }

    if not required_keys.issubset(
        index_data.keys()
    ):

        raise ValueError(
            "BM25 Index 数据结构不完整。"
        )

    if knowledge_base_id is not None:
        stored_kb_id = index_data.get("knowledge_base_id")
        if stored_kb_id not in (None, knowledge_base_id):
            raise ValueError(
                "BM25 Index 的 knowledge_base_id 与请求不一致："
                f"stored={stored_kb_id}, requested={knowledge_base_id}"
            )

    return index_data


def build_and_save_knowledge_base_index(
    knowledge_base_id: str,
    chunks: list[dict],
) -> tuple[dict, Path]:
    """Stage 3：构建并保存一个 KB 专属 BM25 索引。"""
    index_data = build_index(
        chunks,
        knowledge_base_id=knowledge_base_id,
    )
    path = save_index(
        index_data,
        knowledge_base_id=knowledge_base_id,
    )
    return index_data, path


def count_index_chunks(
    index_data: dict,
) -> int:
    return len(index_data.get("chunks") or [])


# ============================================================
# 9. Metadata 统一
# ============================================================

def build_metadata(
    chunk: dict,
) -> dict:
    """
    统一 BM25 返回 metadata。
    """

    return {

        "knowledge_base_id": chunk.get(
            "knowledge_base_id",
            "unknown",
        ),

        "document_id": chunk.get(
            "document_id",
            "unknown",
        ),

        "source": chunk.get(
            "source",
            "unknown",
        ),

        "page_start": chunk.get(
            "page_start",
            "unknown",
        ),

        "page_end": chunk.get(
            "page_end",
            "unknown",
        ),

        "device_model": chunk.get(
            "device_model",
            "unknown",
        ),

        "document_type": chunk.get(
            "document_type",
            "unknown",
        ),

        "version": chunk.get(
            "version",
            "unknown",
        ),

        "section": chunk.get(
            "section",
            "unknown",
        ),

        "heading_path": chunk.get(
            "heading_path",
            [],
        ),

        "section_index": chunk.get(
            "section_index",
            "unknown",
        ),
    }


# ============================================================
# 10. BM25 Search
# ============================================================

def search(
    query: str,
    index_data: dict,
    top_k: int = 5,
) -> list[dict]:
    """
    执行 BM25 搜索。

    返回统一格式：

    {
        "chunk_id": ...,
        "content": ...,
        "bm25_score": ...,
        "bm25_rank": ...,
        "metadata": {...}
    }
    """

    if not isinstance(
        query,
        str,
    ):

        raise TypeError(
            "query 必须是字符串。"
        )

    query = query.strip()

    if not query:

        return []

    if top_k <= 0:

        return []

    bm25 = index_data[
        "bm25"
    ]

    chunks = index_data[
        "chunks"
    ]

    query_tokens = tokenize(
        query
    )

    if not query_tokens:

        return []

    print()
    print(
        f"Query Tokens: "
        f"{query_tokens}"
    )

    # ------------------------------------
    # 计算所有 Chunk 的 BM25 Score
    # ------------------------------------

    scores = bm25.get_scores(
        query_tokens
    )

    # ------------------------------------
    # index + score
    # ------------------------------------

    ranked_indices = sorted(

        range(len(scores)),

        key=lambda i: scores[i],

        reverse=True,
    )

    top_indices = ranked_indices[
        :min(
            top_k,
            len(ranked_indices),
        )
    ]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        chunk = chunks[index]

        result = {

            "chunk_id": chunk.get(
                "chunk_id",
                "unknown",
            ),

            "content": chunk.get(
                "content",
                "",
            ),

            "bm25_score": float(
                scores[index]
            ),

            "bm25_rank": rank,

            "metadata": (
                build_metadata(
                    chunk
                )
            ),
        }

        results.append(
            result
        )

    return results


# ============================================================
# 11. 打印搜索结果
# ============================================================

def print_results(
    results: list[dict],
) -> None:

    if not results:

        print(
            "没有 BM25 搜索结果。"
        )

        return

    for result in results:

        metadata = result[
            "metadata"
        ]

        print()
        print("=" * 70)

        print(
            f'Rank {result["bm25_rank"]}'
        )

        print(
            f'Chunk ID: '
            f'{result["chunk_id"]}'
        )

        print(
            f'BM25 Score: '
            f'{result["bm25_score"]:.4f}'
        )

        print(
            f'Source: '
            f'{metadata["source"]}'
        )

        print(
            f'Page: '
            f'{metadata["page_start"]}'
            f'-'
            f'{metadata["page_end"]}'
        )

        print(
            f'Knowledge Base: '
            f'{metadata.get("knowledge_base_id", "unknown")}'
        )

        print(
            f'Device: '
            f'{metadata["device_model"]}'
        )

        print(
            "Content:"
        )

        print(
            result["content"]
        )


# ============================================================
# 12. Main
# ============================================================

def main():

    print("=" * 70)
    print("BM25 Retriever Test")
    print("=" * 70)

    # ------------------------------------
    # Index 不存在：
    # build → save
    #
    # Index 已存在：
    # load
    # ------------------------------------

    if INDEX_PATH.exists():

        print(
            "发现已有 BM25 Index"
        )

        print(
            "正在加载..."
        )

        index_data = load_index()

    else:

        print(
            "未发现 BM25 Index"
        )

        print(
            "开始读取 Chunk..."
        )

        chunks = load_chunks()

        print(
            f"Chunk 数量：{len(chunks)}"
        )

        index_data = build_index(
            chunks
        )

        save_index(
            index_data
        )

    # ------------------------------------
    # 测试 Query
    # ------------------------------------

    print()
    print("=" * 70)

    query = input(
        "请输入测试问题："
    ).strip()

    results = search(
        query=query,
        index_data=index_data,
        top_k=5,
    )

    print_results(
        results
    )


if __name__ == "__main__":

    main()
