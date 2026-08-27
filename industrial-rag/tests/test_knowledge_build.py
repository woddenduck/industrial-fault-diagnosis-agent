"""
Day20 Stage3 - Knowledge Base Build Test

用途：
1. PyCharm 中可直接点击 Run 运行；
2. 也保留命令行参数，可在 Linux / Windows 终端覆盖默认配置；
3. 严格按照 app/services/knowledge_builder.py 的真实接口执行；
4. 自动检查 Stage3 的最终验收结果。

实际调用链：

create_and_build_knowledge_base(...)
        ↓
create_knowledge_base(...)
        ↓
build_knowledge_base(knowledge_base_id, ...)
        ↓
PDF Parse
        ↓
Section Chunk
        ↓
Embedding
        ↓
Vector Store
        ↓
KB BM25
        ↓
validate_knowledge_base_build(...)
        ↓
PASS / FAIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ============================================================
# 1. 项目路径
# ============================================================

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. PyCharm 默认测试配置
#    直接点击 Run 时使用这里
# ============================================================

DEFAULT_CONFIG = {
    "name": "G120C故障知识库",
    "device_model": "G120C",

    # 当前测试文档的业务 document_id
    "document_id": "G120C_op_instr_0226_zh-CHS",

    # 当前测试 PDF 的真实路径
    "document_path": str(
        PROJECT_ROOT
        / "data"
        / "uploads"
        / "G120C_op_instr_0226_zh-CHS.pdf"
    ),

    # 保持与你当前 Stage3 Builder 的默认/稳定配置一致即可。
    # 如果 Embedding 服务对单批大小有限制，可继续调小。
    "embedding_batch_size": 16,
    "upsert_batch_size": 16,

    # 建议先沿用你当前希望测试的切片参数。
    "chunk_size": 800,
    "overlap": 100,
}


# ============================================================
# 3. 命令行参数
#    不传参数时自动使用 DEFAULT_CONFIG
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day20 Stage3 Knowledge Base Build Test"
    )

    parser.add_argument(
        "--name",
        default=DEFAULT_CONFIG["name"],
        help="知识库名称",
    )
    parser.add_argument(
        "--device-model",
        dest="device_model",
        default=DEFAULT_CONFIG["device_model"],
        help="知识库设备型号",
    )
    parser.add_argument(
        "--document-id",
        dest="document_id",
        default=DEFAULT_CONFIG["document_id"],
        help="待构建文档的 document_id",
    )
    parser.add_argument(
        "--document-path",
        dest="document_path",
        default=DEFAULT_CONFIG["document_path"],
        help="待构建 PDF 的真实路径",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_CONFIG["embedding_batch_size"],
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=DEFAULT_CONFIG["upsert_batch_size"],
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CONFIG["chunk_size"],
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CONFIG["overlap"],
    )

    return parser.parse_args()


# ============================================================
# 4. 测试配置检查
# ============================================================


def validate_test_config(args: argparse.Namespace) -> Path:
    if not str(args.name).strip():
        raise ValueError("name 不能为空")

    if not str(args.device_model).strip():
        raise ValueError("device_model 不能为空")

    if not str(args.document_id).strip():
        raise ValueError("document_id 不能为空")

    document_path = Path(args.document_path).expanduser().resolve()

    if not document_path.exists():
        raise FileNotFoundError(
            "测试 PDF 不存在：\n"
            f"{document_path}\n\n"
            "请修改 test_knowledge_build.py 中：\n"
            "DEFAULT_CONFIG['document_path']"
        )

    if not document_path.is_file():
        raise ValueError(f"document_path 不是文件：{document_path}")

    if document_path.suffix.lower() != ".pdf":
        raise ValueError(
            "当前 knowledge_builder.py 的 Stage3 Parser 只支持 PDF："
            f"{document_path}"
        )

    if args.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size 必须大于 0")

    if args.upsert_batch_size <= 0:
        raise ValueError("upsert_batch_size 必须大于 0")

    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if args.overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if args.overlap >= args.chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    return document_path


# ============================================================
# 5. 输出测试配置
# ============================================================


def print_test_config(
    args: argparse.Namespace,
    document_path: Path,
) -> None:
    print("\n" + "=" * 80)
    print("Day20 Stage3 | Knowledge Base Build Test")
    print("=" * 80)

    print("\n[Test Configuration]")
    print(f"Knowledge Name : {args.name}")
    print(f"Device Model   : {args.device_model}")
    print(f"Document ID    : {args.document_id}")
    print(f"Document Path  : {document_path}")
    print(f"Embedding Batch: {args.embedding_batch_size}")
    print(f"Vector Batch   : {args.upsert_batch_size}")
    print(f"Chunk Size     : {args.chunk_size}")
    print(f"Overlap        : {args.overlap}")
    print("=" * 80)


# ============================================================
# 6. Stage3 最终验收
# ============================================================


def assert_stage3_result(result: dict) -> None:
    """对 knowledge_builder.py 返回结果做最后一道测试脚本级验收。"""

    if not isinstance(result, dict):
        raise AssertionError("Builder 返回结果必须是 dict")

    knowledge_base = result.get("knowledge_base")
    report = result.get("report")

    if not isinstance(knowledge_base, dict):
        raise AssertionError("Builder 返回结果缺少 knowledge_base")

    if not isinstance(report, dict):
        raise AssertionError("Builder 返回结果缺少 report")

    knowledge_base_id = report.get("knowledge_base_id")
    chunk_count = report.get("chunk_count")
    vector_count = report.get("vector_count")
    bm25_count = report.get("bm25_document_count")
    passed = report.get("passed")
    errors = report.get("errors") or []

    # --------------------------------------------------------
    # Stage3 核心验收条件
    # --------------------------------------------------------

    assert knowledge_base_id, "knowledge_base_id 为空"

    assert isinstance(chunk_count, int), "chunk_count 不是 int"
    assert isinstance(vector_count, int), "vector_count 不是 int"
    assert isinstance(bm25_count, int), "bm25_document_count 不是 int"

    assert chunk_count > 0, "Chunk 数量必须大于 0"

    assert chunk_count == vector_count, (
        f"Chunk数量({chunk_count}) != Vector数量({vector_count})"
    )

    assert chunk_count == bm25_count, (
        f"Chunk数量({chunk_count}) != BM25文档数量({bm25_count})"
    )

    assert passed is True, (
        "knowledge_builder.py 内部验收未通过：\n"
        + "\n".join(f"- {item}" for item in errors[:30])
    )

    # Builder 成功后应把 KB 更新为 ready。
    kb_status = str(knowledge_base.get("status", "")).strip().lower()
    assert kb_status == "ready", (
        f"Knowledge Base status 应为 ready，实际为 {kb_status!r}"
    )

    chunk_file = result.get("chunk_file")
    bm25_index_file = result.get("bm25_index_file")

    if chunk_file is not None:
        assert Path(chunk_file).exists(), (
            f"Chunk JSON 不存在：{chunk_file}"
        )

    if bm25_index_file is not None:
        assert Path(bm25_index_file).exists(), (
            f"BM25 Index 不存在：{bm25_index_file}"
        )


# ============================================================
# 7. 结果打印
# ============================================================


def print_final_result(result: dict) -> None:
    kb = result["knowledge_base"]
    report = result["report"]

    print("\n" + "=" * 80)
    print("Stage3 Final Acceptance")
    print("=" * 80)

    print(f"Knowledge Base ID : {report['knowledge_base_id']}")
    print(f"Knowledge Name    : {kb.get('name')}")
    print(f"Status            : {kb.get('status')}")
    print(f"Device Model      : {report.get('device_model')}")
    print(f"Document IDs      : {report.get('document_ids')}")
    print(f"Chunk Count       : {report.get('chunk_count')}")
    print(f"Vector Count      : {report.get('vector_count')}")
    print(f"BM25 Count        : {report.get('bm25_document_count')}")
    print(f"Chunk File        : {result.get('chunk_file')}")
    print(f"BM25 Index File   : {result.get('bm25_index_file')}")
    print(f"Builder Report    : {'PASS' if report.get('passed') else 'FAIL'}")

    errors = report.get("errors") or []
    if errors:
        print("\n[Errors]")
        for item in errors[:30]:
            print(f"- {item}")

    print("=" * 80)
    print("Stage3 TEST RESULT: PASS")
    print("=" * 80)


# ============================================================
# 8. 主流程
# ============================================================


def main() -> None:
    args = parse_args()
    document_path = validate_test_config(args)
    print_test_config(args, document_path)

    # --------------------------------------------------------
    # 严格按照 knowledge_builder.py 的一键入口调用
    # --------------------------------------------------------
    from app.services.knowledge_builder import (
        create_and_build_knowledge_base,
    )

    # knowledge_builder.py 要求：
    # document_paths = {
    #     "doc_xxx": "/path/to/manual.pdf"
    # }
    document_paths = {
        args.document_id: document_path,
    }

    print("\n开始执行 Stage3 完整闭环...")

    result = create_and_build_knowledge_base(
        name=args.name,
        device_model=args.device_model,
        document_ids=[args.document_id],
        document_paths=document_paths,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_batch_size=args.embedding_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        replace_existing_vectors=True,
    )

    # Builder 自身已经做过一次验收；这里再做测试脚本级断言。
    assert_stage3_result(result)
    print_final_result(result)


if __name__ == "__main__":
    main()