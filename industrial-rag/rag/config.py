"""RAG Vector Store 配置。

服务地址和运行目录统一读取 app.config，避免在业务模块中
重复写死端口与本机路径。
"""

import os

from app.config import (
    EMBEDDING_BASE_URL,
    HTTP_TIMEOUT,
    RETRIEVAL_TOP_K,
    VECTOR_DB_DIR,
)


# ============================================================
# 项目路径
# ============================================================

PERSIST_DIRECTORY = VECTOR_DB_DIR

COLLECTION_NAME = "industrial_manual_chunks"


# ============================================================
# Embedding 服务
# ============================================================

EMBEDDING_URL = f"{EMBEDDING_BASE_URL.rstrip('/')}/embed"
EMBEDDING_TIMEOUT = HTTP_TIMEOUT


# ============================================================
# 默认检索参数
# ============================================================

TOP_K = RETRIEVAL_TOP_K

# 当前沿用你现有 config.py 中的值。
# 如果 threshold_test.py 最终实验得到的正式值不是 0.5，
# 只需要修改这里这一处。
DISTANCE_THRESHOLD = float(
    os.getenv("VECTOR_DISTANCE_THRESHOLD", "1.0")
)
