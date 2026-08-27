from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(
    env_name: str,
    default: Path,
) -> Path:
    """读取项目路径；相对路径始终相对于仓库根目录。"""

    path = Path(
        os.getenv(env_name, str(default))
    ).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve(strict=False)


DATA_DIR = _project_path(
    "AGENT_DATA_DIR",
    PROJECT_ROOT / "data",
)


AGENT_HOST = os.getenv(
    "AGENT_HOST",
    "0.0.0.0",
)

AGENT_PORT = int(
    os.getenv(
        "AGENT_PORT",
        "8010",
    )
)


RAG_BASE_URL = os.getenv(
    "RAG_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

RAG_CHAT_URL = f"{RAG_BASE_URL}/chat"
RAG_HEALTH_URL = f"{RAG_BASE_URL}/health"

# 实测一次正常调用约 18.5 秒，设置 60 秒避免误超时。
RAG_TIMEOUT = float(
    os.getenv(
        "RAG_TIMEOUT",
        "60",
    )
)

RAG_MAX_RETRIES = int(
    os.getenv(
        "RAG_MAX_RETRIES",
        "1",
    )
)


DEFAULT_DEVICE_MODEL = os.getenv(
    "DEFAULT_DEVICE_MODEL",
    "G120C",
).strip().upper()

# 正式代码不写死历史 KB ID。
DEFAULT_KNOWLEDGE_BASE_ID = os.getenv(
    "DEFAULT_KNOWLEDGE_BASE_ID",
    "",
).strip()


DEVICE_STATUS_FILE = _project_path(
    "DEVICE_STATUS_FILE",
    DATA_DIR / "device_status.json",
)

MAINTENANCE_HISTORY_FILE = _project_path(
    "MAINTENANCE_HISTORY_FILE",
    DATA_DIR / "maintenance_history.json",
)
