"""
Day21 Stage2 - Centralized cross-platform path management.

Rules
-----
1. Business code imports paths from this module instead of hard-coding OS paths.
2. Relative configured paths are always resolved from PROJECT_ROOT, never cwd.
3. Required persistent directories are validated before runtime directories are created.
4. Importing this module has no mkdir side effect; call initialize_paths() at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from app.config import ConfigurationError, PROJECT_ROOT, settings


def _resolve_path(raw_value: str, *, name: str) -> Path:
    """
    Convert one configured path into an absolute Path.

    Relative values:
        resolved against PROJECT_ROOT.

    Absolute values:
        kept as absolute paths.

    resolve(strict=False) normalizes ".", ".." and separators without requiring
    the target to exist.
    """
    if raw_value is None or not str(raw_value).strip():
        raise ConfigurationError(
            f"Missing required path configuration: {name}"
        )

    path = Path(str(raw_value).strip()).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve(strict=False)


DATA_DIR: Final[Path] = _resolve_path(settings.data_dir, name="DATA_DIR")
MODEL_DIR: Final[Path] = _resolve_path(settings.model_dir, name="MODEL_DIR")
UPLOAD_DIR: Final[Path] = _resolve_path(settings.upload_dir, name="UPLOAD_DIR")
VECTOR_DB_DIR: Final[Path] = _resolve_path(
    settings.vector_db_dir,
    name="VECTOR_DB_DIR",
)
LOG_DIR: Final[Path] = _resolve_path(settings.log_dir, name="LOG_DIR")

# REPORTS_DIR is project-owned and does not need a separate environment variable.
REPORTS_DIR: Final[Path] = (PROJECT_ROOT / "reports").resolve(strict=False)


# These contain persistent/project state. A typo must fail fast rather than
# silently creating a new empty data/model/index location.
REQUIRED_EXISTING_DIRECTORIES: Final[dict[str, Path]] = {
    "DATA_DIR": DATA_DIR,
    "MODEL_DIR": MODEL_DIR,
    "VECTOR_DB_DIR": VECTOR_DB_DIR,
}

# These are runtime-output directories and are safe to create automatically.
CREATABLE_DIRECTORIES: Final[dict[str, Path]] = {
    "UPLOAD_DIR": UPLOAD_DIR,
    "LOG_DIR": LOG_DIR,
    "REPORTS_DIR": REPORTS_DIR,
}


def validate_required_directories() -> None:
    """
    Validate persistent directories that must already exist.

    Important:
        This function runs before create_runtime_directories(), otherwise a
        nested mkdir(parents=True) could accidentally create a misspelled
        DATA_DIR and hide a bad configuration.
    """
    for name, path in REQUIRED_EXISTING_DIRECTORIES.items():
        if not path.exists():
            raise ConfigurationError(
                f"Required directory does not exist: {name}={path}"
            )
        if not path.is_dir():
            raise ConfigurationError(
                f"Configured path is not a directory: {name}={path}"
            )


def create_runtime_directories() -> None:
    """Create only directories whose absence is expected during normal startup."""
    for name, path in CREATABLE_DIRECTORIES.items():
        if path.exists() and not path.is_dir():
            raise ConfigurationError(
                f"Runtime directory path is occupied by a file: {name}={path}"
            )

        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to create runtime directory: {name}={path}: {exc}"
            ) from exc


def initialize_paths() -> None:
    """
    Perform startup filesystem validation.

    Order is intentional:
        1. validate persistent directories
        2. create runtime directories
    """
    validate_required_directories()
    create_runtime_directories()


def path_summary() -> dict[str, str]:
    """Return resolved paths for logs, diagnostics and acceptance tests."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "MODEL_DIR": str(MODEL_DIR),
        "UPLOAD_DIR": str(UPLOAD_DIR),
        "VECTOR_DB_DIR": str(VECTOR_DB_DIR),
        "LOG_DIR": str(LOG_DIR),
        "REPORTS_DIR": str(REPORTS_DIR),
    }


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODEL_DIR",
    "UPLOAD_DIR",
    "VECTOR_DB_DIR",
    "LOG_DIR",
    "REPORTS_DIR",
    "REQUIRED_EXISTING_DIRECTORIES",
    "CREATABLE_DIRECTORIES",
    "validate_required_directories",
    "create_runtime_directories",
    "initialize_paths",
    "path_summary",
]
