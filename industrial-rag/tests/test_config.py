"""
Day21 Stage2 acceptance tests.

Run directly:
    python tests/test_config.py

Or with pytest:
    pytest -q tests/test_config.py

The destructive/failure cases run in isolated subprocesses and temporary
directories, so they do not delete or modify your real data/model/vector store.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make direct execution (`python tests/test_config.py`) independent of cwd/sys.path quirks.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_KEYS = (
    "LLM_BASE_URL",
    "EMBEDDING_BASE_URL",
    "RERANKER_BASE_URL",
    "LLM_PORT",
    "EMBEDDING_PORT",
    "RERANKER_PORT",
    "GATEWAY_PORT",
    "RAG_PORT",
    "DATA_DIR",
    "MODEL_DIR",
    "UPLOAD_DIR",
    "VECTOR_DB_DIR",
    "LOG_DIR",
    "RETRIEVAL_TOP_K",
    "RERANK_TOP_K",
    "SIMILARITY_THRESHOLD",
    "HTTP_TIMEOUT",
    "LLM_TIMEOUT",
    "MODEL_MAX_CONTEXT",
    "CONTEXT_TOKEN_BUDGET",
    "HISTORY_TOKEN_BUDGET",
)


def _write_env(
    path: Path,
    *,
    data_dir: Path,
    model_dir: Path,
    upload_dir: Path,
    vector_dir: Path,
    log_dir: Path,
    omit: set[str] | None = None,
) -> None:
    omit = omit or set()

    values = {
        "LLM_BASE_URL": "http://127.0.0.1:6006",
        "EMBEDDING_BASE_URL": "http://127.0.0.1:6010",
        "RERANKER_BASE_URL": "http://127.0.0.1:6009",
        "LLM_PORT": "6006",
        "EMBEDDING_PORT": "6010",
        "RERANKER_PORT": "6009",
        "GATEWAY_PORT": "6008",
        "RAG_PORT": "8000",
        "DATA_DIR": str(data_dir),
        "MODEL_DIR": str(model_dir),
        "UPLOAD_DIR": str(upload_dir),
        "VECTOR_DB_DIR": str(vector_dir),
        "LOG_DIR": str(log_dir),
        "RETRIEVAL_TOP_K": "10",
        "RERANK_TOP_K": "5",
        "SIMILARITY_THRESHOLD": "0.0",
        "HTTP_TIMEOUT": "30",
        "LLM_TIMEOUT": "120",
        "MODEL_MAX_CONTEXT": "8192",
        "CONTEXT_TOKEN_BUDGET": "4096",
        "HISTORY_TOKEN_BUDGET": "1024",
    }

    lines = [
        f"{key}={value}"
        for key, value in values.items()
        if key not in omit
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _isolated_env(config_file: Path) -> dict[str, str]:
    env = os.environ.copy()

    # Critical: remove inherited config so a shell-exported LLM_BASE_URL cannot
    # accidentally make the "missing configuration" test pass.
    for key in CONFIG_KEYS:
        env.pop(key, None)

    env.pop("RAG_ENV", None)
    env["RAG_CONFIG_FILE"] = str(config_file)

    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not current_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
    )
    return env


def _run_python(code: str, config_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=_isolated_env(config_file),
        text=True,
        capture_output=True,
        check=False,
    )


def test_current_configuration_loads() -> None:
    from app.config import CONFIG_FILE, settings

    assert CONFIG_FILE.is_file()
    assert settings.llm_base_url.startswith(("http://", "https://"))
    assert settings.embedding_port == 6010
    assert 1 <= settings.llm_port <= 65535
    assert settings.retrieval_top_k > 0
    assert 0 < settings.rerank_top_k <= settings.retrieval_top_k
    assert settings.http_timeout > 0
    assert settings.llm_timeout > 0
    assert settings.context_token_budget <= settings.model_max_context


def test_paths_are_absolute_and_project_relative_values_use_project_root() -> None:
    from app.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT, settings
    from common.paths import (
        DATA_DIR,
        LOG_DIR,
        MODEL_DIR,
        UPLOAD_DIR,
        VECTOR_DB_DIR,
    )

    paths = [DATA_DIR, MODEL_DIR, UPLOAD_DIR, VECTOR_DB_DIR, LOG_DIR]
    assert all(isinstance(path, Path) for path in paths)
    assert all(path.is_absolute() for path in paths)

    # Only assert project-root resolution for values that are actually relative.
    configured = {
        "DATA_DIR": (settings.data_dir, DATA_DIR),
        "MODEL_DIR": (settings.model_dir, MODEL_DIR),
        "UPLOAD_DIR": (settings.upload_dir, UPLOAD_DIR),
        "VECTOR_DB_DIR": (settings.vector_db_dir, VECTOR_DB_DIR),
        "LOG_DIR": (settings.log_dir, LOG_DIR),
    }

    for _, (raw, resolved) in configured.items():
        raw_path = Path(raw).expanduser()
        if not raw_path.is_absolute():
            expected = (CONFIG_PROJECT_ROOT / raw_path).resolve(strict=False)
            assert resolved == expected


def test_missing_llm_base_url_fails_fast() -> None:
    with tempfile.TemporaryDirectory(prefix="day21_cfg_missing_") as tmp:
        tmp_path = Path(tmp)
        data = tmp_path / "data"
        model = tmp_path / "models"
        vector = data / "vector_store"
        upload = data / "uploads"
        logs = tmp_path / "logs"

        data.mkdir()
        model.mkdir()
        vector.mkdir(parents=True)

        env_file = tmp_path / "missing_llm.env"
        _write_env(
            env_file,
            data_dir=data,
            model_dir=model,
            upload_dir=upload,
            vector_dir=vector,
            log_dir=logs,
            omit={"LLM_BASE_URL"},
        )

        code = """
try:
    import app.config  # noqa
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("EXPECTED_CONFIGURATION_ERROR_NOT_RAISED")
"""
        result = _run_python(code, env_file)

        assert result.returncode == 0, result.stderr
        assert "ConfigurationError" in result.stdout
        assert (
            "Missing required configuration: LLM_BASE_URL"
            in result.stdout
        )


def test_runtime_dirs_created_but_required_dirs_not_auto_created() -> None:
    with tempfile.TemporaryDirectory(prefix="day21_paths_") as tmp:
        tmp_path = Path(tmp)
        data = tmp_path / "data"
        model = tmp_path / "models"
        vector = data / "vector_store"
        upload = data / "uploads"
        logs = tmp_path / "logs"

        # Required directories already exist.
        data.mkdir()
        model.mkdir()
        vector.mkdir(parents=True)

        env_file = tmp_path / "paths.env"
        _write_env(
            env_file,
            data_dir=data,
            model_dir=model,
            upload_dir=upload,
            vector_dir=vector,
            log_dir=logs,
        )

        success_code = """
from common.paths import LOG_DIR, UPLOAD_DIR, REPORTS_DIR, initialize_paths
initialize_paths()
assert UPLOAD_DIR.is_dir()
assert LOG_DIR.is_dir()
assert REPORTS_DIR.is_dir()
print("RUNTIME_DIRS_CREATED")
"""
        success = _run_python(success_code, env_file)
        assert success.returncode == 0, success.stderr
        assert "RUNTIME_DIRS_CREATED" in success.stdout
        assert upload.is_dir()
        assert logs.is_dir()

        # Now intentionally remove a required directory.
        model.rmdir()

        fail_code = """
try:
    from common.paths import initialize_paths
    initialize_paths()
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("EXPECTED_CONFIGURATION_ERROR_NOT_RAISED")
"""
        failure = _run_python(fail_code, env_file)

        assert failure.returncode == 0, failure.stderr
        assert "ConfigurationError" in failure.stdout
        assert "Required directory does not exist: MODEL_DIR=" in failure.stdout

        # The path layer must NOT recreate required persistent directories.
        assert not model.exists()


def test_invalid_numeric_configuration_fails_fast() -> None:
    with tempfile.TemporaryDirectory(prefix="day21_cfg_invalid_") as tmp:
        tmp_path = Path(tmp)
        data = tmp_path / "data"
        model = tmp_path / "models"
        vector = data / "vector_store"
        upload = data / "uploads"
        logs = tmp_path / "logs"

        data.mkdir()
        model.mkdir()
        vector.mkdir(parents=True)

        env_file = tmp_path / "invalid.env"
        _write_env(
            env_file,
            data_dir=data,
            model_dir=model,
            upload_dir=upload,
            vector_dir=vector,
            log_dir=logs,
        )

        content = env_file.read_text(encoding="utf-8")
        content = content.replace("HTTP_TIMEOUT=30", "HTTP_TIMEOUT=abc")
        env_file.write_text(content, encoding="utf-8")

        code = """
try:
    import app.config  # noqa
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("EXPECTED_CONFIGURATION_ERROR_NOT_RAISED")
"""
        result = _run_python(code, env_file)

        assert result.returncode == 0, result.stderr
        assert "ConfigurationError" in result.stdout
        assert "Invalid float configuration: HTTP_TIMEOUT='abc'" in result.stdout


def test_runtime_python_has_no_known_machine_specific_path_literals() -> None:
    """
    Stage gate for Day21 path migration.

    It intentionally scans runtime source only (app/rag/common), not reports,
    docs, env files or tests, because historical paths are allowed in docs but
    must disappear from executable Python.
    """
    roots = [
        PROJECT_ROOT / "app",
        PROJECT_ROOT / "rag",
        PROJECT_ROOT / "common",
    ]

    forbidden = (
        re.compile(r"/root/autodl-tmp/"),
        re.compile(r"[A-Za-z]:[\\/]+PythonCode"),
        re.compile(r"\.\.[\\/]\.\.[\\/]data"),
    )

    violations: list[str] = []

    for root in roots:
        if not root.exists():
            continue

        for file in root.rglob("*.py"):
            text = file.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden:
                if pattern.search(text):
                    violations.append(
                        f"{file.relative_to(PROJECT_ROOT)} -> {pattern.pattern}"
                    )

    assert not violations, (
        "Machine-specific/hard-coded paths still exist:\n"
        + "\n".join(f"  - {item}" for item in violations)
    )


def main() -> None:
    tests = [
        test_current_configuration_loads,
        test_paths_are_absolute_and_project_relative_values_use_project_root,
        test_missing_llm_base_url_fails_fast,
        test_runtime_dirs_created_but_required_dirs_not_auto_created,
        test_invalid_numeric_configuration_fails_fast,
        test_runtime_python_has_no_known_machine_specific_path_literals,
    ]

    print("=" * 80)
    print("Day21 Stage2 | Configuration & Cross-platform Path Acceptance")
    print("=" * 80)

    passed = 0

    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"[FAIL] {test.__name__}")
            print(f"       {exc}")
            raise
        else:
            passed += 1
            print(f"[PASS] {test.__name__}")

    print("-" * 80)
    print(f"PASS: {passed}/{len(tests)}")
    print("Stage2 acceptance passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
