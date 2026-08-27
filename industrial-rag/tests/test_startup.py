"""
Day21 Stage3 acceptance tests for unified startup and health checking.

AutoDL recommended run order:

    # 1) Static/preflight acceptance; does NOT start services.
    python tests/test_startup.py

    # 2) Start the stack.
    bash deployment/start_all.sh

    # 3) Strict live acceptance after startup.
    STARTUP_LIVE=1 python tests/test_startup.py

Pytest is also supported:

    pytest -q -s tests/test_startup.py
    STARTUP_LIVE=1 pytest -q -s tests/test_startup.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import pytest
except ImportError:  # direct execution does not require pytest
    pytest = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
PLATFORM_ROOT = PROJECT_ROOT.parent
PLATFORM_DEPLOYMENT = PLATFORM_ROOT / "deployment"

START_ALL_SH = DEPLOYMENT_DIR / "start_all.sh"
HEALTH_CHECK_SH = DEPLOYMENT_DIR / "health_check.sh"
START_ALL_BAT = DEPLOYMENT_DIR / "start_all.bat"

DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / ("autodl.env" if os.name != "nt" else "local.env")
)
CONFIG_FILE = Path(os.environ.get("RAG_CONFIG_FILE", DEFAULT_CONFIG)).resolve()
LIVE = os.environ.get("STARTUP_LIVE", "0").strip() == "1"

EXPECTED_PORTS = {
    "LLM_PORT": 6006,
    "EMBEDDING_PORT": 6010,
    "RERANKER_PORT": 6009,
    "GATEWAY_PORT": 6008,
    "RAG_PORT": 8000,
}

EXPECTED_URLS = {
    "LLM_BASE_URL": "http://127.0.0.1:6006",
    "EMBEDDING_BASE_URL": "http://127.0.0.1:6010",
    "RERANKER_BASE_URL": "http://127.0.0.1:6009",
}

REQUIRED_CONFIG_KEYS = {
    *EXPECTED_PORTS,
    *EXPECTED_URLS,
    "DATA_DIR",
    "MODEL_DIR",
    "UPLOAD_DIR",
    "VECTOR_DB_DIR",
    "LOG_DIR",
}

PARENT_LAUNCHERS = (
    "start_llm.sh",
    "start_embedding.sh",
    "start_reranker.sh",
    "start_gateway.sh",
)

AUTODL_CONDA_ENVS = {
    "vllm_qwen3",
    "embedding_service",
    "reranker_service",
    "llm_gateway",
    "rag_system",
}


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AssertionError(f"Invalid env line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


CONFIG = load_env_file(CONFIG_FILE) if CONFIG_FILE.exists() else {}


def resolve_project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_command(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "RAG_CONFIG_FILE": str(CONFIG_FILE)},
    )


def http_get_json(url: str, *, timeout: float = 5.0) -> tuple[int, object | None]:
    try:
        with urlopen(url, timeout=timeout) as response:
            code = int(response.status)
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return int(exc.code), None
    except (URLError, TimeoutError, OSError):
        return 0, None

    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, None


def assert_reachable(name: str, url: str) -> dict | list | None:
    code, payload = http_get_json(url)
    assert 200 <= code < 300, f"{name} is not reachable: {url}, HTTP={code}"
    if isinstance(payload, dict):
        status = str(payload.get("status", "ready")).lower()
        assert status not in {"failed", "unhealthy", "down", "error"}, (
            f"{name} returned failing status={status}: {payload}"
        )
    return payload if isinstance(payload, (dict, list)) else None


def test_required_files_exist() -> None:
    assert CONFIG_FILE.is_file(), f"Config not found: {CONFIG_FILE}"
    assert START_ALL_SH.is_file(), START_ALL_SH
    assert HEALTH_CHECK_SH.is_file(), HEALTH_CHECK_SH
    assert START_ALL_BAT.is_file(), START_ALL_BAT
    assert (PROJECT_ROOT / "app" / "main.py").is_file()


def test_stage2_config_contract() -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - CONFIG.keys())
    assert not missing, f"Missing Stage2 config keys: {missing}"

    actual_ports = {key: int(CONFIG[key]) for key in EXPECTED_PORTS}
    assert actual_ports == EXPECTED_PORTS
    assert len(set(actual_ports.values())) == len(actual_ports), (
        f"Duplicate service ports: {actual_ports}"
    )

    actual_urls = {key: CONFIG[key].rstrip("/") for key in EXPECTED_URLS}
    assert actual_urls == EXPECTED_URLS


def test_autodl_model_dir_exists_when_using_autodl_config() -> None:
    if CONFIG_FILE.name != "autodl.env":
        return
    model_dir = resolve_project_path(CONFIG["MODEL_DIR"])
    assert model_dir.is_dir(), f"AutoDL MODEL_DIR does not exist: {model_dir}"


def test_parent_service_launchers_exist() -> None:
    if CONFIG_FILE.name != "autodl.env":
        return
    missing = [
        str(PLATFORM_DEPLOYMENT / name)
        for name in PARENT_LAUNCHERS
        if not (PLATFORM_DEPLOYMENT / name).is_file()
    ]
    assert not missing, "Missing parent launchers:\n" + "\n".join(missing)


def test_linux_shell_syntax() -> None:
    if os.name == "nt":
        return
    bash = shutil.which("bash")
    assert bash, "bash not found"
    for script in (START_ALL_SH, HEALTH_CHECK_SH):
        result = run_command([bash, "-n", str(script)])
        assert result.returncode == 0, result.stderr


def test_startup_validate_mode() -> None:
    if os.name == "nt":
        return
    bash = shutil.which("bash")
    assert bash
    result = run_command([bash, str(START_ALL_SH), "--validate"], timeout=60)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Validation passed" in output
    assert "No service was started" in output


def test_port_conflict_detection_reports_service() -> None:
    if os.name == "nt":
        return
    bash = shutil.which("bash")
    assert bash

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        result = run_command(
            [bash, str(START_ALL_SH), "--check-port", "LLM", str(port)]
        )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert f"Port {port} is already in use." in output
    assert "Service: LLM" in output


def test_free_port_detection_passes() -> None:
    if os.name == "nt":
        return
    bash = shutil.which("bash")
    assert bash

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    result = run_command(
        [bash, str(START_ALL_SH), "--check-port", "TEST", str(port)]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"port {port} is available" in output


def test_autodl_conda_envs_exist() -> None:
    if CONFIG_FILE.name != "autodl.env":
        return
    conda = shutil.which("conda")
    assert conda, "conda command not found on AutoDL"

    result = run_command([conda, "env", "list", "--json"], timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    env_names = {Path(item).name for item in payload.get("envs", [])}
    missing = sorted(AUTODL_CONDA_ENVS - env_names)
    assert not missing, f"Missing AutoDL conda envs: {missing}; found={sorted(env_names)}"


def test_live_health_endpoints() -> None:
    if not LIVE:
        if pytest is not None:
            pytest.skip("Set STARTUP_LIVE=1 after start_all.sh for live acceptance")
        return

    llm_url = CONFIG["LLM_BASE_URL"].rstrip("/") + "/v1/models"
    embedding_url = CONFIG["EMBEDDING_BASE_URL"].rstrip("/") + "/health"
    reranker_url = CONFIG["RERANKER_BASE_URL"].rstrip("/") + "/health"
    gateway_url = f"http://127.0.0.1:{CONFIG['GATEWAY_PORT']}/health"
    rag_url = f"http://127.0.0.1:{CONFIG['RAG_PORT']}/health"

    assert_reachable("LLM", llm_url)
    assert_reachable("Embedding", embedding_url)
    assert_reachable("Reranker", reranker_url)
    assert_reachable("Gateway", gateway_url)  # degraded is explicitly allowed
    assert_reachable("Industrial RAG", rag_url)


def test_live_health_script() -> None:
    if not LIVE:
        if pytest is not None:
            pytest.skip("Set STARTUP_LIVE=1 after start_all.sh for live acceptance")
        return
    if os.name == "nt":
        return

    bash = shutil.which("bash")
    assert bash
    result = run_command([bash, str(HEALTH_CHECK_SH)], timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Overall: PASS" in output


def _run_direct() -> int:
    tests = [
        test_required_files_exist,
        test_stage2_config_contract,
        test_autodl_model_dir_exists_when_using_autodl_config,
        test_parent_service_launchers_exist,
        test_linux_shell_syntax,
        test_startup_validate_mode,
        test_port_conflict_detection_reports_service,
        test_free_port_detection_passes,
        test_autodl_conda_envs_exist,
    ]
    if LIVE:
        tests.extend([test_live_health_endpoints, test_live_health_script])

    print("=" * 72)
    print("Day21 Stage3 | Startup Acceptance")
    print(f"Config : {CONFIG_FILE}")
    print(f"Live   : {LIVE}")
    print("=" * 72)

    failures = 0
    for fn in tests:
        started = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - acceptance runner wants full report
            failures += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
            print(f"[PASS] {fn.__name__} ({elapsed_ms:.1f} ms)")

    print("=" * 72)
    if failures:
        print(f"RESULT: FAIL ({failures} failed)")
        return 1
    print("RESULT: PASS")
    if not LIVE:
        print("Next: bash deployment/start_all.sh")
        print("Then: STARTUP_LIVE=1 python tests/test_startup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_direct())