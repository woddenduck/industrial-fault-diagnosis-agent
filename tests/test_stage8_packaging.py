"""第八关第一批：配置与部署封装验收。

既可由 pytest 执行，也可直接使用 python 运行并查看完整过程。
本测试只做离线检查，不会启动模型或占用端口。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _title(name: str) -> None:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")


def _load_env_template() -> dict[str, str]:
    result: dict[str, str] = {}

    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, separator, value = line.partition("=")
        assert separator, f"配置行缺少等号：{raw_line}"
        assert key not in result, f"配置项重复：{key}"
        result[key] = value

    return result


def test_01_clean_release_boundary() -> None:
    _title("P01｜发布目录边界")

    forbidden_names = {
        ".git",
        ".ipynb_checkpoints",
    }
    # 正式 GitHub 仓库的根目录必然包含 .git。
    # 这里只禁止项目内部出现嵌套 Git 历史和缓存目录。
    root_git = PROJECT_ROOT / ".git"

    found = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*")
        if path.name in forbidden_names and path != root_git
    ]

    assert not found, f"发布目录仍包含缓存或嵌套 Git 历史：{found}"
    assert not (PROJECT_ROOT / "industrial-fault-diagnosis-agent").exists()
    print("结果：通过。发布副本不包含嵌套 Git 或旧版同名目录。")


def test_02_environment_contract() -> None:
    _title("P02｜统一环境变量契约")
    values = _load_env_template()

    required = {
        "LLM_PORT",
        "EMBEDDING_PORT",
        "RERANKER_PORT",
        "GATEWAY_PORT",
        "RAG_PORT",
        "AGENT_PORT",
        "LLM_MODEL_PATH",
        "EMBEDDING_MODEL_PATH",
        "RERANKER_MODEL_PATH",
        "AGENT_DATA_DIR",
        "DEFAULT_KNOWLEDGE_BASE_ID",
    }
    missing = required - values.keys()
    assert not missing, f".env.example 缺少配置：{sorted(missing)}"

    ports = [
        int(values[name])
        for name in (
            "LLM_PORT",
            "EMBEDDING_PORT",
            "RERANKER_PORT",
            "GATEWAY_PORT",
            "RAG_PORT",
            "AGENT_PORT",
        )
    ]
    assert len(ports) == len(set(ports)), "六项服务端口存在冲突"
    assert values["DEFAULT_KNOWLEDGE_BASE_ID"] == ""
    print("结果：通过。六项服务配置完整、端口唯一、知识库 ID 未写死。")


def test_03_shell_scripts() -> None:
    _title("P03｜部署脚本语法")

    scripts = [
        "start_llm.sh",
        "start_embedding.sh",
        "start_reranker.sh",
        "start_gateway.sh",
        "start_rag.sh",
        "start_agent.sh",
        "start_all.sh",
        "health_check.sh",
        "stop_all.sh",
    ]

    for name in scripts:
        path = DEPLOYMENT_DIR / name
        assert path.is_file(), f"缺少部署脚本：{name}"
        subprocess.run(
            ["bash", "-n", str(path)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        print(f"通过：{name}")

    env = os.environ.copy()
    env["PROFILE_FILE"] = str(ENV_EXAMPLE)
    result = subprocess.run(
        ["bash", str(DEPLOYMENT_DIR / "start_all.sh"), "--validate"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stderr
    print("结果：通过。统一启动脚本静态校验成功，未启动任何服务。")


def test_04_no_machine_specific_path() -> None:
    _title("P04｜跨机器路径")

    checked_files = [
        ENV_EXAMPLE,
        PROJECT_ROOT / "agent" / "config.py",
        PROJECT_ROOT / "gateway" / "app" / "config.py",
        PROJECT_ROOT / "embedding_service" / "app.py",
        PROJECT_ROOT / "industrial-rag" / "config" / "autodl.env",
    ]

    for path in checked_files:
        content = path.read_text(encoding="utf-8")
        assert "/root/autodl-tmp" not in content, f"仍存在 AutoDL 绝对路径：{path}"

    print("结果：通过。正式配置不再绑定原 AutoDL 目录。")


def test_05_http_error_code_fallback() -> None:
    _title("P05｜Agent HTTP 错误码映射")

    # 只提取待测映射函数，避免离线封装验收依赖 FastAPI/HTTPX。
    api_path = PROJECT_ROOT / "agent" / "api.py"
    tree = ast.parse(api_path.read_text(encoding="utf-8"))
    selected_nodes: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if "ERROR_HTTP_STATUS" in names:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "resolve_http_status":
            selected_nodes.append(node)

    namespace: dict[str, object] = {"DiagnoseResponse": object}
    isolated_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(api_path), "exec"), namespace)

    result = SimpleNamespace(
        status="failed",
        error="Knowledge Base 不存在：kb_missing",
        errors=[SimpleNamespace(code="KNOWLEDGE_BASE_NOT_FOUND")],
    )
    resolve_http_status = namespace["resolve_http_status"]

    assert resolve_http_status(result) == 404
    print("结果：通过。可读错误信息不会再把 KB 不存在误映射为 HTTP 500。")


def main() -> int:
    tests = [
        test_01_clean_release_boundary,
        test_02_environment_contract,
        test_03_shell_scripts,
        test_04_no_machine_specific_path,
        test_05_http_error_code_fallback,
    ]

    print("第八关第一批：配置与部署封装验收")
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"\n[失败] {test.__name__}：{exc}")
            print(f"验收结果：{passed}/{len(tests)} 通过")
            return 1

    print(f"\n验收结果：{passed}/{len(tests)} 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
