"""第八关第二批：GitHub 仓库完整性验收。

本程序不访问网络，不启动模型，支持 pytest 和 python 直接运行。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _title(name: str) -> None:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")


def _read(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    assert path.is_file(), f"缺少文件：{relative_path}"
    return path.read_text(encoding="utf-8")


def test_01_service_requirements() -> None:
    _title("R01｜分服务依赖")

    expected = {
        "requirements/agent.txt": {"fastapi", "langgraph", "httpx"},
        "requirements/gateway.txt": {"fastapi", "pydantic-settings", "httpx"},
        "requirements/rag.txt": {"chromadb", "rank-bm25", "pymupdf"},
        "requirements/embedding.txt": {"sentence-transformers", "torch"},
        "requirements/reranker.txt": {"transformers", "torch"},
        "requirements/dev.txt": {"pytest"},
        "requirements/vllm.txt": {"vllm"},
    }

    for relative_path, packages in expected.items():
        content = _read(relative_path).lower()
        missing = {package for package in packages if package not in content}
        assert not missing, f"{relative_path} 缺少：{sorted(missing)}"
        print(f"通过：{relative_path}")

    print("结果：通过。六项服务和开发测试依赖已经分离。")


def test_02_project_and_ci() -> None:
    _title("R02｜项目元数据与 CI")

    pyproject = _read("pyproject.toml")
    workflow = _read(".github/workflows/ci.yml")

    assert 'requires-python = ">=3.10,<3.12"' in pyproject
    assert "not e2e" in pyproject
    assert "python -m pytest" in workflow
    assert "requirements/agent.txt" in workflow
    assert "requirements/dev.txt" in workflow

    print("结果：通过。Python 版本、Pytest 策略和 GitHub CI 已配置。")


def test_03_repository_documents() -> None:
    _title("R03｜仓库文档")

    expected = [
        "README.md",
        "docs/architecture.md",
        "docs/deployment.md",
        "docs/api.md",
        "docs/testing.md",
    ]

    for relative_path in expected:
        content = _read(relative_path)
        assert len(content.strip()) >= 200, f"文档内容过少：{relative_path}"
        print(f"通过：{relative_path}")

    readme = _read("README.md")
    for keyword in ("LangGraph", "Hybrid RAG", "Human Review", "快速开始"):
        assert keyword in readme, f"README 缺少核心说明：{keyword}"

    print("结果：通过。架构、部署、接口和测试说明完整。")


def test_04_test_layering() -> None:
    _title("R04｜测试分层")

    conftest = _read("tests/conftest.py")
    pyproject = _read("pyproject.toml")

    assert "test_agent_http_e2e.py" in conftest
    assert "test_s01_device_status_tool_call" in conftest
    assert "test_s02_maintenance_history_tool_call" in conftest
    assert "test_s03_greeting_final_answer" in conftest
    assert "pytest.mark.e2e" in conftest
    assert "not e2e" in pyproject

    superseded = [
        "tests/test_agent_runner.py",
        "tests/test_rag_node.py",
        "tests/test_risk_control.py",
        "tests/test_three_tool_agent.py",
        "tests/graph_demo.py",
    ]
    remaining = [
        path
        for path in superseded
        if (PROJECT_ROOT / path).exists()
    ]
    assert not remaining, f"仍包含已被正式链路替代的早期脚本：{remaining}"

    print("结果：通过。默认测试保持离线，真实 HTTP 测试必须显式执行。")


def test_05_git_exclusion_policy() -> None:
    _title("R05｜Git 排除策略")

    gitignore = _read(".gitignore")
    required_rules = [
        ".env",
        "logs/",
        "deployment/run/",
        "industrial-rag/data/uploads/",
        "industrial-rag/data/vector_store/",
        "*.safetensors",
        "*.pdf",
    ]

    missing = [rule for rule in required_rules if rule not in gitignore]
    assert not missing, f".gitignore 缺少规则：{missing}"
    # 根目录 .git 属于正式仓库自身，只禁止项目内部嵌套其他 Git 仓库。
    root_git = PROJECT_ROOT / ".git"
    nested_git_dirs = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob(".git")
        if path != root_git
    ]

    assert not nested_git_dirs, (
        f"发布仓库不应包含嵌套 Git 历史：{nested_git_dirs}"
    )

    print("结果：通过。配置、模型、文档原件和运行数据不会进入 Git。")


def test_06_agent_import_contracts() -> None:
    _title("R06｜测试导入契约")

    missing: list[str] = []

    for test_path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
        test_tree = ast.parse(
            test_path.read_text(encoding="utf-8"),
            filename=str(test_path),
        )

        for node in test_tree.body:
            if (
                not isinstance(node, ast.ImportFrom)
                or not node.module
                or not node.module.startswith("agent.")
            ):
                continue

            module_path = (
                PROJECT_ROOT
                / Path(*node.module.split("."))
            ).with_suffix(".py")

            if not module_path.is_file():
                missing.append(
                    f"{test_path.name}: 模块不存在 {node.module}"
                )
                continue

            module_tree = ast.parse(
                module_path.read_text(encoding="utf-8"),
                filename=str(module_path),
            )
            exported_names: set[str] = set()

            for item in module_tree.body:
                if isinstance(
                    item,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    exported_names.add(item.name)
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        item.targets
                        if isinstance(item, ast.Assign)
                        else [item.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            exported_names.add(target.id)
                elif isinstance(item, (ast.Import, ast.ImportFrom)):
                    for alias in item.names:
                        exported_names.add(
                            alias.asname or alias.name.split(".")[-1]
                        )

            for imported_name in node.names:
                if (
                    imported_name.name != "*"
                    and imported_name.name not in exported_names
                ):
                    missing.append(
                        f"{test_path.name}: "
                        f"{node.module}.{imported_name.name} 不存在"
                    )

    assert not missing, "发现过时测试导入：" + "; ".join(missing)
    print("结果：通过。测试引用的 Agent 模块与公开符号全部存在。")


def main() -> int:
    tests = [
        test_01_service_requirements,
        test_02_project_and_ci,
        test_03_repository_documents,
        test_04_test_layering,
        test_05_git_exclusion_policy,
        test_06_agent_import_contracts,
    ]

    print("第八关第二批：GitHub 仓库完整性验收")
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
