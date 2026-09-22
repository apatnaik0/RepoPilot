from __future__ import annotations

from pathlib import Path

import pytest

from swe_agent.tools import RepositoryTools, Tool


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def repository_tools(repository: Path) -> RepositoryTools:
    return RepositoryTools(repository)


def test_list_files_returns_sorted_repository_relative_paths(
    repository_tools: RepositoryTools,
) -> None:
    assert repository_tools.list_files() == "README.md\nsrc/calculator.py"


def test_list_files_can_start_from_subdirectory(
    repository_tools: RepositoryTools,
) -> None:
    assert repository_tools.list_files("src") == "src/calculator.py"


def test_read_file_returns_text(repository_tools: RepositoryTools) -> None:
    assert repository_tools.read_file("README.md") == "# Example\n"


def test_read_file_rejects_missing_file(repository_tools: RepositoryTools) -> None:
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        repository_tools.read_file("missing.py")


def test_read_file_rejects_directory(repository_tools: RepositoryTools) -> None:
    with pytest.raises(IsADirectoryError, match="Path is not a file"):
        repository_tools.read_file("src")


def test_list_files_rejects_file(repository_tools: RepositoryTools) -> None:
    with pytest.raises(NotADirectoryError, match="Path is not a directory"):
        repository_tools.list_files("README.md")


def test_parent_traversal_is_rejected(repository_tools: RepositoryTools) -> None:
    with pytest.raises(PermissionError, match="escapes the repository root"):
        repository_tools.read_file("../secret.txt")


def test_absolute_path_is_rejected(repository_tools: RepositoryTools) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        repository_tools.read_file("/etc/passwd")


def test_symlink_escape_is_rejected(
    repository: Path, repository_tools: RepositoryTools, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (repository / "outside-link.txt").symlink_to(outside)

    with pytest.raises(PermissionError, match="escapes the repository root"):
        repository_tools.read_file("outside-link.txt")


def test_registry_rejects_duplicate_names(repository_tools: RepositoryTools) -> None:
    registry = repository_tools.build_registry()
    duplicate = Tool(
        name="read_file",
        description="Duplicate",
        parameters={"type": "object"},
        function=lambda: "",
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(duplicate)


def test_registry_lookup_rejects_unknown_tool(
    repository_tools: RepositoryTools,
) -> None:
    registry = repository_tools.build_registry()

    with pytest.raises(KeyError, match="Unknown tool"):
        registry.get("delete_file")


def test_model_schemas_do_not_expose_python_callables(
    repository_tools: RepositoryTools,
) -> None:
    registry = repository_tools.build_registry()
    schemas = registry.model_schemas()

    assert [schema["name"] for schema in schemas] == [
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_tests",
        "finish",
    ]
    assert all("function" not in schema for schema in schemas)
