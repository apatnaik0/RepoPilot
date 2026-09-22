from __future__ import annotations

from pathlib import Path
import sys

import pytest

from swe_agent.executor import ToolCall, ToolExecutor
from swe_agent.tools import RepositoryTools


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calculator.py").write_text(
        "def add(left, right):\n"
        "    return left + right\n\n"
        "def subtract(left, right):\n"
        "    return left - right\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("calculator example\n", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"\xff\xfe\x00")
    return tmp_path


def test_search_code_returns_paths_line_numbers_and_lines(repository: Path) -> None:
    tools = RepositoryTools(repository)

    result = tools.search_code("return", path="src")

    assert result == (
        "src/calculator.py:2:    return left + right\n"
        "src/calculator.py:5:    return left - right"
    )


def test_search_code_can_search_one_file(repository: Path) -> None:
    tools = RepositoryTools(repository)

    assert tools.search_code("calculator", path="README.md") == (
        "README.md:1:calculator example"
    )


def test_search_code_returns_empty_string_when_no_match(repository: Path) -> None:
    assert RepositoryTools(repository).search_code("multiply") == ""


def test_search_code_skips_non_utf8_files(repository: Path) -> None:
    assert RepositoryTools(repository).search_code("calculator") == (
        "README.md:1:calculator example"
    )


def test_search_code_enforces_max_results(repository: Path) -> None:
    result = RepositoryTools(repository).search_code("return", max_results=1)

    assert result == "src/calculator.py:2:    return left + right"


def test_search_code_truncates_long_output(repository: Path) -> None:
    tools = RepositoryTools(repository, search_output_limit=45)

    result = tools.search_code("return")

    assert result.endswith("[search output truncated]")


def test_search_code_rejects_empty_query(repository: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RepositoryTools(repository).search_code("")


def test_search_code_rejects_repository_escape(repository: Path) -> None:
    with pytest.raises(PermissionError, match="escapes the repository root"):
        RepositoryTools(repository).search_code("secret", path="..")


def test_edit_file_replaces_one_exact_occurrence(repository: Path) -> None:
    tools = RepositoryTools(repository)

    result = tools.edit_file(
        "src/calculator.py",
        "return left + right",
        "return left + right + 1",
    )

    assert result == "Updated src/calculator.py"
    assert "return left + right + 1" in (repository / "src/calculator.py").read_text()


def test_edit_file_rejects_missing_text(repository: Path) -> None:
    with pytest.raises(ValueError, match="old_text was not found"):
        RepositoryTools(repository).edit_file(
            "src/calculator.py", "return left * right", "return 0"
        )


def test_edit_file_rejects_ambiguous_text(repository: Path) -> None:
    with pytest.raises(ValueError, match="occurs 4 times"):
        RepositoryTools(repository).edit_file(
            "src/calculator.py", "left", "first"
        )


def test_edit_file_rejects_repository_escape(repository: Path) -> None:
    with pytest.raises(PermissionError, match="escapes the repository root"):
        RepositoryTools(repository).edit_file("../outside.py", "old", "new")


def test_run_tests_captures_success(repository: Path) -> None:
    tools = RepositoryTools(
        repository,
        test_command=(sys.executable, "-c", "print('all tests passed')"),
    )

    result = tools.run_tests()

    assert "timed_out: false" in result
    assert "exit_code: 0" in result
    assert "all tests passed" in result


def test_run_tests_captures_failure(repository: Path) -> None:
    tools = RepositoryTools(
        repository,
        test_command=(
            sys.executable,
            "-c",
            "import sys; print('failed', file=sys.stderr); sys.exit(3)",
        ),
    )

    result = tools.run_tests()

    assert "timed_out: false" in result
    assert "exit_code: 3" in result
    assert "failed" in result


def test_run_tests_captures_timeout(repository: Path) -> None:
    tools = RepositoryTools(
        repository,
        test_command=(sys.executable, "-c", "import time; time.sleep(1)"),
        test_timeout_seconds=0.01,
    )

    result = tools.run_tests()

    assert "timed_out: true" in result
    assert "exit_code: none" in result


def test_run_tests_does_not_accept_model_supplied_command(repository: Path) -> None:
    executor = ToolExecutor(RepositoryTools(repository).build_registry())

    result = executor.execute(
        ToolCall(
            id="unsafe-command",
            name="run_tests",
            arguments={"command": "rm -rf important-files"},
        )
    )

    assert result.success is False
    assert result.error == "ValueError: Unexpected argument(s): command"


def test_finish_returns_nonempty_summary(repository: Path) -> None:
    assert RepositoryTools(repository).finish("Fixed the bug; tests pass.") == (
        "Fixed the bug; tests pass."
    )


def test_finish_rejects_blank_summary(repository: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RepositoryTools(repository).finish("   ")


def test_registry_contains_complete_v0_tool_set(repository: Path) -> None:
    registry = RepositoryTools(repository).build_registry()

    assert [tool.name for tool in registry] == [
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_tests",
        "finish",
    ]
