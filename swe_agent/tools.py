"""Tool definitions and repository-scoped file tools."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


ToolFunction = Callable[..., str]


@dataclass(frozen=True)
class Tool:
    """Connect a model-facing tool description to executable Python code."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    def model_schema(self) -> dict[str, Any]:
        """Return the serializable portion of the tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Store the tools that an agent is allowed to request."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def model_schemas(self) -> list[dict[str, Any]]:
        """Return model-facing schemas in deterministic registration order."""
        return [tool.model_schema() for tool in self._tools.values()]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


class RepositoryTools:
    """File tools confined to a single repository root."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        test_command: tuple[str, ...] | None = None,
        test_timeout_seconds: float = 30.0,
        search_output_limit: int = 20_000,
    ) -> None:
        root = Path(repository_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Repository root is not a directory: {root}")
        if test_timeout_seconds <= 0:
            raise ValueError("Test timeout must be greater than zero")
        if search_output_limit <= 0:
            raise ValueError("Search output limit must be greater than zero")

        self.repository_root = root
        self.test_command = test_command or (sys.executable, "-m", "pytest", "-q")
        self.test_timeout_seconds = test_timeout_seconds
        self.search_output_limit = search_output_limit

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a user-supplied path and reject repository escapes."""
        requested_path = Path(relative_path)
        if requested_path.is_absolute():
            raise ValueError("Path must be relative to the repository root")

        resolved_path = (self.repository_root / requested_path).resolve()
        if not resolved_path.is_relative_to(self.repository_root):
            raise PermissionError(
                f"Path escapes the repository root: {relative_path}"
            )
        return resolved_path

    def list_files(self, path: str = ".") -> str:
        """List all files under a repository-relative directory."""
        directory = self._resolve_path(path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        files = sorted(
            candidate.relative_to(self.repository_root).as_posix()
            for candidate in directory.rglob("*")
            if candidate.is_file()
        )
        return "\n".join(files)

    def read_file(self, path: str) -> str:
        """Read a UTF-8 file at a repository-relative path."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")
        return file_path.read_text(encoding="utf-8")

    def search_code(self, query: str, path: str = ".", max_results: int = 50) -> str:
        """Find literal text in repository files and return matching lines."""
        if not query:
            raise ValueError("Search query must not be empty")
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")

        search_root = self._resolve_path(path)
        if not search_root.exists():
            raise FileNotFoundError(f"Search path does not exist: {path}")

        candidates = [search_root] if search_root.is_file() else search_root.rglob("*")
        matches: list[str] = []
        output_length = 0

        for candidate in candidates:
            if not candidate.is_file():
                continue

            relative_path = candidate.relative_to(self.repository_root).as_posix()
            safe_candidate = self._resolve_path(relative_path)
            try:
                lines = safe_candidate.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if query not in line:
                    continue
                match = f"{relative_path}:{line_number}:{line}"
                if output_length + len(match) + 1 > self.search_output_limit:
                    matches.append("[search output truncated]")
                    return "\n".join(matches)
                matches.append(match)
                output_length += len(match) + 1
                if len(matches) >= max_results:
                    return "\n".join(matches)

        return "\n".join(matches)

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace exactly one occurrence of text in a repository file."""
        if not old_text:
            raise ValueError("old_text must not be empty")

        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")

        content = file_path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise ValueError(f"old_text was not found in: {path}")
        if occurrences > 1:
            raise ValueError(
                f"old_text occurs {occurrences} times in {path}; provide more context"
            )

        file_path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Updated {path}"

    def run_tests(self) -> str:
        """Run the configured test command without invoking a shell."""
        try:
            completed = subprocess.run(
                self.test_command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=self.test_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = self._coerce_subprocess_output(error.stdout)
            stderr = self._coerce_subprocess_output(error.stderr)
            return self._format_test_result(
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )

        return self._format_test_result(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )

    @staticmethod
    def _coerce_subprocess_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(encoding="utf-8", errors="replace")
        return output

    @staticmethod
    def _format_test_result(
        *, exit_code: int | None, stdout: str, stderr: str, timed_out: bool
    ) -> str:
        return "\n".join(
            [
                f"timed_out: {str(timed_out).lower()}",
                f"exit_code: {exit_code if exit_code is not None else 'none'}",
                "stdout:",
                stdout.rstrip(),
                "stderr:",
                stderr.rstrip(),
            ]
        )

    @staticmethod
    def finish(summary: str) -> str:
        """Return the model's final task summary for the agent loop."""
        if not summary.strip():
            raise ValueError("Finish summary must not be empty")
        return summary

    def build_registry(self) -> ToolRegistry:
        """Create the initial registry of repository tools."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="list_files",
                description=(
                    "List all files recursively beneath a directory inside the "
                    "repository. Paths are relative to the repository root."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Directory relative to the repository root. "
                                "Defaults to '.'."
                            ),
                        }
                    },
                    "additionalProperties": False,
                },
                function=self.list_files,
            )
        )
        registry.register(
            Tool(
                name="read_file",
                description="Read a UTF-8 text file inside the repository.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File relative to the repository root.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                function=self.read_file,
            )
        )
        registry.register(
            Tool(
                name="search_code",
                description=(
                    "Search for literal text in repository files. Returns matching "
                    "repository-relative paths, line numbers, and lines."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Case-sensitive literal text to find.",
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "File or directory relative to the repository root. "
                                "Defaults to '.'."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum matches to return. Defaults to 50.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                function=self.search_code,
            )
        )
        registry.register(
            Tool(
                name="edit_file",
                description=(
                    "Replace one exact, uniquely occurring text block in a UTF-8 "
                    "repository file. Include enough old_text context to make it unique."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File relative to the repository root.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Exact existing text to replace.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text, which may be empty.",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
                function=self.edit_file,
            )
        )
        registry.register(
            Tool(
                name="run_tests",
                description=(
                    "Run the application's preconfigured test command and return its "
                    "exit code, standard output, standard error, and timeout status."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                function=self.run_tests,
            )
        )
        registry.register(
            Tool(
                name="finish",
                description=(
                    "Finish the task with a concise summary after verifying the work."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Final summary of changes and verification.",
                        }
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                function=self.finish,
            )
        )
        return registry
