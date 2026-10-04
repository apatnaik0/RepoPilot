from pathlib import Path

import pytest

import swe_agent.cli as cli
from swe_agent.executor import ToolCall
from swe_agent.model import ModelResponse, ScriptedModelClient


def test_cli_runs_agent_to_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted_model = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="finish-1",
                        name="finish",
                        arguments={"summary": "Task completed."},
                    ),
                )
            )
        ]
    )
    monkeypatch.setattr(cli, "OpenAIModelClient", lambda model: scripted_model)

    exit_code = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--task",
            "Inspect the repository",
            "--model",
            "test-model",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Status: completed\nFinal answer: Task completed.\n"
    )


def test_cli_returns_failure_when_step_limit_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted_model = ScriptedModelClient([ModelResponse(content="Still working.")])
    monkeypatch.setattr(cli, "OpenAIModelClient", lambda model: scripted_model)

    exit_code = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--task",
            "Keep investigating",
            "--model",
            "test-model",
            "--max-steps",
            "1",
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().out == "Status: max_steps\n"


def test_cli_rejects_invalid_repository(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--repo",
                str(tmp_path / "missing"),
                "--task",
                "Anything",
                "--model",
                "test-model",
            ]
        )

    assert error.value.code == 2


def test_cli_rejects_nonpositive_max_steps(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--repo",
                str(tmp_path),
                "--task",
                "Anything",
                "--model",
                "test-model",
                "--max-steps",
                "0",
            ]
        )

    assert error.value.code == 2

