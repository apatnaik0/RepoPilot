from pathlib import Path

import pytest

from swe_agent.agent import Agent
from swe_agent.executor import ToolCall
from swe_agent.model import ModelResponse, ScriptedModelClient
from swe_agent.state import AgentState, AgentStatus
from swe_agent.tools import RepositoryTools


def make_agent(repository: Path, *responses: ModelResponse) -> Agent:
    registry = RepositoryTools(repository).build_registry()
    return Agent(ScriptedModelClient(responses), registry)


def test_run_step_records_task_model_response_and_tool_result(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# RepoPilot\n", encoding="utf-8")
    response = ModelResponse(
        content="I will inspect the README.",
        tool_calls=(
            ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}),
        ),
    )
    state = AgentState(task="Understand this repository")

    make_agent(tmp_path, response).run_step(state)

    assert state.step_count == 1
    assert [message["role"] for message in state.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert state.messages[0]["content"] == "Understand this repository"
    assert state.messages[2]["tool_call_id"] == "call-1"
    assert state.messages[2]["content"] == "# RepoPilot\n"
    assert state.messages[2]["success"] is True


def test_run_step_records_tool_failure_as_observation(tmp_path: Path) -> None:
    response = ModelResponse(
        tool_calls=(
            ToolCall(id="call-2", name="read_file", arguments={"path": "missing.py"}),
        )
    )
    state = AgentState(task="Read the missing file")

    make_agent(tmp_path, response).run_step(state)

    tool_message = state.messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["success"] is False
    assert tool_message["content"] == (
        "FileNotFoundError: File does not exist: missing.py"
    )


def test_successful_finish_completes_the_agent(tmp_path: Path) -> None:
    response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-3",
                name="finish",
                arguments={"summary": "The bug is fixed and tests pass."},
            ),
        )
    )
    state = AgentState(task="Fix the bug")

    make_agent(tmp_path, response).run_step(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "The bug is fixed and tests pass."


def test_run_step_rejects_completed_agent(tmp_path: Path) -> None:
    state = AgentState(task="Finished task", status=AgentStatus.COMPLETED)
    agent = make_agent(tmp_path, ModelResponse(content="Unused"))

    with pytest.raises(RuntimeError, match="Cannot step an agent with status: completed"):
        agent.run_step(state)


def test_run_repeats_steps_until_finish(tmp_path: Path) -> None:
    inspect_response = ModelResponse(
        tool_calls=(ToolCall(id="call-4", name="list_files", arguments={}),)
    )
    finish_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-5",
                name="finish",
                arguments={"summary": "Repository inspected."},
            ),
        )
    )
    state = AgentState(task="Inspect the repository")

    returned_state = make_agent(tmp_path, inspect_response, finish_response).run(state)

    assert returned_state is state
    assert state.step_count == 2
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Repository inspected."


def test_run_stops_at_maximum_steps(tmp_path: Path) -> None:
    state = AgentState(task="Keep investigating")
    agent = make_agent(
        tmp_path,
        ModelResponse(content="Still investigating."),
        ModelResponse(content="Still investigating."),
    )

    agent.run(state, max_steps=2)

    assert state.step_count == 2
    assert state.status is AgentStatus.MAX_STEPS
    assert state.final_answer is None


def test_run_rejects_nonpositive_maximum(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)

    with pytest.raises(ValueError, match="max_steps must be greater than zero"):
        agent.run(AgentState(task="Anything"), max_steps=0)
