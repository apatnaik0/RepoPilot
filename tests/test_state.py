from swe_agent.state import AgentState, AgentStatus


def test_new_agent_state_has_running_defaults() -> None:
    state = AgentState(task="Fix the failing tests")

    assert state.task == "Fix the failing tests"
    assert state.messages == []
    assert state.step_count == 0
    assert state.status is AgentStatus.RUNNING
    assert state.final_answer is None


def test_agent_runs_have_independent_message_histories() -> None:
    first = AgentState(task="First task")
    second = AgentState(task="Second task")

    first.messages.append({"role": "user", "content": first.task})

    assert len(first.messages) == 1
    assert second.messages == []


def test_agent_state_can_record_progress_and_completion() -> None:
    state = AgentState(task="Fix the failing tests")

    state.step_count += 1
    state.status = AgentStatus.COMPLETED
    state.final_answer = "Fixed the calculation and all tests pass."

    assert state.step_count == 1
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Fixed the calculation and all tests pass."

