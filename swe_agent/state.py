"""Mutable state for one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    FAILED = "failed"


@dataclass
class AgentState:
    task: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None

