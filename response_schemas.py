"""Typed final responses and application-recorded tool evidence."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposedToolCall(ContractModel):
    """Schema for each individual tool invocation proposed by the LLM."""
    tool: str = Field(
        description="Exact name of the MCP tool to execute (e.g., 'list_systems')"
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of parameters to pass into the tool"
    )
    mutating: bool = Field(
        default=False,
        description="Set to True if this tool call modifies server or database state"
    )


class ExecutionPlan(ContractModel):
    """Top-level schema requested from the LLM by the Agent."""
    thought_process: str = Field(
        description="High-level reasoning explaining why these steps were chosen"
    )
    proposed_calls: list[ProposedToolCall] = Field(
        default_factory=list,
        description="Sequential list of proposed tool calls for validation"
    )

class AgentResponse(ContractModel):
    status: Literal[
        "completed",
        "blocked",
        "clarification_required",
        "failed",
    ]
    answer: str
    steps: list[str]
    total_count: int | None = Field(ge=0)
    citations: list[str]
    completed_actions: list[str] = Field(
        description="Names of mutating tools that actually succeeded"
    )
    missing_fields: list[str]


class ToolObservation(ContractModel):
    ok: bool
    tool: str
    data: Any = None
    error_code: str | None = None
    message: str | None = None


class TraceEvent(ContractModel):
    step: int = Field(ge=1)
    tool: str
    observation: ToolObservation
    mutating: bool = False


class ChatResponse(AgentResponse):
    # Populated by application code, not by the model.
    trace: list[TraceEvent] = Field(default_factory=list)
    execution_plan: ExecutionPlan | None = None
    plan_validation_error: str | None = None


def blocked_response(message: str) -> AgentResponse:
    return AgentResponse(
        status="blocked",
        answer=message,
        steps=[],
        total_count=None,
        citations=[],
        completed_actions=[],
        missing_fields=[],
    )