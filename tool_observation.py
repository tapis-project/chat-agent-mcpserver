"""Normalize MCP results and retain evidence for each request."""
import json
import logging
from mcp.types import CallToolResult, TextContent
from response_schemas import ToolObservation, TraceEvent

logger = logging.getLogger(__name__)
MUTATING_TOOLS = {"submit_job"}


def normalize_observation(tool: str, result: CallToolResult) -> ToolObservation:
    data = result.structuredContent
    if data is None:
        text = "\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = text
    error_code = "mcp_error" if result.isError else None
    if isinstance(data, dict) and "ok" in data:
        observation = ToolObservation.model_validate(data)
        if observation.tool != tool:
            raise ValueError("Observation tool does not match requested tool")
        if result.isError:
            return observation.model_copy(update={"ok": False, "error_code": "mcp_error"})
        if tool == "submit_job" and observation.ok:
            if not isinstance(observation.data, dict) or not observation.data.get("job_uuid"):
                return observation.model_copy(update={"ok": False, "error_code": "missing_job_uuid"})
        return observation
    if isinstance(data, dict):
        status = str(data.get("status", "")).lower()
        if status in {"error", "failed", "failure", "internal error", "bad gateway", "bad request"}:
            error_code = "tapis_error"
        elif status == "with version, appid is required for search":
            error_code = "invalid_arguments"
    if isinstance(data, str) and tool == "read_tapis_file" and data.startswith("Error reading file content:"):
        error_code = "tapis_error"
    # A mutation needs explicit success evidence, not just successful transport.
    if tool in MUTATING_TOOLS and error_code is None:
        error_code = "unverified_action"
    return ToolObservation(
        ok=error_code is None, tool=tool, data=data, error_code=error_code,
        message="Tool execution did not establish success" if error_code else None,
    )


def observation_interceptor(trace: list[TraceEvent]):
    async def intercept(request, handler):
        try:
            result = await handler(request)
        except Exception:
            logger.warning("MCP tool %s raised an exception", request.name)
            observation = ToolObservation(
                ok=False, tool=request.name, error_code="tool_exception",
                message="Tool execution failed",
            )
        else:
            try:
                observation = normalize_observation(request.name, result)
            except (ValueError, AttributeError, TypeError):
                observation = ToolObservation(
                    ok=False, tool=request.name, error_code="invalid_observation",
                    message="The tool returned an invalid observation",
                )
        trace.append(TraceEvent(
            step=len(trace) + 1, tool=request.name, observation=observation,
            mutating=request.name in MUTATING_TOOLS,
        ))
        # Keep failed observations available to the agent instead of aborting its loop.
        return CallToolResult(
            content=[TextContent(type="text", text=observation.model_dump_json())],
            structuredContent=observation.model_dump(mode="json"), isError=False,
        )
    return intercept
