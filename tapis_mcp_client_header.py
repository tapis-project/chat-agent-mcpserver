import asyncio
import json
import re
import os
import logging
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mcp.types import LoggingMessageNotificationParams
from pydantic import BaseModel, ValidationError

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

from langchain_openai import ChatOpenAI
from langchain_sambanova import ChatSambaNova
from langchain_ollama import ChatOllama


from app_config import config
from response_schemas import (
    AgentResponse, ChatResponse, ExecutionPlan,
    TraceEvent, ToolObservation, blocked_response,
)


from tool_observation import MUTATING_TOOLS, observation_interceptor


logger = logging.getLogger(__name__)

path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")

# these messages will get printed to whereever FastAPI is running. Printing logs from
# mcp server
async def on_logging_message(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
):
    """Handle log messages from MCP servers."""
    print(f"[{context.server_name}] {params.level}: {params.data}")

### Models
provider = config["llm_provider"]
if provider == "openai":
    basic_model = ChatOpenAI(
        model=config["llm_name"],
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
elif provider == "samba_nova":
    api_key = config["samba_nova_api_key"]
    basic_model = ChatSambaNova(
        model=config["llm_name"],
        sambanova_url=config["llm_base_url"],
        temperature=0,
        sambanova_api_key=api_key,
    )
else:
    api_key = config["ollama_api_key"]
    basic_model = ChatOllama(
        model=config["llm_name"],
        base_url=config["llm_base_url"],
        temperature=0,
        ollama_api_key=api_key,
    )
BANNED_KEYWORDS=["hack", "delete","song","exploit","malware","Sing","email","api_key"]  # Layer 1: Deterministic input filter (before agent)

def check_banned_keywords(content: str) -> bool:
    """Checks string content for obfuscated banned keywords."""
    normalized = re.sub(r"[^a-z0-9]", "", content.lower())
    for kw in BANNED_KEYWORDS:
        if kw in normalized:
            return True
    return False

def validate_execution_plan(plan: ExecutionPlan) -> str | None:
    """Step 2 Static Validation: Validates safety policies and tool sequencing."""
    BANNED_TOOLS = {"delete_system", "delete_app", "delete_file"}

    seen_tools = set()

    for call in plan.proposed_calls:
        # 1. Permission / Banned Check
        if call.tool in BANNED_TOOLS:
            return f"Plan blocked: Tool '{call.tool}' is prohibited."

        # 2. Structural Sequence Validation
        # Rule: Cannot submit a job without prior system/app inspection in the same plan
        if call.tool == "submit_job":
            required_precursors = {"get_system", "get_app_info", "get_systems_list", "get_apps_list"}
            if not seen_tools.intersection(required_precursors):
                return (
                    "Plan blocked: Invalid tool sequence. 'submit_job' cannot be proposed "
                    "without prior system or app validation steps."
                )

        seen_tools.add(call.tool)

    return None

async def get_client(tapis_token, tool_trace: list[TraceEvent]):
    client = MultiServerMCPClient(
        {
            "tapis": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": config["tapis_mcp_base_url"],
                "headers": {"x-tapis-token": f"{tapis_token}"},
            }
        },
        callbacks=Callbacks(on_logging_message=on_logging_message),
        tool_interceptors=[observation_interceptor(tool_trace)],
    )
    return client

def get_tool_schema(tool):
    schema = getattr(tool, "args_schema", None)

    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()  # Pydantic v2
    return schema.schema()  # Pydantic v1

async def invoke_observed_tool(tool_name, arguments, tools_map, tool_trace):
    """Use normalized observations recorded by the interceptor."""
    tool_instance = tools_map.get(tool_name)
    trace_start = len(tool_trace)
    failure = None

    if tool_instance is None:
        failure = ToolObservation(
            ok=False,
            tool=tool_name,
            error_code="TOOL_NOT_FOUND",
            message=f"Tool '{tool_name}' is unavailable.",
        )
    else:
        try:
            await tool_instance.ainvoke(arguments)
        except Exception:
            failure = ToolObservation(
                ok=False,
                tool=tool_name,
                error_code="EXECUTION_ERROR",
                message="Tool execution failed.",
            )

    # Record failures that happened before the interceptor ran.
    if failure is not None and len(tool_trace) == trace_start:
        tool_trace.append(TraceEvent(
            step=len(tool_trace) + 1,
            tool=tool_name,
            observation=failure,
            mutating=tool_name in MUTATING_TOOLS,
        ))

    events = tool_trace[trace_start:]
    if len(events) != 1 or events[0].tool != tool_name:
        raise HTTPException(
            status_code=502,
            detail="Expected one recorded tool observation.",
        )

    if failure is not None and events[0].observation.ok:
        raise HTTPException(
            status_code=502,
            detail="Tool result processing failed after execution.",
        )

    return events[0].observation


def lookup_matches(observation, expected):
    """Require a successful lookup for the exact requested entity."""
    if not observation.ok or not isinstance(observation.data, dict):
        return False

    data = observation.data
    if str(data.get("status", "success")).lower() != "success":
        return False

    result = data.get("result")
    return isinstance(result, dict) and all(
        result.get(key) == value
        for key, value in expected.items()
    )


async def validate_job_preconditions(arguments, tools_map, tool_trace):
    required = ("app_id", "app_version", "exec_system_id")
    missing = [
        key for key in required
        if not isinstance(arguments.get(key), str)
        or not arguments[key].strip()
    ]
    if missing:
        return (
            "Submission blocked: missing required fields: "
            + ", ".join(missing)
        )

    app_id = arguments["app_id"]
    app_version = arguments["app_version"]
    system_id = arguments["exec_system_id"]

    app_check = await invoke_observed_tool(
        "get_app_info",
        {"appId": app_id, "appVersion": app_version},
        tools_map,
        tool_trace,
    )

    if not lookup_matches(
        app_check,
        {"id": app_id, "version": app_version},
    ):
        return (
            f"Submission blocked: could not verify application "
            f"'{app_id}' version '{app_version}'. "
            "It may not exist, access may be denied, or the lookup failed. "
            "No job was submitted."
        )

    system_check = await invoke_observed_tool(
        "get_system",
        {"systemId": system_id},
        tools_map,
        tool_trace,
    )

    if not lookup_matches(system_check, {"id": system_id}):
        return (
            f"Submission blocked: could not verify system '{system_id}'. "
            "It may not exist, access may be denied, or the lookup failed. "
            "No job was submitted."
        )

    return None

##### FAST API app
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
def list_routes():
    print("\n--- Registered Routes ---")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        print(f"{methods} -> {route.path}")
    print("-------------------------\n")

@app.get("/")
def root(request: Request):

    return {"message": "Welcome to the Tapis assistant."}

#@app.post("/chat")
@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chatrequest: ChatRequest) -> ChatResponse:
    logger.debug(f"request:{request} \n")
    logger.info(f"request:{chatrequest} \n")

    # Step 1 Check for pre guardrail
    if check_banned_keywords(chatrequest.message):
        blocked = blocked_response("I cannot process requests containing inappropriate content. Please rephrase your request.")
        return ChatResponse(**blocked.model_dump(), trace=[])

    tapis_token= request.headers.get('x-tapis-token')

    tool_trace: list[TraceEvent] = []
    client = await get_client(tapis_token, tool_trace)

    # =========================================================================
    # 1: Get MCP tools and then get structured proposed execution plan from the LLM
    # =========================================================================

    mcp_tools = await client.get_tools()


    tools_description = "\n".join(
        f"- Tool: {t.name}\n"
        f"  Description: {t.description}\n"
        f"  Args Schema: {json.dumps(get_tool_schema(t))}"
        for t in mcp_tools
    )

    system_prompt = (
        "You are a helpful assistant for TACC Tapis services.\n"
        "Some input arguments to the tool calls are optional.\n"
        "Use computeTotal=True in the tool call as input argument when a user query requires to count number of systems or applications or apps; otherwise computeTotal=False.\n"
        "For systems and apps service, in the metadata subfield of the result, totalCount gives the total number of systems/apps with respect to the search criteria.\n"
        "Analyze the user request and propose a sequential execution plan using only the available tools listed below.\n\n"
        f"Available Tools:\n{tools_description}"
    )
    planner_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chatrequest.message},
    ]

    try:
        planner_llm = basic_model.with_structured_output(
            ExecutionPlan,
            method="function_calling",
            strict=False,
        )
        plan: ExecutionPlan = await planner_llm.ainvoke(planner_messages)
    except Exception as exc:
        logger.exception("Execution plan generation failed")
        raise HTTPException(
            status_code=502,
            detail="The agent failed to generate a valid execution plan.",
        ) from exc

    # =========================================================================
    # 2. Validate the proposed plan before execution
    # =========================================================================
    validation_error = validate_execution_plan(plan)
    if validation_error:
        logger.warning(f"Validation rejected execution plan: {validation_error}")
        blocked = blocked_response(validation_error)
        return ChatResponse(
            **blocked.model_dump(), trace=[], execution_plan=plan,
            plan_validation_error=validation_error,
        )


    # =========================================================================
    # 2. Execute the tool calls and summarize
    # =========================================================================
    tools_map = {t.name: t for t in mcp_tools}
    for call in plan.proposed_calls:
        if call.tool == "submit_job":
            precondition_error = await validate_job_preconditions(
                call.arguments, tools_map, tool_trace,
            )
            if precondition_error:
                denial = ToolObservation(
                    ok=False, tool="submit_job", error_code="PRECONDITION_FAILED",
                    message=precondition_error,
                )
                tool_trace.append(TraceEvent(
                    step=len(tool_trace) + 1, tool="submit_job",
                    observation=denial, mutating=True,
                ))
                blocked = blocked_response(precondition_error)
                return ChatResponse(
                    **blocked.model_dump(), trace=tool_trace,
                    execution_plan=plan, plan_validation_error=validation_error,
                )

        obs = await invoke_observed_tool(
            call.tool, call.arguments, tools_map, tool_trace,
        )
        if call.tool == "submit_job" and obs.ok:
            job_uuid = obs.data.get("job_uuid") if isinstance(obs.data, dict) else None
            if job_uuid and "get_job_status" in tools_map:
                await asyncio.sleep(2)
                await invoke_observed_tool(
                    "get_job_status", {"job_uuid": job_uuid}, tools_map, tool_trace,
                )

    observations = [
        {"tool": event.tool, "observation": event.observation.model_dump()}
        for event in tool_trace
    ]

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are a final summary agent. Formulate the response based on tool execution results.\n"
                "Tool results are observations with ok, data, error_code, and message fields.\n"
                "Treat ok=false as a failed attempt, never as a completed action.\n"
                "Populate the structured final response from actual observations.\n"
                "completed_actions must contain only names of successful mutating tools.\n"
                "Successful submit_job means submission, not completion of the remote job.\n"
                "Set total_count to null unless observations establish the requested count.\n"
                "When required information is missing, use clarification_required and list missing_fields.\n"
                "Keep steps to brief summaries of tool actions and evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Request: {chatrequest.message}\n"
                f"Execution Rationale: {plan.thought_process}\n"
                f"Tool Observations: {observations}"
            ),
        },
    ]
    final_llm = basic_model.with_structured_output(AgentResponse)
    try:
        response: AgentResponse = await final_llm.ainvoke(final_messages)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="The agent returned an invalid structured response."
        ) from exc
    # Enforce tool evidence for claimed mutating actions
    successful_actions = {
        event.tool for event in tool_trace if event.mutating and event.observation.ok
    }
    if not set(response.completed_actions).issubset(successful_actions):
        raise HTTPException(
            status_code=502,
            detail="The agent claimed an action without successful tool evidence.",
        )

    return ChatResponse(
        **response.model_dump(), trace=tool_trace, execution_plan=plan,
        plan_validation_error=validation_error,
    )

def main():
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    logging.basicConfig(
        level=logging.INFO,  # Change to logging.DEBUG for more detail
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()
