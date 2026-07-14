from app_config import config
from langchain_openai import ChatOpenAI
from langchain_sambanova import ChatSambaNova
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool
from langchain.agents.middleware import ModelRequest, dynamic_prompt #wrap_model_call, wrap_tool_call, ModelResponse
from typing import TypedDict
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient  
import asyncio
import json
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.types import LoggingMessageNotificationParams
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
#from tapipy.tapis import Tapis, TapisResult
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
#from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
import logging
logger = logging.getLogger(__name__)

from typing import Any
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-5.1")

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the final AI response
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a model to evaluate safety
        safety_prompt = f"""Evaluate if this response is related to TACC Tapis services and appropriate for response. Any other response should considered unsafe. 
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. It is either not relevant to Tapis or not appropriate. Please rephrase your request."

        return None

#### Change the project_name in phoenix for benchmark dataset
#### project name in phoenix
PROJECT_NAME="llama4-eval"   #"user-specific-qa-eval"    #"tapis-job-submit-eval" #"tapis-eval"
client_tracer_provider= register(
    project_name=PROJECT_NAME,
    auto_instrument=True,
    batch=True,
    set_global_tracer_provider=False)

# Get a tracer to add additional instrumentattion
tracer = client_tracer_provider.get_tracer(__name__)

LangChainInstrumentor(tracer_provider=client_tracer_provider).instrument(skip_dep_check=True)

path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")



class Context(TypedDict):
    user_role: str

async def on_logging_message(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
):
    """Handle log messages from MCP servers."""
    print(f"[{context.server_name}] {params.level}: {params.data}")

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Logs the tool call request details."""
    logger.info(f"[Tool Call Request] Tool: {request.name}, Args: {request.args}")
    # Proceed with the actual tool call
    response = await handler(request)
    logger.info(f"[Tool Call Response] Tool: {request.name}, Response: {response}")
    print("\n --------")
    return response

### Models
provider = config["llm_provider"]
if  provider == "openai":
    basic_model = ChatOpenAI(
            model=config["llm_name"],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
elif provider == "samba_nova":
        api_key = config["samba_nova_api_key"]
        print(f"Using SambaNova with api key {api_key[:5]}...{api_key[-5:]}")
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
            ollama_api_key= api_key,
        )    
#basic_model = ChatOpenAI(model="gpt-4o-mini")

#advanced_model = ChatOpenAI(model="gpt-4o")



@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    if request.runtime.context is not None:
        user_role = request.runtime.context.get("user_role", "user")
    else:
    	user_role = "user" # Default value if context is missing
    # Log a warning or handle the error as appropriate
    #print("Warning: request.runtime.context is None. Using default user role.")

    #user_role = request.runtime.context.get("user_role", "user")
    #base_prompt = "You are a helpful assistant. Use computeTotal=True when required to count number of objects in the call tool results. "
    base_prompt = "" \
    " You are a helpful assistant. " \
    " Some input arguments to the tool calls are optional. "\
    " Use computeTotal=True in the tool call as input argument when a user query requires to count number of systems or applications or apps.otherwise computeTotal= False" \
    " For systems and apps service, in the metadata subfield of the result, totalCount gives the total number of systems/apps with respect to the search criteria." \
    " For security service, the count on the records can be found in the message field of the result. " \
    " In summary, give tool call name, arguments, and give steps taken to reach that total count and which field of datastructure. " \
    " If the user query require specific information from the result json, extract the field and the corresponding field value."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


guard_middleware= ContentFilterMiddleware(banned_keywords=["hack", "exploit","malware","Sing","email","api_key"])  # Layer 1: Deterministic input filter (before agent)





async def get_client(tapis_token):
    logger.debug("---token ---")
    logger.debug(print(tapis_token))
    client = MultiServerMCPClient(
        {
            "tapis": {
            "transport": "streamable_http",  # HTTP-based remote server
            # Ensure you start your tapis server on port 8000
            "url": config["tapis_mcp_base_url"],
            "headers":{"x-tapis-token":f"{tapis_token}"}
            }
    },
    callbacks=Callbacks(on_logging_message=on_logging_message),
    tool_interceptors=[logging_interceptor]
   )
    return client
    

def printAImessage(result):
    for msg in result["messages"]:
        if isinstance(msg,AIMessage):
            if msg.content=="":
                logger.info("printAImessage: tool_call msg:" + str(msg.tool_calls))
            else:
                print(msg.content)
                return msg.content
            print("\n")

def returnAImessage(result):
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if msg.content == "":
                logger.info("printAImessage: tool_call msg:" + str(msg.tool_calls))
            else:
                #print(msg.content)
                return msg.content
            print("\n")

                #async def main():
#@tracer.agent
async def create_tapis_agent(client):
    # get the tools from MCP server
    tools = await client.get_tools()
    for tool in tools:
        if isinstance(tool, StructuredTool):
            print(tool.name +"\n")
            print(tool.args_schema)
            print("\n")
       

    # create an agent
    agent = create_agent(
    model= basic_model,  # Default model
    tools = tools,
    middleware=[user_role_prompt,guard_middleware],#SafetyGuardrailMiddleware()],
    context_schema = Context
    )
    return agent

##### FAST API app    
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root(request: Request):
    
    tapis_header = request.headers.get('x-tapis-token')
    return {"message": "Welcome to your first agent and header: "+ str(tapis_header)}



@app.post("/chat")
#@tracer.agent
async def chat(request: Request, chatrequest: ChatRequest):
    logger.debug(f"request:{request} \n")
    logger.info(f"request:{chatrequest} \n")
    tapis_token= request.headers.get('x-tapis-token')
    logger.debug(f"Tapis token: {tapis_token}")
    messages=[{"role": "user", "content": chatrequest.message}]
    client = await get_client(tapis_token)

    agent = await create_tapis_agent(client);
    #with tracer.start_as_current_span(
    #        "agent-span3",
    #        openinference_span_kind="agent",
    #) as span:
    #    span.set_input(messages)
    #    try:
    result = await agent.ainvoke({"messages":messages })
    #    except Exception as error:
    #        span.record_exception(error)
    #        span.set_status(Status(StatusCode.ERROR))
    #    else:
    #        span.set_output(returnAImessage(result))
    #       span.set_status(Status(StatusCode.OK))


    return {"answer":printAImessage(result) }

def main():
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    main()