from app_config import config
from langchain_openai import ChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud
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

import logging
logger = logging.getLogger(__name__)

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
        llm = ChatSambaNovaCloud(
            model=config["llm_name"],
            sambanova_url=config["llm_base_url"],
            temperature=0,
            sambanova_api_key=api_key,
        )
else:
        api_key = config["ollama_api_key"]
        llm = ChatOllama(
            model=config["llm_name"],
            base_url=config["llm_base_url"],
            temperature=0,
            ollama_api_key= api_key,
        )    
basic_model = ChatOpenAI(model="gpt-4o-mini")

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
    " If the user query require specific information from the result json, extract the field and the corresponding field."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


async def get_client(tapis_token):
    logger.debug("---token ---")
    logger.debug(print(tapis_token))
    client = MultiServerMCPClient(
        {
            "tapis": {
            "transport": "streamable_http",  # HTTP-based remote server
            # Ensure you start your tapis server on port 8000
            #"url": "http://127.0.0.1:8000/mcp",
            "url": config["tapis_mcp_base_url"], #"http://host.docker.internal:8000/mcp",
            "headers":{"x-tapis-token":f"{tapis_token}"}
            #"url":"http://0.0.0.0:8000/mcp"
        }
    },
    callbacks=Callbacks(on_logging_message=on_logging_message),
    #tool_interceptors=[logging_interceptor]
   )
    return client
    

def printAImessage(result):
    for msg in result["messages"]:
        if isinstance(msg,AIMessage):
            if msg.content=="":
                print(msg.tool_calls)
            else:
                print(msg.content)
                return msg.content
            print("\n")    
                 

#async def main():
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
    middleware=[user_role_prompt], 
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
    #return {"message": my_header}
    return {"message": "Welcome to your first agent and header: "+ str(tapis_header)}

@app.post("/chat")
async def chat(request: Request, chatrequest: ChatRequest):
    logger.debug(f"request:{request} \n")
    logger.info(f"request:{chatrequest} \n")
    tapis_token= request.headers.get('x-tapis-token')
    logger.debug(f"Tapis token: {tapis_token}")
    client = await get_client(tapis_token)
    agent = await create_tapis_agent(client);
    result = await agent.ainvoke({"messages":[{"role":"user","content":chatrequest.message}]})
    return {"reply":printAImessage(result) }

def main():
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    main()