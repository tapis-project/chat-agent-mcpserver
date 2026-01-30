from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, wrap_tool_call,ModelRequest, ModelResponse,dynamic_prompt
from typing import TypedDict
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient  
import asyncio
import json
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.types import LoggingMessageNotificationParams
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from tapipy.tapis import Tapis, TapisResult
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
# async def logging_interceptor(
#     request: MCPToolCallRequest,
#     handler,
# ):
#     """Log tool calls before and after execution."""
#     print(f"Calling tool: {request.name} with args: {request.args}")
#     result = await handler(request)
#     print(f"Tool {request.name} returned: {result}")
#     return result

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")



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

# def get_token_for_tool(name):
#     print("getting token for tool : " + name)
#     t = Tapis(base_url=os.getenv("TAPIS_BASE_URL"), username=os.getenv("USERNAME"), password=os.getenv("PASSWORD"))
#     t.get_tokens()
#     return t.access_token.access_token

# async def auth_header_interceptor(
#     request: MCPToolCallRequest,
#     handler,
# ):
#     """Add authentication headers based on the tool being called."""
#     token = get_token_for_tool(request.name)
#     modified_request = request.override(
#         headers={"x-tapis-token": f"{token}"}  
#     )
#     return await handler(modified_request)

async def get_client(tapis_token):
    logger.info("---token ---")
    logger.info(print(tapis_token))
    client = MultiServerMCPClient(
        {
            "tapis": {
            "transport": "streamable_http",  # HTTP-based remote server
            # Ensure you start your tapis server on port 8000
            #"url": "http://127.0.0.1:8000/mcp",
            "url": "http://host.docker.internal:8000/mcp",
            "headers":{"x-tapis-token":f"{tapis_token}"}
            #"url":"http://0.0.0.0:8000/mcp"
        }
    },
    callbacks=Callbacks(on_logging_message=on_logging_message),
    tool_interceptors=[logging_interceptor]
   )
    return client
    

def printAImessage(result):
    for msg in result["messages"]:
        # if isinstance(msg, ToolMessage):
        #     #print(msg.content)
        #     print("\n ----")
        #     print(json.loads(msg.content[0]['text']))
        #     print("\n\n")
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
    middleware=[user_role_prompt], #dynamic_model_selection, handle_tool_errors
    #system_prompt="You are a helpful assistant. Be concise and accurate."
    context_schema = Context
    )
    return agent
    
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
    logger.info(f"request:{request} \n")
    logger.info(f"request:{chatrequest} \n")
    tapis_token= request.headers.get('x-tapis-token')
    logger.info(f"Tapis token: {tapis_token}")
    client = await get_client(tapis_token)
    agent = await create_tapis_agent(client);
    result = await agent.ainvoke({"messages":[{"role":"user","content":chatrequest.message}]})
    return {"reply":printAImessage(result) }

def main():
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

# if __name__ == "__main__":
#     #logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
#     asyncio.run(main())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    main()