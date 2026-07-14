import os
from fastmcp import FastMCP
from tapipy.tapis import Tapis, TapisResult
import json
from langchain_core.tools import StructuredTool
import requests
from typing import Optional, Any
from dotenv import load_dotenv
from fastmcp.server.dependencies import get_http_headers
import logging
logger = logging.getLogger(__name__)
import time
from phoenix.otel import register

path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")

TAPIS_BASE_URL = os.getenv("TAPIS_BASE_URL","https://public.tapis.io")
RAG_BASE_URL=os.getenv("RAG_BASE_URL")

PROJECT_NAME="llama4-eval" #"user-specific-qa-eval" #"tapis-job-submit-eval"
tracer_provider = register(
  project_name=PROJECT_NAME,
  auto_instrument=True,
  batch=True,
 set_global_tracer_provider=False
)
# Get a tracer to add additional instrumentattion
tracer = tracer_provider.get_tracer(__name__)


# Initialize tapipy object
def get_tapis_tapipy_object(tapis_base_url:str, access_token:str) -> Any:
    t = Tapis(base_url=tapis_base_url,
              access_token=access_token)
    return t

mcp = FastMCP(
    name="TapisMCP",    
    log_level="ERROR"
)

@mcp.tool
@tracer.tool(name="MCP.get_tapis_help")
async def get_tapis_help(question: str) -> Any:
    # """
    # This tool uses Tapis documentation to answer questions on Tapis v3 Service and Frameworks, SDKs and its usage
    # Args:
    # question: question/message from the user

    # """

    url = RAG_BASE_URL
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "")
    logger.info(auth_header)
    jsonobj = {"question": question}
    tapis_headers = {'Content-type': 'application/json', 'x-tapis-token': auth_header}
    response = requests.post(url, json=jsonobj, headers=tapis_headers)
    logger.info(response.text)
    logger.info(response.status_code)
    if response.status_code == 500:
        return {"status": "Internal Error"}
    elif response.status_code == 502:
        return {"status": "Bad Gateway"}
    elif response.status_code == 400:
        return {"status": "Bad Request"}
    else:
        return json.loads(response.text)

@mcp.tool
@tracer.tool(name="MCP.get_system")
async def get_system(systemId:str) -> TapisResult:   
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "") 
    logger.debug("auth_header: ---")
    logger.debug(auth_header)
    tapis_headers = {'x-tapis-token':auth_header}
    #t = Tapis(base_url=tapis_base_url, username=tapis_user_name, password=tapis_password)
    #t.get_tokens()
    #t.access_token
    url = TAPIS_BASE_URL + "/v3/systems/" + systemId
    response = requests.get(url, headers=tapis_headers)
    return (response.json())


     
@mcp.tool
@tracer.tool(name="MCP.get_systems_list")
async def get_systems_list(listType: str = "OWNED", 
                           computeTotal: bool = False, 
                           systemType: str = None,
                           defaultAuthnMethod : str = None,
                           canExec : bool = None) -> TapisResult:  
    """
    Get list of systems a user owns. 
    Args:
        listType: It is three options - OWNED (Include only items owned by requester), this is default, 
        SHARED_PUBLIC (Include only items shared publicly), ALL (Include all items requester is authorized to view)
        computeTotal: Compute total number of systems, default is False
        systemType: System type, default is None
        defaultAuthnMethod: Default authentication method to authenticate to the system, default is None
        canExec: Indicates if system can be used to execute jobs, default is None
    """  
    url = TAPIS_BASE_URL +'/v3/systems/search'
    first_query_param = 0
    if listType != "OWNED":
        url = url + "?listType=" + listType
        first_query_param = 1
    if computeTotal != False:
        if first_query_param == 0:
                url = url + "?computeTotal=" + str(computeTotal)
                first_query_param = 1
        else:
                url = url + "&computeTotal=" + str(computeTotal)   
    if systemType is not None:
        if first_query_param == 0:
                url = url + "?systemType.eq=" + systemType
                first_query_param = 1
        else:
                url = url + "&systemType.eq=" + systemType    
    if  defaultAuthnMethod is not None:
        if first_query_param == 0:
                url = url + "?defaultAuthnMethod.eq=" + defaultAuthnMethod
                first_query_param = 1
        else:
                url = url + "&defaultAuthnMethod.eq=" + defaultAuthnMethod
    if canExec is not None:
        if first_query_param == 0:
                url = url + "?canExec.eq=" + str(canExec)
                first_query_param = 1
        else:
                url = url + "&canExec.eq=" + str(canExec)

    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "") 
    logger.info(auth_header)
    tapis_headers = {'x-tapis-token':auth_header}
    response = requests.get(url, headers=tapis_headers)
    print("\n")
    if response.status_code == 500:
        return {"status":"error"}
    else:
        if response.status_code == 400:
            return {"status":"bad request"}
    
    #return (json.loads(response.text))
    #logger.info(response.text)
    return (response.json())

@mcp.tool
@tracer.tool(name="MCP.get_app_info")
async def get_app_info(appId: str, appVersion:str) -> TapisResult:
    """
     Get application(apps) information or details given appId and appVersion
     Args: appId: application id
           appVersion: application version
    """
    url = TAPIS_BASE_URL +'/v3/apps/'+ appId + '/' + appVersion
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "") 
    logger.info(auth_header)
    tapis_headers = {'x-tapis-token':auth_header}
    response = requests.get(url, headers=tapis_headers)
    print("\n")
    if response.status_code == 500:
        return {"status":"error"}
    else:
        if response.status_code == 400:
            return {"status":"bad request"}
    
    #return (json.loads(response.text))
    return (response.json())                  
    
     
@mcp.tool
@tracer.tool(name="MCP.get_apps_list")
async def get_apps_list(listType: str = "OWNED",
                        computeTotal: bool = False,
                        appId: str = None,
                        appVersion: str = None,
                        ) -> TapisResult:
    
    """
     Get list of applications details based on some search query.
     Args:
        listType: Options for listType are - OWNED (Include only items owned by requester), (Default), SHARED_PUBLIC (Include only items shared publicly),
                SHARED_DIRECT (Include only items shared directly with requester Exclude publicly shared items.),
                READ_PERM (Include only items for which requester was granter READ or MODIFY permission),
                MINE (Include items owned or shared directly with requester. Exclude publicly shared items.),
                ALL (Include all items requester is authorized to view. Includes check for READ or MODIFY permission.)
        computeTotal: Compute total number of applications, default is False,
        appId: application id, default is None,
        appVersion: application version, This also needs an appId specified. default is None. 
    
    """
    url = TAPIS_BASE_URL +'/v3/apps/search'
    first_query_param = 0
    if listType != "OWNED":
        url = url + "?listType=" + listType
        first_query_param = 1
    if computeTotal != False:
        if first_query_param == 0:
                url = url + "?computeTotal=" + str(computeTotal)
                first_query_param = 1
        else:
                url = url + "&computeTotal=" + str(computeTotal)  
    
    if appId is not None:
        if first_query_param == 0:
                url = url + "?id.eq=" + appId + "&select=allAttributes"
                first_query_param = 1
        else:
                url = url + "&id.eq=" + appId + "&select=allAttributes"    
    if appId is not None and appVersion is not None:
        url = url + "&version.eq=" + appVersion + "&select=allAttributes"
    elif appId is None and appVersion is not None:
        logger.info("app Id is needed for version query")
        return {"status": "With version, appID is required for search"}
    else:
          pass
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "") 
    logger.info(auth_header)
    tapis_headers = {'x-tapis-token':auth_header}
    response = requests.get(url, headers=tapis_headers)
    print("\n")
    if response.status_code == 500:
        return {"status":"error"}
    else:
        if response.status_code == 400:
            return {"status":"bad request"}
    
    #return (json.loads(response.text))
    return (response.json())                            

# @mcp.tool()
# def get_system_status(system_id: str):
#     """Check if a specific Tapis system is enabled."""
#     headers = get_http_headers()
#     auth_header = headers.get("x-tapis-token", "")
#     t = get_tapis_tapipy_object(TAPIS_BASE_URL,auth_header)
#     sys_info = t.systems.getSystem(systemId=system_id)
#     return f"System {system_id} is {'Enabled' if sys_info.enabled else 'Disabled'}"

@mcp.tool()
@tracer.tool(name="MCP.list_files")
def list_files(system_id: str, path: str):
    """
    Step 1/2 in Workflow: List files in a specific Tapis system and path.
    Useful for verifying data existence before running a job.
    """
    headers = get_http_headers()
    auth_header = headers.get("x-tapis-token", "")
    t = get_tapis_tapipy_object(TAPIS_BASE_URL,auth_header)
    files = t.files.listFiles(systemId=system_id, path=path)
    return [f.name for f in files]


@mcp.tool()
@tracer.tool(name="MCP.submit_job")
def submit_job(
        name: str,
        description: str,
        app_id: str,
        app_version: str,
        exec_system_id: str,
        # Use | None instead of Optional, and lowercase list/dict
        file_inputs: list[dict[str, str]] | None = None,
        app_args: list[dict[str, Any]] | None = None,

):
    """
    Step 4 in Workflow: Submits a job to Tapis.
    Requires IDs gathered from list_systems and search_apps.
    """
    parameterSet ={}
    if app_args is not None:
        parameterSet["appArgs"] = app_args
    job_request = {
        "name": name,
        "description": description,
        "appId": app_id,
        "appVersion": app_version,
        "execSystemId": exec_system_id,
        "fileInputs": file_inputs if file_inputs else [],
        "parameterSet": parameterSet
    }
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "")
    logger.info(auth_header)
    t = get_tapis_tapipy_object(TAPIS_BASE_URL, auth_header)
    # Validation point: This is where CodeBERT would inspect 'job_request'
    res = t.jobs.submitJob(**job_request)
    return f"Job submitted. Status: {res.status}. UUID: {res.uuid}"


@mcp.tool
@tracer.tool(name="MCP.read_tapis_file")
def read_tapis_file(system_id: str, path: str):
    """
    Reads the text content of a Tapis file and returns it to the terminal.
    Best for small files like logs, scripts, or CSVs.
    """
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "")
    logger.info(auth_header)
    t = get_tapis_tapipy_object(TAPIS_BASE_URL, auth_header)
    try:
        # Fetch content from Tapis
        content = t.files.getContents(systemId=system_id, path=path)

        # content is usually returned as bytes; decode to string
        text_content = content.decode('utf-8')

        # Limit length to avoid hitting LLM context limits if file is huge
        if len(text_content) > 10000:
            return text_content[:10000] + "\n\n... [Content truncated for length] ..."

        return text_content

    except Exception as e:
        return f"Error reading file content: {str(e)}"
@mcp.tool
@tracer.tool(name="MCP.get_job_output_listing")
def get_job_output_listing(job_uuid:str, path:str):
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "")
    logger.info(auth_header)
    t = get_tapis_tapipy_object(TAPIS_BASE_URL, auth_header)
    output = t.jobs.getJobOutputList(jobUuid=job_uuid, outputPath= path)
    return f"Job {job_uuid} output listing is: str({output})"
@mcp.tool
@tracer.tool(name="MCP.get_job_status")
def get_job_status(job_uuid: str):
    """
    Step 5 in Workflow: Poll the status of a submitted job.
    """
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("x-tapis-token", "")
    logger.info(auth_header)
    t = get_tapis_tapipy_object(TAPIS_BASE_URL, auth_header)
    status = t.jobs.getJobStatus(jobUuid=job_uuid)
    return f"Job {job_uuid} is currently: {status.status}"
@mcp.tool
@tracer.tool(name="MCP.wait_for_few_seconds")
def wait_for_few_seconds(secs: int):
    print(f"waiting for {secs} secs before checking the job status....")
    time.sleep(secs)
    print("finished waiting.....")
    return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    mcp.run(transport="http", host='0.0.0.0', port=9001)