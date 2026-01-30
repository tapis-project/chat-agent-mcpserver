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


path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")

TAPIS_BASE_URL = os.getenv("TAPIS_BASE_URL")
RAG_ACCESS_TOKEN=os.getenv("TAPIS_TACC_SPADHY_TOKEN")


mcp = FastMCP(
    name="TapisMCP",    
    log_level="ERROR"
)

@mcp.tool
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
    return (response.json())

@mcp.tool
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
                url = url + "?id.eq=" + appId
                first_query_param = 1
        else:
                url = url + "&appId.eq=" + appId    
    if appId is not None and appVersion is not None:
        url = url + "&appVersion.eq=" + appVersion
    elif appId is None and appVersion is not None:
          print("app Id is needed")
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

@mcp.tool
async def get_tapis_help(question:str) -> Any:
    
    url = 'https://rag.pods.tacc.tapis.io/chat'
    jsonobj = {"question":question}
    headers = {'Content-type': 'application/json','x-tapis-token':RAG_ACCESS_TOKEN}
    response = requests.post(url, json = jsonobj, headers=headers)
    if response.status_code == 500:
        return {"status":"error"}
    return json.loads(response.text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    mcp.run(transport="http", host='0.0.0.0', port=9001)