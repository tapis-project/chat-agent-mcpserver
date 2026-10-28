"""Build and run the Tapis MCP server."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

from tapis_service import TapisService
from tapis_tools import register_tools

# Load environment variables from .env
load_dotenv(Path(__file__).with_name(".env"), override=False)

# Initialize server and service directly
mcp = FastMCP(name="TapisMCP")
service = TapisService(
    base_url=os.getenv("TAPIS_BASE_URL", "https://public.tapis.io"),
    rag_url=os.getenv("RAG_BASE_URL"),
)

# Register tools
register_tools(mcp, service)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    mcp.run(transport="http", host="0.0.0.0", port=9001)


if __name__ == "__main__":
    main()