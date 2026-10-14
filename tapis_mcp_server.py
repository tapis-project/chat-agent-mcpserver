"""Build and run the Tapis MCP server."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

from tapis_service import TapisService
from tapis_tools import register_tools


def create_server(service: TapisService | None = None) -> FastMCP:
    service = service or TapisService(
        base_url=os.getenv("TAPIS_BASE_URL", "https://public.tapis.io"),
        rag_url=os.getenv("RAG_BASE_URL"),
    )

    @asynccontextmanager
    async def lifespan(server):
        provider = None

        if os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true":
            from phoenix.otel import register

            provider = register(
                project_name=os.getenv(
                    "PHOENIX_PROJECT_NAME", "llama4-eval"
                ),
                auto_instrument=True,
                batch=True,
                set_global_tracer_provider=False,
            )
            service.tracer = provider.get_tracer(__name__)

        try:
            async with service.lifespan():
                yield
        finally:
            service.tracer = None
            if provider is not None:
                provider.shutdown()

    server = FastMCP(name="TapisMCP", lifespan=lifespan)
    register_tools(server, service)
    return server


# Exposes the MCP object without starting services.
mcp = create_server()


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s\t%(message)s",
    )
    create_server().run(
        transport="http",
        host="0.0.0.0",
        port=9001,
    )


if __name__ == "__main__":
    main()