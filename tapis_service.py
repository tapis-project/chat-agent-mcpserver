"""Tapis transport, authentication, serialization, and error handling."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
import logging
from typing import Any, Callable

import httpx
from tapipy.tapis import Tapis, TapisResult

from response_schemas import ToolObservation

logger = logging.getLogger(__name__)


def json_data(value: Any) -> Any:
    """Convert SDK result objects into JSON-compatible data."""
    if isinstance(value, TapisResult):
        return {
            key: json_data(item)
            for key, item in vars(value).items()
        }

    if isinstance(value, dict):
        return {
            key: json_data(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError("Unsupported Tapis result type")


class TapisService:
    def __init__(
        self,
        base_url: str,
        rag_url: str | None = None,
        *,
        timeout: float = 30,
        tapis_factory: Callable[..., Tapis] = Tapis,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.rag_url = rag_url
        self.timeout = timeout
        self.tapis_factory = tapis_factory
        self.transport = transport
        self.http: httpx.AsyncClient | None = None
        self.tracer = None

    @asynccontextmanager
    async def lifespan(self):
        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            self.http = client
            try:
                yield
            finally:
                self.http = None

    def _span(self, tool: str):
        if self.tracer:
            return self.tracer.start_as_current_span(f"MCP.{tool}")
        return nullcontext()

    @staticmethod
    def failure(
        tool: str,
        code: str,
        message: str,
    ) -> ToolObservation:
        return ToolObservation(
            ok=False,
            tool=tool,
            error_code=code,
            message=message,
        )

    async def get(
        self,
        tool: str,
        path: str,
        token: str,
        *,
        params=None,
    ) -> ToolObservation:
        if not token:
            return self.failure(
                tool,
                "authentication_required",
                "A Tapis access token is required",
            )

        return await self._get_json(
            tool,
            f"{self.base_url}{path}",
            params=params,
            headers={"x-tapis-token": token},
        )

    async def help(
        self,
        question: str,
        token: str = "",
    ) -> ToolObservation:
        if not self.rag_url:
            return self.failure(
                "get_tapis_help",
                "configuration_missing",
                "RAG_BASE_URL is not configured",
            )

        return await self._get_json(
            "get_tapis_help",
            self.rag_url,
            method="POST",
            body={"question": question},
            headers={"x-tapis-token": token},
        )

    async def _get_json(
        self,
        tool,
        url,
        *,
        method="GET",
        params=None,
        headers=None,
        body=None,
    ) -> ToolObservation:
        if self.http is None:
            return self.failure(
                tool,
                "service_unavailable",
                "The HTTP client has not been started",
            )

        try:
            with self._span(tool):
                response = await self.http.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict):
                    status = str(data.get("status", "")).lower()
                    if status in {"error", "failed", "failure"}:
                        return self.failure(
                            tool,
                            "tapis_error",
                            "The upstream service reported an error",
                        )

                return ToolObservation(
                    ok=True,
                    tool=tool,
                    data=data,
                )

        except httpx.HTTPStatusError as exc:
            return self.failure(
                tool,
                "http_error",
                f"Upstream service returned HTTP "
                f"{exc.response.status_code}",
            )
        except httpx.TimeoutException:
            return self.failure(
                tool,
                "timeout",
                "The upstream request timed out",
            )
        except httpx.RequestError:
            return self.failure(
                tool,
                "connection_error",
                "Could not reach the upstream service",
            )
        except ValueError:
            return self.failure(
                tool,
                "invalid_response",
                "The upstream service returned invalid JSON",
            )

    async def sdk(
        self,
        tool: str,
        token: str,
        operation: Callable[[Tapis], Any],
    ) -> ToolObservation:
        if not token:
            return self.failure(
                tool,
                "authentication_required",
                "A Tapis access token is required",
            )

        def execute():
            # Each operation gets its own token-bound SDK client.
            client = self.tapis_factory(
                base_url=self.base_url,
                access_token=token,
                debug_prints=False,
            )
            with self._span(tool):
                return json_data(operation(client))

        try:
            # Blocking SDK calls run outside the event loop.
            data = await asyncio.to_thread(execute)
            return ToolObservation(
                ok=True,
                tool=tool,
                data=data,
            )
        except Exception:
            # SDK exception text can contain credentials.
            logger.warning("Tapis SDK operation %s failed", tool)
            return self.failure(
                tool,
                "sdk_error",
                "The Tapis operation failed",
            )