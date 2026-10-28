"""Deterministic contract and HTTP tests; no model credentials or remote jobs."""
import asyncio
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from response_schemas import AgentResponse, ToolObservation, TraceEvent, blocked_response
from tool_observation import normalize_observation, observation_interceptor


def response(**overrides):
    data = dict(
        status="completed", answer="You own two systems.", steps=[], total_count=2,
        citations=[], completed_actions=[], missing_fields=[],
    )
    data.update(overrides)
    return AgentResponse(**data)


def mcp_result(data, is_error=False):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data))],
        isError=is_error,
    )


def load_client_module():
    # Avoid provider construction and telemetry side effects during import.
    with patch("langchain_openai.ChatOpenAI"), patch("phoenix.otel.register"), patch(
        "openinference.instrumentation.langchain.LangChainInstrumentor.instrument"
    ):
        return importlib.import_module("tapis_mcp_client_header")


class SchemaTests(unittest.TestCase):
    def test_rejects_negative_count_and_extra_fields(self):
        for extra in ({"total_count": -1}, {"execute_immediately": True}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                response(**extra)

    def test_blocked_response_serializes(self):
        data = blocked_response("Request blocked").model_dump(mode="json")
        self.assertEqual(data["status"], "blocked")
        self.assertEqual(data["completed_actions"], [])
        self.assertIsNone(data["total_count"])


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    def test_tapis_failure_inside_successful_transport(self):
        result = normalize_observation("get_system", mcp_result({"status": "error"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "tapis_error")

    def test_successful_transport_does_not_prove_submission(self):
        result = normalize_observation("submit_job", mcp_result("Job submitted"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "unverified_action")

    def test_submission_requires_job_identifier(self):
        for data, expected in (({}, False), ({"job_uuid": "job-123"}, True)):
            observed = ToolObservation(ok=True, tool="submit_job", data=data)
            result = normalize_observation("submit_job", mcp_result(observed.model_dump()))
            self.assertEqual(result.ok, expected)

    def test_mcp_error_overrides_explicit_success(self):
        data = ToolObservation(ok=True, tool="get_system", data={}).model_dump()
        self.assertFalse(normalize_observation("get_system", mcp_result(data, True)).ok)

    async def test_exception_is_recorded_and_returned_to_model(self):
        trace = []
        request = MCPToolCallRequest(name="get_system", args={}, server_name="tapis")
        result = await observation_interceptor(trace)(request, AsyncMock(side_effect=RuntimeError("private detail")))
        self.assertFalse(result.isError)
        self.assertFalse(result.structuredContent["ok"])
        self.assertEqual(trace[0].observation.error_code, "tool_exception")
        self.assertNotIn("private detail", result.content[0].text)

    async def test_request_traces_are_isolated(self):
        traces = [[], []]
        request = MCPToolCallRequest(name="get_system", args={}, server_name="tapis")
        for trace in traces:
            await observation_interceptor(trace)(request, AsyncMock(return_value=mcp_result({"status": "success"})))
        self.assertEqual([len(trace) for trace in traces], [1, 1])
        self.assertEqual([trace[0].step for trace in traces], [1, 1])
