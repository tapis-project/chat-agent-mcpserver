"""Small MCP tool definitions (Synchronous)."""

import time
from typing import Annotated, Any
from urllib.parse import quote

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from pydantic import Field

from response_schemas import ToolObservation
from tapis_service import TapisService

# Export mutating tools for application tracking
MUTATING_TOOLS = {"submit_job"}


def access_token() -> str:
    """Read the current request's token without logging it."""
    return get_http_headers().get("x-tapis-token", "")


def search_params(
    list_type: str,
    compute_total: bool,
    **filters,
) -> dict[str, Any]:
    params = {}

    if list_type != "OWNED":
        params["listType"] = list_type

    if compute_total:
        params["computeTotal"] = "true"

    for name, value in filters.items():
        if value is not None:
            params[name] = (
                str(value).lower()
                if isinstance(value, bool)
                else value
            )

    return params


def register_tools(
    mcp: FastMCP,
    service: TapisService,
) -> None:

    @mcp.tool
    def get_tapis_help(
        question: str,
    ) -> ToolObservation:
        """Answer a question using Tapis documentation."""
        return service.help(question, access_token())

    @mcp.tool
    def get_system(
        systemId: str,
    ) -> ToolObservation:
        """Get a Tapis system by ID."""
        return service.get(
            "get_system",
            f"/v3/systems/{quote(systemId, safe='')}",
            access_token(),
        )

    @mcp.tool
    def get_systems_list(
        listType: str = "OWNED",
        computeTotal: bool = False,
        systemType: str | None = None,
        defaultAuthnMethod: str | None = None,
        canExec: bool | None = None,
    ) -> ToolObservation:
        """Search systems.

        listType: OWNED, SHARED_PUBLIC, ALL.
        computeTotal adds a count.
        """
        params = search_params(
            listType,
            computeTotal,
            **{
                "systemType.eq": systemType,
                "defaultAuthnMethod.eq": defaultAuthnMethod,
                "canExec.eq": canExec,
            },
        )

        return service.get(
            "get_systems_list",
            "/v3/systems/search",
            access_token(),
            params=params,
        )

    @mcp.tool
    def get_app_info(
        appId: str,
        appVersion: str,
    ) -> ToolObservation:
        """Get a specific version of a Tapis application."""
        path = (
            f"/v3/apps/{quote(appId, safe='')}/"
            f"{quote(appVersion, safe='')}"
        )
        return service.get(
            "get_app_info",
            path,
            access_token(),
        )

    @mcp.tool
    def get_apps_list(
        listType: str = "OWNED",
        computeTotal: bool = False,
        appId: str | None = None,
        appVersion: str | None = None,
    ) -> ToolObservation:
        """Search applications.

        listType: OWNED, SHARED_PUBLIC, SHARED_DIRECT,
        READ_PERM, MINE, ALL.
        appVersion requires appId.
        computeTotal adds a count to the result metadata.
        """
        if appVersion is not None and appId is None:
            return service.failure(
                "get_apps_list",
                "invalid_arguments",
                "appId is required when appVersion is supplied",
            )

        params = search_params(
            listType,
            computeTotal,
            **{
                "id.eq": appId,
                "version.eq": appVersion,
                "select": (
                    "allAttributes" if appId is not None else None
                ),
            },
        )

        return service.get(
            "get_apps_list",
            "/v3/apps/search",
            access_token(),
            params=params,
        )

    @mcp.tool
    def list_files(
        system_id: str,
        path: str,
    ) -> ToolObservation:
        """List file names at a path on a Tapis system."""
        return service.sdk(
            "list_files",
            access_token(),
            lambda client: [
                entry.name
                for entry in client.files.listFiles(
                    systemId=system_id,
                    path=path,
                )
            ],
        )

    @mcp.tool
    def submit_job(
        name: str,
        description: str,
        app_id: str,
        app_version: str,
        exec_system_id: str,
        file_inputs: list[dict[str, str]] | None = None,
        app_args: list[dict[str, Any]] | None = None,
    ) -> ToolObservation:
        """Submit a Tapis job.

        Submission does not mean execution has completed.
        """
        job_request = {
            "name": name,
            "description": description,
            "appId": app_id,
            "appVersion": app_version,
            "execSystemId": exec_system_id,
            "fileInputs": file_inputs or [],
            "parameterSet": (
                {"appArgs": app_args}
                if app_args is not None
                else {}
            ),
        }

        def submit(client):
            result = client.jobs.submitJob(**job_request)
            return {
                "job_uuid": result.uuid,
                "job_status": result.status,
            }

        observation = service.sdk(
            "submit_job",
            access_token(),
            submit,
        )

        if observation.ok and not observation.data.get("job_uuid"):
            return service.failure(
                "submit_job",
                "missing_job_uuid",
                "Tapis did not return a job identifier",
            )

        if observation.ok:
            observation.message = (
                "Job submitted; execution may still be pending"
            )

        return observation

    @mcp.tool
    def read_tapis_file(
        system_id: str,
        path: str,
    ) -> ToolObservation:
        """Read up to 10,000 characters from a UTF-8 text file."""
        def read(client):
            content = client.files.getContents(
                systemId=system_id,
                path=path,
            )
            text = content.decode("utf-8")

            return {
                "content": text[:10000],
                "truncated": len(text) > 10000,
            }

        return service.sdk(
            "read_tapis_file",
            access_token(),
            read,
        )

    @mcp.tool
    def get_job_output_listing(
        job_uuid: str,
        path: str,
    ) -> ToolObservation:
        """List job output files at the requested output path."""
        return service.sdk(
            "get_job_output_listing",
            access_token(),
            lambda client: client.jobs.getJobOutputList(
                jobUuid=job_uuid,
                outputPath=path,
            ),
        )

    @mcp.tool
    def get_job_status(
        job_uuid: str,
    ) -> ToolObservation:
        """Get the current execution status of a Tapis job."""
        return service.sdk(
            "get_job_status",
            access_token(),
            lambda client: {
                "job_uuid": job_uuid,
                "status": client.jobs.getJobStatus(
                    jobUuid=job_uuid
                ).status,
            },
        )

    @mcp.tool
    def wait_for_few_seconds(
        secs: Annotated[int, Field(ge=0, le=60)],
    ) -> ToolObservation:
        """Wait between zero and 60 seconds before polling again."""
        time.sleep(secs)

        return ToolObservation(
            ok=True,
            tool="wait_for_few_seconds",
            data={"waited_seconds": secs},
        )