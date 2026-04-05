from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.types import EnvironmentMetadata, HealthResponse, SchemaResponse

from baseline import run_baseline
from graders import grade_episode
from models import AgentAction, BaselineRequest, GraderRequest, ResetPayload, StepPayload
from server.environment import PromptShieldEnvironment
from tasks import ENV_VERSION, list_tasks_payload


class AppServices:
    def __init__(self) -> None:
        self.http_env = PromptShieldEnvironment()
        self.http_lock = asyncio.Lock()


def _serialize_observation(observation) -> dict[str, Any]:
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
    }


def _metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="PromptShield Environment",
        description=(
            "Deterministic security environment for prompt injection,"
            " jailbreak, and data exfiltration defense."
        ),
        version=ENV_VERSION,
        author="promptshield_env",
    )


def _schema() -> SchemaResponse:
    from models import PromptShieldObservation, PromptShieldState

    return SchemaResponse(
        action=AgentAction.model_json_schema(),
        observation=PromptShieldObservation.model_json_schema(),
        state=PromptShieldState.model_json_schema(),
    )


def create_promptshield_app() -> FastAPI:
    app = FastAPI(
        title="PromptShield Env",
        version=ENV_VERSION,
        description="OpenEnv environment for prompt injection and data exfiltration defense.",
    )
    services = AppServices()
    app.state.services = services

    @app.get("/")
    async def index() -> HTMLResponse:
        html = """
        <html>
          <head><title>PromptShield Env</title></head>
          <body style="font-family: sans-serif; margin: 32px;">
            <h1>PromptShield Env</h1>
            <p>Deterministic OpenEnv environment for prompt injection and data exfiltration defense.</p>
            <ul>
              <li><a href="/docs">/docs</a></li>
              <li><a href="/tasks">/tasks</a></li>
              <li><a href="/schema">/schema</a></li>
              <li><a href="/metadata">/metadata</a></li>
              <li><a href="/health">/health</a></li>
            </ul>
          </body>
        </html>
        """
        return HTMLResponse(html)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy")

    @app.get("/metadata", response_model=EnvironmentMetadata)
    async def metadata() -> EnvironmentMetadata:
        return _metadata()

    @app.get("/schema", response_model=SchemaResponse)
    async def schema() -> SchemaResponse:
        return _schema()

    @app.get("/tasks")
    async def tasks() -> list[dict[str, object]]:
        return list_tasks_payload()

    @app.post("/reset")
    async def reset(payload: ResetPayload = Body(default_factory=ResetPayload)) -> JSONResponse:
        async with services.http_lock:
            observation = services.http_env.reset(
                seed=payload.seed,
                episode_id=payload.episode_id,
                task_id=payload.task_id,
                case_id=payload.case_id,
            )
            return JSONResponse(_serialize_observation(observation))

    @app.post("/step")
    async def step(payload: StepPayload) -> JSONResponse:
        async with services.http_lock:
            try:
                observation = services.http_env.step(payload.action, timeout_s=payload.timeout_s)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return JSONResponse(_serialize_observation(observation))

    @app.get("/state")
    async def state() -> JSONResponse:
        async with services.http_lock:
            try:
                return JSONResponse(services.http_env.state.model_dump())
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/grader")
    async def grader(payload: GraderRequest = Body(default_factory=GraderRequest)) -> JSONResponse:
        async with services.http_lock:
            if payload.use_current_episode:
                try:
                    case = services.http_env.current_case()
                    actions = services.http_env.current_actions()
                except RuntimeError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            else:
                if not payload.case_id or not payload.actions:
                    raise HTTPException(
                        status_code=400,
                        detail="case_id and actions are required when use_current_episode is false.",
                    )
                from tasks import get_case

                case = get_case(payload.case_id)
                actions = payload.actions

            grade = grade_episode(case, actions)
            return JSONResponse(grade.model_dump())

    @app.post("/baseline")
    async def baseline(payload: BaselineRequest = Body(default_factory=BaselineRequest)) -> JSONResponse:
        try:
            summary = run_baseline(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(summary.model_dump())

    @app.post("/mcp")
    async def mcp(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            )

        method = body.get("method")
        request_id = body.get("id")
        params = body.get("params") or {}

        if method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "list_promptshield_tasks",
                        "description": "Return the PromptShield task catalog and action schema.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "grade_promptshield_episode",
                        "description": "Grade the current episode or a supplied case and action list.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "use_current_episode": {"type": "boolean"},
                                "case_id": {"type": "string"},
                                "actions": {"type": "array"},
                            },
                        },
                    },
                    {
                        "name": "run_promptshield_baseline",
                        "description": "Run the deterministic heuristic baseline.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "limit_per_task": {"type": "integer"},
                            },
                        },
                    },
                ]
            }
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}

            try:
                if tool_name == "list_promptshield_tasks":
                    content = list_tasks_payload()
                elif tool_name == "grade_promptshield_episode":
                    grade_request = GraderRequest.model_validate(arguments)
                    if grade_request.use_current_episode:
                        case = services.http_env.current_case()
                        actions = services.http_env.current_actions()
                    else:
                        from tasks import get_case

                        case = get_case(grade_request.case_id or "")
                        actions = grade_request.actions
                    content = grade_episode(case, actions).model_dump()
                elif tool_name == "run_promptshield_baseline":
                    baseline_request = BaselineRequest(
                        mode="heuristic",
                        limit_per_task=arguments.get("limit_per_task"),
                    )
                    content = run_baseline(baseline_request).model_dump()
                else:
                    return JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                        }
                    )
            except Exception as exc:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }
                )

            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"content": content}})

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        env = PromptShieldEnvironment()
        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "reset":
                    payload = ResetPayload.model_validate(message.get("data", {}))
                    observation = env.reset(
                        seed=payload.seed,
                        episode_id=payload.episode_id,
                        task_id=payload.task_id,
                        case_id=payload.case_id,
                    )
                    await websocket.send_json({"type": "observation", "data": _serialize_observation(observation)})
                elif message_type == "step":
                    action = AgentAction.model_validate(message.get("data", {}))
                    observation = env.step(action)
                    await websocket.send_json({"type": "observation", "data": _serialize_observation(observation)})
                elif message_type == "state":
                    await websocket.send_json({"type": "state", "data": env.state.model_dump()})
                elif message_type == "close":
                    break
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {"message": f"Unknown message type: {message_type}", "code": "UNKNOWN_TYPE"},
                        }
                    )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": str(exc), "code": "EXECUTION_ERROR"},
                }
            )
        finally:
            await websocket.close()

    return app


app = create_promptshield_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
