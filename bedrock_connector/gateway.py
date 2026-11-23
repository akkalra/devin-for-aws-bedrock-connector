"""HTTP gateway: OpenAI- and Anthropic-compatible endpoints served by Amazon Bedrock,
plus metering, agent sessions and a live dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__
from .agent import Agent
from .bedrock import make_client
from .config import Settings
from .metering import Ledger
from .router import ModelRouter
from .translate import (anthropic_stream_events, anthropic_to_converse, converse_to_anthropic,
                        converse_to_openai, openai_stream_chunks, openai_to_converse)

STATIC = Path(__file__).parent / "static"
SAMPLE_REPO = Path(__file__).resolve().parent.parent / "examples" / "sample_repo"


def create_app(settings: Settings | None = None, client=None) -> FastAPI:
    settings = settings or Settings()
    client = client or make_client(settings)
    router = ModelRouter(settings)
    ledger = Ledger(settings)
    agent = Agent(client, router, ledger)
    app = FastAPI(title="Devin for AWS — Bedrock connector", version=__version__)
    app.state.settings, app.state.ledger, app.state.agent = settings, ledger, agent

    def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
        if not settings.gateway_api_key:
            return
        token = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
        if token != settings.gateway_api_key:
            raise HTTPException(401, "invalid gateway API key")

    def call_bedrock(conv: dict, requested: str, source: str, session_id: str):
        role, model_id = router.resolve(requested)
        try:
            res = client.converse(model_id, conv["messages"], system=conv["system"], tools=conv["tools"],
                                  max_tokens=conv["max_tokens"], temperature=conv["temperature"])
        except Exception as e:  # botocore ClientError etc.
            raise HTTPException(502, f"Bedrock error ({model_id}): {e}") from e
        ledger.record_call(session_id=session_id, role=role, source=source, result=res)
        return res

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__, "mode": settings.resolved_mode, "auth": settings.auth_method,
                "region": settings.region, "role_models": settings.role_models,
                "guardrail": bool(settings.guardrail_id), "inference_provider": "amazon-bedrock"}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models():
        ids = sorted(set(settings.role_models.values())) + ["devin-planner", "devin-coder", "devin-reviewer", "devin-fast"]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "amazon-bedrock"} for i in ids]}

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat_completions(request: Request):
        body = await request.json()
        sid = request.headers.get("x-session-id", "api")
        res = await run_in_threadpool(call_bedrock, openai_to_converse(body), body.get("model"), "openai", sid)
        out = converse_to_openai(res, body.get("model") or res.model_id)
        if body.get("stream"):
            def gen():
                for chunk in openai_stream_chunks(out):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return out

    @app.post("/v1/messages", dependencies=[Depends(auth)])
    async def messages(request: Request):
        body = await request.json()
        sid = request.headers.get("x-session-id", "api")
        res = await run_in_threadpool(call_bedrock, anthropic_to_converse(body), body.get("model"), "anthropic", sid)
        out = converse_to_anthropic(res, body.get("model") or res.model_id)
        if body.get("stream"):
            def gen():
                for event, data in anthropic_stream_events(out):
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return out

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats():
        return {"summary": ledger.summary(), "recent": ledger.recent(40), "health": healthz(),
                "sessions": [s.to_dict() | {"usage": ledger.summary(s.id)}
                             for s in sorted(agent.sessions.values(), key=lambda s: s.started_at or 1e18, reverse=True)]}

    @app.post("/v1/agent/sessions", dependencies=[Depends(auth)])
    async def start_session(request: Request):
        body = await request.json()
        task = body.get("task") or "Fix the failing tests."
        repo = Path(body.get("repo") or SAMPLE_REPO).resolve()
        allowed = Path(os.environ.get("AGENT_REPO_ROOT", SAMPLE_REPO.parent)).resolve()
        if body.get("repo") and repo != allowed and allowed not in repo.parents:
            raise HTTPException(403, f"repo must be under AGENT_REPO_ROOT ({allowed})")
        if not repo.is_dir():
            raise HTTPException(400, f"repo not found: {repo}")
        sess = agent.create_session(task, str(repo), workdir=os.environ.get("AGENT_WORKDIR", "runs"))
        agent.run_async(sess)
        return JSONResponse(sess.to_dict(), status_code=202)

    @app.get("/v1/agent/sessions/{sid}", dependencies=[Depends(auth)])
    def get_session(sid: str):
        if sid not in agent.sessions:
            raise HTTPException(404)
        return agent.sessions[sid].to_dict() | {"usage": ledger.summary(sid)}

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    return app
