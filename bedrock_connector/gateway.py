"""HTTP gateway: OpenAI- and Anthropic-compatible endpoints served by Amazon Bedrock,
plus metering, agent sessions and a live dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__
from .agent import Agent
from .config import Settings
from .identity import MissingCredential
from .inference import CallContext, build_stack
from .runtime import AgentCoreDispatcher, LocalDispatcher
from .tenancy import Org, User
from .translate import (anthropic_stream_events, anthropic_to_converse, converse_to_anthropic,
                        converse_to_openai, openai_stream_chunks, openai_to_converse)

STATIC = Path(__file__).parent / "static"
SAMPLE_REPO = Path(__file__).resolve().parent.parent / "examples" / "sample_repo"


@dataclass
class Principal:
    user: User | None
    org: Org

    @property
    def scope(self) -> str:
        """Org filter: callers see only their own org (anonymous callers get the demo org)."""
        return self.org.id


def create_app(settings: Settings | None = None, client=None, vault=None) -> FastAPI:
    settings = settings or Settings()
    stack = build_stack(settings, base_client=client, vault=vault)
    tenants, ledger, inference, router = stack.tenants, stack.ledger, stack.inference, stack.router
    agent = Agent(inference, tenants.default_org)
    guardrail = {"id": settings.guardrail_id, "version": settings.guardrail_version} if settings.guardrail_id else None
    dispatcher = (AgentCoreDispatcher(settings.agentcore_runtime_arn, settings.region, ledger, guardrail=guardrail)
                  if settings.agentcore_runtime_arn else LocalDispatcher(agent))
    app = FastAPI(title="Devin for AWS — Bedrock connector", version=__version__)
    app.state.settings, app.state.ledger, app.state.agent, app.state.stack = settings, ledger, agent, stack

    def perimeter(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> str | None:
        """GATEWAY_API_KEY guards the whole gateway; returns the bearer token, if any."""
        bearer = (authorization or "").removeprefix("Bearer ").strip() or None
        if settings.gateway_api_key and settings.gateway_api_key not in (x_api_key, bearer):
            raise HTTPException(401, "invalid gateway API key")
        return bearer

    def principal(bearer: str | None = Depends(perimeter), x_devin_session: str | None = Header(None)) -> Principal:
        user = tenants.user_for(x_devin_session) or tenants.user_for(bearer)
        return Principal(user, tenants.orgs[user.org_id] if user else tenants.default_org)

    def org_admin(p: Principal = Depends(principal)) -> Principal:
        if not p.user or not p.user.is_admin:
            raise HTTPException(403, "sign in as an org admin")
        return p

    def call_bedrock(conv: dict, requested: str, source: str, session_id: str, p: Principal):
        ctx = CallContext(p.org, p.user.email if p.user else "anonymous", session_id, source)
        try:
            res, _ = inference.converse(ctx, "coder", conv["messages"], system=conv["system"], tools=conv["tools"],
                                        max_tokens=conv["max_tokens"], temperature=conv["temperature"],
                                        requested=requested or "")
        except MissingCredential as e:
            raise HTTPException(412, str(e)) from e
        except Exception as e:  # botocore ClientError etc.
            raise HTTPException(502, f"Bedrock error: {e}") from e
        return res

    def me(p: Principal) -> dict:
        return {"user": p.user.__dict__ if p.user else None, "org": p.org.public(),
                "routing": {r: router.candidates(r, p.org) for r in settings.role_models},
                "credential_vault": stack.vault.name,
                "agent_runtime": "agentcore" if settings.agentcore_runtime_arn else "local"}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__, "mode": settings.resolved_mode, "auth": settings.auth_method,
                "region": settings.region, "role_models": settings.role_models,
                "guardrail": bool(settings.guardrail_id), "inference_provider": "amazon-bedrock",
                "credential_vault": stack.vault.name,
                "agent_runtime": "agentcore" if settings.agentcore_runtime_arn else "local"}

    @app.get("/v1/models", dependencies=[Depends(perimeter)])
    def models():
        ids = sorted(set(settings.role_models.values())) + ["devin-planner", "devin-coder", "devin-reviewer", "devin-fast"]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "amazon-bedrock"} for i in ids]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, p: Principal = Depends(principal)):
        body = await request.json()
        sid = request.headers.get("x-session-id", "api")
        res = await run_in_threadpool(call_bedrock, openai_to_converse(body), body.get("model"), "openai", sid, p)
        out = converse_to_openai(res, body.get("model") or res.model_id)
        if body.get("stream"):
            def gen():
                for chunk in openai_stream_chunks(out):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return out

    @app.post("/v1/messages")
    async def messages(request: Request, p: Principal = Depends(principal)):
        body = await request.json()
        sid = request.headers.get("x-session-id", "api")
        res = await run_in_threadpool(call_bedrock, anthropic_to_converse(body), body.get("model"), "anthropic", sid, p)
        out = converse_to_anthropic(res, body.get("model") or res.model_id)
        if body.get("stream"):
            def gen():
                for event, data in anthropic_stream_events(out):
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return out

    @app.post("/v1/auth/login", dependencies=[Depends(perimeter)])
    async def login(request: Request):
        """Simulated SSO (stand-in for IAM Identity Center / OIDC): email domain -> org."""
        body = await request.json()
        try:
            token, user = tenants.login(str(body.get("email", "")))
        except KeyError as e:
            raise HTTPException(401, str(e)) from e
        return {"session_token": token} | me(Principal(user, tenants.orgs[user.org_id]))

    @app.post("/v1/auth/logout")
    def logout(x_devin_session: str | None = Header(None), _=Depends(perimeter)):
        if x_devin_session:
            tenants.logout(x_devin_session)
        return {"ok": True}

    @app.get("/v1/me")
    def get_me(p: Principal = Depends(principal)):
        return me(p)

    @app.put("/v1/org/settings")
    async def org_settings(request: Request, p: Principal = Depends(org_admin)):
        body = await request.json()
        if (body.get("credential_source", p.org.credential_source) == "devin_managed"
                and body.get("region", p.org.region) != settings.region):
            raise HTTPException(400, f"Devin-managed inference runs in {settings.region}; "
                                     "use your own key or role for another region")
        try:
            tenants.update_org(p.org, body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return me(p)

    @app.put("/v1/org/bedrock-key")
    async def put_bedrock_key(request: Request, p: Principal = Depends(org_admin)):
        """Store the org's Bedrock API key in the credential vault. It is never returned."""
        key = str((await request.json()).get("api_key", "")).strip()
        if len(key) < 20:
            raise HTTPException(400, "that does not look like a Bedrock API key")
        try:
            await run_in_threadpool(stack.vault.put_api_key, p.org.provider_name, key)
        except Exception as e:
            raise HTTPException(502, f"credential vault error: {type(e).__name__}") from e
        p.org.key_last4, p.org.credential_source = key[-4:], "customer_api_key"
        return me(p)

    @app.post("/v1/org/test-connection")
    def test_connection(p: Principal = Depends(org_admin)):
        ctx = CallContext(p.org, p.user.email, "test-connection", "settings")
        try:
            res, rec = inference.converse(ctx, "fast", [{"role": "user", "content": [{"text": "Reply: Bedrock OK"}]}],
                                          max_tokens=20)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
        return {"ok": True, "model_id": res.model_id, "credential": rec.auth_method, "billed_to": rec.billed_to,
                "request_id": res.request_id, "reply": res.text[:80]}

    @app.get("/v1/stats")
    def stats(p: Principal = Depends(principal)):
        sessions = [s for s in agent.sessions.values() if s.org_id == p.scope]
        return {"summary": ledger.summary(org_id=p.scope), "recent": ledger.recent(40, org_id=p.scope),
                "health": healthz(), "me": me(p),
                "sessions": [s.to_dict() | {"usage": ledger.summary(s.id, org_id=p.scope)}
                             for s in sorted(sessions, key=lambda s: s.started_at or 1e18, reverse=True)]}

    @app.post("/v1/agent/sessions")
    async def start_session(request: Request, p: Principal = Depends(principal)):
        body = await request.json()
        task = body.get("task") or "Fix the failing tests."
        repo = Path(body.get("repo") or SAMPLE_REPO).resolve()
        allowed = Path(os.environ.get("AGENT_REPO_ROOT", SAMPLE_REPO.parent)).resolve()
        if body.get("repo") and settings.agentcore_runtime_arn:
            raise HTTPException(400, "AgentCore Runtime sessions run on the bundled sample; omit repo")
        if body.get("repo") and repo != allowed and allowed not in repo.parents:
            raise HTTPException(403, f"repo must be under AGENT_REPO_ROOT ({allowed})")
        if not repo.is_dir():
            raise HTTPException(400, f"repo not found: {repo}")
        sess = agent.create_session(task, str(repo), workdir=os.environ.get("AGENT_WORKDIR", "runs"),
                                    org=p.org, user_id=p.user.email if p.user else "")
        dispatcher.run_async(sess)
        return JSONResponse(sess.to_dict(), status_code=202)

    @app.get("/v1/agent/sessions/{sid}")
    def get_session(sid: str, p: Principal = Depends(principal)):
        if sid not in agent.sessions or agent.sessions[sid].org_id != p.scope:
            raise HTTPException(404)
        return agent.sessions[sid].to_dict() | {"usage": ledger.summary(sid, org_id=p.scope)}

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    return app
