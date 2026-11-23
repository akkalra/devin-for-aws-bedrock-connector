"""Agent container entrypoint for Amazon Bedrock AgentCore Runtime.

Build with Dockerfile.agentcore (linux/arm64), push to ECR, and create the
runtime with deploy/agentcore/deploy.py. AgentCore serves POST /invocations and
GET /ping on port 8080 and gives each session its own isolated microVM.

Payload: {"task": str, "session_id": str, "user_id": str, "org": {...Org fields}}
"""
from __future__ import annotations

import os
from pathlib import Path

from .agent import Agent
from .config import Settings
from .inference import build_stack
from .tenancy import Org

SAMPLE_REPO = Path(__file__).resolve().parent.parent / "examples" / "sample_repo"


def handle(payload: dict, workload_token: str | None = None, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    stack = build_stack(settings)
    org = Org(**{k: v for k, v in payload["org"].items() if k in Org.__dataclass_fields__})
    agent = Agent(stack.inference, org)
    sess = agent.create_session(payload.get("task") or "Fix the failing tests.", str(SAMPLE_REPO),
                                workdir=os.environ.get("AGENT_WORKDIR", "/tmp/runs"), org=org,
                                user_id=payload.get("user_id", ""), sess_id=payload.get("session_id"))
    sess.workload_token, sess.runtime = workload_token, "agentcore"
    agent.run(sess)
    return {"session": sess.to_dict(), "calls": stack.ledger.recent(10_000)[::-1],
            "runtime_seconds": sess.ended_at - sess.started_at}


def _app():
    from bedrock_agentcore.runtime import BedrockAgentCoreApp, BedrockAgentCoreContext

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload, context):
        return handle(payload, BedrockAgentCoreContext.get_workload_access_token())

    return app


if __name__ == "__main__":
    _app().run()
