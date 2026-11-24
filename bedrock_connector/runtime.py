"""Where agent sessions run: a local thread (demo) or Amazon Bedrock AgentCore Runtime.

With AGENTCORE_RUNTIME_ARN set, the gateway invokes the agent container
deployed on AgentCore Runtime (see agentcore_app.py). `runtimeUserId` makes
AgentCore issue a workload access token for the signed-in user, which the
container exchanges for the org's Bedrock API key in AgentCore Identity.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict

from .agent import Agent, AgentSession


def org_payload(org) -> dict:
    """Org config the runtime needs. Holds no secrets: the key stays in AgentCore Identity."""
    d = asdict(org)
    d.pop("key_last4", None)
    return d


class AgentCoreDispatcher:
    def __init__(self, runtime_arn: str, region: str, ledger, client=None, guardrail: dict | None = None):
        import boto3
        from botocore.config import Config

        self.arn, self.ledger, self.guardrail = runtime_arn, ledger, guardrail
        # The runtime answers only after plan, edits, tests and review finish.
        self.client = client or boto3.client("bedrock-agentcore", region_name=region, config=Config(
            read_timeout=900, retries={"total_max_attempts": 1}))

    def request(self, sess: AgentSession) -> dict:
        return {"agentRuntimeArn": self.arn,
                "runtimeSessionId": f"devin-{sess.id}-{uuid.uuid4().hex}",  # API requires >= 33 chars
                "runtimeUserId": sess.user_id or "anonymous",
                "contentType": "application/json", "accept": "application/json",
                "payload": json.dumps({"task": sess.task, "session_id": sess.id, "user_id": sess.user_id,
                                       "org": org_payload(sess.org), "guardrail": self.guardrail}).encode()}

    def run(self, sess: AgentSession) -> None:
        sess.status, sess.runtime = "running", "agentcore"
        try:
            resp = self.client.invoke_agent_runtime(**self.request(sess))
            out = json.loads(resp["response"].read())
            for k in ("status", "plan", "summary", "diff", "steps", "started_at", "ended_at"):
                setattr(sess, k, out["session"][k])
            self.ledger.ingest(out.get("calls", []))
            self.ledger.record_runtime(sess.id, out.get("runtime_seconds", 0.0), sess.org_id)
        except Exception as e:
            sess.status = "error"
            sess.steps.append({"type": "error", "text": f"AgentCore Runtime: {type(e).__name__}: {e}"})

    def run_async(self, sess: AgentSession) -> threading.Thread:
        t = threading.Thread(target=self.run, args=(sess,), daemon=True)
        t.start()
        return t


class LocalDispatcher:
    def __init__(self, agent: Agent):
        self.agent = agent

    def run_async(self, sess: AgentSession) -> threading.Thread:
        return self.agent.run_async(sess)
