"""A minimal autonomous coding agent whose every model call goes through Amazon Bedrock.

It stands in for Devin's agent loop: plan (planner model), then edit / run /
observe with tools (coder model), then a review summary (reviewer model).
Each call is metered into the shared ledger.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .bedrock import ConverseResult
from .metering import Ledger
from .router import ModelRouter

TOOLS = [
    {"toolSpec": {"name": "list_files", "description": "List files under a directory of the repo.",
                  "inputSchema": {"json": {"type": "object", "properties": {"path": {"type": "string"}}}}}},
    {"toolSpec": {"name": "read_file", "description": "Read a UTF-8 text file from the repo.",
                  "inputSchema": {"json": {"type": "object", "properties": {"path": {"type": "string"}},
                                           "required": ["path"]}}}},
    {"toolSpec": {"name": "write_file", "description": "Create or overwrite a file in the repo with the full new contents.",
                  "inputSchema": {"json": {"type": "object", "properties": {"path": {"type": "string"},
                                                                            "content": {"type": "string"}},
                                           "required": ["path", "content"]}}}},
    {"toolSpec": {"name": "run_command", "description": "Run a shell command in the repo root (60s timeout). Use it to run tests.",
                  "inputSchema": {"json": {"type": "object", "properties": {"command": {"type": "string"}},
                                           "required": ["command"]}}}},
    {"toolSpec": {"name": "finish", "description": "Call when the task is done and verified.",
                  "inputSchema": {"json": {"type": "object", "properties": {"summary": {"type": "string"}},
                                           "required": ["summary"]}}}},
]

SYSTEM = ("You are an autonomous software engineer working in a git repository. "
          "Use the tools to inspect code, make minimal correct edits, and run the tests to verify. "
          "Do not finish until the tests pass. Keep text responses short.")


@dataclass
class AgentSession:
    task: str
    repo_dir: str
    id: str = field(default_factory=lambda: "sess-" + uuid.uuid4().hex[:8])
    status: str = "queued"
    plan: str = ""
    summary: str = ""
    diff: str = ""
    steps: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "task", "repo_dir", "status", "plan", "summary", "diff",
                                              "steps", "started_at", "ended_at")}


class Agent:
    def __init__(self, client, router: ModelRouter, ledger: Ledger, max_steps: int = 25):
        self.client, self.router, self.ledger, self.max_steps = client, router, ledger, max_steps
        self.sessions: dict[str, AgentSession] = {}

    # ---------------------------------------------------------------- public

    def create_session(self, task: str, source_repo: str, workdir: str = "runs") -> AgentSession:
        sess = AgentSession(task=task, repo_dir="")
        dest = Path(workdir) / sess.id
        workroot = Path(workdir).resolve()

        def skip(d, names):
            return {n for n in names if n in ("__pycache__", ".pytest_cache", ".git") or Path(d, n).resolve() == workroot}
        shutil.copytree(source_repo, dest, symlinks=True, ignore=skip)
        subprocess.run("git init -q && printf '__pycache__/\\n*.pyc\\n.pytest_cache/\\n' >> .git/info/exclude && git add -A && git -c user.email=agent@demo -c user.name=agent commit -qm base && git tag base",
                       shell=True, cwd=dest, check=False)
        sess.repo_dir = str(dest.resolve())
        self.sessions[sess.id] = sess
        return sess

    def run_async(self, sess: AgentSession) -> threading.Thread:
        t = threading.Thread(target=self.run, args=(sess,), daemon=True)
        t.start()
        return t

    def run(self, sess: AgentSession) -> AgentSession:
        sess.status, sess.started_at = "running", time.time()
        try:
            self._plan(sess)
            self._execute(sess)
            passed = self._review(sess)
            sess.status = "finished" if passed else "unverified"
        except Exception as e:  # surfaced in the dashboard
            sess.status = "error"
            sess.steps.append({"type": "error", "text": f"{type(e).__name__}: {e}"})
        finally:
            sess.ended_at = time.time()
            self.ledger.record_runtime(sess.id, sess.ended_at - sess.started_at)
        return sess

    # ---------------------------------------------------------------- phases

    def _call(self, sess: AgentSession, role: str, messages, system=None, tools=None) -> ConverseResult:
        model_id = self.router.for_role(role)
        res = self.client.converse(model_id, messages, system=[{"text": system or SYSTEM}], tools=tools)
        rec = self.ledger.record_call(session_id=sess.id, role=role, source="agent", result=res)
        sess.steps.append({"type": "model", "role": role, "model_id": model_id, "tokens_in": res.input_tokens,
                           "tokens_out": res.output_tokens, "acus": round(rec.acus, 4), "text": res.text[:600]})
        return res

    def _plan(self, sess: AgentSession) -> None:
        files = "\n".join(sorted(str(p.relative_to(sess.repo_dir)) for p in Path(sess.repo_dir).rglob("*")
                                 if p.is_file() and ".git" not in p.parts))
        prompt = f"Task: {sess.task}\n\nRepository files:\n{files}\n\nWrite a short numbered plan (max 5 steps)."
        res = self._call(sess, "planner", [{"role": "user", "content": [{"text": prompt}]}],
                         system="You are the planning model for an autonomous coding agent. Be concise.")
        sess.plan = res.text

    def _execute(self, sess: AgentSession) -> None:
        messages = [{"role": "user", "content": [{"text": f"Task: {sess.task}\n\nPlan:\n{sess.plan}\n\nStart now."}]}]
        for _ in range(self.max_steps):
            res = self._call(sess, "coder", messages, tools=TOOLS)
            messages.append(res.message)
            if not res.tool_uses:
                messages.append({"role": "user", "content": [{"text": "Continue using the tools, or call finish."}]})
                continue
            results = []
            for tu in res.tool_uses:
                output = self._run_tool(sess, tu["name"], tu.get("input") or {})
                results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"text": output[:12000]}]}})
                if tu["name"] == "finish":
                    sess.summary = (tu.get("input") or {}).get("summary", "")
                    return
            messages.append({"role": "user", "content": results})
        sess.summary = "Stopped: step limit reached."

    def _review(self, sess: AgentSession) -> bool:
        sess.diff = subprocess.run("git add -A && git diff --cached base", shell=True, cwd=sess.repo_dir, capture_output=True, text=True).stdout
        tests = self._run_tool(sess, "run_command", {"command": "python -m pytest -q"}, log=False)
        prompt = (f"Task: {sess.task}\n\nDiff:\n{sess.diff[:8000] or '(no changes)'}\n\nTest output:\n{tests[-2000:]}\n\n"
                  "Write a 3-bullet PR description: what changed, why, and verification.")
        res = self._call(sess, "reviewer", [{"role": "user", "content": [{"text": prompt}]}],
                         system="You review code changes and write crisp PR descriptions.")
        sess.summary = res.text or sess.summary
        return tests.startswith("exit=0")

    # ---------------------------------------------------------------- tools

    def _safe(self, sess: AgentSession, rel: str) -> Path:
        root = Path(sess.repo_dir).resolve()
        p = (root / (rel or ".")).resolve()
        if root != p and root not in p.parents:
            raise ValueError("path escapes the repository")
        return p

    def _run_tool(self, sess: AgentSession, name: str, args: dict, log: bool = True) -> str:
        try:
            if name == "list_files":
                base = self._safe(sess, args.get("path", "."))
                out = "\n".join(sorted(str(p.relative_to(sess.repo_dir)) for p in base.rglob("*")
                                       if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts))
            elif name == "read_file":
                out = self._safe(sess, args["path"]).read_text()
            elif name == "write_file":
                p = self._safe(sess, args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                out = f"wrote {args['path']} ({len(args['content'])} bytes)"
            elif name == "run_command":
                env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
                r = subprocess.run(args["command"], shell=True, cwd=sess.repo_dir, capture_output=True,
                                   text=True, timeout=60, env=env)
                out = f"exit={r.returncode}\n{r.stdout}\n{r.stderr}".strip()
            elif name == "finish":
                out = "ok"
            else:
                out = f"unknown tool {name}"
        except Exception as e:
            out = f"error: {type(e).__name__}: {e}"
        if log:
            sess.steps.append({"type": "tool", "name": name, "args": {k: (v if len(str(v)) < 300 else str(v)[:300] + "…")
                                                                      for k, v in args.items()},
                               "output": out[:1500]})
        return out
