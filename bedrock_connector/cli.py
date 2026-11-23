"""CLI: serve the gateway, run the agent, or check Bedrock connectivity."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


def _print_usage(ledger, sid=None):
    s = ledger.summary(sid)
    vp = s["value_pools"]
    print(f"\nBedrock calls: {s['calls']}  tokens in/out: {s['input_tokens']}/{s['output_tokens']}  ACUs: {s['acus']:.3f}")
    print(f"Customer ACU revenue ${vp['customer_revenue_usd']:.4f} | AWS rev share ${vp['aws_revenue_share_usd']:.4f} "
          f"| Cognition AWS infra bill ${vp['cognition_aws_infra_usd']:.4f} (Bedrock ${vp['of_which_bedrock_usd']:.4f}) "
          f"| AWS total capture ${vp['aws_total_capture_usd']:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bedrock-connector")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--allow-anonymous", action="store_true",
                    help="serve live Bedrock on a non-loopback host without GATEWAY_API_KEY")
    ap = sub.add_parser("agent")
    ap.add_argument("task")
    ap.add_argument("--repo", default=None)
    sub.add_parser("check")
    args = p.parse_args(argv)
    settings = Settings()

    if args.cmd == "serve":
        import uvicorn

        from .gateway import create_app
        public = args.host not in ("127.0.0.1", "localhost", "::1")
        if public and settings.resolved_mode == "live" and not settings.gateway_api_key and not args.allow_anonymous:
            print("refusing to expose live Bedrock without GATEWAY_API_KEY (or pass --allow-anonymous)", file=sys.stderr)
            return 2
        print(f"mode={settings.resolved_mode} auth={settings.auth_method} region={settings.region}")
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    from .bedrock import make_client
    from .metering import Ledger
    from .router import ModelRouter
    client, router, ledger = make_client(settings), ModelRouter(settings), Ledger(settings)

    if args.cmd == "check":
        model = router.for_role("fast")
        print(f"mode={settings.resolved_mode} auth={settings.auth_method} region={settings.region} model={model}")
        try:
            res = client.converse(model, [{"role": "user", "content": [{"text": "Reply with exactly: Bedrock OK"}]}],
                                  max_tokens=20)
        except Exception as e:  # botocore ClientError, NoCredentialsError, ...
            print(f"Bedrock call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        ledger.record_call(session_id="check", role="fast", source="cli", result=res)
        print(json.dumps({"reply": res.text, "request_id": res.request_id, "latency_ms": res.latency_ms,
                          "tokens": [res.input_tokens, res.output_tokens]}))
        _print_usage(ledger)
        return 0

    from .agent import Agent
    from .gateway import SAMPLE_REPO
    agent = Agent(client, router, ledger)
    sess = agent.create_session(args.task, args.repo or str(SAMPLE_REPO))
    print(f"session {sess.id} in {sess.repo_dir} (mode={settings.resolved_mode}, auth={settings.auth_method})")
    agent.run(sess)
    for st in sess.steps:
        if st["type"] == "model":
            print(f"  [bedrock:{st['role']}] {st['model_id']} in={st['tokens_in']} out={st['tokens_out']} acus={st['acus']}")
        elif st["type"] == "tool":
            print(f"  [tool] {st['name']} {json.dumps(st['args'])[:120]}")
        else:
            print(f"  [error] {st['text']}")
    print(f"\nstatus={sess.status}\n\nPlan:\n{sess.plan}\n\nSummary:\n{sess.summary}\n\nDiff:\n{sess.diff}")
    _print_usage(ledger, sess.id)
    return 0 if sess.status == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())
