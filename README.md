# Devin for AWS — Amazon Bedrock connector (demo)

A working demo of the **"Devin for AWS"** backend: a Devin-style autonomous coding agent
where **every model call goes to Amazon Bedrock**. When you sign in to Devin, your org
defaults to **your own Bedrock API key** (held in **AgentCore Identity**) or a cross-account
IAM role, while **Devin manages model routing** (which model per task, plus fallbacks) in
the backend. Agent sessions can run on **AgentCore Runtime**.
Each call is metered into **ACUs**, and the dashboard splits ACU revenue into AWS revenue
share, Cognition's AWS infra bill (Bedrock + runtime + other AWS), and total AWS capture.

> Demo for an AWS × Cognition partnership discussion. The ACU formula and economics are
> illustrative ([docs/economics.md](docs/economics.md)), not Cognition's real pricing model.

See [docs/architecture.md](docs/architecture.md) for the diagram.

## What it shows

0. **Devin + Bedrock + AgentCore backend.** Sign in (simulated IAM Identity Center SSO) →
   org settings default to *your Amazon Bedrock API key*, stored as an AgentCore Identity
   API-key credential provider (`devin-<org>-bedrock`) and fetched at run time with a workload
   access token for the signed-in user. Alternatives: a cross-account IAM role
   (`deploy/customer-bedrock-role.yaml`, `sts:AssumeRole` + ExternalId) or Devin-managed
   Bedrock. Devin's router picks the model per task and falls back on throttling, capacity
   or model-access errors. Sessions run in-process for the demo or on AgentCore Runtime
   (`Dockerfile.agentcore`, `deploy/agentcore/deploy.py`, `AGENTCORE_RUNTIME_ARN`).
1. **Bedrock as the only inference backend.** `bedrock-runtime:Converse` for planning,
   coding (tool use) and review, with role-based model routing (Sonnet for plan/code, Haiku
   for review/fast). Any other model name a client asks for is still served by Bedrock.
2. **Drop-in connector.** OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`
   endpoints (tool calls + streaming) let existing agents and SDKs run on Bedrock
   without code changes.
3. **ACU metering → AWS consumption.** Every call is recorded with the Bedrock request
   id, tokens, cost and ACUs. Value pools are kept apart to avoid double counting.
4. **Enterprise controls.** IAM / Bedrock API key auth, optional Bedrock Guardrails,
   least-privilege IAM policy, and agent sandbox commands that cannot see the AWS key
   ([docs/security.md](docs/security.md)).

## Quick start

```bash
pip install -e ".[dev]"

# Live: a Bedrock API key (Bedrock console → API keys) or any IAM credentials
export AWS_BEARER_TOKEN_BEDROCK=...      # or: aws sso login / instance role
export AWS_REGION=us-east-1

python -m bedrock_connector check        # one Bedrock call; prints request id + ACUs
python -m bedrock_connector agent "Fix the failing tests"   # agent fixes examples/sample_repo
python -m bedrock_connector serve --port 8080               # dashboard at http://localhost:8080
```

Without credentials it runs in **mock mode** (`BEDROCK_MODE=mock`): same flow, canned responses.

### Use it as a connector for any OpenAI / Anthropic client

```bash
# Anthropic SDK / Claude-compatible agents
export ANTHROPIC_BASE_URL=http://localhost:8080
# OpenAI SDK-compatible agents
export OPENAI_BASE_URL=http://localhost:8080/v1

curl -s localhost:8080/v1/messages -H 'content-type: application/json' -d '{
  "model": "devin-coder", "max_tokens": 200,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a list."}]}'
```

Model names: `devin-planner`, `devin-coder`, `devin-reviewer`, `devin-fast`, or any Bedrock
model id / inference profile (e.g. `amazon.nova-pro-v1:0`). Responses include
`x_bedrock.request_id` (OpenAI format) so each call can be matched to CloudTrail.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/auth/login` | Simulated SSO `{email}` → session token (`x-devin-session` header) |
| `GET /v1/me` | Signed-in user, org credential settings, effective routing |
| `PUT /v1/org/bedrock-key` | Store the org's Bedrock API key in the vault (never returned) |
| `PUT /v1/org/settings` | Credential source, IAM role ARN, region, Devin-managed routing or pinned models |
| `POST /v1/org/test-connection` | One Bedrock call with the org's credentials |
| `POST /v1/chat/completions` | OpenAI-compatible, served by Bedrock Converse |
| `POST /v1/messages` | Anthropic-compatible, served by Bedrock Converse |
| `POST /v1/agent/sessions` | Start an agent session `{task, repo?}` |
| `GET /v1/agent/sessions/{id}` | Steps, diff, summary, ACU usage |
| `GET /v1/stats` | Ledger summary, value pools, recent calls |
| `GET /healthz` | Mode, auth method, region, role → model map |
| `GET /` | Live dashboard |

## Configuration

See [.env.example](.env.example). Main settings: `AWS_REGION`, `BEDROCK_MODE`,
`BEDROCK_MODEL_{PLANNER,CODER,REVIEWER,FAST}`, `BEDROCK_GUARDRAIL_ID`, `GATEWAY_API_KEY`,
`LEDGER_PATH`, `CREDENTIAL_VAULT` (`local` | `agentcore`), `AGENTCORE_WORKLOAD_NAME`,
`AGENTCORE_RUNTIME_ARN`, `BEDROCK_FALLBACKS_JSON`, `MOCK_UNAVAILABLE_MODELS` (demo a fallback
offline), and the economics parameters.

### Run against real AgentCore

```bash
# 1. Agent container on AgentCore Runtime (arm64)
docker buildx build --platform linux/arm64 -f Dockerfile.agentcore -t $ECR/devin-agent:latest --push .
python deploy/agentcore/deploy.py --image $ECR/devin-agent:latest \
    --role-arn arn:aws:iam::$ACCT:role/DevinAgentCoreRuntime   # policy: deploy/agentcore/runtime-role-policy.json
# 2. Devin backend: keys in AgentCore Identity, sessions on AgentCore Runtime
CREDENTIAL_VAULT=agentcore AGENTCORE_RUNTIME_ARN=<arn from step 1> GATEWAY_API_KEY=... \
    bedrock-connector serve --host 0.0.0.0
```

## Docs

- [Architecture](docs/architecture.md): request vs money flows, production additions (VPC endpoints, AgentCore, Marketplace metering)
- [Economics](docs/economics.md): ACU → value-pool math with a worked example
- [Security](docs/security.md): answers for the partner solutions architect
- Companion demo: CUDA → Trainium migration by Devin on [akkalra/cuda-inference-service](https://github.com/akkalra/cuda-inference-service)

## Tests

```bash
pytest -q   # mock mode: translation, metering, routing, gateway, agent loop
```
