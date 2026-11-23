# Security and deployment (for the partner SA)

| Question | Answer in this design |
|---|---|
| Where does inference run? | Only Amazon Bedrock, in the configured region (cross-region inference profiles stay inside a geography, e.g. `us.`). The connector has no other model endpoints. |
| Is customer code used for training? | Bedrock does not store prompts or completions or use them to train models. |
| Auth to Bedrock | Production: IAM role (least privilege, `deploy/iam-policy.json`). Demo: short-lived Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`). |
| Network | Bedrock reached via a VPC interface endpoint (`com.amazonaws.<region>.bedrock-runtime`); no internet egress needed for inference. |
| Content controls | Optional Bedrock Guardrail on every call (`BEDROCK_GUARDRAIL_ID`). |
| Audit | CloudTrail logs Bedrock API calls; optional Bedrock model-invocation logging to S3/CloudWatch; local JSONL ledger (`LEDGER_PATH`) keyed by Bedrock request id. |
| Agent sandbox isolation | Tool commands run with `AWS_*` variables removed, so code the agent runs cannot reach the Bedrock key. Path traversal outside the repo is blocked. |
| Gateway access | Optional `GATEWAY_API_KEY`; production would use IAM SigV4 / Cognito / IAM Identity Center. |

Deployment notes: `serve` refuses to bind a non-loopback host in live mode unless `GATEWAY_API_KEY` is set.
Terminate TLS in front of the gateway (ALB / API Gateway / VPC Lattice); the container itself speaks plain HTTP.
Agent sessions only accept repos under `AGENT_REPO_ROOT`. The demo runs agent tool commands as the gateway
process user; production runs each session in its own isolated sandbox (e.g. a Firecracker microVM / AgentCore Runtime).

## Customer Bedrock credentials (AgentCore Identity)

- A customer's Bedrock API key is sent once to `PUT /v1/org/bedrock-key` and written to
  AgentCore Identity (token vault, encrypted with an AWS-owned or customer KMS key). Devin's
  database keeps only the last four characters. The key is never returned by any endpoint.
- At run time the key is fetched with a workload access token bound to the signed-in user,
  and used only for that tenant's Bedrock client. Agent subprocesses still get no `AWS_*`
  environment variables.
- Preferred for enterprises: a cross-account IAM role (`deploy/customer-bedrock-role.yaml`)
  with an ExternalId, so there is no long-lived key and every call is in the customer's CloudTrail.
- Demo login (`/v1/auth/login`) maps an email domain to an org with no password. Replace it
  with IAM Identity Center / OIDC before any shared deployment.
