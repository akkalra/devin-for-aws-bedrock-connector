"""Create (or update) the Devin agent on Amazon Bedrock AgentCore Runtime.

    python deploy/agentcore/deploy.py --image <acct>.dkr.ecr.us-east-1.amazonaws.com/devin-agent:latest \
        --role-arn arn:aws:iam::<acct>:role/DevinAgentCoreRuntime

Then run the gateway with AGENTCORE_RUNTIME_ARN=<printed arn> and CREDENTIAL_VAULT=agentcore.
"""
import argparse

import boto3


def runtime_request(name: str, image: str, role_arn: str, region: str) -> dict:
    return {
        "agentRuntimeName": name,
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": image}},
        "roleArn": role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "environmentVariables": {"BEDROCK_MODE": "live", "CREDENTIAL_VAULT": "agentcore", "AWS_REGION": region},
        "description": "Devin agent loop; inference on the customer's Amazon Bedrock via AgentCore Identity",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--role-arn", required=True)
    p.add_argument("--name", default="devin_bedrock_agent")
    p.add_argument("--region", default="us-east-1")
    a = p.parse_args()
    ctl = boto3.client("bedrock-agentcore-control", region_name=a.region)
    req = runtime_request(a.name, a.image, a.role_arn, a.region)
    existing = next((r for r in ctl.list_agent_runtimes()["agentRuntimes"] if r["agentRuntimeName"] == a.name), None)
    if existing:
        req.pop("agentRuntimeName")
        out = ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **req)
    else:
        out = ctl.create_agent_runtime(**req)
    print(out["agentRuntimeArn"])


if __name__ == "__main__":
    main()
