"""Maps agent roles and client-requested model names to Bedrock model ids."""
from __future__ import annotations

from .config import Settings

ROLE_ALIASES = {
    "devin-planner": "planner",
    "devin-coder": "coder",
    "devin-reviewer": "reviewer",
    "devin-fast": "fast",
}


def is_bedrock_model_id(name: str) -> bool:
    head = name.split(".", 1)[0]
    if head in ("us", "eu", "apac", "global", "jp", "au", "ca") and "." in name:
        head = name.split(".", 2)[1]
    return head in {"anthropic", "amazon", "meta", "mistral", "cohere", "ai21", "deepseek",
                    "openai", "qwen", "writer", "twelvelabs", "stability", "moonshot", "minimax"} \
        or name.startswith("arn:aws:bedrock")


def geo_profile(model_id: str, region: str) -> str:
    """Point a `us.` cross-region inference profile at the org region's geography."""
    geo = {"eu": "eu", "ap": "apac"}.get(region.split("-", 1)[0], "us")
    return f"{geo}.{model_id[3:]}" if model_id.startswith("us.") else model_id


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def for_role(self, role: str) -> str:
        return self.settings.role_models.get(role, self.settings.role_models["coder"])

    def candidates(self, role: str, org=None) -> list[str]:
        """Ordered Bedrock model ids for a role: Devin's primary pick, then fallbacks.

        An org that turns off Devin-managed routing and pins models for a role
        gets exactly its pinned list.
        """
        if org is not None and not org.devin_managed_routing and org.routing_overrides.get(role):
            return list(org.routing_overrides[role])
        chain = [self.for_role(role)] + self.settings.fallbacks.get(role, [])
        if org is not None:
            chain = [geo_profile(m, org.region) for m in chain]
        return list(dict.fromkeys(chain))

    def resolve(self, requested: str | None) -> tuple[str, str]:
        """Return (role, bedrock_model_id) for any model name a client sends.

        Bedrock ids pass through. `devin-*` aliases pick a role. Anything else
        (e.g. `gpt-4o`, `claude-3-5-sonnet-latest`) is served by the coder
        model on Bedrock, so unmodified clients still never leave AWS.
        """
        name = (requested or "").strip()
        if name in ROLE_ALIASES:
            role = ROLE_ALIASES[name]
            return role, self.for_role(role)
        if name and is_bedrock_model_id(name):
            return "passthrough", name
        lowered = name.lower()
        role = "fast" if any(k in lowered for k in ("haiku", "mini", "nano", "flash", "lite")) else "coder"
        return role, self.for_role(role)
