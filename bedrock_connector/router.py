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


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def for_role(self, role: str) -> str:
        return self.settings.role_models.get(role, self.settings.role_models["coder"])

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
