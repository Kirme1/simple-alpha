"""Registry of selectable models. Cloud providers only appear when their API key is set."""

import os
from dataclasses import dataclass

from .providers import AnthropicProvider, LLMProvider, OpenAIProvider

LOCAL_ID = "local"

ANTHROPIC_MODELS = [
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-haiku-5-5", "Claude Haiku 5.5"),
]

# Override with OPENAI_MODELS="model-a,model-b" to match what your account can use.
DEFAULT_OPENAI_MODELS = "gpt-5,gpt-5-mini"


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    provider: str
    model: str

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "provider": self.provider}


def available_models() -> list[ModelOption]:
    models = [ModelOption(LOCAL_ID, "Local (llama-server)", "local", "local-model")]

    if os.getenv("OPENAI_API_KEY"):
        names = os.getenv("OPENAI_MODELS", DEFAULT_OPENAI_MODELS).split(",")
        models += [
            ModelOption(f"openai:{name.strip()}", name.strip(), "openai", name.strip())
            for name in names
            if name.strip()
        ]

    if os.getenv("ANTHROPIC_API_KEY"):
        models += [
            ModelOption(f"anthropic:{name}", label, "anthropic", name)
            for name, label in ANTHROPIC_MODELS
        ]

    return models


def build_provider(option: ModelOption, local_url: str) -> LLMProvider:
    if option.provider == "anthropic":
        return AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model=option.model)
    if option.provider == "openai":
        return OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),  # None -> api.openai.com
            api_key=os.environ["OPENAI_API_KEY"],
            model=option.model,
        )
    return OpenAIProvider(base_url=local_url, api_key="not-needed", model=option.model)
