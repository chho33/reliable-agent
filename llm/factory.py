"""Model specs ("provider:model-id"), config/models.yaml, pricing and .env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm.anthropic import AnthropicChat
from llm.openai_compat import OpenAICompatChat
from llm.types import ChatModel, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "models.yaml"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing vars."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class ModelConfig:
    spec: str
    provider: str
    model: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    options: dict[str, Any]


def resolve(spec: str, config: dict[str, Any] | None = None) -> ModelConfig:
    if ":" not in spec:
        raise ValueError(f"model spec must look like 'anthropic:<model-id>' or 'openai:<model-id>', got {spec!r}")
    provider, model = spec.split(":", 1)
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"unknown provider {provider!r} (expected anthropic or openai)")
    config = config if config is not None else load_config()
    entry = (config.get("models") or {}).get(spec) or {}
    options = {**(config.get("defaults") or {}), **{k: v for k, v in entry.items() if k not in ("input_per_mtok", "output_per_mtok")}}
    return ModelConfig(spec, provider, model, entry.get("input_per_mtok"), entry.get("output_per_mtok"), options)


def cost_usd(mc: ModelConfig, usage: Usage) -> float | None:
    """USD cost from the config price table, or None if prices are not filled in."""
    if mc.input_per_mtok is None or mc.output_per_mtok is None:
        return None
    return (usage.input_tokens * mc.input_per_mtok + usage.output_tokens * mc.output_per_mtok) / 1_000_000


def make_chat(spec: str, config: dict[str, Any] | None = None) -> ChatModel:
    mc = resolve(spec, config)
    o = mc.options
    common: dict[str, Any] = {"max_tokens": o.get("max_tokens", 1024), "temperature": o.get("temperature"),
                              "timeout": o.get("timeout", 120.0)}
    if mc.provider == "anthropic":
        return AnthropicChat(mc.model, api_key_env=o.get("api_key_env", "ANTHROPIC_API_KEY"),
                             base_url=o.get("base_url", "https://api.anthropic.com"), **common)
    return OpenAICompatChat(
        mc.model, base_url=o.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key_env=o.get("api_key_env", "OPENAI_API_KEY"), max_tokens_field=o.get("max_tokens_field", "max_tokens"),
        require_key=o.get("require_key", True), extra_body=o.get("extra_body"), **common,
    )
