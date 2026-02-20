"""
src/llm/providers/__init__.py
─────────────────────────────
Factory that returns the right LLMClient for a given provider name.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_provider(name: str, **kwargs):
    """
    Parameters
    ----------
    name : "openai" | "ollama"
    **kwargs : override any env-based default (model, base_url, etc.)

    Returns
    -------
    LLMClient instance
    """
    name = name.lower()

    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            model=kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4.1")),
            temperature=float(kwargs.get("temperature", os.getenv("RUN_TEMPERATURE", 0))),
            max_tokens=int(kwargs.get("max_tokens", os.getenv("MAX_TOKENS", 800))),
            seed=int(kwargs.get("seed", os.getenv("RUN_SEED", 42))),
        )

    elif name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(
            model=kwargs.get("model", os.getenv("OLLAMA_MODEL", "mistral:latest")),
            base_url=kwargs.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            temperature=float(kwargs.get("temperature", os.getenv("RUN_TEMPERATURE", 0))),
            max_tokens=int(kwargs.get("max_tokens", os.getenv("MAX_TOKENS", 800))),
        )

    else:
        raise ValueError(
            f"Unknown provider '{name}'. Supported: 'openai', 'ollama'."
        )
