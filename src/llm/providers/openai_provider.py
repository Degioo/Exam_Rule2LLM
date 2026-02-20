"""
src/llm/providers/openai_provider.py
─────────────────────────────────────
OpenAI Chat Completions provider (requests-based, no SDK required).
Reads OPENAI_API_KEY, OPENAI_MODEL, RUN_TEMPERATURE, MAX_TOKENS, RUN_SEED
from environment / .env — override via constructor kwargs.
"""
from __future__ import annotations

import os
import requests

from ..base import LLMClient


class OpenAIProvider(LLMClient):
    """Calls OpenAI Chat Completions API using plain requests."""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 800,
        seed: int = 42,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to .env or export it as an "
                "environment variable."
            )

    def generate(self, system: str, user: str) -> str:
        """Send a chat request and return the assistant reply text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "response_format": {"type": "json_object"},  # JSON mode where supported
        }

        resp = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"]
