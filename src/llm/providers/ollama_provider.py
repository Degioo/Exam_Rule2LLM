"""
src/llm/providers/ollama_provider.py
─────────────────────────────────────
Ollama local LLM provider — calls /api/chat (multi-turn, system message support).
Reads OLLAMA_BASE_URL, OLLAMA_MODEL, RUN_TEMPERATURE, MAX_TOKENS from .env.
"""
from __future__ import annotations

import os
import requests

from ..base import LLMClient


class OllamaProvider(LLMClient):
    """
    Calls Ollama's /api/chat endpoint.

    Requires:
    - Ollama running: `ollama serve`
    - Model pulled:   `ollama pull <model>`

    Default base URL: http://localhost:11434
    """

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral:latest")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        """Send system + user messages and return assistant reply."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # Ollama /api/chat response format:
        # {"message": {"role": "assistant", "content": "..."}, ...}
        return data["message"]["content"]
