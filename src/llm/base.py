"""Abstract LLM client interface."""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Call the model and return raw text output."""
        raise NotImplementedError
