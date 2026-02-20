from .base import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Add your OpenAI API call here.")
