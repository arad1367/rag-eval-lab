"""A thin wrapper around any OpenAI-compatible chat API.

Concepts:
- Groq, Ollama, vLLM, Together, OpenAI ... all speak the same "chat completions" API,
  so changing provider = changing base_url, api_key and model in .env. No code change.
- temperature=0 makes answers as repeatable as possible (important for evaluation).
- We record token usage: tokens are what you pay for and what drives latency.
"""

from dataclasses import dataclass

from openai import OpenAI

from rag_lab.config import settings


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model=None):
        api_key = api_key or settings.llm_api_key
        if not api_key:
            raise RuntimeError("LLM_API_KEY is empty. Copy .env.example to .env and add a key.")
        self.model = model or settings.llm_model
        # max_retries: the SDK waits and retries automatically on rate limits (HTTP 429).
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url, api_key=api_key, max_retries=5, timeout=60
        )

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        usage = response.usage
        return LLMResponse(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
