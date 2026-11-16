"""LLM client: send a user message, get back a validated Pydantic object.

    provider = GeminiProvider(settings, system_prompt)
    llm = LLMClient(provider, settings.max_concurrency)

    result = llm.generate(user_message, ResourceMetadata)
    results = await llm.agenerate_many(user_messages, ResourceMetadata)

To use another vendor, write a class with the two methods of `LLMProvider`.
"""

import asyncio
import logging
from typing import Protocol, Sequence, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from src.config import LLMSettings

logger = logging.getLogger(__name__)

Output = TypeVar("Output", bound=BaseModel)


class LLMProvider(Protocol):
    """Turns a user message into JSON text that follows the given JSON schema."""

    def generate(self, user_message: str, json_schema: dict) -> str: ...

    async def agenerate(self, user_message: str, json_schema: dict) -> str: ...


class GeminiProvider:
    """Gemini on Vertex AI. The system prompt is cached once and reused by every call."""

    def __init__(self, settings: LLMSettings, system_prompt: str):
        self.settings = settings
        self.system_prompt = system_prompt
        self.client = genai.Client(
            enterprise=True,
            project=settings.project,
            location=settings.location,
            http_options=types.HttpOptions(
                timeout=settings.timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=settings.max_attempts),
            ),
        )
        self.cache_name = self._create_cache()

    def generate(self, user_message: str, json_schema: dict) -> str:
        response = self.client.models.generate_content(
            model=self.settings.model,
            contents=user_message,
            config=self._build_config(json_schema),
        )
        self._log_usage(response)
        return response.text

    async def agenerate(self, user_message: str, json_schema: dict) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.settings.model,
            contents=user_message,
            config=self._build_config(json_schema),
        )
        self._log_usage(response)
        return response.text

    def _create_cache(self) -> str | None:
        """Cache the system prompt. Returns None if the prompt is too small to cache."""
        try:
            cache = self.client.caches.create(
                model=self.settings.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_prompt,
                    ttl=f"{self.settings.cache_ttl_seconds}s",
                ),
            )
        except errors.ClientError as error:
            if error.code != 400:
                raise
            logger.warning("System prompt not cached (%s). Sending it with every call.", error.message)
            return None
        return cache.name

    def _build_config(self, json_schema: dict) -> types.GenerateContentConfig:
        # A cached system prompt must not be sent again in the request.
        system_instruction = None if self.cache_name else self.system_prompt
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            cached_content=self.cache_name,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=json_schema,
        )

    @staticmethod
    def _log_usage(response: types.GenerateContentResponse) -> None:
        usage = response.usage_metadata
        logger.info(
            "Tokens: prompt=%s cached=%s output=%s",
            usage.prompt_token_count,
            usage.cached_content_token_count,
            usage.candidates_token_count,
        )


class LLMClient:
    """Calls a provider and validates its answer against a Pydantic model."""

    def __init__(self, provider: LLMProvider, max_concurrency: int = 4):
        self.provider = provider
        self.max_concurrency = max_concurrency

    def generate(self, user_message: str, output_model: type[Output]) -> Output:
        text = self.provider.generate(user_message, output_model.model_json_schema())
        return self._parse(text, output_model)

    async def agenerate(self, user_message: str, output_model: type[Output]) -> Output:
        text = await self.provider.agenerate(user_message, output_model.model_json_schema())
        return self._parse(text, output_model)

    def generate_many(self, user_messages: Sequence[str], output_model: type[Output]) -> list[Output | Exception]:
        """One at a time. A failed message gives its exception; the rest still run."""
        results: list[Output | Exception] = []
        for user_message in user_messages:
            try:
                results.append(self.generate(user_message, output_model))
            except Exception as error:
                results.append(error)
        return results

    async def agenerate_many(
        self, user_messages: Sequence[str], output_model: type[Output]
    ) -> list[Output | Exception]:
        """Several at a time (up to max_concurrency). A failed message gives its exception."""
        limit = asyncio.Semaphore(self.max_concurrency)

        async def generate_one(user_message: str) -> Output:
            async with limit:
                return await self.agenerate(user_message, output_model)

        return await asyncio.gather(*(generate_one(m) for m in user_messages), return_exceptions=True)

    @staticmethod
    def _parse(text: str | None, output_model: type[Output]) -> Output:
        if not text:
            raise ValueError("The model returned no text.")
        return output_model.model_validate_json(text)
