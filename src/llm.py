"""LLM client: send a user message, get back a validated Pydantic object.

    provider = GeminiProvider(settings, system_prompt)
    llm = LLMClient(provider, settings.max_concurrency, settings.retry)

    result = llm.generate(user_message, ResourceMetadata)
    results = await llm.agenerate_many(user_messages, ResourceMetadata)

LLMClient retries the failures that can go away (quota, server errors, timeouts, an empty or
off-schema answer) as RetryPolicy in src/config.py says. When the quota stays exhausted it raises
QuotaExhaustedError, for this call and every later one, so the run can stop.

To use another vendor, write a class with the two methods of `LLMProvider`.
"""

import asyncio
import logging
import random
import threading
import time
from collections import Counter
from typing import Protocol, Sequence, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from src.config import LLMSettings, RetryPolicy

try:  # the HTTP library google-genai sends requests with
    import httpx2

    NETWORK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, httpx2.TransportError)
except ImportError:  # pragma: no cover - another google-genai version
    NETWORK_ERRORS = (TimeoutError, ConnectionError)

logger = logging.getLogger(__name__)

Output = TypeVar("Output", bound=BaseModel)

QUOTA_STATUS_CODE = 429
# Request timeout, quota, and server errors: asking again later can work.
RETRYABLE_STATUS_CODES = frozenset({408, QUOTA_STATUS_CODE, 500, 502, 503, 504})

# What went wrong with a call, for deciding whether and how to retry it.
QUOTA = "quota"
TRANSIENT = "transient"
INVALID_ANSWER = "invalid answer"


class InvalidResponseError(ValueError):
    """The model's answer was empty or did not match the output model."""


class QuotaExhaustedError(Exception):
    """The model quota stayed exhausted through every retry. Stop the run; a later run carries on."""


def classify_failure(error: Exception) -> str | None:
    """QUOTA, TRANSIENT or INVALID_ANSWER for a failure worth retrying; None for any other."""
    if isinstance(error, errors.APIError):
        if error.code == QUOTA_STATUS_CODE:
            return QUOTA
        return TRANSIENT if error.code in RETRYABLE_STATUS_CODES else None
    if isinstance(error, NETWORK_ERRORS):
        return TRANSIENT
    if isinstance(error, InvalidResponseError):
        return INVALID_ANSWER
    return None


# Replaced in tests, so that no test really waits.
_now = time.monotonic
_sleep = time.sleep
_asleep = asyncio.sleep


def _jitter() -> float:
    """A random factor for a wait, so that calls paused together do not all retry at once."""
    return random.uniform(0.8, 1.2)


class LLMProvider(Protocol):
    """Turns a user message into JSON text that follows the given JSON schema."""

    def generate(self, user_message: str, json_schema: dict) -> str: ...

    async def agenerate(self, user_message: str, json_schema: dict) -> str: ...


class GeminiProvider:
    """Gemini on Vertex AI. The system prompt is cached and reused by every call.

    The cache lives `cache_ttl_seconds`. A call that finds it expired makes a new cache and is
    tried again once, so a run longer than the TTL keeps working.
    """

    def __init__(self, settings: LLMSettings, system_prompt: str):
        self.settings = settings
        self.system_prompt = system_prompt
        self.client = genai.Client(
            enterprise=True,
            project=settings.project,
            location=settings.location,
            http_options=types.HttpOptions(
                timeout=settings.timeout_seconds * 1000,
                # One attempt: LLMClient does the retrying, with longer waits than the SDK's.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._renew_lock = threading.Lock()
        self.cache_name = self._create_cache()

    def generate(self, user_message: str, json_schema: dict) -> str:
        cache_name = self.cache_name
        try:
            return self._generate_once(user_message, json_schema, cache_name)
        except errors.ClientError as error:
            if not self._is_cache_expired(error):
                raise
            self._renew_cache(cache_name)
            return self._generate_once(user_message, json_schema, self.cache_name)

    async def agenerate(self, user_message: str, json_schema: dict) -> str:
        cache_name = self.cache_name
        try:
            return await self._agenerate_once(user_message, json_schema, cache_name)
        except errors.ClientError as error:
            if not self._is_cache_expired(error):
                raise
            self._renew_cache(cache_name)
            return await self._agenerate_once(user_message, json_schema, self.cache_name)

    def _generate_once(self, user_message: str, json_schema: dict, cache_name: str | None) -> str:
        response = self.client.models.generate_content(
            model=self.settings.model,
            contents=user_message,
            config=self._build_config(json_schema, cache_name),
        )
        self._log_usage(response)
        return response.text

    async def _agenerate_once(
        self, user_message: str, json_schema: dict, cache_name: str | None
    ) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.settings.model,
            contents=user_message,
            config=self._build_config(json_schema, cache_name),
        )
        self._log_usage(response)
        return response.text

    @staticmethod
    def _is_cache_expired(error: errors.ClientError) -> bool:
        return error.code == 400 and "expired" in (error.message or "").lower()

    def _renew_cache(self, expired_name: str | None) -> None:
        """Make a new cache, unless another call already replaced `expired_name`.

        Many calls run at once and all fail together when the cache expires; only the first
        makes the new cache, the others reuse it.
        """
        with self._renew_lock:
            if self.cache_name != expired_name:
                return
            logger.warning("Cache %s expired. Creating a new one.", expired_name)
            self.cache_name = self._create_cache()

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

    def _build_config(self, json_schema: dict, cache_name: str | None) -> types.GenerateContentConfig:
        # A cached system prompt must not be sent again in the request.
        system_instruction = None if cache_name else self.system_prompt
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            cached_content=cache_name,
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
    """Calls a provider, validates its answer against a Pydantic model, and retries the failures
    that can go away (see RetryPolicy).

    A quota error (429) pauses every call made through this client until its wait is over, so
    calls running together do not keep hitting the exhausted quota. When `quota_give_up_after`
    requests in a row have used every attempt on quota errors, the client raises
    QuotaExhaustedError, and from then on raises it for every call without calling the model.
    """

    def __init__(
        self, provider: LLMProvider, max_concurrency: int = 4, retry: RetryPolicy | None = None
    ):
        self.provider = provider
        self.max_concurrency = max_concurrency
        self.retry = retry or RetryPolicy()
        self.quota_exhausted = False
        self._paused_until = 0.0  # no call starts before this _now() time (set by a 429)
        self._quota_give_ups = 0  # requests in a row that used every attempt on quota errors

    def generate(self, user_message: str, output_model: type[Output]) -> Output:
        json_schema = output_model.model_json_schema()
        failures: Counter[str] = Counter()
        attempt = 0
        while True:
            self._wait_out_pause_sync()
            attempt += 1
            try:
                result = self._parse(self.provider.generate(user_message, json_schema), output_model)
            except Exception as error:
                delay = self._retry_delay(error, attempt, failures)
                if delay is None:
                    raise
                _sleep(delay)
                continue
            self._quota_give_ups = 0
            return result

    async def agenerate(self, user_message: str, output_model: type[Output]) -> Output:
        json_schema = output_model.model_json_schema()
        failures: Counter[str] = Counter()
        attempt = 0
        while True:
            await self._wait_out_pause()
            attempt += 1
            try:
                text = await self.provider.agenerate(user_message, json_schema)
                result = self._parse(text, output_model)
            except Exception as error:
                delay = self._retry_delay(error, attempt, failures)
                if delay is None:
                    raise
                await _asleep(delay)
                continue
            self._quota_give_ups = 0
            return result

    # --- Retrying --------------------------------------------------------------------------

    def _pause_left(self) -> float:
        """Seconds until calls may start again; raises QuotaExhaustedError once given up."""
        if self.quota_exhausted:
            raise QuotaExhaustedError("The model quota is exhausted; not calling the model.")
        return max(0.0, self._paused_until - _now())

    def _wait_out_pause_sync(self) -> None:
        while (left := self._pause_left()) > 0:
            _sleep(left)

    async def _wait_out_pause(self) -> None:
        # Checked again after each wait: another call may have made the pause longer.
        while (left := self._pause_left()) > 0:
            await _asleep(left)

    def _retry_delay(self, error: Exception, attempt: int, failures: Counter[str]) -> float | None:
        """Seconds to wait before trying again, or None if `error` must be raised as it is.

        Raises QuotaExhaustedError when this request makes `quota_give_up_after` in a row that
        used every attempt on quota errors.
        """
        kind = classify_failure(error)
        if kind is None:
            return None
        failures[kind] += 1
        out_of_attempts = attempt >= self.retry.attempts or (
            kind == INVALID_ANSWER and failures[kind] >= self.retry.invalid_response_attempts
        )
        if out_of_attempts:
            logger.warning("Giving up after %d attempts: %s", attempt, describe(error))
            if kind == QUOTA:
                self._count_quota_give_up(error)
            return None

        delay = self._backoff(attempt)
        if kind == QUOTA:
            self._paused_until = max(self._paused_until, _now() + delay)
            logger.warning(
                "Quota exhausted (429): pausing every call for %.0fs (attempt %d of %d).",
                delay,
                attempt,
                self.retry.attempts,
            )
            return 0.0  # the shared pause is waited out before the next attempt
        logger.warning(
            "%s. Retrying in %.0fs (attempt %d of %d).",
            describe(error),
            delay,
            attempt,
            self.retry.attempts,
        )
        return delay

    def _backoff(self, attempt: int) -> float:
        """initial_delay, doubled after each attempt, with jitter, at most max_delay."""
        delay = self.retry.initial_delay_seconds * 2 ** (attempt - 1) * _jitter()
        return min(self.retry.max_delay_seconds, delay)

    def _count_quota_give_up(self, error: Exception) -> None:
        self._quota_give_ups += 1
        if self._quota_give_ups >= self.retry.quota_give_up_after:
            self.quota_exhausted = True
            raise QuotaExhaustedError(
                f"{self._quota_give_ups} requests in a row failed on quota after "
                f"{self.retry.attempts} attempts each."
            ) from error

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
            raise InvalidResponseError("The model returned no text.")
        try:
            return output_model.model_validate_json(text)
        except ValidationError as error:
            raise InvalidResponseError(
                f"The answer does not match {output_model.__name__}: {error}"
            ) from error


def describe(error: Exception, limit: int = 300) -> str:
    """A one-line description of `error` for the log."""
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
