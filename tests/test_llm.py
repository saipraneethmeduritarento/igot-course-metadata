import asyncio
import json
import re
import threading
from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from src import llm as llm_module
from src.config import RetryPolicy
from src.llm import GeminiProvider, InvalidResponseError, LLMClient, QuotaExhaustedError
from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT
from src.resource_metadata.schema import ResourceMetadata


class Answer(BaseModel):
    value: int


NO_WAIT = RetryPolicy(initial_delay_seconds=0, max_delay_seconds=0)


class FakeProvider:
    """Answers {"value": <the message as an int>}. The message "bad" fails."""

    def generate(self, user_message, json_schema):
        if user_message == "bad":
            raise RuntimeError("boom")
        return json.dumps({"value": int(user_message)})

    async def agenerate(self, user_message, json_schema):
        await asyncio.sleep(0)
        return self.generate(user_message, json_schema)


@pytest.fixture
def client():
    return LLMClient(FakeProvider(), max_concurrency=2, retry=NO_WAIT)


def test_generate_returns_validated_model(client):
    assert client.generate("7", Answer) == Answer(value=7)


def test_generate_rejects_answer_that_breaks_the_schema():
    client = LLMClient(ScriptedProvider({"m": ['{"value": "not a number"}']}), retry=NO_WAIT)
    with pytest.raises(InvalidResponseError, match="does not match Answer"):
        client.generate("m", Answer)


def test_generate_many_keeps_order_and_returns_failures(client):
    results = client.generate_many(["1", "bad", "3"], Answer)
    assert results[0] == Answer(value=1)
    assert isinstance(results[1], RuntimeError)
    assert results[2] == Answer(value=3)


def test_agenerate_many_keeps_order_and_returns_failures(client):
    results = asyncio.run(client.agenerate_many(["1", "bad", "3"], Answer))
    assert results[0] == Answer(value=1)
    assert isinstance(results[1], RuntimeError)
    assert results[2] == Answer(value=3)


def test_system_prompt_has_the_three_reference_data_placeholders():
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", RESOURCE_METADATA_SYSTEM_PROMPT))
    assert placeholders == {"KCM_CATALOGUE", "SGOS_TREE", "DESIGNATION_LIST"}


def test_prompt_example_output_matches_resource_metadata_schema():
    example = re.search(r"```json\n(.*?)\n```", RESOURCE_METADATA_SYSTEM_PROMPT, re.S).group(1)
    ResourceMetadata.model_validate_json(example)


# --- GeminiProvider: an expired prompt cache is replaced and the call is tried again -------------

def client_error(message, code=400):
    return errors.ClientError(code, {"error": {"code": code, "message": message}})


EXPIRED = "Cache content 123 is expired."


class FakeGemini:
    """Stands in for genai.Client. A call made with an expired cache name raises the 400."""

    def __init__(self, expired_caches=("cache-1",), other_error=None):
        self.expired_caches = set(expired_caches)
        self.other_error = other_error
        self.caches_made = 0
        self.calls = []  # the cache name of every generate_content call
        self.models = SimpleNamespace(generate_content=self._generate)
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._agenerate))
        self.caches = SimpleNamespace(create=self._create_cache)

    def _create_cache(self, **kwargs):
        self.caches_made += 1
        return SimpleNamespace(name=f"cache-{self.caches_made + 1}")

    def _answer(self, config):
        self.calls.append(config.cached_content)
        if self.other_error is not None:
            raise self.other_error
        if config.cached_content in self.expired_caches:
            raise client_error(EXPIRED)
        usage = SimpleNamespace(
            prompt_token_count=1, cached_content_token_count=1, candidates_token_count=1
        )
        return SimpleNamespace(text='{"value": 1}', usage_metadata=usage)

    def _generate(self, model, contents, config):
        return self._answer(config)

    async def _agenerate(self, model, contents, config):
        await asyncio.sleep(0)
        return self._answer(config)


def make_provider(fake):
    provider = GeminiProvider.__new__(GeminiProvider)  # skip the real Vertex client
    provider.settings = SimpleNamespace(
        model="m", temperature=0, max_output_tokens=10, cache_ttl_seconds=3600
    )
    provider.system_prompt = "system"
    provider.client = fake
    provider._renew_lock = threading.Lock()
    provider.cache_name = "cache-1"
    return provider


def test_an_expired_cache_is_replaced_and_the_call_is_tried_again():
    fake = FakeGemini()
    provider = make_provider(fake)
    assert provider.generate("hi", {}) == '{"value": 1}'
    assert fake.calls == ["cache-1", "cache-2"]
    assert provider.cache_name == "cache-2"
    provider.generate("again", {})  # the new cache is kept, no second renewal
    assert fake.caches_made == 1


def test_calls_that_fail_together_share_one_new_cache():
    fake = FakeGemini()
    provider = make_provider(fake)

    async def run():
        return await asyncio.gather(*(provider.agenerate(str(i), {}) for i in range(8)))

    assert asyncio.run(run()) == ['{"value": 1}'] * 8
    assert fake.caches_made == 1
    assert provider.cache_name == "cache-2"


def test_another_400_is_not_retried():
    fake = FakeGemini(other_error=client_error("Request contains an invalid argument."))
    provider = make_provider(fake)
    with pytest.raises(errors.ClientError, match="invalid argument"):
        provider.generate("hi", {})
    assert fake.calls == ["cache-1"]
    assert fake.caches_made == 0


def test_a_second_expiry_is_raised_not_retried_forever():
    fake = FakeGemini(expired_caches=("cache-1", "cache-2"))
    provider = make_provider(fake)
    with pytest.raises(errors.ClientError, match="expired"):
        provider.generate("hi", {})
    assert fake.calls == ["cache-1", "cache-2"]


# --- LLMClient: retrying ---------------------------------------------------------------------------

def api_error(code, message="error"):
    error_class = errors.ServerError if code >= 500 else errors.ClientError
    return error_class(code, {"error": {"code": code, "message": message}})


class ScriptedProvider:
    """Answers each message from its script, in turn: an exception is raised, text is returned.
    The last entry repeats. `on_error` is called just before an exception is raised."""

    def __init__(self, scripts, on_error=lambda: None):
        self.scripts = {message: list(script) for message, script in scripts.items()}
        self.calls = []
        self.on_error = on_error

    def _next(self, user_message):
        self.calls.append(user_message)
        script = self.scripts[user_message]
        answer = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(answer, Exception):
            self.on_error()
            raise answer
        return answer

    def generate(self, user_message, json_schema):
        return self._next(user_message)

    async def agenerate(self, user_message, json_schema):
        await asyncio.sleep(0)
        return self._next(user_message)


@pytest.fixture
def clock(monkeypatch):
    """A fake clock: waiting moves it on at once. `clock.waits` lists every wait, in seconds."""

    class Clock:
        now = 0.0
        waits = []

    def sleep(seconds):
        Clock.waits.append(round(seconds, 3))
        Clock.now += seconds

    async def asleep(seconds):
        Clock.waits.append(round(seconds, 3))
        await asyncio.sleep(0)  # let the other calls run, as a real wait would
        Clock.now += seconds

    Clock.waits = []
    monkeypatch.setattr(llm_module, "_now", lambda: Clock.now)
    monkeypatch.setattr(llm_module, "_sleep", sleep)
    monkeypatch.setattr(llm_module, "_asleep", asleep)
    monkeypatch.setattr(llm_module, "_jitter", lambda: 1.0)
    return Clock


POLICY = RetryPolicy(attempts=4, initial_delay_seconds=5, max_delay_seconds=12, quota_give_up_after=2)
OK = '{"value": 1}'


def test_a_server_error_is_retried_with_growing_waits(clock):
    provider = ScriptedProvider({"m": [api_error(503), api_error(500), api_error(503), OK]})
    client = LLMClient(provider, retry=POLICY)

    assert asyncio.run(client.agenerate("m", Answer)) == Answer(value=1)
    assert len(provider.calls) == 4
    assert [w for w in clock.waits if w] == [5, 10, 12]  # doubled, capped at max_delay


def test_network_errors_and_timeouts_are_retried(clock):
    import httpx2

    provider = ScriptedProvider({"m": [httpx2.ConnectError("refused"), TimeoutError(), OK]})
    assert LLMClient(provider, retry=POLICY).generate("m", Answer) == Answer(value=1)
    assert len(provider.calls) == 3


@pytest.mark.parametrize("code", [400, 403, 404])
def test_a_request_error_is_not_retried(clock, code):
    provider = ScriptedProvider({"m": [api_error(code)]})
    call = LLMClient(provider, retry=POLICY).agenerate("m", Answer)
    with pytest.raises(errors.ClientError):
        asyncio.run(call)
    assert provider.calls == ["m"]


def test_an_invalid_answer_is_tried_only_invalid_response_attempts_times(clock):
    provider = ScriptedProvider({"m": ["", "not json"]})
    policy = RetryPolicy(attempts=6, initial_delay_seconds=1, invalid_response_attempts=3)
    call = LLMClient(provider, retry=policy).agenerate("m", Answer)
    with pytest.raises(InvalidResponseError):
        asyncio.run(call)
    assert len(provider.calls) == 3


def test_a_server_error_is_raised_after_the_last_attempt(clock):
    provider = ScriptedProvider({"m": [api_error(503)]})
    call = LLMClient(provider, retry=POLICY).agenerate("m", Answer)
    with pytest.raises(errors.ServerError):
        asyncio.run(call)
    assert len(provider.calls) == POLICY.attempts


def test_a_quota_error_pauses_every_call_not_only_its_own(clock):
    async def run():
        got_429 = asyncio.Event()
        provider = ScriptedProvider({"a": [api_error(429), OK], "b": [OK]}, on_error=got_429.set)
        client = LLMClient(provider, retry=POLICY)

        async def b_after_the_429():
            await got_429.wait()
            return await client.agenerate("b", Answer)

        return await asyncio.gather(client.agenerate("a", Answer), b_after_the_429())

    assert asyncio.run(run()) == [Answer(value=1), Answer(value=1)]
    # "a" waits out the pause, and so does "b", which never got a 429 itself.
    assert [w for w in clock.waits if w] == [5, 5]


def test_the_run_gives_up_when_the_quota_stays_exhausted(clock):
    provider = ScriptedProvider({"m": [api_error(429)]})
    client = LLMClient(provider, retry=POLICY)

    first, second, third = (client.agenerate("m", Answer) for _ in range(3))
    with pytest.raises(errors.ClientError):  # first request out of attempts: it alone fails
        asyncio.run(first)
    with pytest.raises(QuotaExhaustedError):  # second in a row: quota_give_up_after reached
        asyncio.run(second)
    calls = len(provider.calls)
    with pytest.raises(QuotaExhaustedError):  # from now on the model is not called at all
        asyncio.run(third)
    assert len(provider.calls) == calls == 2 * POLICY.attempts


def test_a_success_resets_the_quota_give_up_count(clock):
    provider = ScriptedProvider({"quota": [api_error(429)], "ok": [OK]})
    client = LLMClient(provider, retry=POLICY)
    for _ in range(3):
        with pytest.raises(errors.ClientError):
            client.generate("quota", Answer)
        client.generate("ok", Answer)
    assert not client.quota_exhausted


def test_the_sdk_does_not_retry_on_its_own(monkeypatch):
    """LLMClient does the retrying; SDK retries on top would multiply the attempts."""
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.caches = SimpleNamespace(create=lambda **_: SimpleNamespace(name="cache-1"))

    monkeypatch.setattr(llm_module.genai, "Client", FakeClient)
    settings = SimpleNamespace(
        project="p", location="global", timeout_seconds=1, model="m", cache_ttl_seconds=60
    )
    GeminiProvider(settings, "system")
    assert captured["http_options"].retry_options.attempts == 1
