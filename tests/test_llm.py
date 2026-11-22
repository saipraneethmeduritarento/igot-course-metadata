import asyncio
import json
import re
import threading
from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from src.llm import GeminiProvider, LLMClient
from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT
from src.resource_metadata.schema import ResourceMetadata


class Answer(BaseModel):
    value: int


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
    return LLMClient(FakeProvider(), max_concurrency=2)


def test_generate_returns_validated_model(client):
    assert client.generate("7", Answer) == Answer(value=7)


def test_generate_rejects_answer_that_breaks_the_schema(client):
    with pytest.raises(ValueError):
        client.generate("not-a-number", Answer)


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
