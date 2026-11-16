import asyncio
import json
import re

import pytest
from pydantic import BaseModel

from src.llm import LLMClient
from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT
from src.resource_metadata_schema import ResourceMetadata


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
