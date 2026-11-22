import re

import pytest

from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT
from src.resource_metadata.extraction import (
    ResourceSkippedError,
    clean_designations,
    clean_video_transcript,
    fetch_user_prompt,
    save_metadata_file,
)
from src.resource_metadata.schema import Designation, ResourceMetadata

PROMPT_EXAMPLE = re.search(r"```json\n(.*?)\n```", RESOURCE_METADATA_SYSTEM_PROMPT, re.S).group(1)


class FakeConnection:
    """Stands in for a psycopg2 connection whose query returns one fixed row."""

    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        pass

    def fetchone(self):
        return self.row


def test_prompt_contains_both_transcripts():
    prompt = fetch_user_prompt(FakeConnection(("video text", "audio text")), "do_1")
    assert "resource_id" not in prompt
    assert "<video_transcript>\nvideo text\n</video_transcript>" in prompt
    assert "<audio_transcript>\naudio text\n</audio_transcript>" in prompt


def test_prompt_leaves_out_missing_transcript():
    prompt = fetch_user_prompt(FakeConnection((None, "audio text")), "do_1")
    assert "video_transcript" not in prompt
    assert "<audio_transcript>" in prompt


def test_resource_without_transcripts_is_skipped():
    with pytest.raises(ResourceSkippedError):
        fetch_user_prompt(FakeConnection(("  ", None)), "do_1")


def test_unknown_resource_is_skipped():
    with pytest.raises(ResourceSkippedError):
        fetch_user_prompt(FakeConnection(None), "do_1")


def test_metadata_file_is_named_after_the_resource_id(tmp_path, monkeypatch):
    monkeypatch.setattr("src.resource_metadata.extraction.RESOURCE_METADATA_OUTPUT_DIR", tmp_path / "out")
    metadata = ResourceMetadata.model_validate_json(PROMPT_EXAMPLE)

    save_metadata_file("do_1", metadata)

    saved = ResourceMetadata.model_validate_json((tmp_path / "out" / "do_1.json").read_text())
    assert saved == metadata


def test_clean_video_transcript_removes_segment_markers_and_extra_spaces():
    text = "**[Segment: Video Time: 0 - 6 sec]** The video  introduces\n  X. **[Segment: Video Time: 6 - 11 sec]** Next."
    assert clean_video_transcript(text) == "The video introduces X. Next."


def test_clean_video_transcript_keeps_other_square_brackets():
    assert clean_video_transcript("**[Segment: Video Time: 0 - 6 sec]** See [Section 20(7)]") == "See [Section 20(7)]"


def test_clean_video_transcript_handles_missing_text():
    assert clean_video_transcript(None) == ""


def test_prompt_uses_cleaned_video_transcript():
    prompt = fetch_user_prompt(FakeConnection(("**[Segment: Video Time: 0 - 6 sec]** hello", "spoken")), "do_1")
    assert "<video_transcript>\nhello\n</video_transcript>" in prompt
    assert "Segment" not in prompt


@pytest.mark.parametrize("marker", ["[Segment: Video Time: 1 - 2 sec]", "**[Segment: Video Time: 1 - 2 sec]", "[Segment: Video Time: 1 - 2 sec]**"])
def test_clean_video_transcript_removes_marker_variants(marker):
    assert clean_video_transcript(f"{marker} Hello") == "Hello"


# --- process_batch / extract_resource_metadata with fakes -----------------------------------------

import asyncio  # noqa: E402

from src.resource_metadata import extraction  # noqa: E402


class FakeLLM:
    """Returns the example metadata for the resource in each prompt; "do_bad" fails."""

    async def agenerate_many(self, user_prompts, output_model):
        results = []
        for prompt in user_prompts:
            resource_id = prompt.split()[1]
            if resource_id == "do_bad":
                results.append(RuntimeError("model failed"))
            else:
                results.append(ResourceMetadata.model_validate_json(PROMPT_EXAMPLE))
        return results


@pytest.fixture
def stubbed_io(monkeypatch):
    """Replace the database and file writes; record which resources get saved."""
    saved = []

    def fake_fetch(connection, resource_id):
        if resource_id == "do_missing":
            raise ResourceSkippedError("resource not found")
        return f"resource_id: {resource_id}"

    monkeypatch.setattr(extraction, "fetch_user_prompt", fake_fetch)
    monkeypatch.setattr(extraction, "save_metadata", lambda conn, rid, metadata: saved.append(rid))
    monkeypatch.setattr(extraction, "save_metadata_file", lambda rid, metadata: None)
    return saved


def test_process_batch_saves_successes_and_survives_skips_and_failures(stubbed_io):
    count = asyncio.run(
        extraction.process_batch(None, FakeLLM(), ["do_1", "do_missing", "do_bad", "do_2"])
    )
    assert count == 2
    assert stubbed_io == ["do_1", "do_2"]


def test_extract_resource_metadata_runs_in_batches(stubbed_io, monkeypatch):
    class FakeConnection:
        closed = False

        def close(self):
            self.closed = True

    connection = FakeConnection()
    monkeypatch.setattr(extraction, "BATCH_SIZE", 2)
    monkeypatch.setattr(extraction, "ensure_metadata_table", lambda connection: None)
    monkeypatch.setattr(extraction, "leave_out_done", lambda connection, ids: ids)
    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: connection)
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "load_llm_settings", lambda: type("S", (), {"max_concurrency": 1})())
    monkeypatch.setattr(extraction, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(extraction, "GeminiProvider", lambda settings, system_prompt: None)
    monkeypatch.setattr(extraction, "LLMClient", lambda provider, max_concurrency: FakeLLM())

    asyncio.run(extraction.extract_resource_metadata(["do_1", "do_2", "do_3"]))

    assert stubbed_io == ["do_1", "do_2", "do_3"]
    assert connection.closed


def test_extract_resource_metadata_does_nothing_when_no_resources_are_pending(monkeypatch):
    class FakeConnection:
        def close(self):
            pass

    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "ensure_metadata_table", lambda connection: None)
    monkeypatch.setattr(extraction, "find_pending_resource_ids", lambda connection: [])
    monkeypatch.setattr(extraction, "build_system_prompt", lambda: pytest.fail("cache must not be built"))

    asyncio.run(extraction.extract_resource_metadata(None))


VALID_DESIGNATIONS = frozenset(
    {"Additional Superintendent of Police", "Additional Superintendent of Police (Civil)",
     "Additional Superintendent of Police (Armed Reserve)"}
)


def _roles(*names):
    return [Designation(name=n, confidence=0.8, rationale="r") for n in names]


def test_clean_designations_expands_compact_form_and_drops_invalid():
    roles = _roles(
        "Additional Superintendent of Police (Civil | Armed Reserve)",
        "Additional Superintendent of Police (Civil)",
        "Additional Superintendent of Police (Cyber Crime)",
        "Made Up Role",
    )
    cleaned = clean_designations(roles, VALID_DESIGNATIONS, "do_1", "target_roles")
    assert [r.name for r in cleaned] == [
        "Additional Superintendent of Police (Civil)",
        "Additional Superintendent of Police (Armed Reserve)",
    ]
    assert all(r.confidence == 0.8 for r in cleaned)


def test_prompt_example_keys_follow_schema_order():
    import json

    example = json.loads(PROMPT_EXAMPLE)
    assert list(example) == list(ResourceMetadata.model_fields)
    assert list(example["sto_level"]) == ["evidence", "rationale", "value", "confidence"]


class DoneConnection:
    """A connection whose query says these resource ids already have metadata."""

    def __init__(self, done):
        self.done = done
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        self.params = params

    def fetchall(self):
        return [(rid,) for rid in self.done]


def test_leave_out_done_keeps_the_order_of_the_rest():
    connection = DoneConnection(["do_2", "do_4"])
    remaining = extraction.leave_out_done(connection, ["do_1", "do_2", "do_3", "do_4", "do_5"])
    assert remaining == ["do_1", "do_3", "do_5"]
    assert connection.params == (["do_1", "do_2", "do_3", "do_4", "do_5"],)


def test_leave_out_done_with_nothing_done_returns_everything():
    assert extraction.leave_out_done(DoneConnection([]), ["do_1", "do_2"]) == ["do_1", "do_2"]


@pytest.mark.parametrize("skip_done, expected", [(True, ["do_1", "do_3"]), (False, ["do_1", "do_2", "do_3"])])
def test_extract_skips_done_resources_unless_told_not_to(stubbed_io, monkeypatch, skip_done, expected):
    class FakeConnection:
        def close(self):
            pass

    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "ensure_metadata_table", lambda connection: None)
    monkeypatch.setattr(
        extraction, "leave_out_done", lambda connection, ids: [i for i in ids if i != "do_2"]
    )
    monkeypatch.setattr(extraction, "load_llm_settings", lambda: type("S", (), {"max_concurrency": 1})())
    monkeypatch.setattr(extraction, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(extraction, "GeminiProvider", lambda settings, system_prompt: None)
    monkeypatch.setattr(extraction, "LLMClient", lambda provider, max_concurrency: FakeLLM())

    asyncio.run(extraction.extract_resource_metadata(["do_1", "do_2", "do_3"], skip_done=skip_done))

    assert stubbed_io == expected
