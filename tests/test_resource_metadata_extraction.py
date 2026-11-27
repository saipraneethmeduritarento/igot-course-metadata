import re

import pytest

from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT
from src.resource_metadata.extraction import (
    ResourceSkippedError,
    clean_designations,
    build_user_prompt,
    clean_video_transcript,
    fetch_user_prompts,
    save_metadata_file,
)
from src.resource_metadata.schema import Designation, ResourceMetadata

PROMPT_EXAMPLE = re.search(r"```json\n(.*?)\n```", RESOURCE_METADATA_SYSTEM_PROMPT, re.S).group(1)


class RowsConnection:
    """Stands in for a psycopg2 connection whose query returns fixed rows."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        self.queries.append(params)

    def fetchall(self):
        return self.rows


def test_prompt_contains_both_transcripts():
    prompt = build_user_prompt("video text", "audio text")
    assert "resource_id" not in prompt
    assert "<video_transcript>\nvideo text\n</video_transcript>" in prompt
    assert "<audio_transcript>\naudio text\n</audio_transcript>" in prompt


def test_prompt_leaves_out_missing_transcript():
    prompt = build_user_prompt(None, "audio text")
    assert "video_transcript" not in prompt
    assert "<audio_transcript>" in prompt


def test_resource_without_transcripts_is_skipped():
    with pytest.raises(ResourceSkippedError):
        build_user_prompt("  ", None)


def test_a_batch_of_prompts_is_read_in_one_query_and_unusable_resources_are_left_out():
    connection = RowsConnection([("do_2", "  ", None), ("do_1", "video text", None)])
    prompts = fetch_user_prompts(connection, ["do_1", "do_2", "do_unknown"])
    assert list(prompts) == ["do_1"]
    assert "video text" in prompts["do_1"]
    assert connection.queries == [(["do_1", "do_2", "do_unknown"],)]


RUN_INFO = {"model_used": "gemini-test", "prompt_version": "v9"}


def test_metadata_file_is_named_after_the_resource_id_in_a_folder_per_run(tmp_path, monkeypatch):
    monkeypatch.setattr("src.resource_metadata.extraction.RESOURCE_METADATA_OUTPUT_DIR", tmp_path / "out")
    metadata = ResourceMetadata.model_validate_json(PROMPT_EXAMPLE)

    save_metadata_file("do_1", metadata, RUN_INFO)

    saved = ResourceMetadata.model_validate_json((tmp_path / "out" / "gemini-test_v9" / "do_1.json").read_text())
    assert saved == metadata


def test_clean_video_transcript_removes_segment_markers_and_extra_spaces():
    text = "**[Segment: Video Time: 0 - 6 sec]** The video  introduces\n  X. **[Segment: Video Time: 6 - 11 sec]** Next."
    assert clean_video_transcript(text) == "The video introduces X. Next."


def test_clean_video_transcript_keeps_other_square_brackets():
    assert clean_video_transcript("**[Segment: Video Time: 0 - 6 sec]** See [Section 20(7)]") == "See [Section 20(7)]"


def test_clean_video_transcript_handles_missing_text():
    assert clean_video_transcript(None) == ""


def test_prompt_uses_cleaned_video_transcript():
    prompt = build_user_prompt("**[Segment: Video Time: 0 - 6 sec]** hello", "spoken")
    assert "<video_transcript>\nhello\n</video_transcript>" in prompt
    assert "Segment" not in prompt


@pytest.mark.parametrize("marker", ["[Segment: Video Time: 1 - 2 sec]", "**[Segment: Video Time: 1 - 2 sec]", "[Segment: Video Time: 1 - 2 sec]**"])
def test_clean_video_transcript_removes_marker_variants(marker):
    assert clean_video_transcript(f"{marker} Hello") == "Hello"


# --- extract_resource_metadata with fakes -------------------------------------------------------

import asyncio  # noqa: E402

from src.config import RetryPolicy  # noqa: E402
from src.llm import QuotaExhaustedError  # noqa: E402
from src.resource_metadata import extraction  # noqa: E402


class FakeLLM:
    """Returns the example metadata for the resource in each prompt. "do_bad" fails and
    "do_quota" raises QuotaExhaustedError."""

    def __init__(self):
        self.prompts = []

    async def agenerate(self, user_prompt, output_model):
        self.prompts.append(user_prompt)
        await asyncio.sleep(0)
        resource_id = user_prompt.split()[1]
        if resource_id == "do_bad":
            raise RuntimeError("model failed")
        if resource_id == "do_quota":
            raise QuotaExhaustedError("quota gone")
        return ResourceMetadata.model_validate_json(PROMPT_EXAMPLE)


def fake_settings(model_env_var=None):
    return type("S", (), {"max_concurrency": 1, "model": "gemini-test", "retry": RetryPolicy()})()


class FakeConnection:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def stubbed_io(monkeypatch, tmp_path):
    """Replace the database, the model and the file writes; record which resources get saved."""
    saved = []

    def fake_fetch(connection, resource_ids):
        return {rid: f"resource_id: {rid}" for rid in resource_ids if rid != "do_missing"}

    llm = FakeLLM()
    monkeypatch.setattr(extraction, "fetch_user_prompts", fake_fetch)
    monkeypatch.setattr(extraction, "save_metadata", lambda conn, rid, metadata, run_info: saved.append(rid))
    monkeypatch.setattr(extraction, "save_metadata_file", lambda rid, metadata, run_info: None)
    monkeypatch.setattr(extraction, "RESOURCE_METADATA_OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "ensure_metadata_table", lambda connection: None)
    monkeypatch.setattr(extraction, "leave_out_done", lambda connection, ids, run_info: ids)
    monkeypatch.setattr(extraction, "load_llm_settings", fake_settings)
    monkeypatch.setattr(extraction, "load_designation_names", lambda: frozenset())
    monkeypatch.setattr(extraction, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(extraction, "GeminiProvider", lambda settings, system_prompt: None)
    monkeypatch.setattr(extraction, "LLMClient", lambda provider, concurrency, retry: llm)
    return saved


def run(ids, **options):
    return asyncio.run(extraction.extract_resource_metadata(ids, **options))


def test_run_saves_successes_and_survives_skips_and_failures(stubbed_io, tmp_path):
    summary = run(["do_1", "do_missing", "do_bad", "do_2", "do_3"], batch_size=2, concurrency=2)

    assert sorted(stubbed_io) == ["do_1", "do_2", "do_3"]
    assert summary.skipped == ["do_missing"]
    assert summary.failed == ["do_bad"]
    assert summary.exit_code == 1
    run_dir = f"gemini-test_{extraction.RESOURCE_PROMPTS.PROMPT_VERSION}"
    [retry_file] = (tmp_path / "out" / run_dir).glob("failed_*.txt")
    assert retry_file.read_text() == "do_bad\n"


def test_a_clean_run_writes_no_retry_file_and_exits_0(stubbed_io, tmp_path):
    summary = run(["do_1", "do_2"], concurrency=3)
    assert summary.exit_code == 0
    assert not list(tmp_path.glob("**/failed_*.txt"))


def test_no_more_calls_than_the_concurrency_run_at_once(stubbed_io, monkeypatch):
    in_flight, peak = 0, 0

    class CountingLLM(FakeLLM):
        async def agenerate(self, user_prompt, output_model):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.001)
            in_flight -= 1
            return ResourceMetadata.model_validate_json(PROMPT_EXAMPLE)

    monkeypatch.setattr(extraction, "LLMClient", lambda provider, concurrency, retry: CountingLLM())
    run([f"do_{i}" for i in range(20)], batch_size=3, concurrency=4)

    assert len(stubbed_io) == 20
    assert peak == 4


def test_the_run_stops_when_the_quota_is_exhausted(stubbed_io):
    ids = ["do_1", "do_quota", *(f"do_{i}" for i in range(2, 30))]
    summary = run(ids, batch_size=4, concurrency=1)

    assert summary.stopped_on_quota
    assert stubbed_io == ["do_1"]
    assert summary.not_started[0] == "do_quota"
    assert sorted(summary.to_retry) == sorted(ids[1:])


def test_limit_caps_how_many_resources_are_processed(stubbed_io):
    run(["do_1", "do_2", "do_3"], limit=2)
    assert stubbed_io == ["do_1", "do_2"]


def test_extract_resource_metadata_does_nothing_when_no_resources_are_pending(monkeypatch):
    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "ensure_metadata_table", lambda connection: None)
    monkeypatch.setattr(extraction, "find_pending_resource_ids", lambda connection, run_info: [])
    monkeypatch.setattr(extraction, "load_llm_settings", fake_settings)
    monkeypatch.setattr(extraction, "build_system_prompt", lambda: pytest.fail("cache must not be built"))

    assert run(None).total == 0


def test_an_empty_id_list_processes_nothing_not_everything(stubbed_io, monkeypatch):
    monkeypatch.setattr(
        extraction, "find_pending_resource_ids", lambda connection, run_info: pytest.fail("not --all")
    )
    assert run([]).total == 0


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
    remaining = extraction.leave_out_done(connection, ["do_1", "do_2", "do_3", "do_4", "do_5"], RUN_INFO)
    assert remaining == ["do_1", "do_3", "do_5"]
    assert connection.params == (["do_1", "do_2", "do_3", "do_4", "do_5"], "gemini-test", "v9")


def test_leave_out_done_with_nothing_done_returns_everything():
    assert extraction.leave_out_done(DoneConnection([]), ["do_1", "do_2"], RUN_INFO) == ["do_1", "do_2"]


@pytest.mark.parametrize("skip_done, expected", [(True, ["do_1", "do_3"]), (False, ["do_1", "do_2", "do_3"])])
def test_extract_skips_done_resources_unless_told_not_to(stubbed_io, monkeypatch, skip_done, expected):
    monkeypatch.setattr(
        extraction, "leave_out_done", lambda connection, ids, run_info: [i for i in ids if i != "do_2"]
    )
    run(["do_1", "do_2", "do_3"], skip_done=skip_done)
    assert stubbed_io == expected


def test_pending_resources_are_those_without_a_row_from_this_model_and_prompt_version():
    connection = DoneConnection([])
    connection.fetchall = lambda: [("do_1", True), ("do_2", False)]
    assert extraction.find_pending_resource_ids(connection, RUN_INFO) == ["do_1"]
    assert connection.params == ("gemini-test", "v9")


def test_metadata_table_is_named_after_the_prompt_version_and_keyed_by_resource_model_and_prompt_version():
    text = repr(extraction.CREATE_TABLE_SQL)
    assert f"resource_metadata_{extraction.RESOURCE_PROMPT_VERSION.replace('.', '_')}" in text
    assert "PRIMARY KEY (resource_id, model_used, prompt_version)" in text


def test_run_saves_with_the_model_and_prompt_version(stubbed_io, monkeypatch):
    runs = []
    monkeypatch.setattr(
        extraction, "save_metadata", lambda conn, rid, metadata, run_info: runs.append(run_info)
    )
    run(["do_1"])
    assert runs == [{"model_used": "gemini-test", "prompt_version": extraction.RESOURCE_PROMPTS.PROMPT_VERSION}]
