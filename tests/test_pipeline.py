import asyncio

import pytest

from src import pipeline as pipeline_module
from src.course_metadata.extraction import CourseSkippedError
from src.llm import QuotaExhaustedError
from src.pipeline import Pipeline, choose_courses, main, read_course_resources, requested_course_ids
from src.runner import RunSummary

# course -> its resources, as the CSV lists them
COURSES = {"c1": ["r1", "r2"], "c2": ["r3"], "c3": ["r_no_transcript"]}
# resources resource_transcripts has a video/mp4 transcript for
TRANSCRIBED = {"r1", "r2", "r3"}


@pytest.fixture
def fake(monkeypatch):
    """Replaces the database and model work with a log of what was done, in order."""
    state = {"events": [], "done": set(), "failing": set(), "quota": set(), "unprepared": set(), "rolled_up_from": {}}

    monkeypatch.setattr(
        pipeline_module,
        "find_transcribed_resource_ids",
        lambda conn, ids: [rid for rid in ids if rid in TRANSCRIBED],
    )
    monkeypatch.setattr(
        pipeline_module.resource_extraction,
        "leave_out_done",
        lambda conn, ids, run: [rid for rid in ids if rid not in state["done"]],
    )
    monkeypatch.setattr(
        pipeline_module.resource_extraction,
        "fetch_user_prompts",
        lambda conn, ids: {rid: f"prompt {rid}" for rid in ids},
    )

    async def process_resource(conn, llm, resource_id, prompt, run, names):
        await asyncio.sleep(0)
        if resource_id in state["quota"]:
            raise QuotaExhaustedError("quota")
        if resource_id in state["failing"]:
            raise RuntimeError("model error")
        state["events"].append(("resource", resource_id))

    def prepare_course(conn, course_id, run, mime_type=None, resource_ids=None):
        assert mime_type == "video/mp4"
        state["rolled_up_from"][course_id] = resource_ids
        if course_id in state["unprepared"]:
            raise CourseSkippedError("no course structure")
        return f"prepared {course_id}"

    async def process_course(conn, llm, course_id, course, run):
        state["events"].append(("course", course_id))

    monkeypatch.setattr(pipeline_module.resource_extraction, "process_resource", process_resource)
    monkeypatch.setattr(pipeline_module.course_extraction, "prepare_course", prepare_course)
    monkeypatch.setattr(pipeline_module.course_extraction, "process_course", process_course)
    return state


def make_pipeline(redo_done=False, concurrency=4):
    return Pipeline(
        None, None, None, {}, {}, frozenset(), COURSES, concurrency=concurrency, redo_done=redo_done
    )


def run(pipeline, course_ids, parallel_courses=1):
    return asyncio.run(pipeline.run(course_ids, parallel_courses))


def test_each_course_is_rolled_up_right_after_its_resources(fake):
    summary = run(make_pipeline(), ["c1", "c2"])

    assert summary.saved == ["c1", "c2"]
    events = fake["events"]
    assert set(events[:2]) == {("resource", "r1"), ("resource", "r2")}
    assert events[2:] == [("course", "c1"), ("resource", "r3"), ("course", "c2")]


def test_a_course_is_rolled_up_from_its_csv_resources_only(fake):
    run(make_pipeline(), ["c1"])
    assert fake["rolled_up_from"]["c1"] == ["r1", "r2"]


def test_resources_with_metadata_are_not_extracted_again(fake):
    fake["done"] = {"r1"}
    pipeline = make_pipeline()

    run(pipeline, ["c1"])

    assert fake["events"] == [("resource", "r2"), ("course", "c1")]
    assert pipeline.resources_saved == ["r2"]


def test_force_extracts_every_resource_again(fake):
    fake["done"] = {"r1", "r2"}

    run(make_pipeline(redo_done=True), ["c1"])

    assert ("resource", "r1") in fake["events"]
    assert ("resource", "r2") in fake["events"]


def test_a_failed_resource_fails_its_course_only(fake):
    fake["failing"] = {"r1"}
    pipeline = make_pipeline()

    summary = run(pipeline, ["c1", "c2"])

    assert summary.failed == ["c1"]
    assert summary.saved == ["c2"]
    assert ("course", "c1") not in fake["events"]
    assert ("resource", "r2") in fake["events"]  # the other resource is still saved
    assert pipeline.resources_failed == ["r1"]
    assert summary.to_retry == ["c1"]


def test_courses_that_cannot_be_rolled_up_are_skipped(fake):
    fake["unprepared"] = {"c2"}

    summary = run(make_pipeline(), ["c3", "c2"])

    assert summary.skipped == ["c3", "c2"]  # c3: no resource with a transcript
    assert summary.exit_code == 0


def test_quota_exhaustion_stops_the_run(fake):
    fake["quota"] = {"r1"}

    summary = run(make_pipeline(), ["c1", "c2"])

    assert summary.stopped_on_quota
    assert summary.not_started == ["c1", "c2"]
    assert ("course", "c1") not in fake["events"]
    assert ("resource", "r3") not in fake["events"]


def test_parallel_courses_all_finish(fake):
    summary = run(make_pipeline(concurrency=1), ["c1", "c2", "c3"], parallel_courses=2)

    assert sorted(summary.saved) == ["c1", "c2"]
    assert summary.skipped == ["c3"]


# --- the CSV and the command line --------------------------------------------------------------


def test_csv_gives_each_courses_video_mp4_resources_in_file_order(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_text(
        "content_id,content_name,resource_id,resource_name,resource_type,mime_type,content_status\n"
        "c2,Two,r3,R3,Learning Resource,Video MP4,Live\n"
        "c1,One,r2,R2,Learning Resource,Video MP4,Live\n"
        "c1,One,r1,R1,Learning Resource,video/mp4,Live\n"
        "c1,One,r_pdf,PDF,Learning Resource,PDF,Live\n"
        "c1,One,r2,R2,Learning Resource,Video MP4,Live\n"
    )
    assert read_course_resources(path) == {"c2": ["r3"], "c1": ["r2", "r1"]}


def test_a_course_id_column_works_like_content_id(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_text("course_id,position,resource_id,mime_type\nc1,1,r1,video/mp4\nc1,2,r2,video/mp4\n")
    assert read_course_resources(path) == {"c1": ["r1", "r2"]}


def test_quotes_around_requested_ids_are_removed(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text('"a"\n \'b\' \n')
    args = pipeline_module.build_arg_parser().parse_args(["--ids-file", str(path)])
    assert requested_course_ids(args) == ["a", "b"]
    args = pipeline_module.build_arg_parser().parse_args(["--ids", '"a", "c"'])
    assert requested_course_ids(args) == ["a", "c"]


def test_requested_courses_not_in_the_csv_are_left_out():
    assert choose_courses(COURSES, ["c2", "elsewhere", "c1"]) == ["c2", "c1"]
    assert choose_courses(COURSES, None) == ["c1", "c2", "c3"]


class QueryLog:
    """Stands in for a psycopg2 connection: keeps the query, returns no rows."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        self.query = repr(query)
        self.params = params

    def fetchall(self):
        return [("r2",), ("r1",)]


def test_only_listed_video_mp4_resources_with_a_transcript_are_selected():
    connection = QueryLog()
    found = pipeline_module.find_transcribed_resource_ids(connection, ["r1", "r2", "r9"])
    assert found == ["r1", "r2"]  # CSV order
    assert connection.params == (["r1", "r2", "r9"],)
    assert "'video/mp4'" in connection.query


def test_resources_are_found_whatever_course_resource_transcripts_files_them_under():
    # A resource reused by a program can be stored under the program's id, not the course's.
    connection = QueryLog()
    pipeline_module.find_transcribed_resource_ids(connection, ["r1", "r2"])
    assert "course_id" not in connection.query


def test_main_passes_the_options_on_and_returns_the_exit_code(monkeypatch, tmp_path):
    calls = []

    async def fake_run(ids, **options):
        calls.append((ids, options))
        return RunSummary(total=1, failed=["x"])

    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_run)
    csv_path = tmp_path / "c.csv"
    argv = ["--csv", str(csv_path), "--ids", "a,b", "--concurrency", "8", "--parallel-courses", "2", "--limit", "1", "--force"]

    assert main(argv) == 1
    assert calls == [
        (
            ["a", "b"],
            {
                "skip_done": False,
                "courses_csv": csv_path,
                "concurrency": 8,
                "parallel_courses": 2,
                "limit": 1,
                "model_env_var": "GEMINI_PRO_MODEL_NAME",
            },
        )
    ]


def test_without_ids_every_csv_course_is_processed(tmp_path):
    args = pipeline_module.build_arg_parser().parse_args([])
    assert requested_course_ids(args) is None
    assert args.csv == pipeline_module.DEFAULT_COURSES_CSV
    path = tmp_path / "ids.txt"
    path.write_text("a\n# note\n\nb\na\n")
    args = pipeline_module.build_arg_parser().parse_args(["--ids-file", str(path)])
    assert requested_course_ids(args) == ["a", "b"]
