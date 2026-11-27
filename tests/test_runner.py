import pytest

from src import runner
from src.course_metadata import extraction as course_extraction
from src.resource_metadata import extraction as resource_extraction
from src.runner import RunSummary, build_arg_parser, parse_args, selected_ids


def parse(*argv):
    return parse_args(build_arg_parser("test", 20, "resources"), list(argv))


def test_defaults():
    args = parse()
    assert args.batch_size == 20
    assert args.concurrency is None  # MAX_CONCURRENCY from .env
    assert args.limit is None
    assert not args.force
    assert args.model_env == "GEMINI_PRO_MODEL_NAME"


def test_sizes_must_be_positive():
    with pytest.raises(SystemExit):
        parse("--batch-size", "0")
    with pytest.raises(SystemExit):
        parse("--concurrency", "-2")


def test_force_needs_chosen_ids():
    with pytest.raises(SystemExit):
        parse("--all", "--force")


def test_all_ids_and_ids_file_exclude_each_other():
    with pytest.raises(SystemExit):
        parse("--all", "--ids", "a")


def test_ids_are_split_trimmed_and_deduplicated():
    assert selected_ids(parse("--ids", " a, b,,a ,c"), "resource_id") == ["a", "b", "c"]


def test_ids_file_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("a\n\n# a note\n b \na\n")
    assert selected_ids(parse("--ids-file", str(path)), "resource_id") == ["a", "b"]


def test_all_selects_every_pending_id():
    assert selected_ids(parse("--all"), "resource_id") is None


def test_without_a_choice_the_sample_csv_is_used(monkeypatch):
    monkeypatch.setattr(runner, "load_sample_ids", lambda column: [f"{column}-1", f"{column}-2"])
    assert selected_ids(parse(), "course_id") == ["course_id-1", "course_id-2"]


@pytest.mark.parametrize(
    "module, function",
    [(resource_extraction, "extract_resource_metadata"), (course_extraction, "rollup_course_metadata")],
)
def test_main_passes_the_options_on_and_returns_the_exit_code(monkeypatch, module, function):
    calls = []

    async def fake_run(ids, **options):
        calls.append((ids, options))
        return RunSummary(total=1, failed=["x"])

    monkeypatch.setattr(module, function, fake_run)
    argv = ["--ids", "a,b", "--batch-size", "50", "--concurrency", "8", "--limit", "1", "--force"]

    assert module.main(argv) == 1
    assert calls == [
        (
            ["a", "b"],
            {
                "skip_done": False,
                "batch_size": 50,
                "concurrency": 8,
                "limit": 1,
                "model_env_var": "GEMINI_PRO_MODEL_NAME",
            },
        )
    ]
