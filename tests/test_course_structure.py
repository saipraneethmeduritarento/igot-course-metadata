import io
import json
import urllib.error
from pathlib import Path

import pytest

from src.config import ApiSettings
from src.course_metadata import course_structure
from src.course_metadata.course_structure import (
    COLLECTION_MIME_TYPE,
    CourseStructureError,
    fetch_course_tree,
    ordered_resource_ids,
    sort_resources_by_order,
)

FIXTURE = Path(__file__).parent / "fixtures" / "learning_pathway_content.json"
SETTINGS = ApiSettings(base_url="https://api.example", authorization="Bearer t", timeout_seconds=5)


def leaf(identifier, index=None, mime_type="video/mp4"):
    node = {"identifier": identifier, "mimeType": mime_type}
    if index is not None:
        node["index"] = index
    return node


def module(identifier, index, children):
    return {
        "identifier": identifier,
        "mimeType": COLLECTION_MIME_TYPE,
        "index": index,
        "children": children,
    }


def test_children_are_ordered_by_index_not_by_list_position():
    content = {"children": [leaf("c", 3), leaf("a", 1), leaf("b", 2)]}
    assert ordered_resource_ids(content) == ["a", "b", "c"]


def test_a_child_without_index_keeps_its_list_position():
    content = {"children": [leaf("a"), leaf("b"), leaf("c")]}
    assert ordered_resource_ids(content) == ["a", "b", "c"]


def test_modules_are_walked_depth_first_and_are_not_listed_themselves():
    content = {
        "children": [
            module("m2", 2, [leaf("m2b", 2), leaf("m2a", 1)]),
            module("m1", 1, [leaf("m1a", 1), module("sub", 2, [leaf("deep", 1)])]),
            leaf("last", 3),
        ]
    }
    assert ordered_resource_ids(content) == ["m1a", "deep", "m2a", "m2b", "last"]


def test_an_empty_collection_is_not_a_resource():
    content = {"children": [leaf("a", 1), module("empty", 2, [])]}
    assert ordered_resource_ids(content) == ["a"]


def test_a_repeated_identifier_keeps_its_first_position():
    content = {"children": [leaf("a", 1), leaf("b", 2), leaf("a", 3)]}
    assert ordered_resource_ids(content) == ["a", "b"]


def test_learning_pathway_is_ordered_by_milestone_then_course_then_child():
    content = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert ordered_resource_ids(content) == [
        "do_114473124676968448114",
        "do_114473125568528384115",
        "do_1144731276433653761102",
        "do_114473131090190336118",
        "do_114473131468234752119",
        "do_1144731316733542401107",
    ]


def test_content_without_children_gives_nothing():
    assert ordered_resource_ids({}) == []


def test_resources_are_sorted_by_the_given_order_and_unknown_ones_go_last():
    resources = [{"resource_id": r} for r in ["z", "b", "x", "a"]]
    ordered, missing = sort_resources_by_order(resources, ["a", "q", "b"])
    assert [r["resource_id"] for r in ordered] == ["a", "b", "x", "z"]
    assert missing == ["x", "z"]


def test_no_unknown_resource_means_nothing_is_reported():
    ordered, missing = sort_resources_by_order([{"resource_id": "a"}], ["a"])
    assert [r["resource_id"] for r in ordered] == ["a"]
    assert missing == []


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def api_body(content):
    return FakeResponse(json.dumps({"result": {"content": content}}).encode())


def test_fetch_calls_the_api_with_the_bearer_header_and_caches_the_content(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        return api_body({"children": [leaf("a", 1)]})

    monkeypatch.setattr(course_structure.urllib.request, "urlopen", fake_urlopen)

    first = fetch_course_tree("do_1", SETTINGS, tmp_path)
    second = fetch_course_tree("do_1", SETTINGS, tmp_path)  # served from the cache

    assert first == second == {"children": [leaf("a", 1)]}
    assert calls == [("https://api.example/api/extended/content/v1/read/do_1", "Bearer t", 5)]
    assert (tmp_path / "do_1.json").exists()


def test_an_unreadable_cache_is_fetched_again(tmp_path, monkeypatch):
    (tmp_path / "do_1.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        course_structure.urllib.request, "urlopen", lambda request, timeout: api_body({"children": []})
    )
    assert fetch_course_tree("do_1", SETTINGS, tmp_path) == {"children": []}


def test_an_http_error_is_a_course_structure_error_and_is_not_cached(tmp_path, monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(course_structure.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(CourseStructureError, match="401"):
        fetch_course_tree("do_1", SETTINGS, tmp_path)
    assert not (tmp_path / "do_1.json").exists()


def test_a_response_without_content_is_a_course_structure_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        course_structure.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b'{"result": {}}'),
    )
    with pytest.raises(CourseStructureError, match="no result.content"):
        fetch_course_tree("do_1", SETTINGS, tmp_path)


def count_calls(monkeypatch, outcomes):
    """Make urlopen return or raise each of `outcomes` in turn; sleeping is skipped."""
    calls, sleeps = [], []

    def fake_urlopen(request, timeout):
        outcome = outcomes[len(calls)]
        calls.append(request.full_url)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(course_structure.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(course_structure.time, "sleep", sleeps.append)
    return calls, sleeps


def http_error(code):
    return urllib.error.HTTPError("https://api.example", code, "x", {}, None)


def test_a_temporary_failure_is_retried_until_it_works(tmp_path, monkeypatch):
    calls, sleeps = count_calls(
        monkeypatch, [http_error(503), TimeoutError(), api_body({"children": []})]
    )
    assert fetch_course_tree("do_1", SETTINGS, tmp_path) == {"children": []}
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_it_gives_up_after_three_attempts(tmp_path, monkeypatch):
    calls, sleeps = count_calls(monkeypatch, [TimeoutError(), TimeoutError(), TimeoutError()])
    with pytest.raises(CourseStructureError, match="could not read"):
        fetch_course_tree("do_1", SETTINGS, tmp_path)
    assert len(calls) == 3
    assert sleeps == [1, 2]  # no wait after the last attempt


@pytest.mark.parametrize("code", [401, 403, 404])
def test_a_refusal_is_not_retried(code, tmp_path, monkeypatch):
    calls, sleeps = count_calls(monkeypatch, [http_error(code)])
    with pytest.raises(CourseStructureError, match=str(code)):
        fetch_course_tree("do_1", SETTINGS, tmp_path)
    assert len(calls) == 1
    assert sleeps == []
