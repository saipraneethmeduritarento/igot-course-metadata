"""The order of a course's resources as a learner sees it in the portal.

The Extended Content Read API returns the course as a tree. Walking that tree gives the portal
order, which `prepare_course` uses to order the resources before they are rolled up and sent to the
model. Without it the resources come in `resource_id` order, which is arbitrary.

    GET {KARMAYOGI_BASE_URL}/api/extended/content/v1/read/<course_id>

A normal course has `children` at the top. A Learning Pathway has `milestones_v1`, each milestone
with `courses`, each course with its own `children`.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.config import COURSE_API_DIR, ApiSettings, load_api_settings

logger = logging.getLogger(__name__)

COLLECTION_MIME_TYPE = "application/vnd.ekstep.content-collection"
READ_PATH = "/api/extended/content/v1/read/{course_id}"

# A failed request is tried MAX_ATTEMPTS times in all, waiting 1s, then 2s, between attempts. Only
# a failure that can go away is retried: a timeout, a network error, an unreadable body, or one of
# these HTTP codes. A 401, 403 or 404 will not change by asking again.
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

Node = dict[str, Any]


class CourseStructureError(Exception):
    """The course structure could not be read from the API (or its cache)."""


# --- Reading the tree ------------------------------------------------------------------------


def sorted_children(nodes: list[Node]) -> list[Node]:
    """Siblings in portal order: by their `index`, a node without one keeps its list position."""
    positioned = list(enumerate(nodes))
    positioned.sort(
        key=lambda item: (item[1]["index"] if isinstance(item[1].get("index"), int) else item[0])
    )
    return [node for _, node in positioned]


def iter_leaves(nodes: list[Node]) -> Iterator[Node]:
    """The leaves under `nodes`, depth first. A module (a node with children) gives its children."""
    for node in sorted_children([n for n in nodes if isinstance(n, dict)]):
        children = node.get("children")
        if children:
            yield from iter_leaves(children)
        elif node.get("mimeType") != COLLECTION_MIME_TYPE:
            yield node


def ordered_resource_ids(content: Node) -> list[str]:
    """The identifiers of the resources of a course, in the order a learner meets them.

    `content` is `result.content` of the API response. Every kind of leaf is listed (video, PDF,
    link, assessment): the transcripts table decides which of them matter. A repeated identifier
    keeps its first position.
    """
    if content.get("milestones_v1"):
        # Learning Pathway: milestones, then their courses, then each course's resources.
        leaves: list[Node] = []
        for milestone in content["milestones_v1"]:
            for course in milestone.get("courses") or []:
                leaves.extend(iter_leaves(course.get("children") or []))
    else:
        leaves = list(iter_leaves(content.get("children") or []))

    seen: dict[str, None] = {}
    for leaf in leaves:
        identifier = leaf.get("identifier")
        if identifier:
            seen.setdefault(identifier, None)
    return list(seen)


def sort_resources_by_order(
    resources: list[dict[str, Any]], ordered_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """The resources in `ordered_ids` order, plus the ids of those that are not in it.

    A resource that is not in `ordered_ids` goes last, sorted by `resource_id`, so that no
    resource is dropped and the result is still deterministic.
    """
    position = {resource_id: index for index, resource_id in enumerate(ordered_ids)}
    known = [r for r in resources if r["resource_id"] in position]
    unknown = [r for r in resources if r["resource_id"] not in position]
    known.sort(key=lambda r: position[r["resource_id"]])
    unknown.sort(key=lambda r: r["resource_id"])
    return known + unknown, [r["resource_id"] for r in unknown]


# --- Fetching it -----------------------------------------------------------------------------


def fetch_course_tree(
    course_id: str,
    settings: ApiSettings | None = None,
    cache_dir: Path = COURSE_API_DIR,
) -> Node:
    """`result.content` of the course, from the cache if it is there, otherwise from the API.

    A fetched response is cached as <cache_dir>/<course_id>.json. Raises CourseStructureError if
    the API cannot be reached or answers with something unusable.
    """
    cache_path = cache_dir / f"{course_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable cache %s, fetching again", cache_path)

    content = _request_course_content(course_id, settings or load_api_settings())

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)
    return content


def _request_course_content(course_id: str, settings: ApiSettings) -> Node:
    request = urllib.request.Request(
        settings.base_url + READ_PATH.format(course_id=course_id),
        headers={"Accept": "application/json", "Authorization": settings.authorization},
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                body = json.load(response)
            break
        except urllib.error.HTTPError as error:
            problem = f"API answered HTTP {error.code} for {course_id}"
            retryable = error.code in RETRYABLE_STATUS_CODES
            cause: Exception = error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            problem = f"could not read {course_id} from the API: {error!r}"
            retryable = True
            cause = error
        if not retryable or attempt == MAX_ATTEMPTS:
            raise CourseStructureError(problem) from cause
        delay = RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
        logger.warning("%s (attempt %d/%d), retrying in %ds", problem, attempt, MAX_ATTEMPTS, delay)
        time.sleep(delay)

    content = (body.get("result") or {}).get("content") if isinstance(body, dict) else None
    if not isinstance(content, dict):
        raise CourseStructureError(f"API response for {course_id} has no result.content")
    return content
