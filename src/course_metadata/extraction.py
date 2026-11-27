"""Course-level roll-up of resource-level metadata.

For each course:
  1. Read its resources from public.resource_transcripts, with their metadata from
     public.resource_metadata_<version>: only the rows of the same model and of RESOURCE_PROMPT_VERSION.
     A course is rolled up only when every resource that has a transcript already has such a
     row; otherwise it is skipped.
  2. Decide in code every field that depends on duration. A value's "share" is the sum of
     (resource duration x confidence) over the resources that give it, divided by the course
     duration.
     - Single-value fields take the value with the highest share.
     - List fields keep the items whose share reaches the thresholds in src/config.py.
     - Competencies come from ONE area, the primary area: the one with the highest share is
       always kept, then up to COURSE_MAX_ITEMS in total whose share reaches COURSE_MIN_SHARE.
       The primary area is always one that has a complete competency, so it is never missing.
     - Presenters and reference resources are merged across resources, never rewritten.
  3. Ask the model only for the text fields (CourseTextMetadata), from the resources' own text
     fields. The decisions of step 2 are not sent, so they cannot bias the text. The prompt
     module is chosen by COURSE_PROMPT_VERSION in src/config.py.
  4. Build `keywords` from the model's `keywords_list`.
  5. Store one row in public.course_metadata_rollup_<version>, keyed by (course_id, model_used,
     prompt_version, resource_prompt_version), so a run with another model or prompt version
     adds rows next to the old ones. Write the same row to
     <COURSE_METADATA_OUTPUT_DIR>/<model>_course-<version>_resource-<version>/<course_id>.json.

Run src.resource_metadata.extraction first. Then, from the project root (see --help):
    python -m src.course_metadata.extraction                        # the courses in the sample CSV
    python -m src.course_metadata.extraction --all --concurrency 8 --batch-size 20
    python -m src.course_metadata.extraction --ids-file <failed_*.txt>
"""

import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from src.config import (
    DEFAULT_MODEL_ENV_VAR,
    COURSE_EXCLUSION_MIN_SHARE,
    COURSE_LEARNING_MODE_MIN_SHARE,
    COURSE_MAX_ITEMS,
    COURSE_METADATA_OUTPUT_DIR,
    COURSE_MIN_SHARE,
    COURSE_PROMPT_VERSION,
    COURSE_RATIONALE_SOURCES,
    RESOURCE_PROMPT_VERSION,
    load_db_dsn,
    load_llm_settings,
)
from src.course_metadata.course_structure import (
    CourseStructureError,
    fetch_course_tree,
    ordered_resource_ids,
    sort_resources_by_order,
)
from src.course_metadata.schema import (
    COMPETENCY_FIELD_OF_AREA,
    CourseMetadata,
    CourseTextMetadata,
)
from src.llm import GeminiProvider, LLMClient
from src.prompt_versions import prompts_for
from src.resource_metadata.extraction import METADATA_COLUMNS as RESOURCE_COLUMNS
from src.resource_metadata.extraction import METADATA_TABLE as RESOURCE_METADATA_TABLE
from src.resource_metadata.extraction import TEXT_COLUMNS
from src.resource_metadata.schema import Keyword
from src.runner import (
    RunSummary,
    build_arg_parser,
    log_summary,
    parse_args,
    run_pool,
    selected_ids,
    write_retry_file,
)
from src.utils.text_matching import (
    has_name_prefix,
    is_short_form,
    name_key,
    normalise_text,
    organisations_compatible,
    reference_key,
)

logger = logging.getLogger(__name__)

Connection = psycopg2.extensions.connection
# One row of resource_transcripts: ids, duration, languages and the metadata columns.
Resource = dict[str, Any]
# (seconds, item) pairs: the seconds a resource adds to a value, and the item it gave for it.
Entries = list[tuple[float, dict[str, Any]]]

# --- Run settings -----------------------------------------------------------------------------

# Default for --batch-size: courses prepared (read from the database and the content API) at a
# time. The number of model calls in flight is --concurrency (default MAX_CONCURRENCY), not this.
DEFAULT_BATCH_SIZE = 5

COURSE_PROMPTS = prompts_for(COURSE_PROMPT_VERSION)

# Named after the course prompt version, so each version has its own table
# (course_metadata_rollup_v3_1 for v3.1: a "." in the version is written as "_").
ROLLUP_TABLE = f"course_metadata_rollup_{COURSE_PROMPT_VERSION.replace('.', '_')}"
# What produced a row: the model (of both the resource rows and the roll-up), the course prompt
# version and the prompt version of the resource rows it was rolled up from. Together with
# course_id, the primary key.
RUN_COLUMNS = ["model_used", "prompt_version", "resource_prompt_version"]
# `model_used`, `prompt_version` and `resource_prompt_version` of a run.
RunInfo = dict[str, str]

# --- Fields and columns -----------------------------------------------------------------------

# The model writes these fields. Code decides every other CourseMetadata field.
TEXT_FIELDS = list(CourseTextMetadata.model_fields)

# One roll-up column per CourseMetadata field. TEXT for the description and course coverage, else JSONB.
COURSE_COLUMNS = [name for name in CourseMetadata.model_fields if name != "course_id"]
JSONB_COLUMNS = {column for column in COURSE_COLUMNS if column not in TEXT_COLUMNS}
# The resource_info list columns are JSONB too.
JSONB_COLUMNS |= {"resource_ids", "list_of_languages"}

# Item keys that identify one competency of each area (two resources giving the same values give
# the same competency).
KCM_COMPETENCY_KEYS = ("competency_id", "theme", "sub_theme")
COMPETENCY_KEYS_OF_AREA = {
    "Behavioural": KCM_COMPETENCY_KEYS,
    "Functional": KCM_COMPETENCY_KEYS,
    "Domain": ("sector", "subsector", "theme"),
}

# --- Decision rules ---------------------------------------------------------------------------

ENGLISH = "English"
OTHER_REFERENCE_TYPE = "Other"
LEARNING_LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]  # on a tie the lower level wins
IMPORTANCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}
STO_SUPPORTING_EVIDENCE = 4  # evidence phrases kept for the chosen STO level
STO_OPPOSING_EVIDENCE = 2  # evidence phrases kept for the other levels
# Seconds of a resource's time that count as online learning, by its learning mode.
ONLINE_FRACTION_OF_MODE = {"Online": 1.0, "Offline": 0.0, "Hybrid": 0.5}

# --- SQL --------------------------------------------------------------------------------------

# A resource without any transcript never gets metadata, so it must not hold its course back.
HAS_TRANSCRIPT_SQL = sql.SQL(
    "(COALESCE(TRIM(t.video_transcript), '') <> '' OR COALESCE(TRIM(t.audio_transcript), '') <> '')"
)

# The metadata row of each resource from one model and resource prompt version (two parameters,
# see resource_metadata_join_parameters). In the ON clause, so a resource without such a row
# still appears, with NULL metadata, and its course is reported as incomplete.
RESOURCE_METADATA_JOIN_SQL = sql.SQL(
    "LEFT JOIN public.{metadata_table} m ON m.resource_id = t.resource_id "
    "AND m.model_used = %s AND m.prompt_version = %s"
).format(metadata_table=sql.Identifier(RESOURCE_METADATA_TABLE))

CREATE_TABLE_SQL = sql.SQL(
    """
    CREATE TABLE IF NOT EXISTS public.{table} (
        course_id               TEXT NOT NULL,
        model_used              TEXT NOT NULL,
        prompt_version          TEXT NOT NULL,
        resource_prompt_version TEXT NOT NULL,
        resource_ids            JSONB NOT NULL,
        language                TEXT,
        list_of_languages       JSONB NOT NULL,
        {metadata_columns},
        PRIMARY KEY (course_id, model_used, prompt_version, resource_prompt_version)
    )
    """
).format(
    table=sql.Identifier(ROLLUP_TABLE),
    metadata_columns=sql.SQL(",\n        ").join(
        sql.SQL("{} {}").format(
            sql.Identifier(column), sql.SQL("JSONB" if column in JSONB_COLUMNS else "TEXT")
        )
        for column in COURSE_COLUMNS
    )
)


class CourseSkippedError(Exception):
    """The course cannot be rolled up.

    Either no resource has a transcript, a resource has no metadata yet, or no resource has a
    complete competency to give the course.
    """


@dataclass
class PreparedCourse:
    """Everything about one course that is known before the model is called."""

    resource_info: dict[str, Any]  # resource_ids, language, list_of_languages
    rolled_up: dict[str, Any]  # the CourseMetadata fields decided by code
    user_prompt: str


class RankedGroup(NamedTuple):
    """One value (or item) of a field, with the resources behind it."""

    share: float  # (duration x confidence) behind the value, as a share of the course duration
    key: Any  # the value, or the identifying keys of an item
    entries: Entries


# --- Step 1: reading the courses and their resources ------------------------------------------


def create_rollup_table(connection: Connection) -> None:
    """Create public.course_metadata_rollup_<version> if it does not exist."""
    with connection, connection.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL)


def resource_metadata_join_parameters(run_info: RunInfo) -> tuple[str, str]:
    """The parameters of RESOURCE_METADATA_JOIN_SQL: the resource rows of this model and of the
    resource prompt version."""
    return run_info["model_used"], run_info["resource_prompt_version"]


def find_pending_course_ids(connection: Connection, run_info: RunInfo) -> list[str]:
    """Courses with no roll-up yet from this run (`run_info`) whose resources (those with a
    transcript) all have metadata from this model and resource prompt version."""
    query = sql.SQL(
        "SELECT t.course_id FROM public.resource_transcripts t {join_metadata} "
        "WHERE t.course_id IS NOT NULL AND {has_transcript} "
        "AND t.course_id NOT IN (SELECT course_id FROM public.{rollup_table} "
        "WHERE model_used = %s AND prompt_version = %s AND resource_prompt_version = %s) "
        "GROUP BY t.course_id HAVING bool_and(m.description IS NOT NULL) ORDER BY t.course_id"
    ).format(
        join_metadata=RESOURCE_METADATA_JOIN_SQL,
        rollup_table=sql.Identifier(ROLLUP_TABLE),
        has_transcript=HAS_TRANSCRIPT_SQL,
    )
    parameters = (
        *resource_metadata_join_parameters(run_info),
        *(run_info[column] for column in RUN_COLUMNS),
    )
    with connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        return [row[0] for row in cursor.fetchall()]


def leave_out_done(connection: Connection, course_ids: list[str], run_info: RunInfo) -> list[str]:
    """`course_ids` without the courses that already have a roll-up from this run (`run_info`),
    order kept. A rerun with the same list then only does what is left."""
    query = sql.SQL(
        "SELECT course_id FROM public.{table} WHERE course_id = ANY(%s) "
        "AND model_used = %s AND prompt_version = %s AND resource_prompt_version = %s"
    ).format(table=sql.Identifier(ROLLUP_TABLE))
    with connection, connection.cursor() as cursor:
        cursor.execute(query, (course_ids, *(run_info[column] for column in RUN_COLUMNS)))
        done = {row[0] for row in cursor.fetchall()}
    if done:
        logger.info("Skipping %d courses that already have a roll-up.", len(done))
    return [course_id for course_id in course_ids if course_id not in done]


def fetch_course_resources(
    connection: Connection,
    course_id: str,
    run_info: RunInfo,
    mime_type: str | None = None,
    resource_ids: list[str] | None = None,
) -> list[Resource]:
    """The resources of the course that have a transcript, each with its metadata from this
    model and resource prompt version. With `mime_type`, only the resources of that type.

    With `resource_ids`, those resources are the course: they are matched on resource_id alone,
    whatever resource_transcripts.course_id says (a resource can be in several courses but that
    column keeps only one of them). Without, the course's resources are the rows whose
    resource_transcripts.course_id is `course_id`.

    Raises CourseSkippedError if there are none, or if any of them has no such metadata yet (the
    message lists the missing resource ids).
    """
    # Ids, duration and languages come from resource_transcripts; the metadata from
    # resource_metadata_<version>. A resource without metadata yet gets NULLs (LEFT JOIN).
    transcript_columns = ["resource_id", "duration_in_sec", "source_language", "source_languages"]
    columns = [*transcript_columns, *RESOURCE_COLUMNS]
    query = sql.SQL(
        "SELECT {columns} FROM public.resource_transcripts t {join_metadata} "
        "WHERE {of_course} AND {has_transcript} {of_type} ORDER BY t.resource_id"
    ).format(
        columns=sql.SQL(", ").join(
            [sql.Identifier("t", column) for column in transcript_columns]
            + [sql.Identifier("m", column) for column in RESOURCE_COLUMNS]
        ),
        join_metadata=RESOURCE_METADATA_JOIN_SQL,
        of_course=sql.SQL("t.resource_id = ANY(%s)" if resource_ids is not None else "t.course_id = %s"),
        has_transcript=HAS_TRANSCRIPT_SQL,
        of_type=sql.SQL("AND t.mime_type = %s" if mime_type else ""),
    )
    parameters = (
        *resource_metadata_join_parameters(run_info),
        list(resource_ids) if resource_ids is not None else course_id,
        *([mime_type] if mime_type else []),
    )
    with connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        resources = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    if not resources:
        raise CourseSkippedError("no resource with a transcript")
    missing = [
        resource["resource_id"] for resource in resources if resource["description"] is None
    ]
    if missing:
        raise CourseSkippedError(
            f"{len(missing)} of {len(resources)} resources have no metadata yet: "
            f"{', '.join(missing)}"
        )
    return resources


def normalise_durations(resources: list[Resource]) -> None:
    """Make every duration a number, so the rest of the code never checks for a missing one.

    A missing duration counts as 0. If no resource has a duration, every resource gets 1 second,
    so all of them count equally.
    """
    for resource in resources:
        resource["duration_in_sec"] = resource["duration_in_sec"] or 0
    if not any(resource["duration_in_sec"] for resource in resources):
        for resource in resources:
            resource["duration_in_sec"] = 1


def course_duration(resources: list[Resource]) -> float:
    return sum(resource["duration_in_sec"] for resource in resources)


# --- Shared helpers for the decisions ---------------------------------------------------------


def heaviest(
    weighted_values: Iterable[tuple[str, float]],
    key: Callable[[str], Any] = lambda value: value,
) -> str | None:
    """The value with the most weight in total, or None if there are no values.

    Values with the same `key` are one value: their weights add up and the first spelling seen is
    returned. On a tie the value seen first wins, so pass the values strongest resource first.
    """
    totals: dict[Any, float] = defaultdict(float)
    first_spelling: dict[Any, str] = {}
    for value, weight in weighted_values:
        value_key = key(value)
        totals[value_key] += weight
        first_spelling.setdefault(value_key, value)
    if not totals:
        return None
    return first_spelling[max(totals, key=totals.get)]


def longest_first(entries: Entries) -> Entries:
    """The entries ordered by resource duration, longest first (order of appearance on a tie)."""
    return sorted(entries, key=lambda entry: entry[0], reverse=True)


@dataclass
class MergedItem:
    """One presenter or reference, gathered from every resource of the course that gives it."""

    entries: Entries = field(default_factory=list)  # (resource duration, item as written there)
    seconds_by_resource: dict[str, float] = field(default_factory=dict)

    def add(self, resource: Resource, item: dict[str, Any]) -> None:
        self.entries.append((resource["duration_in_sec"], item))
        self.seconds_by_resource[resource["resource_id"]] = resource["duration_in_sec"]

    @property
    def covered_seconds(self) -> float:
        """The duration of the resources that give this item, each resource counted once."""
        return sum(self.seconds_by_resource.values())


def by_covered_duration(items: Iterable[MergedItem]) -> list[MergedItem]:
    """Most covered duration first. On a tie, the item that appeared first stays first."""
    return sorted(items, key=lambda item: item.covered_seconds, reverse=True)


# --- Step 2a: languages, presenters and reference resources -----------------------------------


def languages_of(resource: Resource) -> set[str]:
    """The languages of one resource. A resource without a language adds nothing."""
    languages = set(resource["source_languages"] or [])
    if resource["source_language"]:
        languages.add(resource["source_language"])
    return languages


def collect_languages(resources: list[Resource]) -> list[str]:
    """Every language of the course, the one used by most resources first.

    On a tie a regional language comes before English, then the names are in alphabetical order.
    """
    resource_counts = Counter(
        language for resource in resources for language in languages_of(resource)
    )
    return sorted(
        resource_counts,
        key=lambda language: (-resource_counts[language], language == ENGLISH, language),
    )


def collect_resource_info(resources: list[Resource]) -> dict[str, Any]:
    """The columns that say which resources the course was rolled up from, and their languages.

    `language` is the dominant language (the first of `list_of_languages`), None if there is none.
    """
    languages = collect_languages(resources)
    return {
        "resource_ids": [resource["resource_id"] for resource in resources],
        "language": languages[0] if languages else None,
        "list_of_languages": languages,
    }


def collect_reference_resources(resources: list[Resource]) -> list[dict[str, str]]:
    """Every distinct reference of the course, merged across resources and never rewritten.

    Two references are the same when `reference_key` says so. For each reference:
    - `value`: the spelling with the most resource duration behind it, copied as written.
    - `type`: the type with the most resource duration behind it; "Other" only if no resource
      gives another type.
    References are ordered by the duration of the resources that cite them, then by first
    appearance. Nothing is dropped: a broad and a specific reference ("RTI Act, 2005" and
    "RTI Act, 2005, Section 6") are both kept.
    """
    references: dict[str, MergedItem] = {}  # reference key -> the reference; in appearance order
    for resource in resources:
        for reference in resource["reference_resources"] or []:
            if reference["value"].strip():
                key = reference_key(reference["value"], reference["type"])
                references.setdefault(key, MergedItem()).add(resource, reference)

    def merge(reference: MergedItem) -> dict[str, str]:
        entries = longest_first(reference.entries)
        value = heaviest((item["value"].strip(), seconds) for seconds, item in entries)
        reference_type = heaviest(
            (item["type"], seconds)
            for seconds, item in entries
            if item["type"] != OTHER_REFERENCE_TYPE
        )
        return {"value": value, "type": reference_type or OTHER_REFERENCE_TYPE}

    return [merge(reference) for reference in by_covered_duration(references.values())]


@dataclass
class Person(MergedItem):
    """One presenter, gathered from every resource of the course that names them."""

    name_key: tuple[str, ...] = ()
    organisations: set[str] = field(default_factory=set)  # normalised organisations stated

    def is_same_person(self, other_name_key: tuple[str, ...], organisation: str | None) -> bool:
        """Same name words, and an organisation that does not conflict with any already seen."""
        return self.name_key == other_name_key and all(
            organisations_compatible(organisation, known) for known in self.organisations
        )


def collect_presenters(resources: list[Resource]) -> list[dict[str, str | None]]:
    """Every distinct presenter of the course, merged across resources.

    Two entries are the same person when their names have the same words once titles (Dr, Shri,
    Smt, ...), case, punctuation and word order are ignored, and their organisations do not
    conflict (two different stated organisations keep two people with the same name apart). Names
    that only might match, such as initials against a full name, are kept apart: a duplicate is
    safer than merging two people.

    For each person:
    - `name`: the spelling from the longest resources that has a title, otherwise the spelling
      from the longest resources. It is never rewritten.
    - `designation`, `organisation`: the stated value with the most resource duration behind it,
      otherwise None.
    People are ordered by the duration of the resources they appear in, then by first appearance.
    """
    people: list[Person] = []  # in order of first appearance
    for resource in resources:
        for presenter in resource["presenters"] or []:
            name = (presenter.get("name") or "").strip()
            if not name:
                continue
            key = name_key(name) or (name.casefold(),)
            organisation = normalise_text(presenter.get("organisation"))
            person = next((p for p in people if p.is_same_person(key, organisation)), None)
            if person is None:
                person = Person(name_key=key)
                people.append(person)
            if organisation:
                person.organisations.add(organisation)
            person.add(resource, presenter)

    def merge(person: Person) -> dict[str, str | None]:
        entries = longest_first(person.entries)
        names = [presenter["name"].strip() for _, presenter in entries]
        merged = {"name": next((name for name in names if has_name_prefix(name)), names[0])}
        for attribute in ("designation", "organisation"):
            stated = [
                (presenter[attribute].strip(), seconds)
                for seconds, presenter in entries
                if normalise_text(presenter.get(attribute))
            ]
            merged[attribute] = heaviest(stated, key=normalise_text)
        return merged

    return [merge(person) for person in by_covered_duration(people)]


# --- Step 2b: duration-weighted decisions -----------------------------------------------------


def group_by_value(resources: list[Resource], field_name: str) -> dict[str, Entries]:
    """Single-value field: value -> the resources that chose it."""
    groups: dict[str, Entries] = defaultdict(list)
    for resource in resources:
        judged = resource[field_name]
        groups[judged["value"]].append((resource["duration_in_sec"], judged))
    return groups


def group_by_item(
    resources: list[Resource], field_name: str, item_keys: tuple[str, ...]
) -> dict[tuple, Entries]:
    """List field: the values of `item_keys` of an item -> the resources that list that item."""
    groups: dict[tuple, Entries] = defaultdict(list)
    for resource in resources:
        for item in resource[field_name] or []:
            groups[tuple(item[key] for key in item_keys)].append(
                (resource["duration_in_sec"], item)
            )
    return groups


def rank_groups(groups: dict[Any, Entries], resources: list[Resource]) -> list[RankedGroup]:
    """Every group with its share of the course duration, highest share first."""
    total = course_duration(resources)
    ranked = [
        RankedGroup(
            share=sum(seconds * item["confidence"] for seconds, item in entries) / total,
            key=key,
            entries=entries,
        )
        for key, entries in groups.items()
    ]
    return sorted(ranked, key=lambda group: group.share, reverse=True)


def strongest_rationales(entries: Entries, limit: int = COURSE_RATIONALE_SOURCES) -> list[str]:
    """The distinct rationales, from the resource with the most (duration x confidence) down."""
    strongest_first = sorted(
        entries, key=lambda entry: entry[0] * entry[1]["confidence"], reverse=True
    )
    rationales = dict.fromkeys(
        item["rationale"].strip().rstrip(".") + "." for _, item in strongest_first
    )
    return list(rationales)[:limit]


def explain(group: RankedGroup, resources: list[Resource]) -> str:
    """The share of the course duration behind a value, then what the resources said about it."""
    supporting = " ".join(
        f"({number}) {rationale}"
        for number, rationale in enumerate(strongest_rationales(group.entries), start=1)
    )
    return (
        f"Supported by {group.share:.0%} of the course duration (duration weighted by "
        f"confidence), across {len(group.entries)} of {len(resources)} resources. "
        f"Supporting content from the resources: {supporting}"
    )


def judged_value(group: RankedGroup, resources: list[Resource]) -> dict[str, Any]:
    """A course-level judgement: the value, its share as the confidence, and the rationale."""
    return {
        "value": group.key,
        "confidence": round(group.share, 2),
        "rationale": explain(group, resources),
    }


def judged_item(
    group: RankedGroup, item_keys: tuple[str, ...], resources: list[Resource]
) -> dict[str, Any]:
    """A course-level list item: its identifying keys, its share as confidence, the rationale."""
    first_item = group.entries[0][1]
    return {
        **{key: first_item[key] for key in item_keys},
        "confidence": round(group.share, 2),
        "rationale": explain(group, resources),
    }


def roll_up_single_value(
    resources: list[Resource], field_name: str, tie_order: list[str] | None = None
) -> dict[str, Any]:
    """The value with the highest share. Its share is the course-level confidence.

    tie_order: on a tie, the value earlier in this list wins (e.g. the lower learning level).
    """
    ranked = rank_groups(group_by_value(resources, field_name), resources)
    if tie_order:
        # Round the share so that two equal sums that differ only by float error still tie.
        ranked.sort(key=lambda group: (-round(group.share, 9), tie_order.index(group.key)))
    return judged_value(ranked[0], resources)


def roll_up_list(
    resources: list[Resource],
    field_name: str,
    item_keys: tuple[str, ...],
    min_share: float = COURSE_MIN_SHARE,
) -> list[dict[str, Any]]:
    """The items whose share reaches `min_share`, highest first, at most COURSE_MAX_ITEMS.

    item_keys: the item keys that make two entries the same item (e.g. the role name).
    """
    ranked = rank_groups(group_by_item(resources, field_name, item_keys), resources)
    kept = [group for group in ranked if group.share >= min_share]
    return [
        judged_item(group, item_keys, resources) for group in kept[: COURSE_MAX_ITEMS[field_name]]
    ]


def split_learning_time(resources: list[Resource]) -> tuple[Entries, Entries]:
    """(online entries, in-person entries): the seconds each resource adds to each part.

    An Online resource's time is all online, an Offline resource's time is all in-person, and a
    Hybrid resource's time is split half and half, so the two parts add up to the course duration.
    A resource is left out of a part it adds no time to.
    """
    online_entries: Entries = []
    in_person_entries: Entries = []
    for resource in resources:
        seconds = resource["duration_in_sec"]
        judged = resource["learning_mode"]
        online_seconds = seconds * ONLINE_FRACTION_OF_MODE[judged["value"]]
        in_person_seconds = seconds - online_seconds
        if online_seconds:
            online_entries.append((online_seconds, judged))
        if in_person_seconds:
            in_person_entries.append((in_person_seconds, judged))
    return online_entries, in_person_entries


def roll_up_learning_mode(resources: list[Resource]) -> dict[str, Any]:
    """Split the course time into an online and an in-person part (see split_learning_time), then:

      - in-person part below COURSE_LEARNING_MODE_MIN_SHARE -> Online
      - online part below COURSE_LEARNING_MODE_MIN_SHARE    -> Offline
      - otherwise                                           -> Hybrid

    The confidence is the resources' own confidence averaged by duration. It does not change the
    split.
    """
    total = course_duration(resources)
    online_entries, in_person_entries = split_learning_time(resources)
    online_share = sum(seconds for seconds, _ in online_entries) / total
    in_person_share = sum(seconds for seconds, _ in in_person_entries) / total
    confidence = (
        sum(
            resource["duration_in_sec"] * resource["learning_mode"]["confidence"]
            for resource in resources
        )
        / total
    )

    threshold = f"{COURSE_LEARNING_MODE_MIN_SHARE:.0%}"
    online_part = ("Online part", online_entries)
    in_person_part = ("In-person part", in_person_entries)
    if in_person_share < COURSE_LEARNING_MODE_MIN_SHARE:
        value, parts = "Online", [online_part]
        reason = f"The in-person part is below {threshold}, so the course is Online."
    elif online_share < COURSE_LEARNING_MODE_MIN_SHARE:
        value, parts = "Offline", [in_person_part]
        reason = f"The online part is below {threshold}, so the course is Offline."
    else:
        value, parts = "Hybrid", [online_part, in_person_part]
        reason = f"Both parts are at least {threshold}, so the course is Hybrid."

    quotes = " ".join(
        f"{label}: {strongest_rationales(entries, limit=1)[0]}"
        for label, entries in parts
        if entries
    )
    rationale = (
        f"{online_share:.0%} of the course time can be learned online and {in_person_share:.0%} "
        f"needs in-person practice (Hybrid resources counted as half each). {reason} {quotes}"
    )
    return {"value": value, "confidence": round(confidence, 2), "rationale": rationale.strip()}


def roll_up_sto_level(resources: list[Resource]) -> dict[str, Any]:
    """Like a single value, plus evidence phrases from the strongest resources.

    Up to STO_SUPPORTING_EVIDENCE phrases for the chosen level, then up to STO_OPPOSING_EVIDENCE
    for other levels. A phrase given by several resources is kept once.
    """
    sto_level = roll_up_single_value(resources, "sto_level")
    strongest_first = sorted(
        resources,
        key=lambda resource: resource["duration_in_sec"] * resource["sto_level"]["confidence"],
        reverse=True,
    )
    evidence_by_phrase: dict[str, dict] = {}
    for resource in strongest_first:
        for evidence in resource["sto_level"]["evidence"]:
            evidence_by_phrase.setdefault(evidence["phrase"], evidence)
    evidence = list(evidence_by_phrase.values())

    supporting = [item for item in evidence if item["indicates"] == sto_level["value"]]
    opposing = [item for item in evidence if item["indicates"] != sto_level["value"]]
    return {
        **sto_level,
        "evidence": supporting[:STO_SUPPORTING_EVIDENCE] + opposing[:STO_OPPOSING_EVIDENCE],
    }


def complete_competencies(resources: list[Resource], area: str) -> dict[tuple, Entries]:
    """The area's competencies whose identifying values are all filled (a Domain path needs its
    theme)."""
    groups = group_by_item(
        resources, COMPETENCY_FIELD_OF_AREA[area], COMPETENCY_KEYS_OF_AREA[area]
    )
    return {
        key: entries
        for key, entries in groups.items()
        if all(part and part.strip() for part in key)
    }


def roll_up_primary_area(resources: list[Resource]) -> dict[str, Any]:
    """The area with the highest share among the areas that have a complete competency.

    This ensures the course always gets a competency. If a higher-ranked area has none (every
    resource returned `[]`, or a Domain path without a theme), it is passed over and the rationale
    says why. Raises CourseSkippedError if no area has one.
    """
    ranked = rank_groups(group_by_value(resources, "primary_competency_area"), resources)
    position = next(
        (
            index
            for index, group in enumerate(ranked)
            if complete_competencies(resources, group.key)
        ),
        None,
    )
    if position is None:
        raise CourseSkippedError(
            "no resource has a complete competency (every theme filled) in its primary area"
        )

    chosen = ranked[position]
    primary_area = judged_value(chosen, resources)
    passed_over = [group.key for group in ranked[:position]]
    if passed_over:
        primary_area["rationale"] = (
            f"{' and '.join(passed_over)} had more support, but no resource maps a complete "
            f"competency in it, so the course is {chosen.key}. {primary_area['rationale']}"
        )
    return primary_area


def roll_up_competencies(resources: list[Resource], area: str) -> dict[str, Any]:
    """The competencies of the course's primary area, in its field; the other two fields are [].

    Only complete competencies count (see complete_competencies). The one with the highest share
    is always kept, with no minimum share (roll_up_primary_area guarantees that one exists), so the
    course always has a competency. After it come the others whose share reaches COURSE_MIN_SHARE,
    highest first, up to COURSE_MAX_ITEMS of the area's field in total. A KCM competency
    (Behavioural or Functional) also gets its `proficiency_level` by weighted vote; a Domain one
    has none.
    """
    field_name = COMPETENCY_FIELD_OF_AREA[area]
    strongest, *others = rank_groups(complete_competencies(resources, area), resources)
    kept = [strongest, *(group for group in others if group.share >= COURSE_MIN_SHARE)]

    def competency_of(group: RankedGroup) -> dict[str, Any]:
        competency = judged_item(group, COMPETENCY_KEYS_OF_AREA[area], resources)
        if area != "Domain":
            competency["proficiency_level"] = heaviest(
                (item["proficiency_level"], seconds * item["confidence"])
                for seconds, item in group.entries
            )
        return competency

    competencies: dict[str, Any] = {name: [] for name in COMPETENCY_FIELD_OF_AREA.values()}
    competencies[field_name] = [competency_of(group) for group in kept[: COURSE_MAX_ITEMS[field_name]]]
    return competencies


def roll_up_resources(resources: list[Resource]) -> dict[str, Any]:
    """Every CourseMetadata field that code decides. The text fields come from the model."""
    primary_area = roll_up_primary_area(resources)
    target_roles = roll_up_list(resources, "target_roles", ("name",))
    target_role_names = {role["name"] for role in target_roles}
    exclusions = roll_up_list(
        resources, "target_roles_exclusions", ("name",), min_share=COURSE_EXCLUSION_MIN_SHARE
    )
    return {
        "presenters": collect_presenters(resources),
        "reference_resources": collect_reference_resources(resources),
        "learning_level": roll_up_single_value(
            resources, "learning_level", tie_order=LEARNING_LEVEL_ORDER
        ),
        "learning_mode": roll_up_learning_mode(resources),
        "bloom_taxonomy": roll_up_list(resources, "bloom_taxonomy", ("level",)),
        "primary_competency_area": primary_area,
        **roll_up_competencies(resources, primary_area["value"]),
        "sto_level": roll_up_sto_level(resources),
        "target_employee_groups": roll_up_list(resources, "target_employee_groups", ("group",)),
        "target_roles": target_roles,
        # A role is never both targeted and excluded.
        "target_roles_exclusions": [
            role for role in exclusions if role["name"] not in target_role_names
        ],
    }


# --- Step 3: the prompt for the text fields ---------------------------------------------------


def build_user_prompt(resources: list[Resource]) -> str:
    """For each resource, its share of the course duration and its text fields. Nothing else.

    The fields decided by code are not sent: they are lossy aggregates of the same resources, and
    using them as constraints would make the model drop or reshape content the resources cover.
    """
    total = course_duration(resources)
    payload = [
        {
            "duration_share": round(resource["duration_in_sec"] / total, 4),
            **{field_name: resource[field_name] for field_name in TEXT_FIELDS},
        }
        for resource in resources
    ]
    resources_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return COURSE_PROMPTS.COURSE_METADATA_USER_PROMPT.format(resources_json=resources_json)


def order_like_learner_portal(course_id: str, resources: list[Resource]) -> list[Resource]:
    """The resources in the order the course shows them to a learner (from the content API).

    Raises CourseSkippedError if the course structure cannot be read, so that metadata is never
    generated from resources in the wrong order. A resource that is not in the course structure
    is kept, last, and logged.
    """
    try:
        ordered_ids = ordered_resource_ids(fetch_course_tree(course_id))
    except CourseStructureError as error:
        raise CourseSkippedError(f"could not read the course structure: {error}") from error
    ordered, not_in_structure = sort_resources_by_order(resources, ordered_ids)
    if not_in_structure:
        logger.warning(
            "%s: %d resources are not in the course structure and go last: %s",
            course_id,
            len(not_in_structure),
            ", ".join(not_in_structure),
        )
    return ordered


def prepare_course(
    connection: Connection,
    course_id: str,
    run_info: RunInfo,
    mime_type: str | None = None,
    resource_ids: list[str] | None = None,
) -> PreparedCourse:
    """Read a course and do everything that needs no model. With `mime_type` and/or
    `resource_ids`, the course is rolled up from those of its resources only."""
    resources = fetch_course_resources(connection, course_id, run_info, mime_type, resource_ids)
    resources = order_like_learner_portal(course_id, resources)
    normalise_durations(resources)
    return PreparedCourse(
        resource_info=collect_resource_info(resources),
        rolled_up=roll_up_resources(resources),
        user_prompt=build_user_prompt(resources),
    )


# --- Step 4: the keywords (built from the model's keywords_list) ------------------------------


def split_full_and_short_forms(keyword: Keyword) -> tuple[str, list[str]]:
    """The full form of a keyword and its short forms (no repeats, ignoring case).

    If the model's keyword is itself a short form, the longest variant that is not one is the
    full form.
    """
    forms = [keyword.keyword, *keyword.variants]
    full_form = keyword.keyword
    if is_short_form(full_form):
        full_form = max(
            (form for form in forms if not is_short_form(form)), key=len, default=full_form
        )

    short_forms: dict[str, str] = {}  # lower case -> the first spelling seen
    for form in forms:
        if is_short_form(form):
            short_forms.setdefault(form.lower(), form)
    return full_form, [
        form for lowered, form in short_forms.items() if lowered != full_form.lower()
    ]


def format_keyword(keyword: Keyword) -> str:
    """For example: "Quality of Service (QoS, QS)"."""
    full_form, short_forms = split_full_and_short_forms(keyword)
    short_text = f" ({', '.join(short_forms)})" if short_forms else ""
    return f"{full_form}{short_text}"


def build_keywords(keywords: list[Keyword]) -> list[str]:
    """Every keyword as one line: High first, then Medium, then Low (the model's order within)."""
    by_importance = sorted(keywords, key=lambda keyword: IMPORTANCE_ORDER[keyword.importance])
    return [format_keyword(keyword) for keyword in by_importance]


def build_course_metadata(
    course_id: str, rolled_up: dict[str, Any], text: CourseTextMetadata
) -> CourseMetadata:
    """Join the fields decided by code with the text fields written by the model.

    The model does not output `course_id`; it is mapped back from the course of the prompt.
    """
    return CourseMetadata(
        course_id=course_id,
        **rolled_up,
        keywords=build_keywords(text.keywords_list),
        **text.model_dump(),
    )


# --- Step 5: saving the results ---------------------------------------------------------------


def build_rollup_row(
    resource_info: dict[str, Any], metadata: CourseMetadata, run_info: RunInfo
) -> dict[str, Any]:
    """The course_metadata_rollup_<version> row as JSON-ready values, keyed by column, in column order.

    run_info: the RUN_COLUMNS of this run (model, course and resource prompt versions).
    """
    return {
        "course_id": metadata.course_id,
        **{column: run_info[column] for column in RUN_COLUMNS},
        **resource_info,
        **metadata.model_dump(mode="json", exclude={"course_id"}),
    }


def save_rollup(connection: Connection, row: dict[str, Any]) -> None:
    """Insert the course row, or replace it if the course was rolled up before with the same
    model and prompt versions. A row from another model or prompt version is kept."""
    key_columns = ["course_id", *RUN_COLUMNS]
    columns = list(row)
    parameters = [
        Json(row[column]) if column in JSONB_COLUMNS else row[column] for column in columns
    ]
    query = sql.SQL(
        "INSERT INTO public.{table} ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT ({key_columns}) DO UPDATE SET {assignments}"
    ).format(
        table=sql.Identifier(ROLLUP_TABLE),
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        key_columns=sql.SQL(", ").join(sql.Identifier(column) for column in key_columns),
        assignments=sql.SQL(", ").join(
            sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(column))
            for column in columns
            if column not in key_columns
        ),
    )
    with connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)


def run_output_dir(run_info: RunInfo) -> Path:
    """<COURSE_METADATA_OUTPUT_DIR>/<model>_course-<version>_resource-<version>: the files of one
    model and pair of prompt versions, and the failed_*.txt id lists of its runs."""
    return COURSE_METADATA_OUTPUT_DIR / (
        f"{run_info['model_used']}_course-{run_info['prompt_version']}"
        f"_resource-{run_info['resource_prompt_version']}"
    )


def rollup_file_path(row: dict[str, Any]) -> Path:
    """<run_output_dir>/<course_id>.json, so a run with another model or prompt version never
    overwrites the files of an earlier one."""
    return run_output_dir(row) / f"{row['course_id']}.json"


def save_rollup_file(row: dict[str, Any]) -> None:
    """Write the row to its rollup_file_path."""
    output_path = rollup_file_path(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Running ----------------------------------------------------------------------------------


def prepare_courses(
    connection: Connection, course_ids: list[str], run_info: RunInfo
) -> dict[str, PreparedCourse | Exception]:
    """Prepare each course. A skipped course is left out; one that could not be prepared (a
    database error, or a bug in the roll-up) is its error, so it fails alone."""
    courses: dict[str, PreparedCourse | Exception] = {}
    for course_id in course_ids:
        try:
            courses[course_id] = prepare_course(connection, course_id, run_info)
        except CourseSkippedError as error:
            logger.warning("Skipped %s: %s", course_id, error)
        except Exception as error:
            logger.exception("Could not prepare %s", course_id)
            courses[course_id] = error
    return courses


async def process_course(
    connection: Connection,
    llm: LLMClient,
    course_id: str,
    course: PreparedCourse,
    run_info: RunInfo,
) -> None:
    """Ask the model for the course's text fields, then build and save its roll-up. Raises if
    either fails."""
    text_fields = await llm.agenerate(course.user_prompt, CourseTextMetadata)
    metadata = build_course_metadata(course_id, course.rolled_up, text_fields)
    row = build_rollup_row(course.resource_info, metadata, run_info)
    # File first: if the database save fails, the billed response is still kept.
    save_rollup_file(row)
    save_rollup(connection, row)


async def rollup_course_metadata(
    requested_ids: list[str] | None = None,
    skip_done: bool = True,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int | None = None,
    limit: int | None = None,
    model_env_var: str = DEFAULT_MODEL_ENV_VAR,
) -> RunSummary:
    """Roll up and store course metadata.

    With ids, process those that have no roll-up yet from this model and these prompt versions
    (all of them with skip_done=False, which replaces their stored roll-up). With None, process
    every complete course without such a roll-up. `limit` caps how many are processed;
    `concurrency` (default MAX_CONCURRENCY) how many model calls run at once.
    """
    settings = load_llm_settings(model_env_var)
    concurrency = concurrency or settings.max_concurrency
    run_info = {
        "model_used": settings.model,
        "prompt_version": COURSE_PROMPTS.PROMPT_VERSION,
        "resource_prompt_version": prompts_for(RESOURCE_PROMPT_VERSION).PROMPT_VERSION,
    }
    connection = psycopg2.connect(load_db_dsn())
    try:
        create_rollup_table(connection)
        if requested_ids is None:
            course_ids = find_pending_course_ids(connection, run_info)
        elif skip_done:
            course_ids = leave_out_done(connection, requested_ids, run_info)
        else:
            course_ids = list(requested_ids)
        course_ids = course_ids[:limit] if limit else course_ids
        if not course_ids:
            logger.info("No courses to process.")
            return RunSummary()
        logger.info(
            "Processing %d courses with %s, course prompt %s, resource rows of prompt %s: "
            "%d calls at a time, batches of %d",
            len(course_ids),
            run_info["model_used"],
            run_info["prompt_version"],
            run_info["resource_prompt_version"],
            concurrency,
            batch_size,
        )

        llm = LLMClient(
            GeminiProvider(settings, COURSE_PROMPTS.COURSE_METADATA_SYSTEM_PROMPT),
            concurrency,
            settings.retry,
        )

        async def process_one(course_id: str, course: PreparedCourse) -> None:
            await process_course(connection, llm, course_id, course, run_info)

        summary = await run_pool(
            course_ids,
            lambda batch: prepare_courses(connection, batch, run_info),
            process_one,
            concurrency=concurrency,
            batch_size=batch_size,
            label="courses",
        )
        retry_file = write_retry_file(summary, run_output_dir(run_info))
        log_summary(summary, "courses", retry_file)
        return summary
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Roll up resource metadata into course metadata.", DEFAULT_BATCH_SIZE, "courses"
    )
    args = parse_args(parser, argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = asyncio.run(
        rollup_course_metadata(
            selected_ids(args, "course_id"),
            skip_done=not args.force,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            limit=args.limit,
            model_env_var=args.model_env,
        )
    )
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
