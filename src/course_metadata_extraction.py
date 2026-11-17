"""Course-level roll-up of resource-level metadata.

For each course:
  1. read the metadata of its resources from public.resource_transcripts. A course is rolled up only
     when every resource that has a transcript already has metadata, otherwise it is skipped;
  2. decide the course fields that depend on duration IN CODE: each value or item is scored by
     the sum of (resource duration x confidence) over the resources that carry it, divided by the
     course duration. Single-value fields take the highest score; list fields keep the items whose
     score reaches the threshold in src/config.py;
  3. ask the LLM only for the text fields (summary, keywords, presenters, prerequisites,
     responsibilities, activities);
  4. add the fields built by code: the languages and the formatted `keywords_list`;
  5. store one row in public.course_metadata_rollup, and write the full result to
     <COURSE_METADATA_OUTPUT_DIR>/<course_id>.json.

Run resource_metadata_extraction first. Then, from the project root:
    python -m src.course_metadata_extraction
"""

import asyncio
import json
import logging
from collections import Counter, defaultdict

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from pydantic import ValidationError

from src.config import (
    COURSE_EXCLUSION_MIN_SHARE,
    COURSE_MAX_ITEMS,
    COURSE_METADATA_OUTPUT_DIR,
    COURSE_MIN_SHARE,
    COURSE_RATIONALE_SOURCES,
    load_db_dsn,
    load_llm_settings,
)
from src.course_metadata_schema import CourseMetadata, CourseTextMetadata
from src.llm import GeminiProvider, LLMClient
from src.prompts import COURSE_METADATA_SYSTEM_PROMPT, COURSE_METADATA_USER_PROMPT
from src.resource_metadata_extraction import METADATA_COLUMNS as RESOURCE_COLUMNS
from src.resource_metadata_extraction import TEXT_COLUMNS
from src.resource_metadata_schema import Keyword

logger = logging.getLogger(__name__)

Connection = psycopg2.extensions.connection
Resource = dict  # one row of resource_transcripts: ids, duration, languages and metadata columns
Entries = list[tuple[int, dict]]  # (resource duration, the resource's judged item) for one value

# Courses to process. Leave empty to process every complete course that has no roll-up yet.
COURSE_IDS: list[str] = ["do_11371284946833408012", "do_11360057657171968019"]

BATCH_SIZE = 5  # courses sent to the model together; results are saved after each batch

# The model writes these fields. Code decides all the other CourseMetadata fields.
TEXT_FIELDS = [name for name in CourseTextMetadata.model_fields if name != "course_id"]

# One roll-up column per CourseMetadata field. RESOURCE_COLUMNS are the ones read from the resource table.
COURSE_COLUMNS = [name for name in CourseMetadata.model_fields if name != "course_id"]

# A resource without any transcript text never gets metadata, so it must not hold its course back.
HAS_TRANSCRIPT_SQL = sql.SQL("(COALESCE(TRIM(video_transcript), '') <> '' OR COALESCE(TRIM(audio_transcript), '') <> '')")

ENGLISH = "English"
IMPORTANCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}

# Metadata columns are TEXT for the summary and description, JSONB otherwise.
CREATE_TABLE_SQL = sql.SQL(
    """
    CREATE TABLE IF NOT EXISTS public.course_metadata_rollup (
        course_id         TEXT PRIMARY KEY,
        resource_ids      TEXT[] NOT NULL,
        language          TEXT,
        list_of_languages TEXT[] NOT NULL,
        {metadata_columns}
    )
    """
).format(
    metadata_columns=sql.SQL(",\n        ").join(
        sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL("TEXT" if column in TEXT_COLUMNS else "JSONB"))
        for column in COURSE_COLUMNS
    )
)


class CourseSkippedError(Exception):
    """The course cannot be rolled up yet (no resource with a transcript, or a resource has no metadata)."""


# --- Step 1: the table and the courses to process ----------------------------------------------


def create_rollup_table(connection: Connection) -> None:
    """Create public.course_metadata_rollup if it does not exist."""
    with connection, connection.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL)


def find_pending_course_ids(connection: Connection) -> list[str]:
    """Courses with no roll-up yet whose resources (those with a transcript) all have metadata."""
    query = sql.SQL(
        "SELECT course_id FROM public.resource_transcripts "
        "WHERE course_id IS NOT NULL AND {has_transcript} "
        "AND course_id NOT IN (SELECT course_id FROM public.course_metadata_rollup) "
        "GROUP BY course_id HAVING bool_and(course_summary IS NOT NULL) ORDER BY course_id"
    ).format(has_transcript=HAS_TRANSCRIPT_SQL)
    with connection, connection.cursor() as cursor:
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]


# --- Step 2: reading the resources of a course -------------------------------------------------


def fetch_course_resources(connection: Connection, course_id: str) -> list[Resource]:
    """The resources of the course that have a transcript, each with its metadata.

    Raises CourseSkippedError if there are none, or if any of them has no metadata yet (the
    message lists the missing resource ids).
    """
    columns = ["resource_id", "duration_in_sec", "source_language", "source_languages", *RESOURCE_COLUMNS]
    query = sql.SQL(
        "SELECT {columns} FROM public.resource_transcripts "
        "WHERE course_id = %s AND {has_transcript} ORDER BY resource_id"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        has_transcript=HAS_TRANSCRIPT_SQL,
    )

    with connection, connection.cursor() as cursor:
        cursor.execute(query, (course_id,))
        resources = [dict(zip(columns, row)) for row in cursor.fetchall()]

    if not resources:
        raise CourseSkippedError("no resource with a transcript")
    missing = [resource["resource_id"] for resource in resources if resource["course_summary"] is None]
    if missing:
        raise CourseSkippedError(
            f"{len(missing)} of {len(resources)} resources have no metadata yet: {', '.join(missing)}"
        )
    return resources


def fill_missing_durations(resources: list[Resource]) -> None:
    """If no resource has a duration, give each one the same duration so all count equally."""
    if not any(resource["duration_in_sec"] for resource in resources):
        for resource in resources:
            resource["duration_in_sec"] = 1


def course_duration(resources: list[Resource]) -> int:
    return sum(resource["duration_in_sec"] or 0 for resource in resources)


# --- Step 3: fields copied or combined by code -------------------------------------------------


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
    resource_counts = Counter(language for resource in resources for language in languages_of(resource))
    return sorted(resource_counts, key=lambda language: (-resource_counts[language], language == ENGLISH, language))


def collect_resource_info(resources: list[Resource]) -> dict:
    """The columns that say which resources the course was rolled up from, and in which languages.

    `language` is the dominant language (the first of `list_of_languages`), None if no resource has one.
    """
    languages = collect_languages(resources)
    return {
        "resource_ids": [resource["resource_id"] for resource in resources],
        "language": languages[0] if languages else None,
        "list_of_languages": languages,
    }


def collect_reference_resources(resources: list[Resource]) -> list[str]:
    """Every reference of every resource, without duplicates, in order of first appearance."""
    references = (reference for resource in resources for reference in resource["reference_resources"] or [])
    return list(dict.fromkeys(references))


# --- Step 4: duration-weighted decisions -------------------------------------------------------


def group_by_value(resources: list[Resource], field: str) -> dict[str, Entries]:
    """Single-value field: value -> the resources that chose it."""
    groups: dict[str, Entries] = defaultdict(list)
    for resource in resources:
        judged = resource[field]
        groups[judged["value"]].append((resource["duration_in_sec"], judged))
    return groups


def group_by_item(resources: list[Resource], field: str, identity: tuple[str, ...]) -> dict[tuple, Entries]:
    """List field: item identity (e.g. the competency id) -> the resources that list it."""
    groups: dict[tuple, Entries] = defaultdict(list)
    for resource in resources:
        for item in resource[field] or []:
            groups[tuple(item[name] for name in identity)].append((resource["duration_in_sec"], item))
    return groups


def support_seconds(entries: Entries) -> float:
    """Duration weighted by confidence."""
    return sum(duration * item["confidence"] for duration, item in entries)


def rank_groups(groups: dict, resources: list[Resource]) -> list[tuple[float, object, Entries]]:
    """(share of the course duration, group key, entries), highest share first."""
    total = course_duration(resources)
    ranked = [(support_seconds(entries) / total, key, entries) for key, entries in groups.items()]
    return sorted(ranked, key=lambda row: row[0], reverse=True)


def strongest_rationales(entries: Entries, limit: int = COURSE_RATIONALE_SOURCES) -> list[str]:
    """What the resources with the most duration x confidence behind them said about this value."""
    strongest_first = sorted(entries, key=lambda entry: entry[0] * entry[1]["confidence"], reverse=True)
    rationales = dict.fromkeys(item["rationale"].strip().rstrip(".") + "." for _, item in strongest_first)
    return list(rationales)[:limit]


def explain(share: float, entries: Entries, resources: list[Resource]) -> str:
    """The share of the course duration behind a value, then the content from the resources that supports it."""
    supporting = " ".join(f"({number}) {text}" for number, text in enumerate(strongest_rationales(entries), 1))
    return (
        f"Supported by {share:.0%} of the course duration (duration weighted by confidence), "
        f"across {len(entries)} of {len(resources)} resources. Supporting content from the resources: {supporting}"
    )


def weighted_vote(entries: Entries, field: str) -> str:
    """The value of `field` with the most duration x confidence behind it."""
    weights: dict[str, float] = defaultdict(float)
    for duration, item in entries:
        weights[item[field]] += duration * item["confidence"]
    return max(weights, key=weights.get)


def roll_up_single_value(resources: list[Resource], field: str) -> dict:
    """Pick the value with the highest share. Its share is the course-level confidence."""
    share, value, entries = rank_groups(group_by_value(resources, field), resources)[0]
    return {"value": value, "confidence": round(share, 2), "rationale": explain(share, entries, resources)}


def roll_up_learning_mode(resources: list[Resource]) -> dict:
    """Hybrid if both an online part and an in-person part reach the min share, else the weighted vote.

    A Hybrid resource counts for both parts. A course that is half Online and half Offline is Hybrid,
    even though no single resource says so.
    """
    groups = group_by_value(resources, "learning_mode")
    total = course_duration(resources)
    online_entries = groups["Online"] + groups["Hybrid"]
    offline_entries = groups["Offline"] + groups["Hybrid"]
    online = support_seconds(online_entries) / total
    offline = support_seconds(offline_entries) / total
    if min(online, offline) < COURSE_MIN_SHARE:
        return roll_up_single_value(resources, "learning_mode")
    return {
        "value": "Hybrid",
        "confidence": round(min(online, offline), 2),
        "rationale": (
            f"Online learning covers {online:.0%} and in-person practice {offline:.0%} of the course duration "
            "(duration weighted by confidence; Hybrid resources count for both). "
            f"Online part: {strongest_rationales(online_entries, 1)[0]} "
            f"In-person part: {strongest_rationales(offline_entries, 1)[0]}"
        ),
    }


def roll_up_sto_level(resources: list[Resource]) -> dict:
    """Like a single value, plus up to 4 evidence phrases for the level and 2 for other levels."""
    sto_level = roll_up_single_value(resources, "sto_level")
    strongest_first = sorted(
        resources,
        key=lambda r: r["duration_in_sec"] * r["sto_level"]["confidence"],
        reverse=True,
    )
    evidence = []  # strongest resources first, no repeated phrase
    for resource in strongest_first:
        for item in resource["sto_level"]["evidence"]:
            if item["phrase"] not in {e["phrase"] for e in evidence}:
                evidence.append(item)
    for_level = [e for e in evidence if e["indicates"] == sto_level["value"]][:4]
    against = [e for e in evidence if e["indicates"] != sto_level["value"]][:2]
    return {**sto_level, "evidence": for_level + against}


def roll_up_list(
    resources: list[Resource],
    field: str,
    identity: tuple[str, ...],
    min_share: float = COURSE_MIN_SHARE,
    keep_top_one: bool = False,
    vote_field: str | None = None,
) -> list[dict]:
    """Keep the items whose share of the course duration reaches `min_share`, at most COURSE_MAX_ITEMS.

    identity:     the item keys that make two entries the same item (e.g. the competency id).
    keep_top_one: if nothing reaches the threshold, still keep the best item.
    vote_field:   a further item key (e.g. proficiency_level) decided by weighted vote.
    """
    ranked = rank_groups(group_by_item(resources, field, identity), resources)
    kept = [row for row in ranked if row[0] >= min_share][: COURSE_MAX_ITEMS[field]]
    if not kept and keep_top_one:
        kept = ranked[:1]

    items = []
    for share, _, entries in kept:
        first_item = entries[0][1]
        item = {name: first_item[name] for name in identity}
        if vote_field:
            item[vote_field] = weighted_vote(entries, vote_field)
        items.append({**item, "confidence": round(share, 2), "rationale": explain(share, entries, resources)})
    return items


def roll_up_competencies(resources: list[Resource], area: str) -> dict:
    """Only the competency list of the course's primary area is filled. The others stay empty."""
    kcm_identity = ("competency_id", "theme", "sub_theme")
    return {
        "behavioural_competencies": (
            roll_up_list(resources, "behavioural_competencies", kcm_identity, keep_top_one=True, vote_field="proficiency_level")
            if area == "Behavioural"
            else []
        ),
        "functional_competencies": (
            roll_up_list(resources, "functional_competencies", kcm_identity, keep_top_one=True, vote_field="proficiency_level")
            if area == "Functional"
            else []
        ),
        "domain_competencies": (
            roll_up_list(resources, "domain_competencies", ("sector", "subsector", "theme"), keep_top_one=True)
            if area == "Domain"
            else []
        ),
    }


def roll_up_resources(resources: list[Resource]) -> dict:
    """Every CourseMetadata field that code decides. The text fields come from the model."""
    primary_area = roll_up_single_value(resources, "primary_competency_area")
    target_roles = roll_up_list(resources, "target_roles", ("name",))
    target_role_names = {role["name"] for role in target_roles}
    exclusions = roll_up_list(resources, "target_roles_exclusions", ("name",), COURSE_EXCLUSION_MIN_SHARE)

    return {
        "reference_resources": collect_reference_resources(resources),
        "learning_level": roll_up_single_value(resources, "learning_level"),
        "learning_mode": roll_up_learning_mode(resources),
        "bloom_taxonomy": roll_up_list(resources, "bloom_taxonomy", ("level",)),
        "primary_competency_area": primary_area,
        **roll_up_competencies(resources, primary_area["value"]),
        "sto_level": roll_up_sto_level(resources),
        "target_employee_groups": roll_up_list(resources, "target_employee_groups", ("group",)),
        "target_roles": target_roles,
        # A role is never both targeted and excluded.
        "target_roles_exclusions": [role for role in exclusions if role["name"] not in target_role_names],
    }


# --- Step 5: the prompt for the text fields ----------------------------------------------------


def describe_course_profile(rolled_up: dict) -> dict:
    """The decisions made by code, shown to the model as context for the summary."""
    competencies = [
        f"{c['theme']} > {c['sub_theme']}"
        for c in rolled_up["behavioural_competencies"] + rolled_up["functional_competencies"]
    ] + [
        " > ".join(part for part in (d["sector"], d["subsector"], d["theme"]) if part)
        for d in rolled_up["domain_competencies"]
    ]
    return {
        "primary_competency_area": rolled_up["primary_competency_area"]["value"],
        "learning_level": rolled_up["learning_level"]["value"],
        "learning_mode": rolled_up["learning_mode"]["value"],
        "sto_level": rolled_up["sto_level"]["value"],
        "competencies": competencies,
    }


def build_user_prompt(course_id: str, resources: list[Resource], rolled_up: dict) -> str:
    """The course profile plus, for each resource, its duration share and the text-field metadata."""
    total = course_duration(resources)
    payload = [
        {
            "duration_share": round(resource["duration_in_sec"] / total, 4),
            **{field: resource[field] for field in TEXT_FIELDS},
        }
        for resource in resources
    ]
    return COURSE_METADATA_USER_PROMPT.format(
        course_id=course_id,
        course_profile_json=json.dumps(describe_course_profile(rolled_up), ensure_ascii=False),
        resources_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


# --- Step 6: the keywords list (built by code from the model's keywords) ----------------------


def is_short_form(text: str) -> bool:
    """An acronym such as "QoS" or "PDU": one word of at most 8 characters with two or more capitals."""
    return " " not in text and len(text) <= 8 and sum(char.isupper() for char in text) >= 2


def split_full_and_short_forms(keyword: Keyword) -> tuple[str, list[str]]:
    """The full form of a keyword and its short forms (no repeats, ignoring case).

    If the model's keyword is itself an acronym, the longest variant that is not an acronym is the full form.
    """
    forms = [keyword.keyword, *keyword.variants]
    full_form = keyword.keyword
    if is_short_form(full_form):
        full_form = max((form for form in forms if not is_short_form(form)), key=len, default=full_form)

    short_forms: dict[str, str] = {}
    for form in forms:
        if is_short_form(form):
            short_forms.setdefault(form.lower(), form)  # the first spelling wins
    return full_form, [form for lowered, form in short_forms.items() if lowered != full_form.lower()]


def format_keyword(keyword: Keyword) -> str:
    """For example: "Quality of Service (QoS, QS) [High, 7 variants]"."""
    full_form, short_forms = split_full_and_short_forms(keyword)
    short_text = f" ({', '.join(short_forms)})" if short_forms else ""
    variant_count = len(keyword.variants)
    return f"{full_form}{short_text} [{keyword.importance}, {variant_count} variant{'' if variant_count == 1 else 's'}]"


def build_keywords_list(keywords: list[Keyword]) -> list[str]:
    """Every keyword as one line, High importance first, then Medium, then Low (the model's order within each)."""
    by_importance = sorted(keywords, key=lambda keyword: IMPORTANCE_ORDER[keyword.importance])
    return [format_keyword(keyword) for keyword in by_importance]


# --- Step 7: calling the model -----------------------------------------------------------------


async def write_text_fields(llm: LLMClient, user_prompts: list[str]) -> list[CourseTextMetadata | Exception]:
    """Ask the model for the text fields of each prompt. A failed prompt gives its exception."""
    return await llm.agenerate_many(user_prompts, CourseTextMetadata)


def build_course_metadata(course_id: str, rolled_up: dict, text: CourseTextMetadata) -> CourseMetadata:
    """Join the fields decided by code with the text fields written by the model, and add `keywords_list`."""
    return CourseMetadata(
        course_id=course_id,
        **rolled_up,
        keywords_list=build_keywords_list(text.keywords),
        **text.model_dump(exclude={"course_id"}),
    )


# --- Step 8: saving the results ----------------------------------------------------------------


def save_rollup(connection: Connection, resource_info: dict, metadata: CourseMetadata) -> None:
    """Insert the course row, or replace it if the course was rolled up before."""
    values = metadata.model_dump(mode="json")
    columns = ["course_id", *resource_info, *COURSE_COLUMNS]
    parameters = [
        metadata.course_id,
        *resource_info.values(),
        *(values[c] if c in TEXT_COLUMNS else Json(values[c]) for c in COURSE_COLUMNS),
    ]
    query = sql.SQL(
        "INSERT INTO public.course_metadata_rollup ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT (course_id) DO UPDATE SET {assignments}"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        assignments=sql.SQL(", ").join(
            sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in columns[1:]
        ),
    )
    with connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)


def save_rollup_file(resource_info: dict, metadata: CourseMetadata) -> None:
    """Write the full roll-up to <COURSE_METADATA_OUTPUT_DIR>/<course_id>.json."""
    COURSE_METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "course_id": metadata.course_id,
        **resource_info,
        **metadata.model_dump(mode="json", exclude={"course_id"}),
    }
    output_path = COURSE_METADATA_OUTPUT_DIR / f"{metadata.course_id}.json"
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Step 9: running ---------------------------------------------------------------------------


def prepare_course(connection: Connection, course_id: str) -> dict:
    """Read a course and do everything that needs no model."""
    resources = fetch_course_resources(connection, course_id)
    fill_missing_durations(resources)
    rolled_up = roll_up_resources(resources)
    return {
        "resource_info": collect_resource_info(resources),
        "rolled_up": rolled_up,
        "prompt": build_user_prompt(course_id, resources, rolled_up),
    }


async def process_batch(connection: Connection, llm: LLMClient, course_ids: list[str]) -> int:
    """Roll up one batch of courses. Returns how many were saved."""
    courses: dict[str, dict] = {}  # course_id -> resource_info, rolled_up and prompt
    for course_id in course_ids:
        try:
            courses[course_id] = prepare_course(connection, course_id)
        except CourseSkippedError as error:
            logger.warning("Skipped %s: %s", course_id, error)
        except psycopg2.Error:
            logger.exception("Could not read %s", course_id)

    results = await write_text_fields(llm, [course["prompt"] for course in courses.values()])

    saved = 0
    for (course_id, course), result in zip(courses.items(), results):
        if isinstance(result, Exception):
            logger.error("Text fields failed for %s: %r", course_id, result)
            continue
        if result.course_id != course_id:
            logger.error("Model returned course_id %s for %s", result.course_id, course_id)
            continue
        try:
            metadata = build_course_metadata(course_id, course["rolled_up"], result)
            save_rollup(connection, course["resource_info"], metadata)
            save_rollup_file(course["resource_info"], metadata)
            saved += 1
        except (ValidationError, psycopg2.Error, OSError):
            logger.exception("Could not save %s", course_id)
    return saved


async def rollup_course_metadata(requested_ids: list[str] | None = None) -> None:
    """Roll up and store course metadata. With no ids, process every course without a roll-up."""
    connection = psycopg2.connect(load_db_dsn())
    try:
        create_rollup_table(connection)
        course_ids = requested_ids or find_pending_course_ids(connection)
        total = len(course_ids)
        if total == 0:
            logger.info("No courses to process.")
            return
        logger.info("Processing %d courses", total)

        settings = load_llm_settings()
        llm = LLMClient(GeminiProvider(settings, COURSE_METADATA_SYSTEM_PROMPT), settings.max_concurrency)

        saved = 0
        for start in range(0, total, BATCH_SIZE):
            batch = course_ids[start : start + BATCH_SIZE]
            saved += await process_batch(connection, llm, batch)
            logger.info("Progress: %d/%d processed, %d saved", start + len(batch), total, saved)
        logger.info("Done. Saved the roll-up of %d of %d courses.", saved, total)
    finally:
        connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(rollup_course_metadata(COURSE_IDS))
