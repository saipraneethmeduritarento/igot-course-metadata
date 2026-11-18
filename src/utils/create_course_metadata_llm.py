"""Build public.course_metadata_llm: a flat, query-friendly view of public.course_metadata_rollup.

One row per course. Every judged field keeps only its value (no confidence or rationale), and
every list of objects keeps only its name:

    speakers                   <- presenters[].name, unique (ignoring case)
    language                   <- language (the dominant language of the course)
    keywords                   <- keywords[].keyword, unique, High importance first
    prerequisites              <- prerequisites[].topic, unique
    course_summary             <- course_summary
    course_description         <- description
    learning_outcomes          <- learning_outcomes
    learning_level             <- learning_level.value
    learning_mode              <- learning_mode.value
    blooms_taxonomy            <- bloom_taxonomy[].level
    primary_competency_area    <- primary_competency_area.value
    theme, sub_theme           <- the top competency of the primary area:
                                    Behavioural / Functional: theme, sub_theme
                                    Domain:                   sector, subsector
    sto                        <- sto_level.value
    responsibilities_addressed <- responsibilities_addressed
    activities_enabled         <- activities_enabled
    target_employee_group      <- target_employee_groups[].group
    target_roles               <- target_roles[].name
    target_roles_exclusions    <- target_roles_exclusions[].name
    references                 <- reference_resources

Single values are TEXT. Lists are JSONB arrays, e.g. speakers = ["Asha Rao", "Vikram Shah"].

`references` is an SQL keyword, so quote it in queries: SELECT "references" FROM ...

Re-running is safe: a course already in the table is overwritten (upsert).

Run src.course_metadata.extraction first. Then, from the project root:
    python -m src.utils.create_course_metadata_llm                 # every rolled-up course
    python -m src.utils.create_course_metadata_llm do_1 do_2       # only these courses
"""

import argparse
import logging
from collections.abc import Iterable

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from src.config import load_db_dsn

logger = logging.getLogger(__name__)

SOURCE_TABLE = "course_metadata_rollup"
TARGET_TABLE = "course_metadata_llm"

# Columns read from the roll-up table.
SOURCE_COLUMNS = [
    "course_id",
    "language",
    "presenters",
    "keywords",
    "prerequisites",
    "course_summary",
    "description",
    "learning_outcomes",
    "learning_level",
    "learning_mode",
    "bloom_taxonomy",
    "primary_competency_area",
    "behavioural_competencies",
    "functional_competencies",
    "domain_competencies",
    "sto_level",
    "responsibilities_addressed",
    "activities_enabled",
    "target_employee_groups",
    "target_roles",
    "target_roles_exclusions",
    "reference_resources",
]

# Columns of course_metadata_llm, in table order, with their SQL type.
TARGET_COLUMNS = {
    "course_id": "TEXT PRIMARY KEY",
    "speakers": "JSONB",
    "language": "TEXT",
    "keywords": "JSONB",
    "prerequisites": "JSONB",
    "course_summary": "TEXT",
    "course_description": "TEXT",
    "learning_outcomes": "JSONB",
    "learning_level": "TEXT",
    "learning_mode": "TEXT",
    "blooms_taxonomy": "JSONB",
    "primary_competency_area": "TEXT",
    "theme": "TEXT",
    "sub_theme": "TEXT",
    "sto": "TEXT",
    "responsibilities_addressed": "JSONB",
    "activities_enabled": "JSONB",
    "target_employee_group": "JSONB",
    "target_roles": "JSONB",
    "target_roles_exclusions": "JSONB",
    "references": "JSONB",
}

# primary_competency_area -> (roll-up list field, key used as theme, key used as sub_theme)
COMPETENCY_FIELDS = {
    "Behavioural": ("behavioural_competencies", "theme", "sub_theme"),
    "Functional": ("functional_competencies", "theme", "sub_theme"),
    "Domain": ("domain_competencies", "sector", "subsector"),
}

IMPORTANCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}


# --- Turning one roll-up row into one course_metadata_llm row ----------------------------------


def unique(items: Iterable[str | None]) -> list[str]:
    """The non-blank items without repeats (ignoring case and outer spaces); the first spelling wins."""
    seen: dict[str, str] = {}
    for item in items:
        text = (item or "").strip()
        if text:
            seen.setdefault(text.casefold(), text)
    return list(seen.values())


def values_of(items: list[dict] | None, key: str) -> list[str]:
    """The `key` of every object of a roll-up list, unique, in the roll-up's order."""
    return unique(item.get(key) for item in items or [])


def value_of(judged: dict | None) -> str | None:
    """The `value` of a judged field such as {"value": "Beginner", "confidence": ..., "rationale": ...}."""
    return judged.get("value") if judged else None


def unique_keywords(keywords: list[dict] | None) -> list[str]:
    """The keyword names, High importance first, then Medium, then Low (the roll-up's order within each)."""
    by_importance = sorted(keywords or [], key=lambda k: IMPORTANCE_ORDER.get(k.get("importance"), len(IMPORTANCE_ORDER)))
    return values_of(by_importance, "keyword")


def top_competency(rollup: dict) -> tuple[str | None, str | None]:
    """(theme, sub_theme) of the highest-confidence competency of the course's primary area.

    Behavioural and Functional give the KCM theme and sub-theme; Domain gives the sector and subsector.
    (None, None) if the area is unknown or has no competency.
    """
    area = value_of(rollup.get("primary_competency_area"))
    if area not in COMPETENCY_FIELDS:
        return None, None
    field, theme_key, sub_theme_key = COMPETENCY_FIELDS[area]
    competencies = rollup.get(field) or []
    if not competencies:
        return None, None
    top = max(competencies, key=lambda c: c.get("confidence") or 0)  # max keeps the first on a tie
    return top.get(theme_key), top.get(sub_theme_key)


def to_llm_row(rollup: dict) -> dict:
    """One course_metadata_llm row (column -> value) from one course_metadata_rollup row."""
    theme, sub_theme = top_competency(rollup)
    return {
        "course_id": rollup["course_id"],
        "speakers": values_of(rollup.get("presenters"), "name"),
        "language": rollup.get("language"),
        "keywords": unique_keywords(rollup.get("keywords")),
        "prerequisites": values_of(rollup.get("prerequisites"), "topic"),
        "course_summary": rollup.get("course_summary"),
        "course_description": rollup.get("description"),
        "learning_outcomes": list(rollup.get("learning_outcomes") or []),
        "learning_level": value_of(rollup.get("learning_level")),
        "learning_mode": value_of(rollup.get("learning_mode")),
        "blooms_taxonomy": values_of(rollup.get("bloom_taxonomy"), "level"),
        "primary_competency_area": value_of(rollup.get("primary_competency_area")),
        "theme": theme,
        "sub_theme": sub_theme,
        "sto": value_of(rollup.get("sto_level")),
        "responsibilities_addressed": list(rollup.get("responsibilities_addressed") or []),
        "activities_enabled": list(rollup.get("activities_enabled") or []),
        "target_employee_group": values_of(rollup.get("target_employee_groups"), "group"),
        "target_roles": values_of(rollup.get("target_roles"), "name"),
        "target_roles_exclusions": values_of(rollup.get("target_roles_exclusions"), "name"),
        "references": unique(rollup.get("reference_resources") or []),
    }


# --- Database ----------------------------------------------------------------------------------


def create_llm_table(connection) -> None:
    """Create public.course_metadata_llm if it does not exist."""
    query = sql.SQL("CREATE TABLE IF NOT EXISTS public.{table} ({columns}, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())").format(
        table=sql.Identifier(TARGET_TABLE),
        columns=sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(sql_type)) for name, sql_type in TARGET_COLUMNS.items()
        ),
    )
    with connection, connection.cursor() as cursor:
        cursor.execute(query)


def fetch_rollups(connection, course_ids: list[str] | None = None) -> list[dict]:
    """The roll-up rows of the given courses, or of every course if none are given."""
    query = sql.SQL("SELECT {columns} FROM public.{table}").format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in SOURCE_COLUMNS),
        table=sql.Identifier(SOURCE_TABLE),
    )
    parameters: tuple = ()
    if course_ids:
        query += sql.SQL(" WHERE course_id = ANY(%s)")
        parameters = (list(course_ids),)
    query += sql.SQL(" ORDER BY course_id")
    with connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        return [dict(zip(SOURCE_COLUMNS, row)) for row in cursor.fetchall()]


def save_rows(connection, rows: list[dict]) -> None:
    """Insert the rows, replacing any course that is already in the table."""
    if not rows:
        return
    columns = list(TARGET_COLUMNS)
    names = [sql.Identifier(c) for c in columns]
    query = sql.SQL(
        "INSERT INTO public.{table} ({columns}) VALUES %s "
        "ON CONFLICT (course_id) DO UPDATE SET {assignments}, updated_at = now()"
    ).format(
        table=sql.Identifier(TARGET_TABLE),
        columns=sql.SQL(", ").join(names),
        assignments=sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(n) for n in names[1:]),
    )
    with connection, connection.cursor() as cursor:
        execute_values(cursor, query.as_string(cursor), [to_parameters(row, columns) for row in rows])


def to_parameters(row: dict, columns: list[str]) -> tuple:
    """The row's values in column order, with the lists wrapped for the JSONB columns."""
    return tuple(Json(row[c]) if TARGET_COLUMNS[c] == "JSONB" else row[c] for c in columns)


def create_course_metadata_llm(course_ids: list[str] | None = None) -> int:
    """Build or refresh public.course_metadata_llm from the roll-up. Returns how many courses were saved."""
    connection = psycopg2.connect(load_db_dsn())
    try:
        create_llm_table(connection)
        rollups = fetch_rollups(connection, course_ids)
        if course_ids:
            missing = sorted(set(course_ids) - {r["course_id"] for r in rollups})
            if missing:
                logger.warning("No roll-up for %d course(s): %s", len(missing), ", ".join(missing))
        rows = [to_llm_row(rollup) for rollup in rollups]
        save_rows(connection, rows)
        logger.info("public.%s: saved %d course(s).", TARGET_TABLE, len(rows))
        return len(rows)
    finally:
        connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("course_ids", nargs="*", help="courses to process (default: every rolled-up course)")
    args = parser.parse_args()
    create_course_metadata_llm(args.course_ids or None)
