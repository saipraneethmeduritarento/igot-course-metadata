"""Load the LLM metadata JSON files into Postgres: one row per resource / course, one column per key.

    public.resource_metadata_output  <- data/output/resource_metadata_output/<resource_id>.json
    public.course_metadata_output    <- data/output/course_metadata_output/<course_id>.json

The columns are derived from the pydantic schemas (ResourceMetadata, CourseMetadata), flattening
every key of every object into its own column:

    course_summary: "..."                         -> course_summary              TEXT
    learning_outcomes: ["a", "b"]                 -> learning_outcomes           TEXT[]
    learning_level: {value, confidence, rationale}-> learning_level              TEXT
                                                     learning_level_confidence   NUMERIC
                                                     learning_level_rationale    TEXT
    presenters: [{name, designation, organisation}] -> presenters_name           TEXT[]
                                                     presenters_designation      TEXT[]
                                                     presenters_organisation     TEXT[]
    sto_level: {..., evidence: [{phrase, ...}]}   -> sto_level_evidence_phrase   TEXT[]  (etc.)
    target_roles: [{name, confidence, rationale}] -> target_roles                JSONB   (KEEP_AS_JSON)

The arrays of one list line up: element i of presenters_name, presenters_designation and
presenters_organisation all describe the same presenter. A list nested inside a list cannot be a
flat array, so it is stored as JSONB. A key missing from a file is NULL.

The fields in KEEP_AS_JSON are not split: each is one JSONB column holding the list exactly as it
is in the JSON file, e.g. keywords = [{"keyword": ..., "type": ..., "importance": ..., "variants": [...]}].

Re-running is safe: a resource/course already in the table is overwritten (upsert).

    python -m src.utils.load_metadata_to_postgres                 # both levels
    python -m src.utils.load_metadata_to_postgres --level course  # one level
"""

import argparse
import json
import typing
from pathlib import Path

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel

from src.config import COURSE_METADATA_OUTPUT_DIR, RESOURCE_METADATA_OUTPUT_DIR, load_db_dsn
from src.course_metadata.schema import CourseMetadata
from src.resource_metadata.schema import ResourceMetadata
from src.utils.load_transcripts_to_postgres import create_database

# Course columns added by code in src/course_metadata/extraction.py, not part of CourseMetadata.
COURSE_EXTRA_COLUMNS = [
    ("resource_ids", "TEXT[]", ("resource_ids",)),
    ("language", "TEXT", ("language",)),
    ("list_of_languages", "TEXT[]", ("list_of_languages",)),
]


# Fields stored as one JSONB column, the objects exactly as in the JSON, instead of a column per key.
KEEP_AS_JSON = {
    "keywords",
    "prerequisites",
    "bloom_taxonomy",
    "target_employee_groups",
    "target_roles",
    "target_roles_exclusions",
}


def _is_model(annotation):
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _scalar_type(annotation):
    return "NUMERIC(4, 3)" if annotation is float else "TEXT"


def flatten_columns(name, annotation, path, in_list=False):
    """Columns (name, sql type, path of JSON keys) for one schema field, one per leaf key."""
    if _is_model(annotation):
        columns = []
        for key, field in annotation.model_fields.items():
            column = name if key == "value" else f"{name}_{key}"
            columns += flatten_columns(column, field.annotation, (*path, key), in_list)
        return columns
    if typing.get_origin(annotation) is list:
        (item,) = typing.get_args(annotation)
        if in_list:  # a list inside a list is ragged, so it cannot be a flat array
            return [(name, "JSONB", path)]
        if _is_model(item):
            return flatten_columns(name, item, path, in_list=True)
        return [(name, "TEXT[]", path)]
    sql_type = _scalar_type(annotation)
    return [(name, f"{sql_type}[]" if in_list else sql_type, path)]


def schema_columns(model, id_col, extra=()):
    """All columns of a level's table: the id first, then the extras, then every schema field flattened."""
    columns = [(id_col, "TEXT", (id_col,)), *extra]
    for name, field in model.model_fields.items():
        if name == id_col:
            continue
        if name in KEEP_AS_JSON:
            columns.append((name, "JSONB", (name,)))
        else:
            columns += flatten_columns(name, field.annotation, (name,))
    return columns


# level -> (table, id column, directory of JSON files, columns)
LEVELS = {
    "resource": (
        "resource_metadata_output", "resource_id", RESOURCE_METADATA_OUTPUT_DIR,
        schema_columns(ResourceMetadata, "resource_id"),
    ),
    "course": (
        "course_metadata_output", "course_id", COURSE_METADATA_OUTPUT_DIR,
        schema_columns(CourseMetadata, "course_id", COURSE_EXTRA_COLUMNS),
    ),
}


def pick(obj, path):
    """Follow the keys of `path` into obj, mapping over every list on the way."""
    if not path or obj is None:
        return obj
    if isinstance(obj, list):
        return [pick(o, path) for o in obj]
    return pick(obj.get(path[0]), path[1:])


def normalise_course(record):
    """Older course files hold the list of languages in `language`; newer ones split it in two."""
    language = record.get("language")
    if isinstance(language, list):
        record = {**record, "language": language[0] if language else None}
        record.setdefault("list_of_languages", language)
    return record


def to_row(record, columns):
    """One table row (a tuple in column order) for one metadata JSON object."""
    row = []
    for _, sql_type, path in columns:
        value = pick(record, path)
        row.append(Json(value) if sql_type == "JSONB" and value is not None else value)
    return tuple(row)


def read_metadata_files(directory):
    """Every *.json file in the directory, parsed, in file-name order."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.json"))]


def load_level(conn, level, directory=None, batch_size=500):
    """Create the table for one level (adding any new schema column) and upsert every file of its directory."""
    table, id_col, default_dir, columns = LEVELS[level]
    records = read_metadata_files(directory or default_dir)
    if level == "course":
        records = [normalise_course(r) for r in records]
    rows = [to_row(r, columns) for r in records]

    table_id = sql.Identifier(table)
    names = [sql.Identifier(c) for c, _, _ in columns]
    create = sql.SQL("CREATE TABLE IF NOT EXISTS public.{} ({} TEXT PRIMARY KEY, loaded_at TIMESTAMPTZ NOT NULL DEFAULT now())").format(
        table_id, sql.Identifier(id_col)
    )
    # ADD COLUMN IF NOT EXISTS, so a field added to the schema later shows up without dropping the table.
    add_columns = [
        sql.SQL("ALTER TABLE public.{} ADD COLUMN IF NOT EXISTS {} {}").format(table_id, name, sql.SQL(sql_type))
        for name, (_, sql_type, _) in zip(names[1:], columns[1:])
    ]
    insert = sql.SQL(
        "INSERT INTO public.{table} ({cols}) VALUES %s "
        "ON CONFLICT ({id}) DO UPDATE SET {updates}, loaded_at = now()"
    ).format(
        table=table_id,
        cols=sql.SQL(", ").join(names),
        id=sql.Identifier(id_col),
        updates=sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(n) for n in names[1:]),
    )

    with conn, conn.cursor() as cur:
        cur.execute(create)
        for statement in add_columns:
            cur.execute(statement)
        insert_sql = insert.as_string(cur)
        for i in range(0, len(rows), batch_size):
            execute_values(cur, insert_sql, rows[i:i + batch_size])

    print(f"public.{table}: {len(rows)} rows, {len(columns)} columns.")
    return len(rows)


def load_metadata_to_postgres(levels=tuple(LEVELS)):
    dsn = load_db_dsn()
    create_database(dsn)
    conn = psycopg2.connect(dsn)
    try:
        return {level: load_level(conn, level) for level in levels}
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", choices=[*LEVELS, "both"], default="both")
    args = parser.parse_args()
    load_metadata_to_postgres(tuple(LEVELS) if args.level == "both" else (args.level,))
