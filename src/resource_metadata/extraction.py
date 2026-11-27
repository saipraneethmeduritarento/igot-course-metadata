"""Resource-level metadata extraction.

For each resource: read its transcripts from public.resource_transcripts, ask the LLM for metadata,
store one row in public.resource_metadata_<version> (one column per metadata field), and write the full
response to <RESOURCE_METADATA_OUTPUT_DIR>/<model>_<prompt_version>/<resource_id>.json.
public.resource_transcripts itself is only read, never changed.

Every row is keyed by (resource_id, model_used, prompt_version): a run with another model or
prompt version (RESOURCE_PROMPT_VERSION in src/config.py) adds rows next to the old ones, and
"already done" means done with this model and prompt version.

Run from the project root (see --help for the options):
    python -m src.resource_metadata.extraction                      # the resources in the sample CSV
    python -m src.resource_metadata.extraction --all --concurrency 8 --batch-size 100
    python -m src.resource_metadata.extraction --ids-file <failed_*.txt>
"""

import asyncio
import logging
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from src.config import (
    DEFAULT_MODEL_ENV_VAR,
    RESOURCE_METADATA_OUTPUT_DIR,
    RESOURCE_PROMPT_VERSION,
    load_db_dsn,
    load_llm_settings,
)
from src.llm import GeminiProvider, LLMClient
from src.prompt_versions import prompts_for
from src.reference_data import load_designation_names, load_designations, load_kcm_catalogue, load_sgos_tree
from src.resource_metadata.schema import Designation, ResourceMetadata
from src.runner import (
    RunSummary,
    build_arg_parser,
    log_summary,
    parse_args,
    run_pool,
    selected_ids,
    write_retry_file,
)

logger = logging.getLogger(__name__)

Connection = psycopg2.extensions.connection
# `model_used` and `prompt_version` of a run.
RunInfo = dict[str, str]

# Default for --batch-size: resources read from the database at a time. The number of model calls
# in flight is --concurrency (default MAX_CONCURRENCY), not this.
DEFAULT_BATCH_SIZE = 20

RESOURCE_PROMPTS = prompts_for(RESOURCE_PROMPT_VERSION)

# Where the metadata is stored: one row per resource, model and prompt version. Named after the
# prompt version, so each version has its own table (resource_metadata_v3_1 for v3.1: a "." in
# the version is written as "_", so the name needs no quoting in SQL).
METADATA_TABLE = f"resource_metadata_{RESOURCE_PROMPT_VERSION.replace('.', '_')}"
# Which model and prompt version produced a row. Together with resource_id, the primary key.
RUN_COLUMNS = ["model_used", "prompt_version"]

# One table column per ResourceMetadata field. The model never returns resource_id: code owns it.
METADATA_COLUMNS = list(ResourceMetadata.model_fields)
TEXT_COLUMNS = {"course_coverage", "description"}  # every other metadata column is JSONB

# Matches `[Segment: Video Time: ...]`, with or without the surrounding `**`.
SEGMENT_MARKER = re.compile(r"(?:\*\*)?\[Segment:[^\]]*\](?:\*\*)?")


class ResourceSkippedError(Exception):
    """The resource cannot be processed (missing row or no transcript)."""


# --- Choosing resources ------------------------------------------------------------------------


def find_pending_resource_ids(connection: Connection, run_info: RunInfo) -> list[str]:
    """Resources that have no metadata yet from this model and prompt version, and have at least
    one transcript.

    Resources without any transcript are left out (they are never sent to the model);
    how many were left out is logged.
    """
    has_transcript = "(btrim(COALESCE(t.video_transcript, '')) <> '' OR btrim(COALESCE(t.audio_transcript, '')) <> '')"
    query = sql.SQL(
        "SELECT t.resource_id, " + has_transcript + " FROM public.resource_transcripts t "
        "WHERE NOT EXISTS (SELECT 1 FROM public.{table} m WHERE m.resource_id = t.resource_id "
        "AND m.model_used = %s AND m.prompt_version = %s) "
        "ORDER BY t.resource_id"
    ).format(table=sql.Identifier(METADATA_TABLE))
    with connection, connection.cursor() as cursor:
        cursor.execute(query, (run_info["model_used"], run_info["prompt_version"]))
        rows = cursor.fetchall()

    pending = [resource_id for resource_id, has in rows if has]
    if len(pending) < len(rows):
        logger.info("Left out %d resources with no transcript.", len(rows) - len(pending))
    return pending


def leave_out_done(connection: Connection, resource_ids: list[str], run_info: RunInfo) -> list[str]:
    """`resource_ids` without the resources that already have metadata from this model and
    prompt version (order kept).

    A rerun with the same list then only does what is left, instead of paying for every
    resource again.
    """
    query = sql.SQL(
        "SELECT resource_id FROM public.{table} "
        "WHERE resource_id = ANY(%s) AND model_used = %s AND prompt_version = %s"
    ).format(table=sql.Identifier(METADATA_TABLE))
    with connection, connection.cursor() as cursor:
        cursor.execute(query, (resource_ids, run_info["model_used"], run_info["prompt_version"]))
        done = {row[0] for row in cursor.fetchall()}
    if done:
        logger.info("Skipping %d resources that already have metadata.", len(done))
    return [resource_id for resource_id in resource_ids if resource_id not in done]


# --- Building the prompts ----------------------------------------------------------------------


def build_system_prompt() -> str:
    """The system prompt with the reference data filled in. Same for every resource."""
    return (
        RESOURCE_PROMPTS.RESOURCE_METADATA_SYSTEM_PROMPT.replace("{{KCM_CATALOGUE}}", load_kcm_catalogue())
        .replace("{{SGOS_TREE}}", load_sgos_tree())
        .replace("{{DESIGNATION_LIST}}", load_designations())
    )


def clean_video_transcript(text: str | None) -> str:
    """Remove the segment markers and extra whitespace."""
    return re.sub(r"\s+", " ", SEGMENT_MARKER.sub(" ", text or "")).strip()


def fetch_user_prompts(connection: Connection, resource_ids: list[str]) -> dict[str, str]:
    """Read the transcripts of `resource_ids` in one query and build each user prompt.

    A resource that is not in resource_transcripts, or has no usable transcript, is left out
    (and logged).
    """
    with connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT resource_id, video_transcript, audio_transcript "
            "FROM public.resource_transcripts WHERE resource_id = ANY(%s)",
            (resource_ids,),
        )
        rows = {resource_id: transcripts for resource_id, *transcripts in cursor.fetchall()}

    prompts: dict[str, str] = {}
    for resource_id in resource_ids:
        try:
            if resource_id not in rows:
                raise ResourceSkippedError("resource not found in resource_transcripts")
            prompts[resource_id] = build_user_prompt(*rows[resource_id])
        except ResourceSkippedError as error:
            logger.warning("Skipped %s: %s", resource_id, error)
    return prompts


def build_user_prompt(video_transcript: str | None, audio_transcript: str | None) -> str:
    """The user prompt for a resource's transcripts. Raises ResourceSkippedError if both are empty."""
    transcripts = {
        "video_transcript": clean_video_transcript(video_transcript),
        "audio_transcript": (audio_transcript or "").strip(),
    }
    present = {name: text for name, text in transcripts.items() if text}
    if not present:
        raise ResourceSkippedError("no usable transcript")

    transcript_blocks = [f"<{name}>\n{text}\n</{name}>" for name, text in present.items()]
    return RESOURCE_PROMPTS.RESOURCE_METADATA_USER_PROMPT.format(
        transcripts="\n\n".join(transcript_blocks)
    )


# --- Checking designations ---------------------------------------------------------------------

COMPACT_FORM = re.compile(r"^(?P<base>[^()]+?) \((?P<qualifiers>[^()]* \| [^()]*)\)$")


def clean_designations(
    roles: list[Designation], valid_names: frozenset[str], resource_id: str, field: str
) -> list[Designation]:
    """Keep only valid designation names.

    A compact `Base (A | B)` answer is expanded into `Base (A)`, `Base (B)` (each keeps the
    confidence and rationale). Any name that is still not in the list is dropped. Duplicates are removed.
    """
    cleaned: dict[str, Designation] = {}
    for role in roles:
        name = role.name.strip()
        candidates = [name]
        match = COMPACT_FORM.match(name)
        if match:
            candidates = [f"{match['base']} ({q.strip()})" for q in match["qualifiers"].split("|")]
        for candidate in candidates:
            if candidate in valid_names:
                cleaned.setdefault(candidate, role.model_copy(update={"name": candidate}))
            else:
                logger.warning("Dropped invalid designation %r in %s for %s", candidate, field, resource_id)
    return list(cleaned.values())


def validate_designations(
    metadata: ResourceMetadata, valid_names: frozenset[str], resource_id: str
) -> ResourceMetadata:
    """Return the metadata with only valid designations, and none in both role lists."""
    roles = clean_designations(metadata.target_roles, valid_names, resource_id, "target_roles")
    role_names = {role.name for role in roles}
    exclusions = [
        role
        for role in clean_designations(
            metadata.target_roles_exclusions, valid_names, resource_id, "target_roles_exclusions"
        )
        if role.name not in role_names
    ]
    return metadata.model_copy(update={"target_roles": roles, "target_roles_exclusions": exclusions})


# --- Saving the results ------------------------------------------------------------------------


CREATE_TABLE_SQL = sql.SQL(
    "CREATE TABLE IF NOT EXISTS public.{table} ("
    "resource_id TEXT NOT NULL, model_used TEXT NOT NULL, prompt_version TEXT NOT NULL, "
    "extracted_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "PRIMARY KEY (resource_id, model_used, prompt_version))"
).format(table=sql.Identifier(METADATA_TABLE))


def ensure_metadata_table(connection: Connection) -> None:
    """Create public.resource_metadata_<version> if needed, and add any metadata column it lacks.

    A field added to ResourceMetadata later shows up as a new column without dropping the table.
    Run before choosing resources and before calling the model, so a missing table or column
    cannot waste a billed run.
    """
    table = sql.Identifier(METADATA_TABLE)
    with connection, connection.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL)
        for column in METADATA_COLUMNS:
            column_type = "TEXT" if column in TEXT_COLUMNS else "JSONB"
            cursor.execute(
                sql.SQL("ALTER TABLE public.{} ADD COLUMN IF NOT EXISTS {} {}").format(
                    table, sql.Identifier(column), sql.SQL(column_type)
                )
            )


def save_metadata(
    connection: Connection, resource_id: str, metadata: ResourceMetadata, run_info: RunInfo
) -> None:
    """Insert the resource's row, or replace it if it was extracted before with the same model
    and prompt version. A row from another model or prompt version is kept."""
    values = metadata.model_dump(mode="json")
    key_columns = ["resource_id", *RUN_COLUMNS]
    columns = [*key_columns, *METADATA_COLUMNS]
    query = sql.SQL(
        "INSERT INTO public.{table} ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT ({key_columns}) DO UPDATE SET {assignments}, extracted_at = now()"
    ).format(
        table=sql.Identifier(METADATA_TABLE),
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        key_columns=sql.SQL(", ").join(sql.Identifier(column) for column in key_columns),
        assignments=sql.SQL(", ").join(
            sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(column))
            for column in METADATA_COLUMNS
        ),
    )
    parameters = [
        values[column] if column in TEXT_COLUMNS else Json(values[column]) for column in METADATA_COLUMNS
    ]

    with connection, connection.cursor() as cursor:
        cursor.execute(query, [resource_id, *(run_info[c] for c in RUN_COLUMNS), *parameters])


def run_output_dir(run_info: RunInfo) -> Path:
    """<RESOURCE_METADATA_OUTPUT_DIR>/<model_used>_<prompt_version>: the files of one model and
    prompt version, and the failed_*.txt id lists of its runs."""
    return RESOURCE_METADATA_OUTPUT_DIR / f"{run_info['model_used']}_{run_info['prompt_version']}"


def metadata_file_path(resource_id: str, run_info: RunInfo) -> Path:
    """<run_output_dir>/<resource_id>.json, so a run with another model or prompt version never
    overwrites the files of an earlier one."""
    return run_output_dir(run_info) / f"{resource_id}.json"


def save_metadata_file(resource_id: str, metadata: ResourceMetadata, run_info: RunInfo) -> None:
    """Write the full response to its metadata_file_path."""
    output_path = metadata_file_path(resource_id, run_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


# --- Running -----------------------------------------------------------------------------------


async def process_resource(
    connection: Connection,
    llm: LLMClient,
    resource_id: str,
    user_prompt: str,
    run_info: RunInfo,
    valid_names: frozenset[str],
) -> None:
    """Ask the model for the resource's metadata and save it. Raises if either fails."""
    metadata = await llm.agenerate(user_prompt, ResourceMetadata)
    metadata = validate_designations(metadata, valid_names, resource_id)
    # File first: if the database save fails, the billed response is still kept.
    save_metadata_file(resource_id, metadata, run_info)
    save_metadata(connection, resource_id, metadata, run_info)


async def extract_resource_metadata(
    requested_ids: list[str] | None = None,
    skip_done: bool = True,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int | None = None,
    limit: int | None = None,
    model_env_var: str = DEFAULT_MODEL_ENV_VAR,
) -> RunSummary:
    """Extract and store metadata.

    With ids, process those that have no metadata yet from this model and prompt version (all of
    them with skip_done=False, which replaces their stored metadata). With None, process every
    resource without metadata from this model and prompt version. `limit` caps how many are
    processed; `concurrency` (default MAX_CONCURRENCY) how many model calls run at once.
    """
    # Only reads the settings; nothing is billed until the LLM client below is built.
    settings = load_llm_settings(model_env_var)
    concurrency = concurrency or settings.max_concurrency
    run_info = {"model_used": settings.model, "prompt_version": RESOURCE_PROMPTS.PROMPT_VERSION}
    connection = psycopg2.connect(load_db_dsn())
    try:
        # First: find_pending_resource_ids reads the table this creates.
        ensure_metadata_table(connection)
        if requested_ids is None:
            resource_ids = find_pending_resource_ids(connection, run_info)
        elif skip_done:
            resource_ids = leave_out_done(connection, requested_ids, run_info)
        else:
            resource_ids = list(requested_ids)
        resource_ids = resource_ids[:limit] if limit else resource_ids
        if not resource_ids:
            logger.info("No resources to process.")
            return RunSummary()
        logger.info(
            "Processing %d resources with %s, prompt %s: %d calls at a time, batches of %d",
            len(resource_ids),
            run_info["model_used"],
            run_info["prompt_version"],
            concurrency,
            batch_size,
        )

        # Built after the resources are known: this creates the (billed) prompt cache.
        llm = LLMClient(GeminiProvider(settings, build_system_prompt()), concurrency, settings.retry)
        valid_names = load_designation_names()

        async def process_one(resource_id: str, user_prompt: str) -> None:
            await process_resource(connection, llm, resource_id, user_prompt, run_info, valid_names)

        summary = await run_pool(
            resource_ids,
            lambda batch: fetch_user_prompts(connection, batch),
            process_one,
            concurrency=concurrency,
            batch_size=batch_size,
            label="resources",
        )
        retry_file = write_retry_file(summary, run_output_dir(run_info))
        log_summary(summary, "resources", retry_file)
        return summary
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Extract resource-level metadata with the LLM.", DEFAULT_BATCH_SIZE, "resources"
    )
    args = parse_args(parser, argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = asyncio.run(
        extract_resource_metadata(
            selected_ids(args, "resource_id"),
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
