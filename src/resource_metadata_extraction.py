"""Resource-level metadata extraction.

For each resource: read its transcripts from Postgres, ask the LLM for metadata,
store each metadata field in its own column of the same row,
and write the full response to <RESOURCE_METADATA_OUTPUT_DIR>/<resource_id>.json.

Run from the project root:  python -m src.resource_metadata_extraction
"""

import asyncio
import logging
import re

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from src.config import RESOURCE_METADATA_OUTPUT_DIR, load_db_dsn, load_llm_settings
from src.llm import GeminiProvider, LLMClient
from src.prompts import RESOURCE_METADATA_SYSTEM_PROMPT, RESOURCE_METADATA_USER_PROMPT
from src.reference_data import load_designations, load_kcm_catalogue, load_sgos_tree
from src.resource_metadata_schema import ResourceMetadata

logger = logging.getLogger(__name__)

Connection = psycopg2.extensions.connection

# Resources to process. Leave empty to process every resource without metadata yet.
RESOURCE_IDS = [
    "do_1137567857355489281291",
    "do_1137567851239751681435",
    "do_1137567832409784321299",
    "do_1137567823583232001433",
    "do_113756779784151040187",
    "do_1137567793080729601289",
    "do_113756778891182080185",
    "do_1137567747670343681287",
    "do_1137129168580444161100",
    "do_113712913889419264199",
    "do_1137129030786007041113",
    "do_113712901135106048197",
    "do_1137128938371973121110",
    "do_113712870807076864195",
    "do_11371286260652441617",
    "do_1137128602693468161107",
    "do_11371285559701504016",
    "do_11371285322841292814",
    "do_113712850390204416190",
]

BATCH_SIZE = 20  # resources sent to the model together; results are saved after each batch

# One table column per ResourceMetadata field (except resource_id, which identifies the row).
METADATA_COLUMNS = [name for name in ResourceMetadata.model_fields if name != "resource_id"]
TEXT_COLUMNS = {"course_summary"}  # every other metadata column is JSONB

# Matches `[Segment: Video Time: ...]`, with or without the surrounding `**`.
SEGMENT_MARKER = re.compile(r"(?:\*\*)?\[Segment:[^\]]*\](?:\*\*)?")


class ResourceSkippedError(Exception):
    """The resource cannot be processed (missing row or no transcript)."""


# --- Choosing resources ------------------------------------------------------------------------


def find_pending_resource_ids(connection: Connection) -> list[str]:
    """Resources that have no metadata yet."""
    with connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT resource_id FROM public.resource_transcripts "
            "WHERE course_summary IS NULL ORDER BY resource_id"
        )
        return [row[0] for row in cursor.fetchall()]


# --- Building the prompts ----------------------------------------------------------------------


def build_system_prompt() -> str:
    """The system prompt with the reference data filled in. Same for every resource."""
    return (
        RESOURCE_METADATA_SYSTEM_PROMPT.replace("{{KCM_CATALOGUE}}", load_kcm_catalogue())
        .replace("{{SGOS_TREE}}", load_sgos_tree())
        .replace("{{DESIGNATION_LIST}}", load_designations())
    )


def clean_video_transcript(text: str | None) -> str:
    """Remove the segment markers and extra whitespace."""
    return re.sub(r"\s+", " ", SEGMENT_MARKER.sub(" ", text or "")).strip()


def fetch_user_prompt(connection: Connection, resource_id: str) -> str:
    """Read the resource's transcripts from the database and build the user prompt."""
    with connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT video_transcript, audio_transcript FROM public.resource_transcripts "
            "WHERE resource_id = %s",
            (resource_id,),
        )
        row = cursor.fetchone()

    if row is None:
        raise ResourceSkippedError("resource not found in resource_transcripts")

    video_transcript, audio_transcript = row
    transcripts = {
        "video_transcript": clean_video_transcript(video_transcript),
        "audio_transcript": (audio_transcript or "").strip(),
    }
    transcript_blocks = [f"<{name}>\n{text}\n</{name}>" for name, text in transcripts.items() if text]
    if not transcript_blocks:
        raise ResourceSkippedError("no usable transcript")

    return RESOURCE_METADATA_USER_PROMPT.format(
        resource_id=resource_id, transcripts="\n\n".join(transcript_blocks)
    )


# --- Calling the model -------------------------------------------------------------------------


async def extract_metadata(
    llm: LLMClient, user_prompts: list[str]
) -> list[ResourceMetadata | Exception]:
    """Ask the model for the metadata of each prompt. A failed prompt gives its exception."""
    return await llm.agenerate_many(user_prompts, ResourceMetadata)


# --- Saving the results ------------------------------------------------------------------------


def save_metadata(connection: Connection, resource_id: str, metadata: ResourceMetadata) -> None:
    """Store each metadata field in its own column of the resource's row."""
    values = metadata.model_dump(mode="json")
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = %s").format(sql.Identifier(column)) for column in METADATA_COLUMNS
    )
    query = sql.SQL(
        "UPDATE public.resource_transcripts SET {} WHERE resource_id = %s"
    ).format(assignments)
    parameters = [
        values[column] if column in TEXT_COLUMNS else Json(values[column]) for column in METADATA_COLUMNS
    ]

    with connection, connection.cursor() as cursor:
        cursor.execute(query, [*parameters, resource_id])
        if cursor.rowcount != 1:
            raise ResourceSkippedError("resource not found when saving")


def save_metadata_file(resource_id: str, metadata: ResourceMetadata) -> None:
    """Write the full response to <RESOURCE_METADATA_OUTPUT_DIR>/<resource_id>.json."""
    RESOURCE_METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESOURCE_METADATA_OUTPUT_DIR / f"{resource_id}.json"
    output_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


# --- Running -----------------------------------------------------------------------------------


async def process_batch(connection: Connection, llm: LLMClient, resource_ids: list[str]) -> int:
    """Process one batch of resources concurrently. Returns how many were saved."""
    prompts: dict[str, str] = {}
    for resource_id in resource_ids:
        try:
            prompts[resource_id] = fetch_user_prompt(connection, resource_id)
        except ResourceSkippedError as error:
            logger.warning("Skipped %s: %s", resource_id, error)
        except psycopg2.Error:
            logger.exception("Could not read %s", resource_id)

    results = await extract_metadata(llm, list(prompts.values()))

    saved = 0
    for resource_id, result in zip(prompts, results):
        if isinstance(result, Exception):
            logger.error("Extraction failed for %s: %r", resource_id, result)
            continue
        if result.resource_id != resource_id:
            logger.error("Model returned resource_id %s for %s", result.resource_id, resource_id)
            continue
        try:
            save_metadata(connection, resource_id, result)
            save_metadata_file(resource_id, result)
            saved += 1
        except (ResourceSkippedError, psycopg2.Error, OSError):
            logger.exception("Could not save %s", resource_id)
    return saved


async def extract_resource_metadata(requested_ids: list[str] | None = None) -> None:
    """Extract and store metadata. With no ids, process every resource without metadata."""
    connection = psycopg2.connect(load_db_dsn())
    try:
        resource_ids = requested_ids or find_pending_resource_ids(connection)
        total = len(resource_ids)
        if total == 0:
            logger.info("No resources to process.")
            return
        logger.info("Processing %d resources", total)

        # Built after the resources are known: this creates the (billed) prompt cache.
        settings = load_llm_settings()
        llm = LLMClient(GeminiProvider(settings, build_system_prompt()), settings.max_concurrency)

        saved = 0
        for start in range(0, total, BATCH_SIZE):
            batch = resource_ids[start : start + BATCH_SIZE]
            saved += await process_batch(connection, llm, batch)
            logger.info("Progress: %d/%d processed, %d saved", start + len(batch), total, saved)
        logger.info("Done. Saved metadata for %d of %d resources.", saved, total)
    finally:
        connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(extract_resource_metadata(RESOURCE_IDS))
