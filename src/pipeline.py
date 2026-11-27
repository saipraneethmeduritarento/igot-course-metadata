"""Resource extraction and course roll-up together, one course at a time, from a course CSV.

The CSV (--csv, default data/input/courses_run_1_vtt.csv; columns content_id, resource_id,
mime_type) says which courses to process and which resources make up each course. For each
course, in file order:
  1. Take its resources from the CSV: only the video/mp4 ones (RESOURCE_MIME_TYPE), and of those
     only the ones public.resource_transcripts has a transcript for.
  2. Extract metadata for those that have none yet from this model and resource prompt version,
     several at a time (src.resource_metadata.extraction does the work and the saving).
  3. As soon as every one of them has metadata, roll the course up from those resources alone
     (src.course_metadata.extraction does the work and the saving). A resource of the course
     that is in the database but not in the CSV is not used.
  4. Go on to the next course.

The tables, JSON files and prompt versions are the same as those of the two separate scripts,
which can still be used on their own.

If a resource of a course fails, the course is not rolled up and counts as failed; its other
resources are kept. Rerunning the same command only redoes what is missing: resources and courses
that already have a result from this model and these prompt versions are left out (unless
--force).

Run from the project root (see --help for the options):
    python -m src.pipeline                                  # every course in the CSV
    python -m src.pipeline --csv data/input/other_courses.csv
    python -m src.pipeline --ids do_123,do_456              # only these courses of the CSV
    python -m src.pipeline --concurrency 8 --parallel-courses 2
    python -m src.pipeline --ids-file <failed_*.txt>
"""

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

from src.config import DEFAULT_MODEL_ENV_VAR, INPUT_DIR, load_db_dsn, load_llm_settings
from src.course_metadata import extraction as course_extraction
from src.course_metadata.extraction import CourseSkippedError
from src.llm import GeminiProvider, LLMClient, QuotaExhaustedError, describe
from src.reference_data import load_designation_names
from src.resource_metadata import extraction as resource_extraction
from src.runner import RunSummary, log_summary, positive_int, write_retry_file
from src.utils.combine_transcripts import MIME_TYPE_OF_LABEL

logger = logging.getLogger(__name__)

Connection = psycopg2.extensions.connection
RunInfo = dict[str, str]
CourseResources = dict[str, list[str]]  # course id -> its resource ids, in CSV order

# Default for --csv: the courses to process and the resources of each.
DEFAULT_COURSES_CSV = INPUT_DIR / "courses_run_1_vtt.csv"

# Default for --parallel-courses: courses worked on at the same time. 1 finishes each course
# (resources, then roll-up) before starting the next.
DEFAULT_PARALLEL_COURSES = 1

# Only resources of this mime type are processed, in the CSV and in resource_transcripts.
RESOURCE_MIME_TYPE = "video/mp4"
OF_RESOURCE_TYPE_SQL = sql.SQL("t.mime_type = {}").format(sql.Literal(RESOURCE_MIME_TYPE))


class CourseIncompleteError(Exception):
    """A resource of the course could not be extracted, so the course was not rolled up."""


# --- Choosing courses and resources ------------------------------------------------------------


def read_course_resources(path: Path) -> CourseResources:
    """Course id -> its RESOURCE_MIME_TYPE resource ids, from the CSV's content_id (or course_id),
    resource_id and mime_type columns (mime type labels such as "Video MP4" are understood).
    Courses and resources keep the file's order; resources of other types are left out (and
    counted)."""
    courses: CourseResources = {}
    other_types = 0
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            course_id = (row.get("content_id") or row.get("course_id") or "").strip()
            resource_id = (row.get("resource_id") or "").strip()
            if not course_id or not resource_id:
                continue
            label = (row.get("mime_type") or "").strip()
            if MIME_TYPE_OF_LABEL.get(label, label) != RESOURCE_MIME_TYPE:
                other_types += 1
                continue
            resources = courses.setdefault(course_id, [])
            if resource_id not in resources:
                resources.append(resource_id)
    if other_types:
        logger.info("Left out %d resources of %s that are not %s.", other_types, path, RESOURCE_MIME_TYPE)
    return courses


def find_transcribed_resource_ids(connection: Connection, resource_ids: list[str]) -> list[str]:
    """`resource_ids` (order kept) that resource_transcripts has as RESOURCE_MIME_TYPE, with a
    transcript. The others cannot be processed.

    Matched on resource_id alone: resource_transcripts.course_id is not used. A resource can be
    in several courses (e.g. a course and a program that reuses its videos), but that column
    keeps only one of them, and not always this one. The CSV says which resources are in the
    course."""
    query = sql.SQL(
        "SELECT t.resource_id FROM public.resource_transcripts t "
        "WHERE t.resource_id = ANY(%s) AND {has_transcript} AND {of_type}"
    ).format(has_transcript=course_extraction.HAS_TRANSCRIPT_SQL, of_type=OF_RESOURCE_TYPE_SQL)
    with connection, connection.cursor() as cursor:
        cursor.execute(query, (resource_ids,))
        found = {row[0] for row in cursor.fetchall()}
    return [resource_id for resource_id in resource_ids if resource_id in found]


# --- Running -----------------------------------------------------------------------------------


class Pipeline:
    """Runs courses through resource extraction and then roll-up.

    Every model call, of either kind, takes a slot of `concurrency`, so the total number of calls
    in flight never exceeds it however many courses are open.
    """

    def __init__(
        self,
        connection: Connection,
        resource_llm: LLMClient,
        course_llm: LLMClient,
        resource_run: RunInfo,
        course_run: RunInfo,
        valid_designations: frozenset[str],
        course_resources: CourseResources,
        *,
        concurrency: int,
        redo_done: bool = False,
    ):
        self.connection = connection
        self.resource_llm = resource_llm
        self.course_llm = course_llm
        self.resource_run = resource_run
        self.course_run = course_run
        self.valid_designations = valid_designations
        self.course_resources = course_resources
        self.redo_done = redo_done
        self.model_calls = asyncio.Semaphore(concurrency)
        self.stop = asyncio.Event()  # set when the quota is exhausted
        self.resources_saved: list[str] = []
        self.resources_failed: list[str] = []

    async def run(self, course_ids: list[str], parallel_courses: int = 1) -> RunSummary:
        """Process the courses in order, at most `parallel_courses` at a time."""
        summary = RunSummary(total=len(course_ids))
        queue = iter(enumerate(course_ids, start=1))  # shared by the workers

        async def work() -> None:
            for position, course_id in queue:
                if self.stop.is_set():
                    summary.not_started.append(course_id)
                    continue
                logger.info("Course %d/%d: %s", position, len(course_ids), course_id)
                try:
                    await self.run_course(course_id)
                except CourseSkippedError as error:
                    logger.warning("Skipped course %s: %s", course_id, error)
                    summary.skipped.append(course_id)
                except QuotaExhaustedError as error:
                    if not self.stop.is_set():
                        logger.error("Stopping the run: %s", error)
                        self.stop.set()
                    summary.not_started.append(course_id)
                    summary.stopped_on_quota = True
                except Exception as error:
                    logger.error("Failed course %s: %s", course_id, describe(error))
                    summary.failed.append(course_id)
                else:
                    logger.info("Rolled up course %s", course_id)
                    summary.saved.append(course_id)

        await asyncio.gather(*(work() for _ in range(parallel_courses)))
        return summary

    async def run_course(self, course_id: str) -> None:
        """Extract the course's missing resources, then roll it up from its CSV resources.

        Raises CourseSkippedError when there is nothing to roll up, CourseIncompleteError when a
        resource failed, QuotaExhaustedError when the run must stop.
        """
        listed = self.course_resources[course_id]
        resource_ids = find_transcribed_resource_ids(self.connection, listed)
        if len(resource_ids) < len(listed):
            missing = [resource_id for resource_id in listed if resource_id not in resource_ids]
            logger.warning(
                "%s: %d of %d resources in the CSV have no transcript in resource_transcripts "
                "and are left out: %s",
                course_id,
                len(missing),
                len(listed),
                ", ".join(missing),
            )
        if not resource_ids:
            raise CourseSkippedError(f"no {RESOURCE_MIME_TYPE} resource with a transcript")
        to_extract = (
            resource_ids
            if self.redo_done
            else resource_extraction.leave_out_done(self.connection, resource_ids, self.resource_run)
        )
        logger.info(
            "%s: %d resources, %d to extract", course_id, len(resource_ids), len(to_extract)
        )
        if to_extract:
            await self.extract_resources(course_id, to_extract)

        # From these resources only. Raises CourseSkippedError if one still has no metadata
        # (e.g. no usable transcript) or the course structure cannot be read.
        course = course_extraction.prepare_course(
            self.connection, course_id, self.course_run, RESOURCE_MIME_TYPE, resource_ids
        )
        await self.call_model(
            course_extraction.process_course(
                self.connection, self.course_llm, course_id, course, self.course_run
            )
        )

    async def extract_resources(self, course_id: str, resource_ids: list[str]) -> None:
        """Extract and save the resources concurrently. Raises once all have finished if any failed."""
        prompts = resource_extraction.fetch_user_prompts(self.connection, resource_ids)

        async def extract(resource_id: str) -> None:
            await self.call_model(
                resource_extraction.process_resource(
                    self.connection,
                    self.resource_llm,
                    resource_id,
                    prompts[resource_id],
                    self.resource_run,
                    self.valid_designations,
                )
            )

        results = await asyncio.gather(*(extract(rid) for rid in prompts), return_exceptions=True)
        failed: list[str] = []
        quota_error: QuotaExhaustedError | None = None
        for resource_id, result in zip(prompts, results, strict=True):
            if isinstance(result, QuotaExhaustedError):
                quota_error = result
            elif isinstance(result, Exception):
                logger.error("Failed resource %s of %s: %s", resource_id, course_id, describe(result))
                failed.append(resource_id)
            else:
                self.resources_saved.append(resource_id)
        self.resources_failed.extend(failed)
        if quota_error:
            raise quota_error
        if failed:
            raise CourseIncompleteError(
                f"{len(failed)} of {len(prompts)} resources failed: {', '.join(failed)}"
            )

    async def call_model(self, call) -> None:
        """Await `call` (a coroutine that calls the model) in one of the shared slots."""
        async with self.model_calls:
            if self.stop.is_set():
                call.close()  # never started
                raise QuotaExhaustedError("The run is stopping; not calling the model.")
            await call


def choose_courses(course_resources: CourseResources, requested_ids: list[str] | None) -> list[str]:
    """The requested courses that are in the CSV (every CSV course when None), in order."""
    if requested_ids is None:
        return list(course_resources)
    unknown = [course_id for course_id in requested_ids if course_id not in course_resources]
    if unknown:
        logger.warning(
            "Left out %d requested courses that are not in the CSV (or have no %s resource): %s",
            len(unknown),
            RESOURCE_MIME_TYPE,
            ", ".join(unknown),
        )
    return [course_id for course_id in requested_ids if course_id in course_resources]


async def run_pipeline(
    requested_ids: list[str] | None = None,
    skip_done: bool = True,
    *,
    courses_csv: Path = DEFAULT_COURSES_CSV,
    concurrency: int | None = None,
    parallel_courses: int = DEFAULT_PARALLEL_COURSES,
    limit: int | None = None,
    model_env_var: str = DEFAULT_MODEL_ENV_VAR,
) -> RunSummary:
    """Extract resources and roll up courses, course by course, as `courses_csv` lists them.

    Processes the CSV's courses (only `requested_ids` of them, if given) that have no roll-up yet
    from this model and these prompt versions (all of them, with every resource extracted again,
    when skip_done=False). `limit` caps how many courses are processed; `concurrency` (default
    MAX_CONCURRENCY) how many model calls run at once.
    """
    course_resources = read_course_resources(courses_csv)
    course_ids = choose_courses(course_resources, requested_ids)
    settings = load_llm_settings(model_env_var)
    concurrency = concurrency or settings.max_concurrency
    resource_run = {
        "model_used": settings.model,
        "prompt_version": resource_extraction.RESOURCE_PROMPTS.PROMPT_VERSION,
    }
    course_run = {
        "model_used": settings.model,
        "prompt_version": course_extraction.COURSE_PROMPTS.PROMPT_VERSION,
        "resource_prompt_version": resource_run["prompt_version"],
    }
    connection = psycopg2.connect(load_db_dsn())
    try:
        # First: the queries below read these tables.
        resource_extraction.ensure_metadata_table(connection)
        course_extraction.create_rollup_table(connection)
        if skip_done and course_ids:
            course_ids = course_extraction.leave_out_done(connection, course_ids, course_run)
        course_ids = course_ids[:limit] if limit else course_ids
        if not course_ids:
            logger.info("No courses to process.")
            return RunSummary()
        logger.info(
            "Processing %d courses from %s with %s, resource prompt %s, course prompt %s: "
            "%d model calls at a time, %d courses at a time",
            len(course_ids),
            courses_csv,
            settings.model,
            resource_run["prompt_version"],
            course_run["prompt_version"],
            concurrency,
            parallel_courses,
        )

        # Built after the courses are known: each creates a (billed) prompt cache.
        resource_llm = LLMClient(
            GeminiProvider(settings, resource_extraction.build_system_prompt()),
            concurrency,
            settings.retry,
        )
        course_llm = LLMClient(
            GeminiProvider(settings, course_extraction.COURSE_PROMPTS.COURSE_METADATA_SYSTEM_PROMPT),
            concurrency,
            settings.retry,
        )
        pipeline = Pipeline(
            connection,
            resource_llm,
            course_llm,
            resource_run,
            course_run,
            load_designation_names(),
            course_resources,
            concurrency=concurrency,
            redo_done=not skip_done,
        )
        summary = await pipeline.run(course_ids, parallel_courses)

        retry_file = write_retry_file(summary, course_extraction.run_output_dir(course_run))
        logger.info(
            "Resources: %d extracted, %d failed.",
            len(pipeline.resources_saved),
            len(pipeline.resources_failed),
        )
        log_summary(summary, "courses", retry_file)
        return summary
    finally:
        connection.close()


# --- Command line ------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract resource metadata and roll up each course as soon as its resources "
        "are done, for the courses and resources of a CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_COURSES_CSV,
        help="the courses to process and the resources of each (columns content_id, resource_id, "
        f"mime_type; default {DEFAULT_COURSES_CSV})",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="comma-separated course ids: only these courses of the CSV")
    selection.add_argument(
        "--ids-file",
        type=Path,
        help="file with one course id per line (e.g. a failed_*.txt): only these courses of the CSV",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        help="model calls in flight at the same time, resources and courses together "
        "(default MAX_CONCURRENCY from .env)",
    )
    parser.add_argument(
        "--parallel-courses",
        type=positive_int,
        default=DEFAULT_PARALLEL_COURSES,
        help="courses worked on at the same time; with 1, each course is finished before the "
        f"next starts (default {DEFAULT_PARALLEL_COURSES})",
    )
    parser.add_argument(
        "--limit", type=positive_int, help="process at most this many courses (after the done ones are left out)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="also redo courses that already have a roll-up, extracting all their resources again "
        "(this bills them again)",
    )
    parser.add_argument(
        "--model-env",
        default=DEFAULT_MODEL_ENV_VAR,
        help=f"the .env variable that names the model (default {DEFAULT_MODEL_ENV_VAR})",
    )
    return parser


def requested_course_ids(args: argparse.Namespace) -> list[str] | None:
    """The course ids of --ids or --ids-file, in order and without repeats; None for every course
    of the CSV. Surrounding quotes (as copied from a spreadsheet or JSON) are removed."""
    if args.ids:
        ids = args.ids.split(",")
    elif args.ids_file:
        lines = args.ids_file.read_text(encoding="utf-8").splitlines()
        ids = [line for line in lines if not line.strip().startswith("#")]
    else:
        return None
    ids = [item.strip().strip("\"'").strip() for item in ids]
    return list(dict.fromkeys(item for item in ids if item))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = asyncio.run(
        run_pipeline(
            requested_course_ids(args),
            skip_done=not args.force,
            courses_csv=args.csv,
            concurrency=args.concurrency,
            parallel_courses=args.parallel_courses,
            limit=args.limit,
            model_env_var=args.model_env,
        )
    )
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
