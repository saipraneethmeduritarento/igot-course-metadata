"""Running many items through the model, and the command line both extraction scripts share.

`run_pool` keeps a fixed number of model calls in flight. Items are read (prepared) a batch at a
time, and each result is saved as soon as it arrives, so a slow call never holds back the
others. Batch size only sets how many items are read from the database at once and how often
progress is logged; `--concurrency` sets the speed.

    summary = await run_pool(ids, prepare_batch, process_one, concurrency=8, batch_size=50)

  - prepare_batch(ids) -> {id: job}: everything needed before the model is called. An id left
    out was skipped (it logs why); an id whose value is an Exception failed.
  - process_one(id, job): call the model and save the result. An exception fails the item;
    QuotaExhaustedError stops the run, and every item not finished yet is left for a later run.
"""

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DEFAULT_MODEL_ENV_VAR, load_sample_ids
from src.llm import QuotaExhaustedError, describe

logger = logging.getLogger(__name__)

PrepareBatch = Callable[[list[str]], dict[str, Any]]
ProcessOne = Callable[[str, Any], Awaitable[None]]


@dataclass
class RunSummary:
    """What happened to each item of a run."""

    total: int = 0
    saved: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # nothing to send (see the log for why)
    failed: list[str] = field(default_factory=list)
    not_started: list[str] = field(default_factory=list)  # left when the run stopped on quota
    stopped_on_quota: bool = False

    @property
    def to_retry(self) -> list[str]:
        """The ids a later run should try again."""
        return [*self.failed, *self.not_started]

    @property
    def exit_code(self) -> int:
        """0 when every item was saved or skipped, else 1."""
        return 1 if self.to_retry else 0


async def run_pool(
    item_ids: list[str],
    prepare_batch: PrepareBatch,
    process_one: ProcessOne,
    *,
    concurrency: int,
    batch_size: int,
    label: str = "items",
) -> RunSummary:
    """Process every item with at most `concurrency` at a time. See the module docstring."""
    summary = RunSummary(total=len(item_ids))
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue(maxsize=batch_size)
    stop = asyncio.Event()
    finished = 0

    def log_progress() -> None:
        logger.info(
            "Progress: %d/%d %s done (%d saved, %d skipped, %d failed)",
            len(summary.saved) + len(summary.skipped) + len(summary.failed),
            summary.total,
            label,
            len(summary.saved),
            len(summary.skipped),
            len(summary.failed),
        )

    async def produce() -> None:
        try:
            for start in range(0, len(item_ids), batch_size):
                batch = item_ids[start : start + batch_size]
                if stop.is_set():
                    summary.not_started.extend(batch)
                    continue
                try:
                    jobs = prepare_batch(batch)
                except Exception as error:
                    logger.error("Could not prepare %d %s: %s", len(batch), label, describe(error))
                    summary.failed.extend(batch)
                    continue
                for item_id in batch:
                    if item_id not in jobs:
                        summary.skipped.append(item_id)
                    elif isinstance(jobs[item_id], Exception):
                        summary.failed.append(item_id)
                    elif stop.is_set():
                        summary.not_started.append(item_id)
                    else:
                        await queue.put((item_id, jobs[item_id]))
        finally:
            for _ in range(concurrency):  # one end marker per worker
                await queue.put(None)

    async def work() -> None:
        nonlocal finished
        while (entry := await queue.get()) is not None:
            item_id, job = entry
            if stop.is_set():
                summary.not_started.append(item_id)
                continue
            try:
                await process_one(item_id, job)
                summary.saved.append(item_id)
            except QuotaExhaustedError as error:
                if not stop.is_set():
                    logger.error("Stopping the run: %s", error)
                    stop.set()
                summary.not_started.append(item_id)
                summary.stopped_on_quota = True
                continue
            except Exception as error:
                logger.error("Failed %s: %s", item_id, describe(error))
                summary.failed.append(item_id)
            finished += 1
            if finished % batch_size == 0:
                log_progress()

    await asyncio.gather(produce(), *(work() for _ in range(concurrency)))
    log_progress()
    return summary


def write_retry_file(summary: RunSummary, directory: Path) -> Path | None:
    """Write the ids to try again, one per line, to <directory>/failed_<timestamp>.txt.

    Rerun them with --ids-file. Returns None when there is nothing to retry.
    """
    if not summary.to_retry:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"failed_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(summary.to_retry) + "\n", encoding="utf-8")
    return path


def log_summary(summary: RunSummary, label: str, retry_file: Path | None) -> None:
    logger.info(
        "Done. %d of %d %s saved, %d skipped, %d failed, %d not started.",
        len(summary.saved),
        summary.total,
        label,
        len(summary.skipped),
        len(summary.failed),
        len(summary.not_started),
    )
    if summary.stopped_on_quota:
        logger.error(
            "The run stopped because the model quota is exhausted. Run the same command again "
            "later: finished %s are not done again.",
            label,
        )
    if retry_file:
        logger.info("Ids to try again: %s (rerun with --ids-file %s)", retry_file, retry_file)


# --- Command line ------------------------------------------------------------------------------


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_arg_parser(description: str, default_batch_size: int, ids_source: str) -> argparse.ArgumentParser:
    """The options both extraction scripts take. `ids_source` names what the ids are."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=default_batch_size,
        help=f"{ids_source} read from the database at a time, and how often progress is logged "
        f"(default {default_batch_size})",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        help="model calls in flight at the same time; this sets the speed "
        "(default MAX_CONCURRENCY from .env)",
    )
    parser.add_argument(
        "--limit", type=positive_int, help=f"process at most this many {ids_source} (after the done ones are left out)"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help=f"ignore the sample CSV and process every pending {ids_source[:-1]} in the database",
    )
    selection.add_argument("--ids", help="comma-separated ids to process instead of the sample CSV")
    selection.add_argument(
        "--ids-file", type=Path, help="file with one id per line (e.g. a failed_*.txt) to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="also redo ids that already have a result from this model and prompt version",
    )
    parser.add_argument(
        "--model-env",
        default=DEFAULT_MODEL_ENV_VAR,
        help=f"the .env variable that names the model (default {DEFAULT_MODEL_ENV_VAR})",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.all and args.force:
        parser.error("--force needs chosen ids (the sample CSV, --ids or --ids-file), not --all")
    return args


def selected_ids(args: argparse.Namespace, sample_column: str) -> list[str] | None:
    """The ids the command line asks for, in order and without repeats; None for --all."""
    if args.all:
        return None
    if args.ids:
        ids = [item.strip() for item in args.ids.split(",")]
    elif args.ids_file:
        lines = args.ids_file.read_text(encoding="utf-8").splitlines()
        ids = [line.strip() for line in lines if not line.strip().startswith("#")]
    else:
        ids = load_sample_ids(sample_column)
    return list(dict.fromkeys(item for item in ids if item))
