"""Environment and settings loading."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
RESOURCE_METADATA_OUTPUT_DIR = OUTPUT_DIR / "resource_metadata_output"
COURSE_METADATA_OUTPUT_DIR = OUTPUT_DIR / "course_metadata_output"
# Cached Extended Content Read API responses, one <course_id>.json per course.
COURSE_API_DIR = INPUT_DIR / "course_api"

KCM_PATH = INPUT_DIR / "KCM_competency.json"
SGOS_PATH = INPUT_DIR / "sgos_sectors.json"
DESIGNATIONS_PATH = INPUT_DIR / "igot_designations.txt"
# Courses and resources to process: one row per (course_id, resource_id).
# SAMPLE_IDS_PATH = INPUT_DIR / "sample_do_ids_list_for_generation.csv"
SAMPLE_IDS_PATH = INPUT_DIR / "common_courses_with_resources_24-11-2026.csv"


def load_sample_ids(column: str) -> list[str]:
    """The distinct, non-empty values of `column` in SAMPLE_IDS_PATH, in file order."""
    with SAMPLE_IDS_PATH.open(newline="", encoding="utf-8-sig") as file:
        values = (row[column].strip() for row in csv.DictReader(file))
        return list(dict.fromkeys(value for value in values if value))

load_dotenv(PROJECT_ROOT / ".env")

# Course roll-up thresholds. An item reaches the course only if its share of the course duration
# (sum of resource duration x confidence, divided by the course duration) is at least this much.
COURSE_MIN_SHARE = 0.20
# Learning mode: the course time is split into an online part and an in-person part (a Hybrid resource counts
# half for each). A part below this share does not count: Online if the in-person part is below it, Offline if
# the online part is below it, otherwise Hybrid.
COURSE_LEARNING_MODE_MIN_SHARE = 0.35
COURSE_RATIONALE_SOURCES = 2  # resources whose own rationale is quoted in a course-level rationale
COURSE_EXCLUSION_MIN_SHARE = 0.60  # exclusions are stricter: a role is excluded only on strong support
COURSE_MAX_ITEMS = {
    "bloom_taxonomy": 4,
    "target_employee_groups": 4,
    "target_roles": 10,
    "target_roles_exclusions": 5,
    # A course has competencies from ONE area (its primary area): up to this many theme/sub-theme
    # pairs (Behavioural, Functional) or sector/subsector/theme paths (Domain). The strongest one
    # is always kept; the others need COURSE_MIN_SHARE.
    "behavioural_competencies": 5,
    "functional_competencies": 5,
    "domain_competencies": 5,
}

# Which prompt module each pipeline uses: "v1" (src/prompts.py), "v2" (src/prompts_v2.py),
# "v3" (src/prompts_v3.py) or "v3.1" (src/prompts_v3_1.py). Each version also names its table:
# resource_metadata_<RESOURCE_PROMPT_VERSION> and course_metadata_rollup_<COURSE_PROMPT_VERSION>,
# with "." written as "_" (so v3.1 writes resource_metadata_v3_1 and course_metadata_rollup_v3_1;
# the v3 tables are kept).
# Resource extraction: stored in resource_metadata_<version>.prompt_version, next to `model_used`.
RESOURCE_PROMPT_VERSION = "v3.1"
# Course roll-up: stored in course_metadata_rollup_<version>.prompt_version. The roll-up reads the
# resource rows of RESOURCE_PROMPT_VERSION and of the same model, and stores that version in
# course_metadata_rollup_<version>.resource_prompt_version.
COURSE_PROMPT_VERSION = "v3.1"


@dataclass(frozen=True)
class RetryPolicy:
    """How a failed model call is tried again (see LLMClient in src/llm.py).

    A quota error (429), a server error (5xx), a timeout or a network error is retried up to
    `attempts` times in all, waiting `initial_delay_seconds`, then twice as long each time, up to
    `max_delay_seconds` (with some jitter). An answer that is empty or breaks the schema is tried
    up to `invalid_response_attempts` times. Anything else (400, 403, 404...) is not retried.

    A quota error pauses every call, not just the one that got it. When `quota_give_up_after`
    requests in a row have used every attempt on quota errors, the run stops: the quota is gone
    for now, and a later run carries on from where this one stopped.
    """

    attempts: int = 6
    initial_delay_seconds: float = 5.0
    max_delay_seconds: float = 120.0
    invalid_response_attempts: int = 3
    quota_give_up_after: int = 3


def load_retry_policy() -> RetryPolicy:
    """Read the retry policy from environment variables (.env). Every one is optional."""
    defaults = RetryPolicy()
    return RetryPolicy(
        attempts=int(os.environ.get("LLM_RETRY_ATTEMPTS", defaults.attempts)),
        initial_delay_seconds=float(
            os.environ.get("LLM_RETRY_INITIAL_DELAY", defaults.initial_delay_seconds)
        ),
        max_delay_seconds=float(os.environ.get("LLM_RETRY_MAX_DELAY", defaults.max_delay_seconds)),
        invalid_response_attempts=int(
            os.environ.get("LLM_RETRY_INVALID_RESPONSE_ATTEMPTS", defaults.invalid_response_attempts)
        ),
        quota_give_up_after=int(
            os.environ.get("LLM_QUOTA_GIVE_UP_AFTER", defaults.quota_give_up_after)
        ),
    )


@dataclass(frozen=True)
class LLMSettings:
    """Which model to call and how. Change a value with `dataclasses.replace`."""

    model: str
    project: str
    location: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    max_concurrency: int  # default for --concurrency: requests in flight at the same time
    cache_ttl_seconds: int = 3600
    retry: RetryPolicy = RetryPolicy()


DEFAULT_MODEL_ENV_VAR = "GEMINI_PRO_MODEL_NAME"


def load_llm_settings(model_env_var: str = DEFAULT_MODEL_ENV_VAR) -> LLMSettings:
    """Read the LLM settings from environment variables (.env).

    Pass another variable name to use another model, e.g. "GEMINI_FLASH_MODEL_NAME".
    """
    return LLMSettings(
        model=os.environ[model_env_var],
        project=os.environ["GOOGLE_PROJECT_ID"],
        location=os.environ["GOOGLE_PROJECT_LOCATION"],
        temperature=float(os.environ["LLM_TEMPERATURE"]),
        max_output_tokens=int(os.environ["LLM_MAX_OUTPUT_TOKENS"]),
        timeout_seconds=int(os.environ["LLM_TIMEOUT_SECONDS_META"]),
        max_concurrency=int(os.environ["MAX_CONCURRENCY"]),
        retry=load_retry_policy(),
    )


@dataclass(frozen=True)
class ApiSettings:
    """Where and how to call the Karmayogi content API."""

    base_url: str
    authorization: str  # the full Authorization header value, "Bearer <token>"
    timeout_seconds: int


def load_api_settings() -> ApiSettings:
    """Read the content API settings from environment variables (.env)."""
    key = os.environ["KARMAYOGI_API_KEY"].strip()
    # .env.example stores the key with its "Bearer " prefix; accept it with or without.
    if not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return ApiSettings(
        base_url=os.environ["KARMAYOGI_BASE_URL"].rstrip("/"),
        authorization=key,
        timeout_seconds=int(os.environ.get("API_REQUEST_TIMEOUT", "30")),
    )


def load_db_dsn() -> str:
    """Connection string of the Postgres database that holds resource_transcripts."""
    return os.environ["DB_DSN"]
