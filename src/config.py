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

# Which prompt module each pipeline uses: "v1" (src/prompts.py) or "v2" (src/prompts_v2.py).
# Resource extraction: stored in resource_metadata_v2.prompt_version, next to `model_used`.
RESOURCE_PROMPT_VERSION = "v2"
# Course roll-up: stored in course_metadata_rollup_v2.prompt_version. The roll-up reads the
# resource rows of RESOURCE_PROMPT_VERSION and of the same model, and stores that version in
# course_metadata_rollup_v2.resource_prompt_version.
COURSE_PROMPT_VERSION = "v2"


@dataclass(frozen=True)
class LLMSettings:
    """Which model to call and how. Change a value with `dataclasses.replace`."""

    model: str
    project: str
    location: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    max_attempts: int
    max_concurrency: int
    cache_ttl_seconds: int = 3600


def load_llm_settings(model_env_var: str = "GEMINI_PRO_MODEL_NAME") -> LLMSettings:
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
        max_attempts=int(os.environ["MAX_ATTEMPTS"]),
        max_concurrency=int(os.environ["MAX_CONCURRENCY"]),
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
