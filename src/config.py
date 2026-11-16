"""Environment and settings loading."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
RESOURCE_METADATA_OUTPUT_DIR = OUTPUT_DIR / "resource_metadata_output"
COURSE_METADATA_OUTPUT_DIR = OUTPUT_DIR / "course_metadata_output"

KCM_PATH = INPUT_DIR / "KCM_competency.json"
SGOS_PATH = INPUT_DIR / "sgos_sectors.json"
DESIGNATIONS_PATH = INPUT_DIR / "igot_designations.txt"

load_dotenv(PROJECT_ROOT / ".env")

# Course roll-up thresholds. An item reaches the course only if its share of the course duration
# (sum of resource duration x confidence, divided by the course duration) is at least this much.
COURSE_MIN_SHARE = 0.20
COURSE_EXCLUSION_MIN_SHARE = 0.60  # exclusions are stricter: a role is excluded only on strong support
COURSE_MAX_ITEMS = {
    "behavioural_competencies": 5,
    "functional_competencies": 5,
    "domain_competencies": 3,
    "bloom_taxonomy": 4,
    "target_employee_groups": 4,
    "target_roles": 10,
    "target_roles_exclusions": 5,
}


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


def load_db_dsn() -> str:
    """Connection string of the Postgres database that holds resource_transcripts."""
    return os.environ["DB_DSN"]
