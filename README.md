# igot-course-metadata

Generates metadata for iGOT courses with Gemini on Vertex AI. It works at two levels:

- **Resource level.** One LLM call per resource (a video or audio item), made from its transcripts.
- **Course level.** The resource results are combined into one record per course. Code decides
  the fields that depend on duration (weighted by duration × confidence). The LLM writes only the
  text fields: description, course coverage, keywords and so on.

Results are stored in Postgres and also written as JSON files.

## Pipeline

Run every step from the project root, in this order:

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `python -m src.utils.combine_transcripts` | `data/input/transcripts/*.json` | `data/input/resource_transcripts.json` |
| 2 | `python -m src.utils.load_transcripts_to_postgres` | `resource_transcripts.json` | table `resource_transcripts` |
| 3 | `python -m src.resource_metadata.extraction` | `resource_transcripts`, reference masters | table `resource_metadata_v3_1` (one row per resource, model and prompt version), `data/output/resource_metadata_output/<model>_<prompt_version>/<resource_id>.json` |
| 4 | `python -m src.course_metadata.extraction` | `resource_transcripts`, `resource_metadata_v3_1` (rows of the same model and `RESOURCE_PROMPT_VERSION`), content API | table `course_metadata_rollup_v3_1` (one row per course, model, course prompt version and resource prompt version), `data/output/course_metadata_output/<model>_course-<version>_resource-<version>/<course_id>.json` |
| 5 | `python -m src.utils.create_course_metadata_llm [course_id ...]` | `course_metadata_rollup` (the old table; not yet switched to v2) | table `course_metadata_llm` |

Steps 3 and 4 can also run together, course by course, with `python -m src.pipeline` (see
[below](#steps-3-and-4-together-course-by-course)).

Optional: `python -m src.utils.build_transcripts_table` exports `resource_transcripts.json` to a
CSV you can open in Excel. No other step reads this CSV.

### What each step does

1. **combine_transcripts** joins three transcript files on `resource_id`, keeping every resource
   that appears in any of them. It keeps only resources whose status is `Live`, and skips the
   course rows in the video file (`mimeType` `application/vnd.ekstep.content-collection`). A
   resource missing from the video file (it has only a VTT transcript) is kept when it has an
   audio transcript and `data/input/courses_run_1_vtt.csv` lists it as Live; its mime type and
   status come from that CSV, and its duration is unknown. Each
   resource gets one course: the video file's `course_id`, or the audio file's when the video file
   has none. Course ids ending in `_rc` are restricted courses and are kept as is. The three files
   are in `data/input/transcripts/` and each can be a JSON array or JSON Lines:
   - `video_mp4_transcripts.json`
   - `resource_language_from_audio_transcription.json`
   - `audio_vtt_transcripts_en.json`
2. **load_transcripts_to_postgres** creates the database if it doesn't exist, then creates and
   fills `public.resource_transcripts`. **It does nothing if the table already exists.** To
   reload, drop the table first.
3. **resource_metadata.extraction** builds one prompt per resource from its transcripts, sends it
   with the KCM, SGOS and designations reference data, and drops any designation that isn't in
   the master list. Resources that already have metadata are skipped, so a rerun picks up where
   the last one stopped.
4. **course_metadata.extraction** rolls up a course only after every resource in it that has a
   transcript also has metadata; otherwise the course is skipped. It orders the resources the way
   a learner sees them, using the Karmayogi Extended Content Read API. Responses are cached in
   `data/input/course_api/<course_id>.json`. If the API can't be reached, the course is skipped.
5. **create_course_metadata_llm** builds a flat copy of the roll-up that is easy to query. It
   keeps only the values (no confidence or rationale) and one theme/sub-theme per course. Running
   it again overwrites existing rows. With no arguments it processes every rolled-up course.

### Steps 3 and 4 together, course by course

    # every course in data/input/courses_run_1_vtt.csv
    python -m src.pipeline

    # the courses of another CSV
    python -m src.pipeline --csv data/input/other_courses.csv

    # only some courses of the CSV
    python -m src.pipeline --ids do_123,do_456
    python -m src.pipeline --ids-file data/output/course_metadata_output/<model>_course-<version>_resource-<version>/failed_<timestamp>.txt

    # a test run on the first 2 courses not done yet
    python -m src.pipeline --limit 2

    # faster: 8 model calls in flight, 2 courses at a time
    python -m src.pipeline --concurrency 8 --parallel-courses 2

    # redo courses that are already done (bills them again)
    python -m src.pipeline --ids do_123 --force

    # use another model (the .env variable that names it)
    python -m src.pipeline --model-env GEMINI_FLASH_MODEL_NAME

    # all options
    python -m src.pipeline --help

`src.pipeline` does steps 3 and 4 one course at a time. It extracts the course's resources that
have no metadata yet (several at once), rolls the course up as soon as they are all done, then
goes on to the next course. A course's roll-up is therefore ready without waiting for every other
resource. It writes the same tables and files as the two separate scripts.

The courses, and the resources of each course, come from a CSV (`--csv`, default
`data/input/courses_run_1_vtt.csv`; columns `content_id`, `resource_id` and `mime_type`):

- Only the CSV's `video/mp4` resources ("Video MP4") are processed, and of those only the ones
  `resource_transcripts` has a transcript for. The others are logged and left out.
- A course is rolled up from its CSV resources alone. A resource the database has under the
  course but the CSV does not list is not used.
- A course with no such resource is skipped.

Options:

- `--ids a,b` or `--ids-file PATH` (e.g. a `failed_*.txt`): only these courses of the CSV.
  `--limit` counts courses.
- `--concurrency` caps the model calls in flight, resource and course calls together.
- `--parallel-courses N` (default 1) works on N courses at a time. With 1, each course is
  finished before the next one starts.
- If a resource fails, its course is not rolled up and is listed in the `failed_*.txt` file (in
  the course output folder). The course's other resources are kept, so a rerun only redoes the
  missing ones.
- `--force` redoes the chosen courses and extracts all their resources again. **This bills them
  again.**
- `--model-env VAR`: the `.env` variable that names the model (default `GEMINI_PRO_MODEL_NAME`).

### Which IDs are processed

By default, steps 3 and 4 process the IDs in the sample CSV (`SAMPLE_IDS_PATH` in
[src/config.py](src/config.py); columns `course_id`, `category` and `resource_id`). Resource
extraction uses the `resource_id` column and course extraction the `course_id` column. Both
scripts take the same options to choose something else:

| Option | What it does |
| --- | --- |
| `--all` | Ignore the CSV. Process every resource that has a transcript but no metadata yet, or every complete course without a roll-up |
| `--ids a,b,c` | Process these IDs |
| `--ids-file PATH` | Process the IDs in this file, one per line (`#` starts a comment), e.g. a `failed_*.txt` |
| `--limit N` | Process at most N of them, after the finished ones are left out. Useful for a test run |
| `--force` | Also redo IDs that already have a result from this model and prompt version. **This bills them again** |
| `--model-env VAR` | The `.env` variable that names the model (default `GEMINI_PRO_MODEL_NAME`) |

Without `--force`, IDs that already have a result from this model and prompt version are left
out. A rerun of the same command therefore only does what is left.

### Running large batches

    python -m src.resource_metadata.extraction --all --concurrency 8 --batch-size 100
    python -m src.course_metadata.extraction --all --concurrency 8 --batch-size 20

- `--concurrency` (default `MAX_CONCURRENCY` from `.env`) is how many model calls run at the same
  time, and is what makes a run faster. A new call starts as soon as one finishes, and each result
  is saved as soon as it arrives. The limit is your Vertex AI quota: more concurrency means more
  quota errors, which are retried (see below).
- `--batch-size` (default 20 for resources, 5 for courses) is how many items are read from the
  database at a time, and how often progress is logged. It does not change the speed.

**Retries.** A failed model call is retried when retrying can help:

- Retried, waiting 5s, 10s, 20s, ... up to 120s, for 6 attempts in all: quota errors (429),
  server errors (5xx), timeouts and network errors
- Retried up to 3 attempts in all: an answer that is empty or does not match the schema
- Never retried: 400, 403 and 404 errors (bad request, no permission, wrong model)

A quota error pauses every call, not only the one that got it. If 3 requests in a row still get
quota errors after all their attempts, the quota is taken to be exhausted. The run then stops
cleanly and exits with code 1. Every result finished so far is saved, so rerunning the same
command later carries on from where it stopped. The numbers are set by the optional `LLM_RETRY_*`
and `LLM_QUOTA_GIVE_UP_AFTER` variables (see `.env.example`).

**Failed IDs.** At the end, the IDs that failed or were not started are written to
`failed_<timestamp>.txt` in the run's output folder. Rerun them with `--ids-file <that file>`.
The script exits with code 0 when every ID was saved or skipped, and 1 otherwise.

## Setup

Requirements: Python 3.11+, a reachable Postgres, and a Google Cloud service account with Vertex
AI access.

    uv sync --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
    cp .env.example .env         # then fill in the values

Put these files in `data/input/` (it is gitignored):

- `KCM_competency.json`: the KCM competencies (Behavioural / Functional)
- `sgos_sectors.json`: SGOS sectors, subsectors and themes (Domain)
- `igot_designations.txt`: one designation name per line
- the sample CSV named by `SAMPLE_IDS_PATH` in `src/config.py`: the default IDs to process (see above)
- `transcripts/`: the three transcript files read by step 1

## Configuration

- `.env` holds credentials, the model, the database DSN and the API settings. See `.env.example`
  for each variable.
- [src/config.py](src/config.py) holds the input/output paths and the course roll-up thresholds:
  - `COURSE_MIN_SHARE`: the minimum share of course duration an item needs to be kept
  - `COURSE_LEARNING_MODE_MIN_SHARE`: the share that decides Online, Offline or Hybrid
  - `COURSE_EXCLUSION_MIN_SHARE`: the stricter minimum share for excluded roles
  - `COURSE_MAX_ITEMS`: the maximum number of items kept in each list field
- `RESOURCE_PROMPT_VERSION` and `COURSE_PROMPT_VERSION` in [src/config.py](src/config.py) pick
  the prompt file (`v3.1` is [src/prompts_v3_1.py](src/prompts_v3_1.py); `v1` to `v3` are
  `src/prompts.py`, `src/prompts_v2.py` and `src/prompts_v3.py`). Each version writes to its own
  tables, `resource_metadata_<version>` and `course_metadata_rollup_<version>` (with `.` written
  as `_`, e.g. `resource_metadata_v3_1`), so going back to
  `v3` reads and writes the v3 tables again.

## Layout

- `src/config.py`: paths, `.env` loading, thresholds, LLM and API settings
- `src/llm.py`: Gemini client. It caches the system prompt (and renews the cache when it expires),
  runs calls concurrently, checks each answer against a Pydantic model, and retries the failures
  that can go away (quota, server errors, timeouts)
- `src/runner.py`: the worker pool both extraction scripts run on, their shared command-line
  options and the `failed_*.txt` files
- `src/pipeline.py`: steps 3 and 4 together, one course at a time
- `src/prompts.py`: prompts
- `src/reference_data.py`: formats KCM, SGOS and designations as text for the prompts
- `src/resource_metadata/`: resource-level extraction (`extraction.py`) and models (`schema.py`)
- `src/course_metadata/`: course roll-up (`extraction.py`), models (`schema.py`) and learner order
  from the content API (`course_structure.py`)
- `src/utils/`: the transcript preparation scripts, the `course_metadata_llm` builder and
  `text_matching.py` (rules for treating two presenters, references or abbreviations as the same)
- `data/input/`, `data/output/`: inputs and generated metadata (both gitignored)
- `tests/`: unit tests. They need no database, network or credentials

## Tests

    pytest
