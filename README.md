# igot-course-metadata

Generates metadata for iGOT courses with Gemini on Vertex AI. It works at two levels:

- **Resource level.** One LLM call per resource (a video or audio item), made from its transcripts.
- **Course level.** The resource results are combined into one record per course. Code decides
  the fields that depend on duration (weighted by duration × confidence). The LLM writes only the
  text fields: summary, description, keywords and so on.

Results are stored in Postgres and also written as JSON files.

## Pipeline

Run every step from the project root, in this order:

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `python -m src.utils.combine_transcripts` | `data/input/transcripts/*.json` | `data/input/resource_transcripts.json` |
| 2 | `python -m src.utils.load_transcripts_to_postgres` | `resource_transcripts.json` | table `resource_transcripts` |
| 3 | `python -m src.resource_metadata.extraction` | `resource_transcripts`, reference masters | table `resource_metadata`, `data/output/resource_metadata_output/<resource_id>.json` |
| 4 | `python -m src.course_metadata.extraction` | `resource_transcripts`, `resource_metadata`, content API | table `course_metadata_rollup`, `data/output/course_metadata_output/<course_id>.json` |
| 5 | `python -m src.utils.create_course_metadata_llm [course_id ...]` | `course_metadata_rollup` | table `course_metadata_llm` |

Optional: `python -m src.utils.build_transcripts_table` exports `resource_transcripts.json` to a
CSV you can open in Excel. No other step reads this CSV.

### What each step does

1. **combine_transcripts** joins three transcript files on `resource_id`, keeping every resource
   that appears in any of them. It keeps only resources whose status is `Live`. The three files
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

### Which IDs are processed

Steps 3 and 4 read the IDs to process from `data/input/sample_do_ids_list_for_generation.csv`
(columns `course_id`, `category` and `resource_id`). The file must exist, because it is read when
the module is imported.

- **Resource extraction** uses the `resource_id` column. If the column is empty, every resource
  that has a transcript but no metadata yet is processed.
- **Course extraction** uses the `course_id` column. If the column is empty, every complete course
  without a roll-up is processed. **Courses already in the list are processed again and billed
  again**, because course extraction does not skip finished courses.

## Setup

Requirements: Python 3.11+, a reachable Postgres, and a Google Cloud service account with Vertex
AI access.

    uv sync --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
    cp .env.example .env         # then fill in the values

Put these files in `data/input/` (it is gitignored):

- `KCM_competency.json`: the KCM competencies (Behavioural / Functional)
- `sgos_sectors.json`: SGOS sectors, subsectors and themes (Domain)
- `igot_designations.txt`: one designation name per line
- `sample_do_ids_list_for_generation.csv`: the IDs to process (see above)
- `transcripts/`: the three transcript files read by step 1

## Configuration

- `.env` holds credentials, the model, the database DSN and the API settings. See `.env.example`
  for each variable.
- [src/config.py](src/config.py) holds the input/output paths and the course roll-up thresholds:
  - `COURSE_MIN_SHARE`: the minimum share of course duration an item needs to be kept
  - `COURSE_LEARNING_MODE_MIN_SHARE`: the share that decides Online, Offline or Hybrid
  - `COURSE_EXCLUSION_MIN_SHARE`: the stricter minimum share for excluded roles
  - `COURSE_MAX_ITEMS`: the maximum number of items kept in each list field
- [src/prompts.py](src/prompts.py) holds the system and user prompts for both levels.

## Layout

- `src/config.py`: paths, `.env` loading, thresholds, LLM and API settings
- `src/llm.py`: Gemini client. It caches the system prompt (and renews the cache when it expires),
  runs calls concurrently and checks each answer against a Pydantic model
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
