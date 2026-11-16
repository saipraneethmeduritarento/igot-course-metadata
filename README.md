# igot-course-metadata

Generates course-level and resource-level metadata for iGOT courses using an LLM.

## Layout

- `src/` – config, LLM client, reference masters, course/resource logic, CLI
- `src/utils/` – shared helpers
- `src/scripts/` – existing entry-point scripts
- `prompts/v1/` – prompt templates
- `data/input/` – reference masters (KCM, SGOS, STO, designations) and transcripts
- `data/output/` – generated metadata
- `tests/` – tests

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r <(python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
    cp .env.example .env   # then fill in values

Run from the project root, e.g. `python -m src.course`.
