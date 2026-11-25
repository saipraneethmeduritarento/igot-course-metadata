"""The prompt modules, by the version stored with every generated row (`prompt_version`).

Each prompt module defines PROMPT_VERSION. RESOURCE_PROMPT_VERSION and COURSE_PROMPT_VERSION in
src/config.py pick the module each pipeline uses. To add a version, create the module (with its own
PROMPT_VERSION) and add it to PROMPT_MODULES.
"""

from types import ModuleType

from src import prompts, prompts_v2

PROMPT_MODULES: dict[str, ModuleType] = {
    module.PROMPT_VERSION: module for module in (prompts, prompts_v2)
}


def prompts_for(version: str) -> ModuleType:
    """The prompt module of `version`. Raises ValueError for an unknown version."""
    try:
        return PROMPT_MODULES[version]
    except KeyError:
        raise ValueError(
            f"Unknown prompt version {version!r}; known versions: {', '.join(PROMPT_MODULES)}"
        ) from None
