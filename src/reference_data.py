"""Reference data (KCM, SGOS, designations) formatted as text for the prompts."""

import json
from pathlib import Path

from src.config import DESIGNATIONS_PATH, KCM_PATH, SGOS_PATH


def load_kcm_catalogue(path: Path = KCM_PATH) -> str:
    """One line per competency: ID | Type | Theme > SubTheme | definition | L-O | L-T | L-S."""
    competencies = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(_format_competency(competency) for competency in competencies)


def load_sgos_tree(path: Path = SGOS_PATH) -> str:
    """One line per subsector: Sector > Subsector > Theme: description | Theme: description."""
    sectors = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        f"{sector} > {subsector} > " + " | ".join(f"{t['theme']}: {t['description']}" for t in themes)
        for sector, subsectors in sectors.items()
        for subsector, themes in subsectors.items()
    )


def load_designations(path: Path = DESIGNATIONS_PATH) -> str:
    """One designation name per line."""
    return path.read_text(encoding="utf-8").strip()


def _format_competency(competency: dict) -> str:
    labels = {level["level"]: level["label"] for level in competency["proficiency_levels"]}
    return " | ".join(
        [
            competency["competency_id"],
            competency["type"],
            f"{competency['theme']} > {competency['sub_theme']}",
            competency["sub_theme_description"],
            f"L-O: {labels['Operational']}",
            f"L-T: {labels['Tactical']}",
            f"L-S: {labels['Strategic']}",
        ]
    )
