"""Reference data (KCM, SGOS, designations) formatted as text for the prompts."""

import json
import re
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
    """Compact designation list: `Base (A | B | C)` stands for `Base (A)`, `Base (B)`, `Base (C)`.

    Names that share a base and differ only in a trailing parenthetical are merged into one line.
    Names without a trailing parenthetical (including the plain base) stay on their own line.
    """
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[str]] = {}
    lines: list[tuple[str, bool]] = []  # (text or base, is_group)
    for name in names:
        match = None if name.startswith("(") else re.match(r"^([^()]+?) \(([^()|]*)\)$", name)
        if not match:
            lines.append((name, False))
            continue
        base, qualifier = match.groups()
        if base not in groups:
            groups[base] = []
            lines.append((base, True))
        if qualifier not in groups[base]:
            groups[base].append(qualifier)
    return "\n".join(f"{text} ({' | '.join(groups[text])})" if is_group else text for text, is_group in lines)


def load_designation_names(path: Path = DESIGNATIONS_PATH) -> frozenset[str]:
    """Every valid full designation name, exactly as written in the source file."""
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
