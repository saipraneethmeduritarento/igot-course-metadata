import json
import re

from src.course_metadata_extraction import (
    METADATA_COLUMNS,
    TEXT_FIELDS,
    build_course_metadata,
    build_user_prompt,
    collect_languages,
    collect_reference_resources,
    fill_missing_durations,
    roll_up_resources,
)
from src.course_metadata_schema import CourseMetadata, CourseTextMetadata
from src.prompts import COURSE_METADATA_SYSTEM_PROMPT

PROMPT_EXAMPLE = re.search(r"```json\n(.*?)\n```", COURSE_METADATA_SYSTEM_PROMPT, re.S).group(1)


def judged(value, confidence=1.0):
    return {"value": value, "confidence": confidence, "rationale": "r"}


def kcm(competency_id, level="Operational", confidence=1.0):
    return {
        "competency_id": competency_id, "theme": "T", "sub_theme": f"S-{competency_id}",
        "proficiency_level": level, "confidence": confidence, "rationale": "r",
    }


def resource(resource_id, duration, area="Domain", level="Operational", **overrides):
    """A resource with complete metadata. Pass fields to override."""
    row = {
        "resource_id": resource_id, "duration_in_sec": duration, "source_language": None, "source_languages": None,
        "presenters": [], "course_summary": "summary", "keywords": [], "reference_resources": [],
        "learning_level": judged("Beginner"), "learning_mode": judged("Online"),
        "prerequisites": [], "bloom_taxonomy": [],
        "primary_competency_area": judged(area),
        "behavioural_competencies": [], "functional_competencies": [], "domain_competencies": [],
        "sto_level": {**judged(level), "evidence": []},
        "responsibilities_addressed": [], "activities_enabled": [],
        "target_employee_groups": [], "target_roles": [], "target_roles_exclusions": [],
    }
    row.update(overrides)
    return row


# --- prompt and schema -------------------------------------------------------------------------


def test_prompt_example_matches_text_schema_and_key_order():
    CourseTextMetadata.model_validate_json(PROMPT_EXAMPLE)
    assert list(json.loads(PROMPT_EXAMPLE)) == list(CourseTextMetadata.model_fields)


def test_code_and_text_fields_cover_every_course_field():
    rolled_up = roll_up_resources([resource("a", 100)])
    assert set(rolled_up) | set(TEXT_FIELDS) | {"course_id"} == set(CourseMetadata.model_fields)
    assert METADATA_COLUMNS == [n for n in CourseMetadata.model_fields if n != "course_id"]


# --- fields done by code -----------------------------------------------------------------------


def test_collect_languages_is_unique_sorted_and_skips_missing():
    resources = [
        resource("a", 10, source_language="Hindi", source_languages=["Hindi"]),
        resource("b", 10, source_language="English", source_languages=["English", "Hindi"]),
        resource("c", 10),
    ]
    assert collect_languages(resources) == ["English", "Hindi"]


def test_reference_resources_are_deduplicated_in_order():
    resources = [
        resource("a", 10, reference_resources=["x", "y"]),
        resource("b", 10, reference_resources=["y", "z"]),
    ]
    assert collect_reference_resources(resources) == ["x", "y", "z"]


def test_course_with_no_durations_counts_resources_equally():
    resources = [resource("a", 0), resource("b", None, area="Functional")]
    fill_missing_durations(resources)
    assert [r["duration_in_sec"] for r in resources] == [1, 1]


# --- duration-weighted decisions ---------------------------------------------------------------


def test_primary_area_is_decided_by_total_duration_not_resource_count():
    resources = [resource("long", 900, area="Domain")] + [
        resource(f"short{i}", 30, area="Functional") for i in range(3)
    ]
    area = roll_up_resources(resources)["primary_competency_area"]
    assert area["value"] == "Domain"
    assert area["confidence"] == round(900 / 990, 2)


def test_confidence_weights_the_vote():
    resources = [
        resource("a", 500, level="Strategic", sto_level={**judged("Strategic", 0.4), "evidence": []}),
        resource("b", 400, level="Operational", sto_level={**judged("Operational", 0.9), "evidence": []}),
    ]
    assert roll_up_resources(resources)["sto_level"]["value"] == "Operational"  # 360 > 200


def test_learning_mode_is_hybrid_when_online_and_offline_parts_are_both_substantial():
    resources = [
        resource("a", 550, learning_mode=judged("Online")),
        resource("b", 450, learning_mode=judged("Offline")),
    ]
    mode = roll_up_resources(resources)["learning_mode"]
    assert mode["value"] == "Hybrid"
    assert mode["confidence"] == 0.45


def test_learning_mode_follows_the_vote_when_the_other_part_is_small():
    resources = [
        resource("a", 850, learning_mode=judged("Online")),
        resource("b", 150, learning_mode=judged("Offline")),
    ]
    assert roll_up_resources(resources)["learning_mode"]["value"] == "Online"


def test_only_the_primary_area_competencies_are_kept():
    resources = [
        resource("a", 800, area="Functional", functional_competencies=[kcm("F1")]),
        resource("b", 200, area="Behavioural", behavioural_competencies=[kcm("B1")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert [c["competency_id"] for c in rolled_up["functional_competencies"]] == ["F1"]
    assert rolled_up["behavioural_competencies"] == []
    assert rolled_up["domain_competencies"] == []


def test_list_items_below_the_min_share_are_dropped_but_the_top_competency_is_kept():
    resources = [
        resource("a", 500, area="Functional", functional_competencies=[kcm("F1"), kcm("F2")]),
        resource("b", 500, area="Functional", functional_competencies=[kcm("F1")]),
        resource("c", 100, area="Functional", functional_competencies=[kcm("F3")]),
    ]
    ids = [c["competency_id"] for c in roll_up_resources(resources)["functional_competencies"]]
    assert ids == ["F1", "F2"]  # F1 91%, F2 45%, F3 9%


def test_top_competency_is_kept_when_nothing_reaches_the_min_share():
    resources = [resource(str(i), 100, area="Functional", functional_competencies=[kcm(f"F{i}")]) for i in range(10)]
    assert len(roll_up_resources(resources)["functional_competencies"]) == 1


def test_proficiency_level_is_a_weighted_vote():
    resources = [
        resource("a", 700, area="Functional", functional_competencies=[kcm("F1", "Tactical")]),
        resource("b", 300, area="Functional", functional_competencies=[kcm("F1", "Strategic")]),
    ]
    competency = roll_up_resources(resources)["functional_competencies"][0]
    assert competency["proficiency_level"] == "Tactical"
    assert competency["confidence"] == 1.0


def test_exclusions_need_strong_support_and_never_overlap_target_roles():
    role = lambda name: {"name": name, "confidence": 1.0, "rationale": "r"}
    resources = [
        resource("a", 700, target_roles=[role("Clerk")], target_roles_exclusions=[role("Driver"), role("Clerk")]),
        resource("b", 300, target_roles_exclusions=[role("Peon")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert [r["name"] for r in rolled_up["target_roles"]] == ["Clerk"]
    assert [r["name"] for r in rolled_up["target_roles_exclusions"]] == ["Driver"]  # Peon 30% < 60%


def test_sto_evidence_has_supporting_and_opposing_phrases():
    evidence = [{"phrase": f"op{i}", "indicates": "Operational"} for i in range(6)] + [
        {"phrase": "strat", "indicates": "Strategic"}
    ]
    resources = [resource("a", 100, sto_level={**judged("Operational"), "evidence": evidence})]
    phrases = [e["phrase"] for e in roll_up_resources(resources)["sto_level"]["evidence"]]
    assert phrases == ["op0", "op1", "op2", "op3", "strat"]


# --- prompt and assembly -----------------------------------------------------------------------


def test_user_prompt_has_profile_and_duration_shares_but_no_resource_ids():
    resources = [resource("a", 300), resource("b", 100)]
    prompt = build_user_prompt("do_course", resources, roll_up_resources(resources))
    assert "course_id: do_course" in prompt
    assert '"primary_competency_area": "Domain"' in prompt
    assert '"duration_share":0.75' in prompt
    assert '"resource_id"' not in prompt


def test_build_course_metadata_joins_code_and_text_fields():
    resources = [resource("a", 100)]
    text = CourseTextMetadata(
        course_id="c1", presenters=[], course_summary="s", keywords=[], prerequisites=[],
        responsibilities_addressed=[], activities_enabled=[],
    )
    metadata = build_course_metadata("c1", roll_up_resources(resources), text)
    assert metadata.course_id == "c1"
    assert metadata.primary_competency_area.value == "Domain"
    assert metadata.course_summary == "s"
