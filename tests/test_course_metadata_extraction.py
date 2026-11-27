import json
import re

import pytest
from pydantic import ValidationError

from src.config import COURSE_MAX_ITEMS, COURSE_PROMPT_VERSION
from src.course_metadata.extraction import (
    COURSE_COLUMNS,
    COURSE_PROMPTS,
    CREATE_TABLE_SQL,
    RESOURCE_COLUMNS,
    CourseSkippedError,
    TEXT_FIELDS,
    build_course_metadata,
    build_keywords,
    build_rollup_row,
    build_user_prompt,
    collect_languages,
    collect_resource_info,
    collect_reference_resources,
    fetch_course_resources,
    normalise_durations,
    roll_up_resources,
    rollup_file_path,
)
from src.course_metadata.schema import CourseMetadata, CourseTextMetadata
from src.prompt_versions import PROMPT_MODULES, prompts_for
from src.prompts import COURSE_METADATA_SYSTEM_PROMPT
from src.resource_metadata.schema import Keyword

PROMPT_EXAMPLE = re.search(r"```json\n(.*?)\n```", COURSE_METADATA_SYSTEM_PROMPT, re.S).group(1)


def judged(value, confidence=1.0):
    return {"value": value, "confidence": confidence, "rationale": "r"}


def kcm(competency_id, level="Operational", confidence=1.0):
    return {
        "competency_id": competency_id, "theme": "T", "sub_theme": f"S-{competency_id}",
        "proficiency_level": level, "confidence": confidence, "rationale": "r",
    }


def domain_path(subsector="Telecommunications", theme="Next-Generation Telecom Rollout", confidence=1.0):
    return {"sector": "Technology", "subsector": subsector, "theme": theme, "confidence": confidence, "rationale": "r"}


def resource(resource_id, duration, area="Domain", level="Operational", **overrides):
    """A resource with complete metadata (a Domain path by default). Pass fields to override."""
    row = {
        "resource_id": resource_id, "duration_in_sec": duration, "source_language": None, "source_languages": None,
        "presenters": [], "course_coverage": "d", "description": "summary", "learning_outcomes": [], "keywords_list": [], "reference_resources": [],
        "learning_level": judged("Beginner"), "learning_mode": judged("Online"),
        "prerequisites": [], "bloom_taxonomy": [],
        "primary_competency_area": judged(area),
        "behavioural_competencies": [], "functional_competencies": [],
        "domain_competencies": [domain_path()] if area == "Domain" else [],
        "sto_level": {**judged(level), "evidence": []},
        "responsibilities_addressed": [], "activities_enabled": [],
        "target_employee_groups": [], "target_roles": [], "target_roles_exclusions": [],
    }
    row.update(overrides)
    return row


def minimal_text():
    return CourseTextMetadata(
        course_coverage="d", description="s", learning_outcomes=[], keywords_list=[], prerequisites=[],
        responsibilities_addressed=[], activities_enabled=[],
    )


# --- prompt and schema -------------------------------------------------------------------------


def test_prompt_example_matches_text_schema_and_key_order():
    CourseTextMetadata.model_validate_json(PROMPT_EXAMPLE)
    assert list(json.loads(PROMPT_EXAMPLE)) == list(CourseTextMetadata.model_fields)


def test_code_and_text_fields_cover_every_course_field():
    rolled_up = roll_up_resources([resource("a", 100)])
    built_after_the_model = {"keywords"}
    assert set(rolled_up) | set(TEXT_FIELDS) | built_after_the_model | {"course_id"} == set(CourseMetadata.model_fields)
    assert COURSE_COLUMNS == [n for n in CourseMetadata.model_fields if n != "course_id"]


# --- fields done by code -----------------------------------------------------------------------


def test_languages_are_ordered_by_resource_count_then_regional_before_english():
    resources = [
        resource("a", 10, source_language="Hindi", source_languages=["Hindi"]),
        resource("b", 10, source_language="English", source_languages=["English", "Hindi"]),
        resource("c", 10, source_language="Telugu"),
        resource("d", 10),
    ]
    # Hindi in 2 resources; English and Telugu in 1 each, so the regional Telugu comes before English.
    assert collect_languages(resources) == ["Hindi", "Telugu", "English"]


def test_dominant_language_is_the_first_one_and_tie_prefers_regional_over_english():
    tied = [resource("a", 10, source_language="English"), resource("b", 10, source_language="Marathi")]
    assert collect_resource_info(tied)["language"] == "Marathi"
    both_regional = [resource("a", 10, source_language="Telugu"), resource("b", 10, source_language="Hindi")]
    assert collect_resource_info(both_regional)["language"] == "Hindi"  # still tied: the first in the list


def test_course_without_any_language_has_no_dominant_language():
    info = collect_resource_info([resource("a", 10)])
    assert info == {"resource_ids": ["a"], "language": None, "list_of_languages": []}


def reference(value, type_="URL"):
    return {"value": value, "type": type_}


def test_reference_resources_are_deduplicated_and_ordered_by_duration_then_first_appearance():
    resources = [
        resource("a", 10, reference_resources=[reference("x.in"), reference("y.in")]),
        resource("b", 10, reference_resources=[reference("y.in"), reference("z.in")]),
    ]
    assert collect_reference_resources(resources) == [reference("y.in"), reference("x.in"), reference("z.in")]


def test_web_address_variants_are_one_reference_shown_as_the_longest_resources_wrote_it():
    resources = [
        resource("a", 100, reference_resources=[reference("https://www.GeM.gov.in/")]),
        resource("b", 500, reference_resources=[reference("gem.gov.in")]),
        resource("c", 100, reference_resources=[reference("www.gem.gov.in.")]),
    ]
    assert collect_reference_resources(resources) == [reference("gem.gov.in")]


def test_different_paths_on_one_site_stay_apart():
    resources = [resource("a", 100, reference_resources=[reference("gem.gov.in"), reference("gem.gov.in/registration")])]
    assert len(collect_reference_resources(resources)) == 2


def test_helpline_numbers_match_on_digits():
    resources = [
        resource("a", 300, reference_resources=[reference("1930", "Helpline")]),
        resource("b", 100, reference_resources=[reference("1-9-3-0", "Helpline")]),
    ]
    assert collect_reference_resources(resources) == [reference("1930", "Helpline")]


def test_act_punctuation_variants_merge_but_a_section_is_a_separate_reference():
    resources = [
        resource("a", 300, reference_resources=[reference("RTI Act, 2005", "Act/Rule")]),
        resource("b", 100, reference_resources=[
            reference("RTI Act 2005", "Act/Rule"), reference("RTI Act, 2005, Section 6", "Act/Rule"),
        ]),
    ]
    assert collect_reference_resources(resources) == [
        reference("RTI Act, 2005", "Act/Rule"), reference("RTI Act, 2005, Section 6", "Act/Rule"),
    ]


def test_type_is_the_weighted_vote_and_other_only_wins_alone():
    resources = [
        resource("a", 900, reference_resources=[reference("gem.gov.in", "Other"), reference("Manual X", "Other")]),
        resource("b", 100, reference_resources=[reference("gem.gov.in", "Portal/App")]),
        resource("c", 300, reference_resources=[reference("GEM.GOV.IN", "URL")]),
    ]
    assert collect_reference_resources(resources) == [reference("gem.gov.in", "URL"), reference("Manual X", "Other")]


def test_course_without_references_has_an_empty_list():
    assert collect_reference_resources([resource("a", 100), resource("b", 100, reference_resources=None)]) == []


def test_course_with_no_durations_counts_resources_equally():
    resources = [resource("a", 0), resource("b", None, area="Functional")]
    normalise_durations(resources)
    assert [r["duration_in_sec"] for r in resources] == [1, 1]


def test_a_single_missing_duration_counts_as_zero():
    resources = [resource("a", 100), resource("b", None)]
    normalise_durations(resources)
    assert [r["duration_in_sec"] for r in resources] == [100, 0]
    assert roll_up_resources(resources)["primary_competency_area"]["confidence"] == 1.0
    assert '"duration_share":0.0' in build_user_prompt(resources)


# --- duration-weighted decisions ---------------------------------------------------------------


def test_primary_area_is_decided_by_total_duration_not_resource_count():
    resources = [resource("long", 900, area="Domain")] + [
        resource(f"short{i}", 30, area="Functional") for i in range(3)
    ]
    area = roll_up_resources(resources)["primary_competency_area"]
    assert area["value"] == "Domain"
    assert area["confidence"] == round(900 / 990, 2)


@pytest.mark.parametrize("first, second, expected", [
    ("Intermediate", "Beginner", "Beginner"),
    ("Advanced", "Intermediate", "Intermediate"),
    ("Advanced", "Beginner", "Beginner"),
])
def test_learning_level_tie_goes_to_the_lower_level(first, second, expected):
    resources = [
        resource("a", 300, learning_level=judged(first, 0.7)),
        resource("b", 700, learning_level=judged(second, 0.3)),  # 300 x 0.7 = 700 x 0.3 = 210
    ]
    assert roll_up_resources(resources)["learning_level"]["value"] == expected


def test_learning_level_without_a_tie_follows_the_weighted_vote():
    resources = [resource("a", 600, learning_level=judged("Advanced")), resource("b", 400)]
    assert roll_up_resources(resources)["learning_level"]["value"] == "Advanced"


def test_confidence_weights_the_vote():
    resources = [
        resource("a", 500, level="Strategic", sto_level={**judged("Strategic", 0.4), "evidence": []}),
        resource("b", 400, level="Operational", sto_level={**judged("Operational", 0.9), "evidence": []}),
    ]
    assert roll_up_resources(resources)["sto_level"]["value"] == "Operational"  # 360 > 200


def learning_mode_of(*modes):
    """modes: (duration, value, confidence) per resource."""
    resources = [
        resource(f"r{i}", duration, learning_mode={**judged(value, confidence), "rationale": f"{value} {i}"})
        for i, (duration, value, confidence) in enumerate(modes)
    ]
    return roll_up_resources(resources)["learning_mode"]


@pytest.mark.parametrize("modes, expected", [
    ([(600, "Online", 1.0), (400, "Offline", 1.0)], "Hybrid"),                          # 60% / 40%
    ([(700, "Online", 1.0), (300, "Offline", 1.0)], "Online"),                          # in-person 30%
    ([(650, "Online", 1.0), (350, "Offline", 1.0)], "Hybrid"),                          # in-person exactly 35%
    ([(200, "Online", 1.0), (800, "Offline", 1.0)], "Offline"),                         # online 20%
    ([(500, "Hybrid", 1.0), (500, "Hybrid", 1.0)], "Hybrid"),                           # 50% / 50%
    ([(600, "Online", 1.0), (400, "Hybrid", 1.0)], "Online"),                           # in-person 200 of 1000
    ([(300, "Online", 1.0), (700, "Hybrid", 1.0)], "Hybrid"),                           # in-person exactly 35%
    ([(600, "Online", 1.0), (400, "Offline", 1.0), (400, "Hybrid", 1.0)], "Hybrid"),    # 57% / 43%
])
def test_learning_mode_splits_the_course_time_and_hybrid_counts_half(modes, expected):
    assert learning_mode_of(*modes)["value"] == expected


def test_learning_mode_confidence_does_not_change_the_split():
    assert learning_mode_of((600, "Online", 0.1), (400, "Offline", 1.0))["value"] == "Hybrid"


def test_learning_mode_confidence_is_the_duration_weighted_average():
    mode = learning_mode_of((900, "Online", 0.9), (300, "Hybrid", 0.6), (300, "Offline", 0.8))
    assert mode["confidence"] == 0.82  # (810 + 180 + 240) / 1500


def test_learning_mode_rationale_states_both_shares_and_the_rule():
    rationale = learning_mode_of((900, "Online", 0.9), (300, "Hybrid", 0.6), (300, "Offline", 0.8))["rationale"]
    assert rationale.startswith("70% of the course time can be learned online and 30% needs in-person practice")
    assert "Both parts are at least 35%, so the course is Hybrid." not in rationale  # 30% is below 35%
    assert "The in-person part is below 35%, so the course is Online." in rationale
    assert "Online part: Online 0." in rationale and "In-person part" not in rationale


def test_only_the_primary_area_has_competencies():
    resources = [
        resource("a", 800, area="Functional", functional_competencies=[kcm("F1")]),
        resource("b", 200, area="Behavioural", behavioural_competencies=[kcm("B1")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert [c["competency_id"] for c in rolled_up["functional_competencies"]] == ["F1"]
    assert rolled_up["behavioural_competencies"] == []
    assert rolled_up["domain_competencies"] == []


def test_competencies_reaching_the_min_share_are_kept_strongest_first():
    resources = [
        resource("a", 500, area="Functional", functional_competencies=[kcm("F1"), kcm("F2")]),
        resource("b", 500, area="Functional", functional_competencies=[kcm("F1")]),
        resource("c", 100, area="Functional", functional_competencies=[kcm("F3")]),
    ]
    competencies = roll_up_resources(resources)["functional_competencies"]
    first = competencies[0]
    assert (first["competency_id"], first["theme"], first["sub_theme"]) == ("F1", "T", "S-F1")
    # F1 91%, F2 45%; F3 (9%) is below COURSE_MIN_SHARE
    assert [(c["competency_id"], c["confidence"]) for c in competencies] == [("F1", 0.91), ("F2", 0.45)]


def test_competencies_are_capped_by_course_max_items():
    ids = [f"F{i}" for i in range(7)]
    resources = [resource("a", 100, area="Functional", functional_competencies=[kcm(i) for i in ids])]
    competencies = roll_up_resources(resources)["functional_competencies"]
    assert len(competencies) == COURSE_MAX_ITEMS["functional_competencies"] == 5
    assert [c["competency_id"] for c in competencies] == ids[:5]


def test_strongest_competency_is_kept_even_with_a_small_share():
    resources = [resource(str(i), 100, area="Functional", functional_competencies=[kcm(f"F{i}")]) for i in range(10)]
    competencies = roll_up_resources(resources)["functional_competencies"]
    assert [c["confidence"] for c in competencies] == [0.1]  # the others are below COURSE_MIN_SHARE


def test_domain_course_has_sector_paths_and_no_proficiency_level():
    resources = [
        resource("a", 600, domain_competencies=[domain_path(), domain_path("IT", "Cloud", 0.7)]),
        resource("b", 400, domain_competencies=[domain_path(confidence=0.9)]),
    ]
    competencies = roll_up_resources(resources)["domain_competencies"]
    assert [(c["sector"], c["subsector"], c["theme"]) for c in competencies] == [
        ("Technology", "Telecommunications", "Next-Generation Telecom Rollout"),  # 96%
        ("Technology", "IT", "Cloud"),  # 42%
    ]
    assert all("proficiency_level" not in c for c in competencies)


def test_domain_path_without_a_theme_is_never_chosen():
    resources = [
        resource("a", 900, domain_competencies=[domain_path(theme=None)]),
        resource("b", 100, domain_competencies=[domain_path("IT", "Cloud")]),
    ]
    assert [c["theme"] for c in roll_up_resources(resources)["domain_competencies"]] == ["Cloud"]


def test_primary_area_falls_back_to_the_next_area_with_a_competency():
    resources = [
        resource("a", 700, area="Functional"),  # Functional, but no competency fits
        resource("b", 300, area="Behavioural", behavioural_competencies=[kcm("B1")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert rolled_up["primary_competency_area"]["value"] == "Behavioural"
    assert rolled_up["primary_competency_area"]["rationale"].startswith("Functional had more support")
    assert [c["competency_id"] for c in rolled_up["behavioural_competencies"]] == ["B1"]
    assert rolled_up["functional_competencies"] == []


def test_course_without_any_complete_competency_is_skipped():
    resources = [resource("a", 100, domain_competencies=[domain_path(theme=None)]), resource("b", 100, area="Functional")]
    with pytest.raises(CourseSkippedError, match="no resource has a complete competency"):
        roll_up_resources(resources)


def test_course_metadata_rejects_a_missing_or_extra_competency():
    rolled_up = roll_up_resources([resource("a", 100)])
    with pytest.raises(ValidationError, match="domain_competencies is empty"):
        build_course_metadata("c1", {**rolled_up, "domain_competencies": []}, minimal_text())
    with pytest.raises(ValidationError, match="functional_competencies is set"):
        build_course_metadata("c1", {**rolled_up, "functional_competencies": [kcm("F1")]}, minimal_text())


def test_proficiency_level_is_a_weighted_vote_per_competency():
    resources = [
        resource("a", 700, area="Functional", functional_competencies=[kcm("F1", "Tactical"), kcm("F2", "Strategic")]),
        resource("b", 300, area="Functional", functional_competencies=[kcm("F1", "Strategic")]),
    ]
    f1, f2 = roll_up_resources(resources)["functional_competencies"]
    assert (f1["competency_id"], f1["proficiency_level"], f1["confidence"]) == ("F1", "Tactical", 1.0)
    assert (f2["competency_id"], f2["proficiency_level"]) == ("F2", "Strategic")


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


# --- presenters --------------------------------------------------------------------------------


def presenter(name, designation=None, organisation=None):
    return {"name": name, "designation": designation, "organisation": organisation}


def course_presenters(*resources):
    return roll_up_resources(list(resources))["presenters"]


def test_presenters_with_and_without_a_title_are_merged_and_the_title_is_shown():
    assert course_presenters(
        resource("a", 600, presenters=[presenter("Anil Kumar", "Director")]),
        resource("b", 100, presenters=[presenter("Dr. Anil Kumar")]),
    ) == [presenter("Dr. Anil Kumar", "Director")]


def test_hindi_titles_and_reordered_names_are_the_same_person():
    assert course_presenters(
        resource("a", 300, presenters=[presenter("Smt. Sunita Sharma")]),
        resource("b", 300, presenters=[presenter("Sharma Sunita", organisation="ISTM")]),
        resource("c", 300, presenters=[presenter("SHRI RAMESH RAO"), presenter("Ramesh Rao, IAS")]),
    ) == [presenter("Smt. Sunita Sharma", organisation="ISTM"), presenter("SHRI RAMESH RAO")]


def test_same_name_with_different_organisations_stays_two_people():
    result = course_presenters(
        resource("a", 300, presenters=[presenter("Rajesh Kumar", organisation="NIC")]),
        resource("b", 200, presenters=[presenter("Rajesh Kumar", organisation="ISTM")]),
    )
    assert [p["organisation"] for p in result] == ["NIC", "ISTM"]


def test_an_acronym_and_its_full_form_are_the_same_organisation():
    result = course_presenters(
        resource("a", 300, presenters=[presenter("Rajesh Kumar", organisation="NIC")]),
        resource("b", 200, presenters=[presenter("Rajesh Kumar", organisation="National Informatics Centre")]),
    )
    assert result == [presenter("Rajesh Kumar", organisation="NIC")]


def test_initials_are_not_merged_with_a_full_name():
    result = course_presenters(resource("a", 300, presenters=[presenter("Anil Kumar"), presenter("A. Kumar")]))
    assert [p["name"] for p in result] == ["Anil Kumar", "A. Kumar"]


def test_designation_comes_from_the_longest_resources_that_state_one():
    result = course_presenters(
        resource("a", 100, presenters=[presenter("Anil Kumar", "Deputy Director")]),
        resource("b", 500, presenters=[presenter("Anil Kumar", "Director")]),
        resource("c", 900, presenters=[presenter("Anil Kumar")]),
    )
    assert result == [presenter("Anil Kumar", "Director")]


def test_presenters_are_ordered_by_duration_then_first_appearance():
    result = course_presenters(
        resource("a", 100, presenters=[presenter("Host"), presenter("Guest One")]),
        resource("b", 800, presenters=[presenter("Main Faculty")]),
        resource("c", 100, presenters=[presenter("Guest Two")]),
    )
    assert [p["name"] for p in result] == ["Main Faculty", "Host", "Guest One", "Guest Two"]


def test_presenter_order_uses_the_sum_of_their_resource_durations():
    result = course_presenters(
        resource("a", 500, presenters=[presenter("Single Long")]),
        resource("b", 300, presenters=[presenter("Dr. Many Short")]),
        resource("c", 300, presenters=[presenter("Many Short")]),
        resource("d", 300, presenters=[presenter("Short Many"), presenter("Many Short")]),  # counted once here
    )
    assert [p["name"] for p in result] == ["Dr. Many Short", "Single Long"]  # 900 s before 500 s


def test_course_without_presenters_has_an_empty_list():
    assert course_presenters(resource("a", 100), resource("b", 100, presenters=None)) == []


def test_presenters_are_not_sent_to_the_model():
    resources = [resource("a", 100, presenters=[presenter("Dr. Anil Kumar")])]
    assert "Anil Kumar" not in build_user_prompt(resources)


# --- prompt and assembly -----------------------------------------------------------------------


def test_user_prompt_has_duration_shares_but_no_resource_ids():
    resources = [resource("a", 300), resource("b", 100)]
    prompt = build_user_prompt(resources)
    assert '"duration_share":0.75' in prompt
    assert '"resource_id"' not in prompt


def test_user_prompt_sends_only_the_resources_text_fields():
    """No code decision (levels, competencies, roles, ...) reaches the model."""
    resources = [
        resource(
            "a", 600, target_roles=[{"name": "Section Officer", "confidence": 1.0, "rationale": "r"}],
        ),
        resource("b", 400),
    ]
    prompt = build_user_prompt(resources)
    assert "course_profile" not in prompt
    assert "Section Officer" not in prompt
    payload = json.loads(prompt.split("<resources>")[1].split("</resources>")[0])
    assert all(set(item) == {"duration_share", *TEXT_FIELDS} for item in payload)


def test_build_course_metadata_joins_code_and_text_fields():
    resources = [resource("a", 100)]
    text = CourseTextMetadata(
        description="s", course_coverage="d", learning_outcomes=[],
        keywords_list=[], prerequisites=[],
        responsibilities_addressed=[], activities_enabled=[],
    )
    metadata = build_course_metadata("c1", roll_up_resources(resources), text)
    assert metadata.course_id == "c1"
    assert metadata.primary_competency_area.value == "Domain"
    assert metadata.description == "s"


RUN_INFO = {"model_used": "gemini-test", "prompt_version": "v9", "resource_prompt_version": "v8"}


def test_rollup_row_starts_with_the_key_columns_model_and_prompt_versions():
    resources = [resource("a", 100)]
    metadata = build_course_metadata("c1", roll_up_resources(resources), minimal_text())
    row = build_rollup_row(collect_resource_info(resources), metadata, RUN_INFO)
    assert list(row)[:5] == ["course_id", "model_used", "prompt_version", "resource_prompt_version", "resource_ids"]
    assert (row["model_used"], row["prompt_version"], row["resource_prompt_version"]) == ("gemini-test", "v9", "v8")
    assert list(row)[7:] == COURSE_COLUMNS


def test_rollup_file_is_kept_per_model_and_prompt_versions():
    path = rollup_file_path({"course_id": "c1", **RUN_INFO})
    assert path.parts[-2:] == ("gemini-test_course-v9_resource-v8", "c1.json")


def test_create_table_has_the_run_columns_in_its_primary_key():
    text = repr(CREATE_TABLE_SQL)  # a psycopg2 Composed; its repr holds the literal SQL parts
    assert f"course_metadata_rollup_{COURSE_PROMPT_VERSION.replace('.', '_')}" in text
    assert "PRIMARY KEY (course_id, model_used, prompt_version, resource_prompt_version)" in text


def test_every_prompt_module_has_its_own_version_and_the_configured_one_is_used():
    assert set(PROMPT_MODULES) == {"v1", "v2", "v3", "v3.1"}
    assert COURSE_PROMPTS is prompts_for(COURSE_PROMPT_VERSION)
    with pytest.raises(ValueError, match="Unknown prompt version 'v0'"):
        prompts_for("v0")


# --- keywords, rationales and field order ------------------------------------------------------


def keyword(name, importance, variants):
    return Keyword(keyword=name, type="Concept", importance=importance, variants=variants)


def test_keywords_are_sorted_by_importance_with_short_forms():
    keywords = [
        keyword("phishing", "Low", ["phishing"]),
        keyword("Quality of Service", "High", ["quality of service", "QoS", "QS", "QOS"]),
        keyword("QoS flow", "Medium", ["QoS flow", "QoS flows"]),
    ]
    assert build_keywords(keywords) == [
        "Quality of Service (QoS, QS)",
        "QoS flow",
        "phishing",
    ]


def test_keywords_use_the_expanded_form_when_the_keyword_is_an_acronym():
    keywords = [keyword("NCRP", "High", ["NCRP", "National Cybercrime Reporting Portal", "Cybercrime Portal"])]
    assert build_keywords(keywords) == ["National Cybercrime Reporting Portal (NCRP)"]


def test_rationale_quotes_the_strongest_resources_content():
    resources = [
        resource("a", 600, learning_level={**judged("Beginner"), "rationale": "Defines every term from scratch."}),
        resource("b", 300, learning_level={**judged("Beginner"), "rationale": "Assumes no prior knowledge"}),
        resource("c", 100, learning_level={**judged("Beginner"), "rationale": "Third resource, not quoted."}),
    ]
    rationale = roll_up_resources(resources)["learning_level"]["rationale"]
    assert "100% of the course duration" in rationale and "across 3 of 3 resources" in rationale
    assert "(1) Defines every term from scratch. (2) Assumes no prior knowledge." in rationale
    assert "Third resource" not in rationale


def test_hybrid_rationale_quotes_both_parts():
    resources = [
        resource("a", 550, learning_mode={**judged("Online"), "rationale": "Pure concepts."}),
        resource("b", 450, learning_mode={**judged("Offline"), "rationale": "Role-play practice."}),
    ]
    rationale = roll_up_resources(resources)["learning_mode"]["rationale"]
    assert "Online part: Pure concepts." in rationale and "In-person part: Role-play practice." in rationale


def test_rationale_comes_before_value_and_confidence():
    resources = [resource("a", 100, area="Functional", functional_competencies=[kcm("F1")])]
    text = CourseTextMetadata(
        description="s", course_coverage="d", learning_outcomes=[], keywords_list=[],
        prerequisites=[{"topic": "t", "recommended_by": "AI", "confidence": 0.8, "rationale": "r"}],
        responsibilities_addressed=[], activities_enabled=[],
    )
    document = build_course_metadata("c1", roll_up_resources(resources), text).model_dump(mode="json")
    assert list(document["learning_level"])[:3] == ["rationale", "value", "confidence"]
    assert list(document["prerequisites"][0])[0] == "topic"
    assert list(document["functional_competencies"][0])[0] == "competency_id"
    assert list(document["sto_level"])[:4] == ["evidence", "rationale", "value", "confidence"]


# --- only complete courses are rolled up -------------------------------------------------------


class FakeConnection:
    """Stands in for a psycopg2 connection whose query returns the given resources."""

    def __init__(self, resources):
        self.columns = ["resource_id", "duration_in_sec", "source_language", "source_languages", *RESOURCE_COLUMNS]
        self.rows = [tuple(r[column] for column in self.columns) for r in resources]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows


def test_complete_course_is_fetched():
    resources = [resource("a", 100), resource("b", 200)]
    assert [r["resource_id"] for r in fetch_course_resources(FakeConnection(resources), "c1", RUN_INFO)] == ["a", "b"]


def test_course_with_a_resource_missing_metadata_is_skipped_and_lists_the_missing_ids():
    resources = [resource("a", 100), resource("b", 200, description=None), resource("c", 50, description=None)]
    with pytest.raises(CourseSkippedError, match=r"2 of 3 resources have no metadata yet: b, c"):
        fetch_course_resources(FakeConnection(resources), "c1", RUN_INFO)


def test_course_with_no_transcribed_resource_is_skipped():
    with pytest.raises(CourseSkippedError, match="no resource with a transcript"):
        fetch_course_resources(FakeConnection([]), "c1", RUN_INFO)


def test_resource_metadata_is_read_for_the_runs_model_and_resource_prompt_version():
    connection = FakeConnection([resource("a", 100)])
    fetch_course_resources(connection, "c1", RUN_INFO)
    assert connection.params == ("gemini-test", "v8", "c1")


def test_a_mime_type_limits_the_course_to_resources_of_that_type():
    connection = FakeConnection([resource("a", 100)])
    fetch_course_resources(connection, "c1", RUN_INFO, "video/mp4")
    assert connection.params == ("gemini-test", "v8", "c1", "video/mp4")
    assert "mime_type" in repr(connection.query)


def test_resource_ids_are_the_course_whatever_course_resource_transcripts_files_them_under():
    # A resource reused by a program can be stored under the program's id, not the course's.
    connection = FakeConnection([resource("a", 100)])
    fetch_course_resources(connection, "c1", RUN_INFO, "video/mp4", ["a", "b"])
    assert connection.params == ("gemini-test", "v8", ["a", "b"], "video/mp4")
    assert "resource_id = ANY" in repr(connection.query)
    assert "course_id" not in repr(connection.query)


def test_without_resource_ids_the_course_is_read_by_its_id():
    connection = FakeConnection([resource("a", 100)])
    fetch_course_resources(connection, "c1", RUN_INFO)
    assert "course_id" in repr(connection.query)
    assert "resource_id = ANY" not in repr(connection.query)


def test_without_a_mime_type_every_resource_type_is_read():
    connection = FakeConnection([resource("a", 100)])
    fetch_course_resources(connection, "c1", RUN_INFO)
    assert "mime_type" not in repr(connection.query)


# --- running: rollup_course_metadata with fakes ------------------------------------------------

import asyncio  # noqa: E402

from src.config import RetryPolicy  # noqa: E402
from src.course_metadata import extraction  # noqa: E402
from src.course_metadata.extraction import PreparedCourse  # noqa: E402
from src.llm import QuotaExhaustedError  # noqa: E402


class DoneConnection:
    """A connection whose query says these course ids already have a roll-up."""

    def __init__(self, done):
        self.done = done
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self

    def execute(self, query, params):
        self.params = params

    def fetchall(self):
        return [(course_id,) for course_id in self.done]

    def close(self):
        pass


def test_leave_out_done_drops_courses_rolled_up_by_this_run_and_keeps_the_order():
    connection = DoneConnection(["c2"])
    assert extraction.leave_out_done(connection, ["c1", "c2", "c3"], RUN_INFO) == ["c1", "c3"]
    assert connection.params == (["c1", "c2", "c3"], "gemini-test", "v9", "v8")


class FakeCourseLLM:
    """Answers every course; "c_bad" fails and "c_quota" raises QuotaExhaustedError."""

    async def agenerate(self, user_prompt, output_model):
        await asyncio.sleep(0)
        if user_prompt == "c_bad":
            raise RuntimeError("model failed")
        if user_prompt == "c_quota":
            raise QuotaExhaustedError("quota gone")
        return minimal_text()


@pytest.fixture
def course_run(monkeypatch, tmp_path):
    """Replace the database, the content API, the model and the saves; record saved courses."""
    saved = []

    def fake_prepare(connection, course_id, run_info):
        if course_id == "c_incomplete":
            raise CourseSkippedError("1 of 2 resources have no metadata yet")
        if course_id == "c_broken":
            raise KeyError("a bug in the roll-up")
        return PreparedCourse(resource_info={}, rolled_up={}, user_prompt=course_id)

    settings = type("S", (), {"max_concurrency": 2, "model": "gemini-test", "retry": RetryPolicy()})()
    monkeypatch.setattr(extraction, "load_llm_settings", lambda model_env_var: settings)
    monkeypatch.setattr(extraction.psycopg2, "connect", lambda dsn: DoneConnection(["c_done"]))
    monkeypatch.setattr(extraction, "load_db_dsn", lambda: "dsn")
    monkeypatch.setattr(extraction, "create_rollup_table", lambda connection: None)
    monkeypatch.setattr(extraction, "prepare_course", fake_prepare)
    monkeypatch.setattr(extraction, "GeminiProvider", lambda settings, system_prompt: None)
    monkeypatch.setattr(extraction, "LLMClient", lambda provider, concurrency, retry: FakeCourseLLM())
    monkeypatch.setattr(extraction, "build_course_metadata", lambda course_id, rolled_up, text: course_id)
    monkeypatch.setattr(extraction, "build_rollup_row", lambda info, metadata, run_info: {"course_id": metadata})
    monkeypatch.setattr(extraction, "save_rollup_file", lambda row: None)
    monkeypatch.setattr(extraction, "save_rollup", lambda connection, row: saved.append(row["course_id"]))
    monkeypatch.setattr(extraction, "COURSE_METADATA_OUTPUT_DIR", tmp_path)
    return saved


def roll_up(ids, **options):
    return asyncio.run(extraction.rollup_course_metadata(ids, **options))


def test_course_run_skips_done_courses_and_survives_skips_and_failures(course_run, tmp_path):
    ids = ["c1", "c_done", "c_incomplete", "c_broken", "c_bad", "c2"]
    summary = roll_up(ids, batch_size=2)

    assert sorted(course_run) == ["c1", "c2"]  # c_done was rolled up by an earlier run
    assert summary.skipped == ["c_incomplete"]
    assert sorted(summary.failed) == ["c_bad", "c_broken"]
    [retry_file] = tmp_path.glob("*/failed_*.txt")
    assert sorted(retry_file.read_text().split()) == ["c_bad", "c_broken"]


def test_course_run_with_force_redoes_done_courses(course_run):
    roll_up(["c1", "c_done"], skip_done=False)
    assert sorted(course_run) == ["c1", "c_done"]


def test_course_run_stops_when_the_quota_is_exhausted(course_run):
    summary = roll_up(["c1", "c_quota", "c2", "c3"], batch_size=1, concurrency=1)
    assert course_run == ["c1"]
    assert summary.stopped_on_quota
    assert summary.to_retry == ["c_quota", "c2", "c3"]
    assert summary.exit_code == 1
