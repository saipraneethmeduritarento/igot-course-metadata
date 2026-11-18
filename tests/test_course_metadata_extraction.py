import json
import re

import pytest
from pydantic import ValidationError

from src.course_metadata_extraction import (
    COURSE_COLUMNS,
    RESOURCE_COLUMNS,
    CourseSkippedError,
    TEXT_FIELDS,
    build_course_metadata,
    build_keywords_list,
    build_user_prompt,
    collect_languages,
    collect_resource_info,
    collect_reference_resources,
    fetch_course_resources,
    normalise_durations,
    roll_up_resources,
)
from src.course_metadata_schema import CourseMetadata, CourseTextMetadata
from src.prompts import COURSE_METADATA_SYSTEM_PROMPT
from src.resource_metadata_schema import Keyword

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
        "presenters": [], "course_summary": "summary", "description": "d", "learning_outcomes": [], "keywords": [], "reference_resources": [],
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
        description="d", course_summary="s", learning_outcomes=[], keywords=[], prerequisites=[],
        responsibilities_addressed=[], activities_enabled=[],
    )


# --- prompt and schema -------------------------------------------------------------------------


def test_prompt_example_matches_text_schema_and_key_order():
    CourseTextMetadata.model_validate_json(PROMPT_EXAMPLE)
    assert list(json.loads(PROMPT_EXAMPLE)) == list(CourseTextMetadata.model_fields)


def test_code_and_text_fields_cover_every_course_field():
    rolled_up = roll_up_resources([resource("a", 100)])
    built_after_the_model = {"keywords_list"}
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


def test_only_the_primary_area_has_a_competency():
    resources = [
        resource("a", 800, area="Functional", functional_competencies=[kcm("F1")]),
        resource("b", 200, area="Behavioural", behavioural_competencies=[kcm("B1")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert rolled_up["functional_competencies"]["competency_id"] == "F1"
    assert rolled_up["behavioural_competencies"] is None
    assert rolled_up["domain_competencies"] is None


def test_course_has_only_the_strongest_competency():
    resources = [
        resource("a", 500, area="Functional", functional_competencies=[kcm("F1"), kcm("F2")]),
        resource("b", 500, area="Functional", functional_competencies=[kcm("F1")]),
        resource("c", 100, area="Functional", functional_competencies=[kcm("F3")]),
    ]
    competency = roll_up_resources(resources)["functional_competencies"]
    assert (competency["competency_id"], competency["theme"], competency["sub_theme"]) == ("F1", "T", "S-F1")
    assert competency["confidence"] == 0.91  # F1 91%; F2 (45%) is dropped


def test_strongest_competency_is_kept_even_with_a_small_share():
    resources = [resource(str(i), 100, area="Functional", functional_competencies=[kcm(f"F{i}")]) for i in range(10)]
    assert roll_up_resources(resources)["functional_competencies"]["confidence"] == 0.1


def test_domain_course_has_one_sector_path_and_no_proficiency_level():
    resources = [
        resource("a", 600, domain_competencies=[domain_path(), domain_path("IT", "Cloud", 0.7)]),
        resource("b", 400, domain_competencies=[domain_path(confidence=0.9)]),
    ]
    competency = roll_up_resources(resources)["domain_competencies"]
    assert (competency["sector"], competency["subsector"], competency["theme"]) == (
        "Technology", "Telecommunications", "Next-Generation Telecom Rollout",
    )
    assert "proficiency_level" not in competency


def test_domain_path_without_a_theme_is_never_chosen():
    resources = [
        resource("a", 900, domain_competencies=[domain_path(theme=None)]),
        resource("b", 100, domain_competencies=[domain_path("IT", "Cloud")]),
    ]
    assert roll_up_resources(resources)["domain_competencies"]["theme"] == "Cloud"


def test_primary_area_falls_back_to_the_next_area_with_a_competency():
    resources = [
        resource("a", 700, area="Functional"),  # Functional, but no competency fits
        resource("b", 300, area="Behavioural", behavioural_competencies=[kcm("B1")]),
    ]
    rolled_up = roll_up_resources(resources)
    assert rolled_up["primary_competency_area"]["value"] == "Behavioural"
    assert rolled_up["primary_competency_area"]["rationale"].startswith("Functional had more support")
    assert rolled_up["behavioural_competencies"]["competency_id"] == "B1"
    assert rolled_up["functional_competencies"] is None


def test_course_without_any_complete_competency_is_skipped():
    resources = [resource("a", 100, domain_competencies=[domain_path(theme=None)]), resource("b", 100, area="Functional")]
    with pytest.raises(CourseSkippedError, match="no resource has a complete competency"):
        roll_up_resources(resources)


def test_course_metadata_rejects_a_missing_or_extra_competency():
    rolled_up = roll_up_resources([resource("a", 100)])
    with pytest.raises(ValidationError, match="domain_competencies is missing"):
        build_course_metadata("c1", {**rolled_up, "domain_competencies": None}, minimal_text())
    with pytest.raises(ValidationError, match="functional_competencies is set"):
        build_course_metadata("c1", {**rolled_up, "functional_competencies": kcm("F1")}, minimal_text())


def test_proficiency_level_is_a_weighted_vote():
    resources = [
        resource("a", 700, area="Functional", functional_competencies=[kcm("F1", "Tactical")]),
        resource("b", 300, area="Functional", functional_competencies=[kcm("F1", "Strategic")]),
    ]
    competency = roll_up_resources(resources)["functional_competencies"]
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
        course_summary="s", description="d", learning_outcomes=[],
        keywords=[], prerequisites=[],
        responsibilities_addressed=[], activities_enabled=[],
    )
    metadata = build_course_metadata("c1", roll_up_resources(resources), text)
    assert metadata.course_id == "c1"
    assert metadata.primary_competency_area.value == "Domain"
    assert metadata.course_summary == "s"


# --- keywords_list, rationales and field order -------------------------------------------------


def keyword(name, importance, variants):
    return Keyword(keyword=name, type="Concept", importance=importance, variants=variants)


def test_keywords_list_is_sorted_by_importance_with_short_forms_and_variant_counts():
    keywords = [
        keyword("phishing", "Low", ["phishing"]),
        keyword("Quality of Service", "High", ["quality of service", "QoS", "QS", "QOS"]),
        keyword("QoS flow", "Medium", ["QoS flow", "QoS flows"]),
    ]
    assert build_keywords_list(keywords) == [
        "Quality of Service (QoS, QS) [High, 4 variants]",
        "QoS flow [Medium, 2 variants]",
        "phishing [Low, 1 variant]",
    ]


def test_keywords_list_uses_the_expanded_form_when_the_keyword_is_an_acronym():
    keywords = [keyword("NCRP", "High", ["NCRP", "National Cybercrime Reporting Portal", "Cybercrime Portal"])]
    assert build_keywords_list(keywords) == ["National Cybercrime Reporting Portal (NCRP) [High, 3 variants]"]


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
        course_summary="s", description="d", learning_outcomes=[], keywords=[],
        prerequisites=[{"topic": "t", "recommended_by": "AI", "confidence": 0.8, "rationale": "r"}],
        responsibilities_addressed=[], activities_enabled=[],
    )
    document = build_course_metadata("c1", roll_up_resources(resources), text).model_dump(mode="json")
    assert list(document["learning_level"])[:3] == ["rationale", "value", "confidence"]
    assert list(document["prerequisites"][0])[0] == "topic"
    assert list(document["functional_competencies"])[0] == "competency_id"
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
        pass

    def fetchall(self):
        return self.rows


def test_complete_course_is_fetched():
    resources = [resource("a", 100), resource("b", 200)]
    assert [r["resource_id"] for r in fetch_course_resources(FakeConnection(resources), "c1")] == ["a", "b"]


def test_course_with_a_resource_missing_metadata_is_skipped_and_lists_the_missing_ids():
    resources = [resource("a", 100), resource("b", 200, course_summary=None), resource("c", 50, course_summary=None)]
    with pytest.raises(CourseSkippedError, match=r"2 of 3 resources have no metadata yet: b, c"):
        fetch_course_resources(FakeConnection(resources), "c1")


def test_course_with_no_transcribed_resource_is_skipped():
    with pytest.raises(CourseSkippedError, match="no resource with a transcript"):
        fetch_course_resources(FakeConnection([]), "c1")
