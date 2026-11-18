from src.utils.create_course_metadata_llm import TARGET_COLUMNS, to_llm_row, top_competency, unique


def judged(value):
    return {"value": value, "confidence": 0.8, "rationale": "r"}


ROLLUP = {
    "course_id": "do_1",
    "language": "Hindi",
    "presenters": [
        {"name": "Asha Rao", "designation": "Director", "organisation": "NIC"},
        {"name": " asha rao ", "designation": None, "organisation": None},
        {"name": "Vikram Shah", "designation": None, "organisation": None},
    ],
    "keywords": [
        {"keyword": "Load dispatch", "type": "Concept", "importance": "Low", "variants": []},
        {"keyword": "Merit order", "type": "Concept", "importance": "High", "variants": []},
        {"keyword": "merit order", "type": "Concept", "importance": "Medium", "variants": []},
    ],
    "prerequisites": [{"topic": "Basic electricity", "recommended_by": "AI", "confidence": 0.6, "rationale": "r"}],
    "course_summary": "Summary.",
    "description": "Description.",
    "learning_outcomes": ["Outcome"],
    "learning_level": judged("Intermediate"),
    "learning_mode": judged("Online"),
    "bloom_taxonomy": [
        {"level": "Apply", "confidence": 0.7, "rationale": "r"},
        {"level": "Understand", "confidence": 0.5, "rationale": "r"},
    ],
    "primary_competency_area": judged("Domain"),
    "behavioural_competencies": [],
    "functional_competencies": [],
    "domain_competencies": [
        {"sector": "Resources", "subsector": "Coal", "theme": "Mining", "confidence": 0.4, "rationale": "r"},
        {"sector": "Resources", "subsector": "Power", "theme": "Grid", "confidence": 0.9, "rationale": "r"},
    ],
    "sto_level": {**judged("Operational"), "evidence": []},
    "responsibilities_addressed": ["Schedule generation"],
    "activities_enabled": ["Prepare schedules"],
    "target_employee_groups": [{"group": "Group A", "confidence": 0.8, "rationale": "r"}],
    "target_roles": [{"name": "Executive Engineer", "confidence": 0.8, "rationale": "r"}],
    "target_roles_exclusions": [{"name": "Clerk", "confidence": 0.3, "rationale": "r"}],
    "reference_resources": ["IEGC 2023", "IEGC 2023"],
}


def test_row_has_every_column_and_flattens_values():
    row = to_llm_row(ROLLUP)
    assert list(row) == list(TARGET_COLUMNS)
    assert row["speakers"] == ["Asha Rao", "Vikram Shah"]
    assert row["language"] == "Hindi"
    assert row["keywords"] == ["Merit order", "Load dispatch"]  # High first, case-insensitive duplicate dropped
    assert row["prerequisites"] == ["Basic electricity"]
    assert row["course_description"] == "Description."
    assert row["learning_level"] == "Intermediate"
    assert row["learning_mode"] == "Online"
    assert row["blooms_taxonomy"] == ["Apply", "Understand"]
    assert row["primary_competency_area"] == "Domain"
    assert row["sto"] == "Operational"
    assert row["target_employee_group"] == ["Group A"]
    assert row["target_roles"] == ["Executive Engineer"]
    assert row["target_roles_exclusions"] == ["Clerk"]
    assert row["references"] == ["IEGC 2023"]


def test_domain_theme_is_sector_and_subsector_of_top_competency():
    assert top_competency(ROLLUP) == ("Resources", "Power")


def test_kcm_theme_is_theme_and_sub_theme_of_primary_area():
    rollup = {
        **ROLLUP,
        "primary_competency_area": judged("Behavioural"),
        "behavioural_competencies": [
            {"competency_id": "BEH-1", "theme": "Communication", "sub_theme": "Active Listening",
             "proficiency_level": "Operational", "confidence": 0.44, "rationale": "r"},
            {"competency_id": "BEH-2", "theme": "Collaboration", "sub_theme": "Teamwork",
             "proficiency_level": "Tactical", "confidence": 0.28, "rationale": "r"},
        ],
    }
    assert top_competency(rollup) == ("Communication", "Active Listening")


def test_missing_competencies_and_nulls_give_empty_values():
    rollup = {"course_id": "do_2", "primary_competency_area": judged("Functional"), "functional_competencies": []}
    row = to_llm_row(rollup)
    assert (row["theme"], row["sub_theme"]) == (None, None)
    assert row["speakers"] == []
    assert row["references"] == []
    assert row["learning_level"] is None


def test_unique_skips_blanks_and_keeps_first_spelling():
    assert unique(["QoS", "qos ", "", None, "PDU"]) == ["QoS", "PDU"]
