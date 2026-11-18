from src.utils.load_metadata_to_postgres import LEVELS, normalise_course, to_row

RESOURCE_COLUMNS = LEVELS["resource"][3]
COURSE_COLUMNS = LEVELS["course"][3]

RECORD = {
    "resource_id": "do_1",
    "course_summary": "A summary.",
    "learning_outcomes": ["Outcome one", "Outcome two"],
    "learning_level": {"value": "Beginner", "confidence": 0.8, "rationale": "Basic terms."},
    "keywords": [
        {"keyword": "QoS", "type": "Concept", "importance": "High", "variants": ["QoS", "quality of service"]},
        {"keyword": "5QI", "type": "Concept", "importance": "Low", "variants": ["5QI"]},
    ],
    "domain_competencies": [
        {"sector": "Energy", "subsector": "Power", "theme": None, "confidence": 0.7, "rationale": "Grid."}
    ],
    "sto_level": {"value": "Tactical", "confidence": 0.7, "rationale": "r",
                  "evidence": [{"phrase": "plan", "indicates": "Tactical"}]},
    "target_roles": [],
}


def as_dict(record, columns=RESOURCE_COLUMNS):
    """The row of a record keyed by column name, with Json wrappers unwrapped."""
    return {c: getattr(v, "adapted", v) for (c, _, _), v in zip(columns, to_row(record, columns))}


def column_types(columns=RESOURCE_COLUMNS):
    return {name: sql_type for name, sql_type, _ in columns}


def test_one_column_per_key_with_matching_types():
    types = column_types()
    assert types["resource_id"] == "TEXT"
    assert types["course_summary"] == "TEXT"
    assert types["learning_outcomes"] == "TEXT[]"
    assert types["learning_level"] == "TEXT"
    assert types["learning_level_confidence"] == "NUMERIC(4, 3)"
    assert types["presenters_name"] == "TEXT[]"
    assert types["domain_competencies_confidence"] == "NUMERIC(4, 3)[]"
    assert types["sto_level_evidence_phrase"] == "TEXT[]"
    assert types["behavioural_competencies_competency_id"] == "TEXT[]"
    assert "learning_level_value" not in types


def test_keep_as_json_fields_are_one_jsonb_column_each():
    types = column_types()
    for field in ("keywords", "prerequisites", "bloom_taxonomy", "target_employee_groups",
                  "target_roles", "target_roles_exclusions"):
        assert types[field] == "JSONB"
        assert [name for name, _, path in RESOURCE_COLUMNS if path[0] == field] == [field]
    assert as_dict(RECORD)["keywords"] == RECORD["keywords"]
    assert as_dict(RECORD)["target_roles"] == []


def test_single_object_is_flattened():
    row = as_dict(RECORD)
    assert (row["learning_level"], row["learning_level_confidence"], row["learning_level_rationale"]) == (
        "Beginner", 0.8, "Basic terms.")


def test_list_of_objects_becomes_parallel_arrays():
    row = as_dict(RECORD)
    assert row["domain_competencies_sector"] == ["Energy"]
    assert row["domain_competencies_theme"] == [None]
    assert row["sto_level_evidence_indicates"] == ["Tactical"]


def test_strings_lists_empty_lists_and_missing_keys():
    row = as_dict(RECORD)
    assert row["learning_outcomes"] == ["Outcome one", "Outcome two"]
    assert row["description"] is None and row["presenters_name"] is None


def test_course_columns_include_code_added_fields():
    types = column_types(COURSE_COLUMNS)
    assert types["course_id"] == "TEXT" and "resource_id" not in types
    assert (types["resource_ids"], types["language"], types["list_of_languages"], types["keywords_list"]) == (
        "TEXT[]", "TEXT", "TEXT[]", "TEXT[]")


def test_old_course_language_list_is_split():
    record = normalise_course({"course_id": "do_c", "language": ["English", "Hindi"]})
    assert record["language"] == "English"
    assert record["list_of_languages"] == ["English", "Hindi"]
