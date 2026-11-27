"""Output schema for course-level metadata (the roll-up of resource-level metadata).

CourseMetadata has the fields of ResourceMetadata (src/resource_metadata/schema.py), with
`course_id` in place of `resource_id`, plus `keywords`. Unlike a resource, a course has
competencies from ONE area: the field of its primary area holds 1 to COURSE_MAX_ITEMS of them
(src/config.py), and the other two competency fields are `[]`. A Domain competency of a course
always has its `theme`.

The `model_used`, `prompt_version`, `resource_ids`, `language` and `list_of_languages` columns of
course_metadata_rollup_<version> are added by code, so they are not part of this schema.

CourseTextMetadata mirrors Section 4 of COURSE_METADATA_SYSTEM_PROMPT in src/prompts.py.
Keep the two in sync.
"""

from pydantic import BaseModel, model_validator

from src.resource_metadata.schema import (
    BloomLevel,
    Designation,
    DomainCompetency,
    EmployeeGroup,
    KcmCompetency,
    Keyword,
    LearningLevel,
    LearningMode,
    Prerequisite,
    Presenter,
    PrimaryCompetencyArea,
    ReferenceResource,
    StoLevelJudgement,
)

# Primary competency area -> the CourseMetadata field that holds its one competency.
COMPETENCY_FIELD_OF_AREA = {
    "Behavioural": "behavioural_competencies",
    "Functional": "functional_competencies",
    "Domain": "domain_competencies",
}


class CourseDomainCompetency(DomainCompetency):
    """A course's sector path always has its theme (a resource's may not)."""

    theme: str


class CourseTextMetadata(BaseModel):
    """The course fields the model writes. Code decides every other CourseMetadata field."""

    course_coverage: str
    description: str
    learning_outcomes: list[str]
    keywords_list: list[Keyword]
    prerequisites: list[Prerequisite]
    responsibilities_addressed: list[str]
    activities_enabled: list[str]


class CourseMetadata(BaseModel):
    """The full course-level metadata: the model's text fields plus the fields decided by code.

    The text fields are listed again here, rather than inherited from CourseTextMetadata, because
    the field order sets the key order of the output JSON and the column order of the table.
    """

    course_id: str
    presenters: list[Presenter]
    course_coverage: str
    description: str
    learning_outcomes: list[str]
    keywords_list: list[Keyword]
    keywords: list[str]  # built by code from `keywords_list`, see build_keywords
    reference_resources: list[ReferenceResource]
    learning_level: LearningLevel
    learning_mode: LearningMode
    prerequisites: list[Prerequisite]
    bloom_taxonomy: list[BloomLevel]
    primary_competency_area: PrimaryCompetencyArea
    # Filled only in the field of `primary_competency_area`; the other two are [].
    behavioural_competencies: list[KcmCompetency]
    functional_competencies: list[KcmCompetency]
    domain_competencies: list[CourseDomainCompetency]
    sto_level: StoLevelJudgement
    responsibilities_addressed: list[str]
    activities_enabled: list[str]
    target_employee_groups: list[EmployeeGroup]
    target_roles: list[Designation]
    target_roles_exclusions: list[Designation]

    @model_validator(mode="after")
    def has_competencies_only_in_its_primary_area(self) -> "CourseMetadata":
        """The primary area's competency list is not empty; the other two are empty."""
        primary_area = self.primary_competency_area.value
        for area, field_name in COMPETENCY_FIELD_OF_AREA.items():
            is_primary = area == primary_area
            is_set = bool(getattr(self, field_name))
            if is_set != is_primary:
                state = "empty" if is_primary else "set"
                raise ValueError(
                    f"{field_name} is {state}, but the primary competency area is {primary_area}"
                )
        return self
