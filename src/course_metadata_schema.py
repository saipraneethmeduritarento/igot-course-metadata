"""Output schema for course-level metadata (the roll-up of resource-level metadata).

The fields are the same as ResourceMetadata (src/resource_metadata_schema.py), with
`course_id` in place of `resource_id`. The `language` and `resource_ids` columns of the
course_metadata_rollup table are added by code, not by the model, so they are not here.
CourseTextMetadata mirrors Section 4 of COURSE_METADATA_SYSTEM_PROMPT in src/prompts.py. Keep the two in sync.
"""

from pydantic import BaseModel

from src.resource_metadata_schema import (
    BloomLevel,
    DomainCompetency,
    Designation,
    EmployeeGroup,
    KcmCompetency,
    Keyword,
    LearningLevel,
    LearningMode,
    Prerequisite,
    PrimaryCompetencyArea,
    Presenter,
    StoLevelJudgement,
)


class CourseTextMetadata(BaseModel):
    """The course fields the model writes. Everything else in CourseMetadata is decided by code."""

    course_id: str
    presenters: list[Presenter]
    course_summary: str
    keywords: list[Keyword]
    prerequisites: list[Prerequisite]
    responsibilities_addressed: list[str]
    activities_enabled: list[str]


class CourseMetadata(BaseModel):
    """The full course-level metadata: the model's text fields plus the fields decided by code."""

    course_id: str
    presenters: list[Presenter]
    course_summary: str
    keywords: list[Keyword]
    reference_resources: list[str]
    learning_level: LearningLevel
    learning_mode: LearningMode
    prerequisites: list[Prerequisite]
    bloom_taxonomy: list[BloomLevel]
    primary_competency_area: PrimaryCompetencyArea
    behavioural_competencies: list[KcmCompetency]
    functional_competencies: list[KcmCompetency]
    domain_competencies: list[DomainCompetency]
    sto_level: StoLevelJudgement
    responsibilities_addressed: list[str]
    activities_enabled: list[str]
    target_employee_groups: list[EmployeeGroup]
    target_roles: list[Designation]
    target_roles_exclusions: list[Designation]
