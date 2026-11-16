"""Output schema for resource-level metadata.

Field names, order and allowed values mirror Section 5 of
RESOURCE_METADATA_SYSTEM_PROMPT in src/prompts.py. Keep the two in sync.

Resources with no usable transcript should be skipped before calling the model,
so the schema has no "error" shape.
"""

from typing import Literal

from pydantic import BaseModel, Field

StoLevel = Literal["Strategic", "Tactical", "Operational"]


class Judged(BaseModel):
    """A decision the model made, with how sure it is and why."""

    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# --- Part 1: content understanding ----------------------------------------- #


class Presenter(BaseModel):
    name: str
    designation: str | None
    organisation: str | None


class Keyword(BaseModel):
    keyword: str
    type: Literal["Concept", "Procedure", "Scheme/Law/Policy", "Tool/Portal", "Organisation", "Other"]
    importance: Literal["High", "Medium", "Low"]
    variants: list[str]


# --- Part 2: learner profile ----------------------------------------------- #


class LearningLevel(Judged):
    value: Literal["Beginner", "Intermediate", "Advanced"]


class LearningMode(Judged):
    value: Literal["Online", "Offline", "Hybrid"]


class Prerequisite(Judged):
    topic: str
    recommended_by: Literal["Speaker", "AI"]


class BloomLevel(Judged):
    level: Literal["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]


# --- Part 3: competency mapping and STO level ------------------------------ #


class PrimaryCompetencyArea(Judged):
    value: Literal["Domain", "Functional", "Behavioural"]


class KcmCompetency(Judged):
    """Behavioural or Functional competency from the KCM catalogue."""

    competency_id: str
    theme: str
    sub_theme: str
    proficiency_level: StoLevel


class DomainCompetency(Judged):
    sector: str
    subsector: str
    theme: str | None


class StoEvidence(BaseModel):
    phrase: str
    indicates: StoLevel


class StoLevelJudgement(Judged):
    value: StoLevel
    evidence: list[StoEvidence]


# --- Part 4: role targeting ------------------------------------------------ #


class EmployeeGroup(Judged):
    group: Literal["Group A", "Group B", "Group C", "Group D"]


class Designation(Judged):
    name: str


# --- Root ------------------------------------------------------------------- #


class ResourceMetadata(BaseModel):
    resource_id: str
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
