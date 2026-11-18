"""Output schema for resource-level metadata.

Field names, order and allowed values mirror Section 5 of
RESOURCE_METADATA_SYSTEM_PROMPT in src/prompts.py. Keep the two in sync.

Resources with no usable transcript should be skipped before calling the model,
so the schema has no "error" shape.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

StoLevel = Literal["Strategic", "Tactical", "Operational"]

# Every decision the model makes has a `confidence` and a `rationale`. They are declared last in each
# class (value or identifying fields first) because pydantic keeps the field order of the class, and
# the model writes its JSON in schema order.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


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


class LearningLevel(BaseModel):
    value: Literal["Beginner", "Intermediate", "Advanced"]
    confidence: Confidence
    rationale: str


class LearningMode(BaseModel):
    value: Literal["Online", "Offline", "Hybrid"]
    confidence: Confidence
    rationale: str


class Prerequisite(BaseModel):
    topic: str
    recommended_by: Literal["Speaker", "AI"]
    confidence: Confidence
    rationale: str


class BloomLevel(BaseModel):
    level: Literal["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]
    confidence: Confidence
    rationale: str


# --- Part 3: competency mapping and STO level ------------------------------ #


class PrimaryCompetencyArea(BaseModel):
    value: Literal["Domain", "Functional", "Behavioural"]
    confidence: Confidence
    rationale: str


class KcmCompetency(BaseModel):
    """Behavioural or Functional competency from the KCM catalogue."""

    competency_id: str
    theme: str
    sub_theme: str
    proficiency_level: StoLevel
    confidence: Confidence
    rationale: str


class DomainCompetency(BaseModel):
    sector: str
    subsector: str
    theme: str | None
    confidence: Confidence
    rationale: str


class StoEvidence(BaseModel):
    phrase: str
    indicates: StoLevel


class StoLevelJudgement(BaseModel):
    value: StoLevel
    confidence: Confidence
    rationale: str
    evidence: list[StoEvidence]


# --- Part 4: role targeting ------------------------------------------------ #


class EmployeeGroup(BaseModel):
    group: Literal["Group A", "Group B", "Group C", "Group D"]
    confidence: Confidence
    rationale: str


class Designation(BaseModel):
    name: str
    confidence: Confidence
    rationale: str


# --- Root ------------------------------------------------------------------- #


class ResourceMetadata(BaseModel):
    resource_id: str
    presenters: list[Presenter]
    course_summary: str
    description: str
    learning_outcomes: list[str]
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
