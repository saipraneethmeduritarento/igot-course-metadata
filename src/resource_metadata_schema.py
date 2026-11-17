"""Output schema for resource-level metadata.

Field names, order and allowed values mirror Section 5 of
RESOURCE_METADATA_SYSTEM_PROMPT in src/prompts.py. Keep the two in sync.

Resources with no usable transcript should be skipped before calling the model,
so the schema has no "error" shape.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

StoLevel = Literal["Strategic", "Tactical", "Operational"]

# Every decision the model makes has a `rationale` and a `confidence`. The model writes its JSON in
# schema order (pydantic keeps the field order of the class), so in each class the identifying fields
# come first, then the `rationale` (and `evidence`), then the decided value and its `confidence`:
# the model reasons before it commits to a value.
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


class ReferenceResource(BaseModel):
    value: str
    type: Literal[
        "URL", "Act/Rule", "Circular/Order/Guideline", "Report/Book/Manual", "Portal/App", "Helpline", "Course", "Other"
    ]


# --- Part 2: learner profile ----------------------------------------------- #


class LearningLevel(BaseModel):
    rationale: str
    value: Literal["Beginner", "Intermediate", "Advanced"]
    confidence: Confidence


class LearningMode(BaseModel):
    rationale: str
    value: Literal["Online", "Offline", "Hybrid"]
    confidence: Confidence


class Prerequisite(BaseModel):
    topic: str
    rationale: str
    recommended_by: Literal["Speaker", "AI"]
    confidence: Confidence


class BloomLevel(BaseModel):
    level: Literal["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]
    rationale: str
    confidence: Confidence


# --- Part 3: competency mapping and STO level ------------------------------ #


class PrimaryCompetencyArea(BaseModel):
    rationale: str
    value: Literal["Domain", "Functional", "Behavioural"]
    confidence: Confidence


class KcmCompetency(BaseModel):
    """Behavioural or Functional competency from the KCM catalogue."""

    competency_id: str
    theme: str
    sub_theme: str
    rationale: str
    proficiency_level: StoLevel
    confidence: Confidence


class DomainCompetency(BaseModel):
    sector: str
    subsector: str
    theme: str | None
    rationale: str
    confidence: Confidence


class StoEvidence(BaseModel):
    phrase: str
    indicates: StoLevel


class StoLevelJudgement(BaseModel):
    evidence: list[StoEvidence]
    rationale: str
    value: StoLevel
    confidence: Confidence


# --- Part 4: role targeting ------------------------------------------------ #


class EmployeeGroup(BaseModel):
    group: Literal["Group A", "Group B", "Group C", "Group D"]
    rationale: str
    confidence: Confidence


class Designation(BaseModel):
    name: str
    rationale: str
    confidence: Confidence


# --- Root ------------------------------------------------------------------- #


class ResourceMetadata(BaseModel):
    presenters: list[Presenter]
    description: str
    course_summary: str
    learning_outcomes: list[str]
    keywords: list[Keyword]
    reference_resources: list[ReferenceResource]
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
