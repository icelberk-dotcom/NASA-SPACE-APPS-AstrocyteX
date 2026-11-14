"""
Pydantic schemas shared by the FastAPI layer, the Gemini agent and the UI client.

Open-source attribution: Pydantic (MIT) for validation / serialisation, FastAPI (MIT)
consumes these models for request/response documentation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------------- inputs
class ReadingIn(BaseModel):
    """One snapshot of an astronaut's data (any subset of the 43 parameters: 31 physiological + 12 environment)."""

    values: Dict[str, float] = Field(default_factory=dict, description="parameter key -> value in its unit")
    sensor_series: Optional[Dict[str, List[float]]] = Field(
        default=None, description="optional recent high-rate stream per continuous sensor (newest last)")
    subjective_log: str = Field(default="", max_length=4000, description="free-text log (EN/TR)")
    mission_day: float = Field(default=0.0, ge=0, le=1500)
    gravity: float = Field(default=0.0, ge=0, le=1.0,
                           description="0 = microgravity, 1 = Earth (used only if values has no 'gravity_g')")
    ecg: Optional[Dict[str, Any]] = Field(default=None, description="ECGResult dict to merge (optional)")
    save: bool = True


# ------------------------------------------------------------------------ Gemini
class TaskDraft(BaseModel):
    """A mitigation task proposed by Gemini (or by the offline reasoner)."""

    parameter_key: str
    title: str = Field(max_length=120)
    instructions: str = Field(default="", max_length=600)
    duration_min: int = Field(default=10, ge=1, le=90)
    tag: str = Field(default="general", max_length=24)


class AdvisorOutput(BaseModel):
    summary: str = Field(description="3-4 sentence situation briefing for the crew member")
    recommendations: List[str] = Field(default_factory=list, description="up to 4 short bullet recommendations")
    tasks: List[TaskDraft] = Field(default_factory=list)
    escalate_to_flight_surgeon: bool = False


# ----------------------------------------------------------------------- outputs
class ParameterAssessment(BaseModel):
    key: str
    name: str
    name_tr: str
    category: str
    unit: str
    kind: str
    value: Optional[float]
    normal_range: str
    risk_pct: Optional[float]
    rule_risk_pct: Optional[float]
    model_risk_pct: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    epistemic_pct: Optional[float] = None
    aleatoric_pct: Optional[float] = None
    level: str
    color: str
    direction: str


class TaskOut(BaseModel):
    id: int
    astronaut_id: str
    parameter_key: str
    title: str
    description: str
    severity_level: str
    risk_pct: float
    duration_min: int
    points: int
    tag: str
    status: str
    source: str
    created_at: str
    completed_at: Optional[str] = None


class BadgeOut(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    earned: bool
    earned_at: Optional[str] = None


class GamificationState(BaseModel):
    astronaut_id: str
    xp: int
    level: int
    rank: str
    level_xp: int
    next_level_xp: int
    progress: float
    streak_days: int
    tasks_completed: int
    badges: List[BadgeOut]


class CompletionResult(BaseModel):
    task: TaskOut
    points_awarded: int
    already_completed: bool = False
    leveled_up: bool = False
    new_badges: List[BadgeOut] = Field(default_factory=list)
    state: GamificationState


class Assessment(BaseModel):
    astronaut_id: str
    timestamp: str
    model_version: str
    model_trained_on: str
    overall: Dict[str, Any]
    systems: Dict[str, Dict[str, Any]]
    parameters: List[ParameterAssessment]
    forecast: List[float]
    physics: Dict[str, Any]
    modality_attention: Dict[str, float]
    trend_drivers: List[Dict[str, Any]]
    ecg: Optional[Dict[str, Any]] = None
    environment: Dict[str, Any] = Field(default_factory=dict,
                                        description="space-environment hazard index, RIDGE themes, derived values and "
                                                    "the counterfactual environmental risk uplift")
    advisor: Dict[str, Any]
    new_tasks: List[TaskOut]
    subjective_symptoms: List[str] = Field(default_factory=list)
