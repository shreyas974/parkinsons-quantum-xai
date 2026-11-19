"""Request and response shapes for the API. The frontend mirrors these in src/lib/types.ts."""

from __future__ import annotations

from pydantic import BaseModel


class Contribution(BaseModel):
    feature: str
    signal: str  # "voice" or "spiral"
    value: float  # the reduced feature, already scaled to [0, pi] for the qubit rotation
    shap: float  # push towards Parkinson's (+) or towards healthy (-)


class PredictResponse(BaseModel):
    demo: bool
    model: str
    model_label: str
    model_kind: str
    probability: float
    label: str  # "higher" or "lower"
    base_value: float
    contributions: list[Contribution]
    angles: list[float]  # one RY rotation per qubit, same order as feature_names
    feature_names: list[str]
    voice_share: float
    spiral_share: float
    elapsed_ms: int
    note: str = ""


class ModelInfo(BaseModel):
    id: str
    label: str
    kind: str  # "quantum" or "classical"
    balanced_accuracy: float | None = None


class Health(BaseModel):
    status: str
    demo: bool
    reason: str
    models: list[str]
    voice_columns: list[str]


class VoiceSample(BaseModel):
    id: str
    label: str
    description: str
    values: dict[str, float]


class ResultRow(BaseModel):
    input: str
    model: str
    label: str
    kind: str
    folds: int
    seeds: int | None = None
    balanced_accuracy: float | None = None
    balanced_accuracy_std: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    f1: float | None = None
    auc: float | None = None
    train_time_s: float | None = None


class ResultsSummary(BaseModel):
    available: bool
    note: str
    rows: list[ResultRow]
