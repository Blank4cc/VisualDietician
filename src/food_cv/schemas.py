from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class PortionEstimation:
    label: str
    weight_g: float
    confidence: float


@dataclass(frozen=True)
class NutritionBreakdown:
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float


@dataclass(frozen=True)
class MealItemResult:
    label: str
    confidence: float
    weight_g: float
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float

