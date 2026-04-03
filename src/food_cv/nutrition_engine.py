from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

from .schemas import NutritionBreakdown


@dataclass(frozen=True)
class NutritionConfig:
    foundation_dir: Path


class USDANutritionEngine:
    def __init__(self, config: NutritionConfig) -> None:
        self.config = config
        self.food_df: pd.DataFrame | None = None
        self.food_nutrient_df: pd.DataFrame | None = None
        self._name_index: dict[str, int] = {}
        self._all_keys: list[str] = []
        self._food101_alias: dict[str, str] = {
            "french_fries": "french fries",
            "fried_rice": "fried rice",
            "grilled_cheese_sandwich": "grilled cheese sandwich",
            "hamburger": "burger",
            "hot_dog": "hot dog",
            "ice_cream": "ice cream",
            "macaroni_and_cheese": "macaroni and cheese",
            "mashed_potatoes": "mashed potatoes",
            "pancakes": "pancake",
            "spaghetti_bolognese": "spaghetti",
            "spaghetti_carbonara": "spaghetti",
            "strawberry_shortcake": "strawberry cake",
        }
        self._load()

    def _load(self) -> None:
        food_csv = self.config.foundation_dir / "food.csv"
        food_nutrient_csv = self.config.foundation_dir / "food_nutrient.csv"
        if not food_csv.exists():
            raise FileNotFoundError(f"找不到 USDA food.csv: {food_csv}")
        if not food_nutrient_csv.exists():
            raise FileNotFoundError(f"找不到 USDA food_nutrient.csv: {food_nutrient_csv}")

        self.food_df = pd.read_csv(food_csv, usecols=["fdc_id", "description"], low_memory=False)
        self.food_nutrient_df = pd.read_csv(
            food_nutrient_csv,
            usecols=["fdc_id", "nutrient_id", "amount"],
            low_memory=False,
        )
        self.food_df = self.food_df.dropna(subset=["description"]).copy()
        self.food_df["key"] = self.food_df["description"].str.lower().str.strip()
        self._name_index = dict(zip(self.food_df["key"], self.food_df["fdc_id"]))
        self._all_keys = list(self._name_index.keys())

    @staticmethod
    def _normalize_food_name(food_name: str) -> str:
        return food_name.lower().strip().replace("_", " ").replace("-", " ")

    def _find_fdc_id(self, food_name: str) -> int | None:
        if not food_name:
            return None
        normalized = self._normalize_food_name(food_name)
        key = self._food101_alias.get(normalized, normalized)
        if key in self._name_index:
            return int(self._name_index[key])
        partial_candidates = [x for x in self._all_keys if key in x or x in key]
        if partial_candidates:
            return int(self._name_index[partial_candidates[0]])
        matches = get_close_matches(key, self._all_keys, n=1, cutoff=0.6)
        if not matches:
            return None
        return int(self._name_index[matches[0]])

    def nutrition_for(self, food_name: str, weight_g: float) -> NutritionBreakdown:
        if weight_g <= 0:
            raise ValueError("weight_g 必须大于 0")
        if self.food_nutrient_df is None:
            raise RuntimeError("USDA 数据尚未加载")
        fdc_id = self._find_fdc_id(food_name)
        if fdc_id is None:
            return NutritionBreakdown(calories=0.0, protein_g=0.0, fat_g=0.0, carbs_g=0.0)

        subset = self.food_nutrient_df[self.food_nutrient_df["fdc_id"] == fdc_id]
        if subset.empty:
            return NutritionBreakdown(calories=0.0, protein_g=0.0, fat_g=0.0, carbs_g=0.0)

        nutrients = {
            1008: "calories",
            1003: "protein_g",
            1004: "fat_g",
            1005: "carbs_g",
        }
        per_100g = {"calories": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
        for nutrient_id, field in nutrients.items():
            values = subset.loc[subset["nutrient_id"] == nutrient_id, "amount"]
            if not values.empty:
                per_100g[field] = float(values.iloc[0])

        scale = weight_g / 100.0
        return NutritionBreakdown(
            calories=per_100g["calories"] * scale,
            protein_g=per_100g["protein_g"] * scale,
            fat_g=per_100g["fat_g"] * scale,
            carbs_g=per_100g["carbs_g"] * scale,
        )
