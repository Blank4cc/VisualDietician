from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import re

import pandas as pd

from .schemas import NutritionBreakdown


@dataclass(frozen=True)
class NutritionConfig:
    foundation_dir: Path
    sr_legacy_dir: Path | None = None


class USDANutritionEngine:
    def __init__(self, config: NutritionConfig) -> None:
        self.config = config
        self.food_df: pd.DataFrame | None = None
        self.food_nutrient_df: pd.DataFrame | None = None
        self._name_index: dict[str, int] = {}
        self._all_keys: list[str] = []
        self._fdc_tokens: dict[int, set[str]] = {}
        self._token_index: dict[str, set[int]] = {}
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
            "greek_salad": "greek salad",
            "tuna_tartare": "tuna",
            "bruschetta": "tomato bread",
            "fish_and_chips": "fried fish and french fries",
            "macarons": "macaroon",
            "foie_gras": "duck liver",
            "garlic_bread": "bread garlic",
            "chicken_quesadilla": "quesadilla chicken",
            "beef_tartare": "beef",
            "beet_salad": "beet salad",
            "club_sandwich": "sandwich",
            "eggs_benedict": "egg benedict",
            "pulled_pork_sandwich": "pork sandwich",
            "pork_chop": "pork chop",
            "scallops": "scallop",
            "seaweed_salad": "seaweed salad",
            "shrimp_and_grits": "shrimp",
            "waffles": "waffle",
            "donuts": "doughnut",
            "churros": "fried dough",
            "huevos_rancheros": "egg",
            "samosa": "samosa",
            "edamame": "soybean",
            "apple_pie": "apple pie",
            "baby_back_ribs": "pork ribs",
            "beignets": "fried dough",
            "bibimbap": "rice beef vegetable",
            "bread_pudding": "bread pudding",
            "breakfast_burrito": "burrito",
            "caesar_salad": "caesar salad",
            "caprese_salad": "mozzarella tomato basil salad",
            "carrot_cake": "carrot cake",
            "cheesecake": "cheesecake",
            "chicken_curry": "chicken curry",
            "chicken_wings": "chicken wings",
            "clam_chowder": "clam chowder",
            "croque_madame": "ham cheese sandwich",
            "cup_cakes": "cupcake",
            "deviled_eggs": "egg",
            "dumplings": "dumpling",
            "escargots": "snail",
            "falafel": "falafel",
            "filet_mignon": "beef steak",
            "frozen_yogurt": "frozen yogurt",
            "gnocchi": "gnocchi",
            "grilled_salmon": "salmon",
            "guacamole": "avocado",
            "gyoza": "dumpling",
            "hamburger": "hamburger",
            "hot_and_sour_soup": "hot and sour soup",
            "lobster_bisque": "lobster bisque",
            "lobster_roll_sandwich": "lobster sandwich",
            "miso_soup": "miso soup",
            "nachos": "nachos",
            "oysters": "oyster",
            "pad_thai": "pad thai",
            "paella": "paella",
            "pho": "pho",
            "poutine": "french fries gravy cheese curd",
            "ramen": "ramen",
            "ravioli": "ravioli",
            "risotto": "risotto",
            "sashimi": "raw fish",
            "spaghetti_bolognese": "spaghetti bolognese",
            "takoyaki": "octopus ball",
            "tiramisu": "tiramisu",
            "waffles": "waffle",
        }
        self._fallback_per_100g: dict[str, NutritionBreakdown] = {
            "greek salad": NutritionBreakdown(150.0, 3.8, 12.0, 6.0),
            "salad": NutritionBreakdown(90.0, 2.5, 5.5, 8.5),
            "tuna": NutritionBreakdown(132.0, 28.0, 1.0, 0.0),
            "bruschetta": NutritionBreakdown(210.0, 6.0, 8.0, 27.0),
            "burger": NutritionBreakdown(250.0, 12.0, 11.0, 24.0),
            "fried rice": NutritionBreakdown(185.0, 4.2, 6.8, 26.5),
            "spaghetti": NutritionBreakdown(158.0, 5.8, 0.9, 30.8),
            "pizza": NutritionBreakdown(266.0, 11.0, 10.0, 33.0),
            "ice cream": NutritionBreakdown(207.0, 3.5, 11.0, 24.0),
            "french fries": NutritionBreakdown(312.0, 3.4, 15.0, 41.0),
            "croque madame": NutritionBreakdown(271.0, 12.0, 15.0, 21.0),
            "risotto": NutritionBreakdown(166.0, 4.0, 6.0, 24.0),
            "huevos rancheros": NutritionBreakdown(156.0, 8.0, 8.0, 13.0),
            "prime rib": NutritionBreakdown(355.0, 24.0, 28.0, 0.0),
        }
        self._load()

    def _load(self) -> None:
        food_frames: list[pd.DataFrame] = []
        nutrient_frames: list[pd.DataFrame] = []

        foundation_food_csv = self.config.foundation_dir / "food.csv"
        foundation_food_nutrient_csv = self.config.foundation_dir / "food_nutrient.csv"
        if not foundation_food_csv.exists():
            raise FileNotFoundError(f"找不到 USDA food.csv: {foundation_food_csv}")
        if not foundation_food_nutrient_csv.exists():
            raise FileNotFoundError(f"找不到 USDA food_nutrient.csv: {foundation_food_nutrient_csv}")
        food_frames.append(pd.read_csv(foundation_food_csv, usecols=["fdc_id", "description"], low_memory=False))
        nutrient_frames.append(
            pd.read_csv(
                foundation_food_nutrient_csv,
                usecols=["fdc_id", "nutrient_id", "amount"],
                low_memory=False,
            )
        )

        if self.config.sr_legacy_dir:
            sr_food_csv = self.config.sr_legacy_dir / "food.csv"
            sr_food_nutrient_csv = self.config.sr_legacy_dir / "food_nutrient.csv"
            if sr_food_csv.exists() and sr_food_nutrient_csv.exists():
                food_frames.append(pd.read_csv(sr_food_csv, usecols=["fdc_id", "description"], low_memory=False))
                nutrient_frames.append(
                    pd.read_csv(
                        sr_food_nutrient_csv,
                        usecols=["fdc_id", "nutrient_id", "amount"],
                        low_memory=False,
                    )
                )

        self.food_df = pd.concat(food_frames, ignore_index=True)
        self.food_nutrient_df = pd.concat(nutrient_frames, ignore_index=True)
        self.food_nutrient_df = self.food_nutrient_df.dropna(subset=["fdc_id", "nutrient_id", "amount"])
        self.food_df = self.food_df.dropna(subset=["description"]).copy()
        self.food_df["key"] = self.food_df["description"].str.lower().str.strip()
        self.food_df = self.food_df.drop_duplicates(subset=["fdc_id"], keep="first")
        self._name_index = dict(zip(self.food_df["key"], self.food_df["fdc_id"], strict=False))
        self._all_keys = list(self._name_index.keys())
        for key, fdc_id in self._name_index.items():
            tokens = self._tokenize(key)
            self._fdc_tokens[int(fdc_id)] = tokens
            for token in tokens:
                self._token_index.setdefault(token, set()).add(int(fdc_id))

    @staticmethod
    def _normalize_food_name(food_name: str) -> str:
        return food_name.lower().strip().replace("_", " ").replace("-", " ")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {x for x in re.findall(r"[a-zA-Z]+", text.lower()) if len(x) >= 3}

    def _best_by_token_overlap(self, key: str) -> int | None:
        query_tokens = self._tokenize(key)
        if not query_tokens:
            return None
        candidate_ids: set[int] = set()
        for token in query_tokens:
            candidate_ids.update(self._token_index.get(token, set()))
        if not candidate_ids:
            return None

        best_id: int | None = None
        best_score = -1.0
        for fdc_id in candidate_ids:
            desc_tokens = self._fdc_tokens.get(fdc_id, set())
            if not desc_tokens:
                continue
            intersect = len(query_tokens & desc_tokens)
            union = len(query_tokens | desc_tokens)
            jaccard = intersect / max(union, 1)
            coverage = intersect / max(len(query_tokens), 1)
            score = 0.7 * coverage + 0.3 * jaccard
            if score > best_score:
                best_score = score
                best_id = fdc_id
        if best_score < 0.35:
            return None
        return best_id

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
        token_match_id = self._best_by_token_overlap(key)
        if token_match_id is not None:
            return token_match_id
        matches = get_close_matches(key, self._all_keys, n=1, cutoff=0.6)
        if not matches:
            return None
        return int(self._name_index[matches[0]])

    def _fallback_nutrition_for(self, food_name: str, weight_g: float) -> NutritionBreakdown | None:
        normalized = self._normalize_food_name(food_name)
        key = self._food101_alias.get(normalized, normalized)
        profile = self._fallback_per_100g.get(key)
        if profile is None:
            for fallback_key, item in self._fallback_per_100g.items():
                if fallback_key in key or key in fallback_key:
                    profile = item
                    break
        if profile is None:
            return None
        scale = weight_g / 100.0
        return NutritionBreakdown(
            calories=profile.calories * scale,
            protein_g=profile.protein_g * scale,
            fat_g=profile.fat_g * scale,
            carbs_g=profile.carbs_g * scale,
        )

    def nutrition_for(self, food_name: str, weight_g: float) -> NutritionBreakdown:
        if weight_g <= 0:
            raise ValueError("weight_g 必须大于 0")
        if self.food_nutrient_df is None:
            raise RuntimeError("USDA 数据尚未加载")
        fdc_id = self._find_fdc_id(food_name)
        if fdc_id is None:
            fallback = self._fallback_nutrition_for(food_name, weight_g)
            if fallback is not None:
                return fallback
            return NutritionBreakdown(calories=0.0, protein_g=0.0, fat_g=0.0, carbs_g=0.0)

        subset = self.food_nutrient_df[self.food_nutrient_df["fdc_id"] == fdc_id]
        if subset.empty:
            fallback = self._fallback_nutrition_for(food_name, weight_g)
            if fallback is not None:
                return fallback
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

        if per_100g["calories"] <= 0.0:
            fallback = self._fallback_nutrition_for(food_name, weight_g)
            if fallback is not None:
                return fallback

        scale = weight_g / 100.0
        return NutritionBreakdown(
            calories=per_100g["calories"] * scale,
            protein_g=per_100g["protein_g"] * scale,
            fat_g=per_100g["fat_g"] * scale,
            carbs_g=per_100g["carbs_g"] * scale,
        )
