from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    food_data_dir: Path
    foundation_dir: Path
    sr_legacy_dir: Path
    model_dir: Path

    @staticmethod
    def from_root(root_dir: str | Path) -> "ProjectPaths":
        root = Path(root_dir).resolve()
        food_data_dir = root / "food_data"
        foundation_dir = food_data_dir / "FoodData_Central_foundation_food_csv_2025-12-18"
        sr_legacy_dir = food_data_dir / "FoodData_Central_sr_legacy_food_csv_2018-04"
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return ProjectPaths(
            root_dir=root,
            food_data_dir=food_data_dir,
            foundation_dir=foundation_dir,
            sr_legacy_dir=sr_legacy_dir,
            model_dir=model_dir,
        )
