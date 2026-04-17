from __future__ import annotations

from pathlib import Path
from typing import Any

from .classifier import FoodClassifier, build_resnet50_classifier
from .config import ProjectPaths
from .nutrition_engine import NutritionConfig, USDANutritionEngine
from .portion_estimator import PortionEstimator
from .schemas import MealItemResult


class MealPredictor:
    def __init__(
        self,
        paths: ProjectPaths,
        checkpoint_path: str | Path | None = None,
        labels: list[str] | None = None,
    ) -> None:
        model = build_resnet50_classifier(num_classes=101, freeze_backbone=True)
        self.classifier = FoodClassifier(model=model, labels=labels)
        if checkpoint_path:
            self.classifier.load_checkpoint(checkpoint_path)
        self.portion_estimator = PortionEstimator()
        self.nutrition_engine = USDANutritionEngine(
            NutritionConfig(
                foundation_dir=paths.foundation_dir,
                sr_legacy_dir=paths.sr_legacy_dir,
            )
        )

    def predict_meal(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        class_preds = self.classifier.predict_topk(path, topk=3)
        labels = [x.label for x in class_preds]
        portion_preds = self.portion_estimator.estimate(path, label_candidates=labels)

        items: list[MealItemResult] = []
        for item in portion_preds:
            nutrition = self.nutrition_engine.nutrition_for(item.label, item.weight_g)
            items.append(
                MealItemResult(
                    label=item.label,
                    confidence=item.confidence,
                    weight_g=item.weight_g,
                    calories=nutrition.calories,
                    protein_g=nutrition.protein_g,
                    fat_g=nutrition.fat_g,
                    carbs_g=nutrition.carbs_g,
                )
            )

        total = {
            "calories": float(sum(x.calories for x in items)),
            "protein_g": float(sum(x.protein_g for x in items)),
            "fat_g": float(sum(x.fat_g for x in items)),
            "carbs_g": float(sum(x.carbs_g for x in items)),
        }
        return {
            "image_path": str(path),
            "top3_classification": [x.__dict__ for x in class_preds],
            "items": [x.__dict__ for x in items],
            "total": total,
        }
