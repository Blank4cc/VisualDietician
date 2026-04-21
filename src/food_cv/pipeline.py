from __future__ import annotations

import json
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
        confidence_threshold: float = 0.35,
        block_nutrition_when_low_confidence: bool = True,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold 必须在 [0, 1] 区间内")
        resolved_labels = labels
        if resolved_labels is None:
            labels_file = paths.model_dir / "class_names.json"
            if labels_file.exists():
                try:
                    payload = json.loads(labels_file.read_text(encoding="utf-8"))
                    if isinstance(payload, list) and payload:
                        resolved_labels = [str(x) for x in payload]
                except Exception:
                    resolved_labels = None
        model = build_resnet50_classifier(num_classes=101, freeze_backbone=True)
        self.classifier = FoodClassifier(model=model, labels=resolved_labels)
        if checkpoint_path:
            self.classifier.load_checkpoint(checkpoint_path)
        self.portion_estimator = PortionEstimator()
        self.nutrition_engine = USDANutritionEngine(
            NutritionConfig(
                foundation_dir=paths.foundation_dir,
                sr_legacy_dir=paths.sr_legacy_dir,
            )
        )
        self.confidence_threshold = confidence_threshold
        self.block_nutrition_when_low_confidence = block_nutrition_when_low_confidence

    def predict_meal(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        class_preds = self.classifier.predict_topk(path, topk=3)
        labels = [x.label for x in class_preds]
        top1_confidence = float(class_preds[0].confidence) if class_preds else 0.0
        is_trusted = top1_confidence >= self.confidence_threshold
        if self.block_nutrition_when_low_confidence and not is_trusted:
            return {
                "image_path": str(path),
                "top3_classification": [x.__dict__ for x in class_preds],
                "items": [],
                "total": {"calories": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0},
                "meta": {
                    "trusted_prediction": False,
                    "confidence_threshold": self.confidence_threshold,
                    "reason": "top1_confidence_below_threshold",
                },
            }

        label_candidates = labels[:1] if self.portion_estimator.config.single_item_mode else labels
        try:
            portion_preds = self.portion_estimator.estimate(path, label_candidates=label_candidates)
        except Exception:
            portion_preds = []

        items: list[MealItemResult] = []
        for item in portion_preds:
            try:
                nutrition = self.nutrition_engine.nutrition_for(item.label, item.weight_g)
            except Exception:
                continue
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
            "meta": {
                "trusted_prediction": is_trusted,
                "confidence_threshold": self.confidence_threshold,
            },
        }
