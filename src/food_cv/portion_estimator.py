from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from .schemas import PortionEstimation


@dataclass(frozen=True)
class PortionConfig:
    coin_diameter_cm: float = 2.5
    default_weight_g: float = 180.0
    fallback_confidence: float = 0.4
    min_detection_confidence: float = 0.35
    max_items: int = 3
    max_weight_g_per_item: float = 800.0


DEFAULT_DENSITY_G_PER_CM2: dict[str, float] = {
    "pizza": 2.8,
    "burger": 1.8,
    "salad": 0.9,
    "rice": 1.5,
    "fried_rice": 1.6,
    "spaghetti": 1.3,
    "steak": 2.2,
    "sushi": 1.7,
    "ramen": 1.1,
    "cake": 1.4,
}


class PortionEstimator:
    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        config: PortionConfig | None = None,
        density_map: dict[str, float] | None = None,
    ) -> None:
        self.model = YOLO(yolo_model_name)
        self.config = config or PortionConfig()
        self.density_map = density_map or DEFAULT_DENSITY_G_PER_CM2

    def estimate(
        self,
        image_path: str | Path,
        label_candidates: Iterable[str],
        pixel_per_cm: float | None = None,
    ) -> list[PortionEstimation]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        candidates = [x for x in label_candidates if x]
        if not candidates:
            candidates = ["unknown_food"]
        if pixel_per_cm is not None and pixel_per_cm <= 0:
            raise ValueError("pixel_per_cm 必须大于 0")

        results = self.model.predict(source=str(path), verbose=False)
        if not results:
            return [
                PortionEstimation(
                    label=candidates[0],
                    weight_g=self.config.default_weight_g,
                    confidence=self.config.fallback_confidence,
                )
            ]

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return [
                PortionEstimation(
                    label=candidates[0],
                    weight_g=self.config.default_weight_g,
                    confidence=self.config.fallback_confidence,
                )
            ]

        ppc = pixel_per_cm if pixel_per_cm else 40.0
        estimations: list[PortionEstimation] = []
        for idx, xyxy in enumerate(boxes.xyxy.tolist()):
            confidence = float(boxes.conf[idx].item()) if boxes.conf is not None else self.config.fallback_confidence
            if confidence < self.config.min_detection_confidence:
                continue
            x1, y1, x2, y2 = xyxy
            bbox_w = max(x2 - x1, 1.0)
            bbox_h = max(y2 - y1, 1.0)
            area_cm2 = (bbox_w / ppc) * (bbox_h / ppc)
            label = candidates[min(idx, len(candidates) - 1)]
            density = self.density_map.get(label, 1.2)
            estimated_weight = max(area_cm2 * density, 1.0)
            weight_g = min(estimated_weight, self.config.max_weight_g_per_item)
            estimations.append(
                PortionEstimation(
                    label=label,
                    weight_g=weight_g,
                    confidence=confidence,
                )
            )
            if len(estimations) >= self.config.max_items:
                break
        if not estimations:
            return [
                PortionEstimation(
                    label=candidates[0],
                    weight_g=self.config.default_weight_g,
                    confidence=self.config.fallback_confidence,
                )
            ]
        return estimations
