from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Any

from .schemas import PortionEstimation


@dataclass(frozen=True)
class PortionConfig:
    coin_diameter_cm: float = 2.5
    default_weight_g: float = 180.0
    fallback_confidence: float = 0.4
    min_detection_confidence: float = 0.35
    max_items: int = 3
    max_weight_g_per_item: float = 800.0
    single_item_mode: bool = True
    min_bbox_area_cm2: float = 1.0
    max_bbox_area_cm2: float = 1500.0


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
        self.model: Any | None = None
        try:
            from ultralytics import YOLO  # Lazy import for better fault tolerance.

            self.model = YOLO(yolo_model_name)
        except Exception:
            self.model = None
        self.config = config or PortionConfig()
        self.density_map = density_map or DEFAULT_DENSITY_G_PER_CM2

    def _fallback(self, label: str) -> list[PortionEstimation]:
        return [
            PortionEstimation(
                label=label,
                weight_g=self.config.default_weight_g,
                confidence=self.config.fallback_confidence,
            )
        ]

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
        if self.config.min_bbox_area_cm2 <= 0:
            raise ValueError("min_bbox_area_cm2 必须大于 0")
        if self.config.max_bbox_area_cm2 <= self.config.min_bbox_area_cm2:
            raise ValueError("max_bbox_area_cm2 必须大于 min_bbox_area_cm2")

        primary_label = candidates[0]
        if self.model is None:
            return self._fallback(primary_label)

        try:
            results = self.model.predict(source=str(path), verbose=False)
        except Exception:
            return self._fallback(primary_label)
        if not results:
            return self._fallback(primary_label)

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return self._fallback(primary_label)

        ppc = pixel_per_cm if pixel_per_cm else 40.0
        estimations: list[PortionEstimation] = []
        max_items = 1 if self.config.single_item_mode else self.config.max_items
        for idx, xyxy in enumerate(boxes.xyxy.tolist()):
            confidence = float(boxes.conf[idx].item()) if boxes.conf is not None else self.config.fallback_confidence
            if confidence < self.config.min_detection_confidence:
                continue
            x1, y1, x2, y2 = xyxy
            bbox_w = max(x2 - x1, 1.0)
            bbox_h = max(y2 - y1, 1.0)
            area_cm2 = (bbox_w / ppc) * (bbox_h / ppc)
            if area_cm2 < self.config.min_bbox_area_cm2 or area_cm2 > self.config.max_bbox_area_cm2:
                continue
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
            if len(estimations) >= max_items:
                break
        if not estimations:
            return self._fallback(primary_label)
        return estimations
