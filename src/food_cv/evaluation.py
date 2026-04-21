from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .nutrition_engine import NutritionConfig, USDANutritionEngine


@torch.no_grad()
def evaluate_classification_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches 必须大于 0")
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * int(y.numel())

        top1 = torch.argmax(logits, dim=1)
        top1_correct += int((top1 == y).sum().item())

        k = min(5, logits.shape[1])
        top5 = torch.topk(logits, k=k, dim=1).indices
        top5_correct += int(top5.eq(y.view(-1, 1)).any(dim=1).sum().item())
        total += int(y.numel())

        if max_batches is not None and batch_idx >= max_batches:
            break

    if total == 0:
        return {"top1": 0.0, "top5": 0.0, "avg_loss": 0.0}

    return {
        "top1": top1_correct / total,
        "top5": top5_correct / total,
        "avg_loss": total_loss / total,
    }


def evaluate_nutrition_hit_rate(
    predictor: Any,
    image_paths: list[str | Path],
) -> dict[str, float]:
    if not image_paths:
        return {"image_count": 0.0, "item_count": 0.0, "nonzero_calorie_rate": 0.0}

    image_count = 0
    item_count = 0
    nonzero_count = 0
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        result = predictor.predict_meal(path)
        items = result.get("items", [])
        for item in items:
            item_count += 1
            if float(item.get("calories", 0.0)) > 0.0:
                nonzero_count += 1
        image_count += 1

    if item_count == 0:
        return {"image_count": float(image_count), "item_count": 0.0, "nonzero_calorie_rate": 0.0}

    return {
        "image_count": float(image_count),
        "item_count": float(item_count),
        "nonzero_calorie_rate": nonzero_count / item_count,
    }


def evaluate_end_to_end_from_json(
    predictor: Any,
    sample_json_path: str | Path,
) -> dict[str, float]:
    path = Path(sample_json_path)
    if not path.exists():
        raise FileNotFoundError(f"评估样本文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("评估样本 JSON 顶层必须是 list")

    image_count = 0
    usable_count = 0
    calorie_abs_error_sum = 0.0
    calorie_abs_pct_error_sum = 0.0
    weight_abs_error_sum = 0.0
    weight_abs_pct_error_sum = 0.0

    for row in payload:
        if not isinstance(row, dict):
            continue
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            continue
        gt_total_calories = float(row.get("gt_total_calories", 0.0))
        gt_total_weight = float(row.get("gt_total_weight_g", 0.0))
        pred = predictor.predict_meal(image_path)
        pred_total_calories = float(pred.get("total", {}).get("calories", 0.0))
        pred_total_weight = float(sum(float(item.get("weight_g", 0.0)) for item in pred.get("items", [])))

        calorie_abs_error = abs(pred_total_calories - gt_total_calories)
        weight_abs_error = abs(pred_total_weight - gt_total_weight)
        calorie_abs_error_sum += calorie_abs_error
        weight_abs_error_sum += weight_abs_error

        if gt_total_calories > 0:
            calorie_abs_pct_error_sum += calorie_abs_error / gt_total_calories
        if gt_total_weight > 0:
            weight_abs_pct_error_sum += weight_abs_error / gt_total_weight

        usable_count += 1
        image_count += 1

    if usable_count == 0:
        return {
            "sample_count": 0.0,
            "calorie_mae": 0.0,
            "calorie_mape": 0.0,
            "weight_mae_g": 0.0,
            "weight_mape": 0.0,
        }

    return {
        "sample_count": float(usable_count),
        "calorie_mae": calorie_abs_error_sum / usable_count,
        "calorie_mape": calorie_abs_pct_error_sum / usable_count,
        "weight_mae_g": weight_abs_error_sum / usable_count,
        "weight_mape": weight_abs_pct_error_sum / usable_count,
    }


def create_eval_template(
    image_paths: list[str | Path],
    output_json_path: str | Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        rows.append(
            {
                "image_path": str(path),
                "gt_total_calories": 0.0,
                "gt_total_weight_g": 0.0,
                "notes": "fill ground-truth values",
            }
        )
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_path


def auto_fill_eval_samples_from_usda(
    sample_json_path: str | Path,
    foundation_dir: str | Path,
    sr_legacy_dir: str | Path | None = None,
    default_weight_g: float = 180.0,
    overwrite_existing_gt: bool = True,
) -> dict[str, float]:
    if default_weight_g <= 0:
        raise ValueError("default_weight_g 必须大于 0")
    sample_path = Path(sample_json_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"评估样本文件不存在: {sample_path}")

    with sample_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("评估样本 JSON 顶层必须是 list")

    engine = USDANutritionEngine(
        NutritionConfig(
            foundation_dir=Path(foundation_dir),
            sr_legacy_dir=Path(sr_legacy_dir) if sr_legacy_dir else None,
        )
    )

    updated_count = 0
    skipped_count = 0
    unresolved_count = 0

    for row in rows:
        if not isinstance(row, dict):
            skipped_count += 1
            continue
        current_cal = float(row.get("gt_total_calories", 0.0))
        current_w = float(row.get("gt_total_weight_g", 0.0))
        if not overwrite_existing_gt and current_cal > 0 and current_w > 0:
            skipped_count += 1
            continue

        image_path = Path(str(row.get("image_path", "")))
        label = image_path.parent.name if image_path.parent.name else ""
        if not label:
            skipped_count += 1
            continue

        nutrition = engine.nutrition_for(label, default_weight_g)
        if nutrition.calories <= 0:
            unresolved_count += 1
            continue

        row["gt_total_calories"] = round(float(nutrition.calories), 3)
        row["gt_total_weight_g"] = round(float(default_weight_g), 3)
        note = str(row.get("notes", "")).strip()
        auto_note = "auto-filled from USDA CSV by class label"
        row["notes"] = f"{note} | {auto_note}" if note else auto_note
        updated_count += 1

    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return {
        "total_rows": float(len(rows)),
        "updated_rows": float(updated_count),
        "skipped_rows": float(skipped_count),
        "unresolved_rows": float(unresolved_count),
    }


def get_food101_test_images(
    food101_root: str | Path,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    stratified: bool = False,
    per_class_limit: int | None = None,
) -> list[Path]:
    root = Path(food101_root)
    meta_test = root / "meta" / "test.txt"
    images_dir = root / "images"
    if not meta_test.exists():
        raise FileNotFoundError(f"找不到 Food-101 测试清单: {meta_test}")
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 Food-101 图片目录: {images_dir}")

    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")
    if per_class_limit is not None and per_class_limit <= 0:
        raise ValueError("per_class_limit 必须大于 0")

    rng = random.Random(seed)
    paths: list[Path] = []
    by_class: dict[str, list[Path]] = {}
    with meta_test.open("r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if not key:
                continue
            p = images_dir / f"{key}.jpg"
            if p.exists():
                if stratified:
                    cls = p.parent.name
                    by_class.setdefault(cls, []).append(p)
                else:
                    paths.append(p)
                    if limit is not None and len(paths) >= limit and not shuffle:
                        break

    if not stratified:
        if shuffle:
            rng.shuffle(paths)
        if limit is not None:
            paths = paths[:limit]
        return paths

    classes = sorted(by_class.keys())
    if not classes:
        return []
    for cls in classes:
        if shuffle:
            rng.shuffle(by_class[cls])

    # Stratified sampling: pull samples in round-robin order to keep class balance.
    max_take_per_class = per_class_limit
    if max_take_per_class is None and limit is not None:
        max_take_per_class = max(1, (limit + len(classes) - 1) // len(classes))
    if max_take_per_class is None:
        max_take_per_class = max(len(by_class[c]) for c in classes)

    selected: list[Path] = []
    for round_idx in range(max_take_per_class):
        for cls in classes:
            cls_items = by_class[cls]
            if round_idx < len(cls_items):
                selected.append(cls_items[round_idx])
                if limit is not None and len(selected) >= limit:
                    return selected
    return selected


def run_scheme_a_batch_test(
    predictor: Any,
    image_paths: list[str | Path],
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    confidence_threshold: float = 0.35,
) -> dict[str, float]:
    if not image_paths:
        return {
            "image_count": 0.0,
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "nonzero_total_calorie_rate": 0.0,
            "nonzero_item_calorie_rate": 0.0,
            "avg_total_calories": 0.0,
            "low_confidence_rate": 0.0,
            "trusted_nonzero_total_calorie_rate": 0.0,
            "trusted_avg_total_calories": 0.0,
        }
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold 必须在 [0, 1] 区间内")

    rows: list[dict[str, Any]] = []
    image_count = 0
    top1_correct = 0
    top3_correct = 0
    nonzero_total_calorie = 0
    trusted_nonzero_total_calorie = 0
    item_count = 0
    nonzero_item_calorie = 0
    trusted_image_count = 0
    low_confidence_count = 0
    total_calories_sum = 0.0
    trusted_total_calories_sum = 0.0

    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        gt_label = path.parent.name
        result = predictor.predict_meal(path)
        preds = result.get("top3_classification", [])
        pred_labels = [str(x.get("label", "")) for x in preds]
        top1_confidence = float(preds[0].get("confidence", 0.0)) if preds else 0.0
        top1 = pred_labels[0] if pred_labels else ""
        top1_hit = float(top1 == gt_label)
        top3_hit = float(gt_label in pred_labels)
        top1_correct += int(top1_hit)
        top3_correct += int(top3_hit)
        is_trusted = top1_confidence >= confidence_threshold
        if is_trusted:
            trusted_image_count += 1
        else:
            low_confidence_count += 1

        total = result.get("total", {})
        total_calories = float(total.get("calories", 0.0))
        if total_calories > 0:
            nonzero_total_calorie += 1
            if is_trusted:
                trusted_nonzero_total_calorie += 1
        total_calories_sum += total_calories
        if is_trusted:
            trusted_total_calories_sum += total_calories

        items = result.get("items", [])
        for item in items:
            item_count += 1
            if float(item.get("calories", 0.0)) > 0.0:
                nonzero_item_calorie += 1

        rows.append(
            {
                "image_path": str(path),
                "gt_label": gt_label,
                "pred_top1": top1,
                "pred_top3": ";".join(pred_labels),
                "pred_top1_confidence": top1_confidence,
                "top1_hit": int(top1_hit),
                "top3_hit": int(top3_hit),
                "total_calories": total_calories,
                "total_protein_g": float(total.get("protein_g", 0.0)),
                "total_fat_g": float(total.get("fat_g", 0.0)),
                "total_carbs_g": float(total.get("carbs_g", 0.0)),
            }
        )
        image_count += 1

    if image_count == 0:
        return {
            "image_count": 0.0,
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "nonzero_total_calorie_rate": 0.0,
            "nonzero_item_calorie_rate": 0.0,
            "avg_total_calories": 0.0,
            "low_confidence_rate": 0.0,
            "trusted_nonzero_total_calorie_rate": 0.0,
            "trusted_avg_total_calories": 0.0,
        }

    if output_json_path:
        out_json = Path(output_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    if output_csv_path:
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_path",
                    "gt_label",
                    "pred_top1",
                    "pred_top3",
                    "pred_top1_confidence",
                    "top1_hit",
                    "top3_hit",
                    "total_calories",
                    "total_protein_g",
                    "total_fat_g",
                    "total_carbs_g",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return {
        "image_count": float(image_count),
        "top1_acc": top1_correct / image_count,
        "top3_acc": top3_correct / image_count,
        "nonzero_total_calorie_rate": nonzero_total_calorie / image_count,
        "nonzero_item_calorie_rate": (nonzero_item_calorie / item_count) if item_count > 0 else 0.0,
        "avg_total_calories": total_calories_sum / image_count,
        "low_confidence_rate": low_confidence_count / image_count,
        "trusted_nonzero_total_calorie_rate": (
            trusted_nonzero_total_calorie / trusted_image_count if trusted_image_count > 0 else 0.0
        ),
        "trusted_avg_total_calories": (trusted_total_calories_sum / trusted_image_count if trusted_image_count > 0 else 0.0),
    }
