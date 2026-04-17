from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


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
