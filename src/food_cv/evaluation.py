from __future__ import annotations

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
