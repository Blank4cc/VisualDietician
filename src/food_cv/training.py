from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-4
    device: str = "cuda"


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
    return running_loss / max(total, 1)


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str | Path,
    config: TrainConfig | None = None,
) -> dict[str, float]:
    cfg = config or TrainConfig()
    if cfg.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if cfg.lr <= 0:
        raise ValueError("lr 必须大于 0")

    device = _resolve_device(cfg.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    best_acc = 0.0
    best_loss = float("inf")

    for _ in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_loss = min(best_loss, train_loss)
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, path)

    return {"best_val_acc": best_acc, "best_train_loss": best_loss}
