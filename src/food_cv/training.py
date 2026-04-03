from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def _emit(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-4
    device: str = "auto"
    log_every_n_steps: int = 50
    max_steps_per_epoch: int | None = None
    use_amp: bool = True
    require_accelerator: bool = False


def _resolve_device(device: str) -> tuple[Any, str]:
    preferred = device.lower().strip()
    if preferred in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if preferred in {"auto", "mps"} and mps_backend and mps_backend.is_available():
        return torch.device("mps"), "mps"
    if preferred in {"auto", "directml", "dml"}:
        try:
            import torch_directml

            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: Any,
    backend: str,
    log_every_n_steps: int = 50,
    max_steps_per_epoch: int | None = None,
    use_amp: bool = True,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    total_steps = len(loader)
    if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
        total_steps = min(total_steps, max_steps_per_epoch)
    _emit(f"[train] epoch_loop start total_steps={total_steps}")
    use_cuda_amp = backend == "cuda" and use_amp
    if use_cuda_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    start_ts = perf_counter()
    for step, (x, y) in enumerate(loader, start=1):
        if step == 1:
            _emit("[train] first_batch_loaded")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp:
            try:
                autocast_ctx = torch.amp.autocast("cuda")
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast()
            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        batch_size = x.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        if log_every_n_steps > 0 and (step % log_every_n_steps == 0 or step == total_steps):
            avg_loss = running_loss / max(total, 1)
            elapsed = perf_counter() - start_ts
            step_per_sec = step / max(elapsed, 1e-6)
            remaining_steps = max(total_steps - step, 0)
            eta_sec = remaining_steps / max(step_per_sec, 1e-6)
            _emit(
                f"[train] step={step}/{total_steps} avg_loss={avg_loss:.4f} "
                f"speed={step_per_sec:.2f}step/s eta={eta_sec/60:.1f}min"
            )
        if max_steps_per_epoch is not None and max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
            break
    return running_loss / max(total, 1)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
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
) -> dict[str, float | str]:
    cfg = config or TrainConfig()
    if cfg.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if cfg.lr <= 0:
        raise ValueError("lr 必须大于 0")
    if cfg.log_every_n_steps < 0:
        raise ValueError("log_every_n_steps 不能小于 0")

    device, backend = _resolve_device(cfg.device)
    if cfg.require_accelerator and backend == "cpu":
        raise RuntimeError("未检测到可用加速后端（CUDA/MPS/DirectML），训练被中止")
    _emit(f"[train] backend={backend} device={device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        _emit(f"[train] epoch={epoch}/{cfg.epochs} start")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            backend=backend,
            log_every_n_steps=cfg.log_every_n_steps,
            max_steps_per_epoch=cfg.max_steps_per_epoch,
            use_amp=cfg.use_amp,
        )
        try:
            val_acc = evaluate_accuracy(model, val_loader, device)
        except RuntimeError as exc:
            if "version_counter" not in str(exc):
                raise
            _emit("[train] 验证阶段在当前后端失败，回退到 CPU 验证")
            model.to(torch.device("cpu"))
            val_acc = evaluate_accuracy(model, val_loader, torch.device("cpu"))
            model.to(device)
        _emit(f"[train] epoch={epoch}/{cfg.epochs} done train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_loss = min(best_loss, train_loss)
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, path)

    return {"best_val_acc": best_acc, "best_train_loss": best_loss, "backend": backend}
