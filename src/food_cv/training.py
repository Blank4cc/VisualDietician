from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Callable

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
    optimizer_foreach: bool | None = None
    optimizer: str = "adamw"  # adamw | adam
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = 1.0
    lr_schedule: str = "cosine"  # cosine | none
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.1
    min_effective_epochs: float = 1.0
    enforce_min_effective_epochs: bool = True


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
    grad_clip_norm: float | None = None,
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
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def _compute_epoch_lr(cfg: TrainConfig, epoch: int) -> float:
    """Compute epoch-level learning rate with optional warmup and cosine decay."""
    if epoch <= 0:
        raise ValueError("epoch 必须从 1 开始")
    if cfg.lr_schedule.lower() == "none":
        return cfg.lr

    warmup_epochs = max(int(cfg.warmup_epochs), 0)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return cfg.lr * (epoch / warmup_epochs)

    if cfg.epochs <= warmup_epochs:
        return cfg.lr

    progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = min(max(cfg.min_lr_ratio, 0.0), 1.0)
    return cfg.lr * (floor + (1.0 - floor) * cosine)


def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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


def evaluate_topk_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            topk = min(k, logits.shape[1])
            pred_topk = torch.topk(logits, k=topk, dim=1).indices
            match = pred_topk.eq(y.view(-1, 1))
            correct += int(match.any(dim=1).sum().item())
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
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if not (0.0 <= cfg.label_smoothing < 1.0):
        raise ValueError("label_smoothing 必须在 [0, 1) 之间")
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm 必须大于 0，或设置为 None")
    if cfg.lr_schedule.lower() not in {"cosine", "none"}:
        raise ValueError("lr_schedule 仅支持 'cosine' 或 'none'")
    if cfg.warmup_epochs < 0:
        raise ValueError("warmup_epochs 不能小于 0")
    if not (0.0 <= cfg.min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio 必须在 [0, 1] 之间")
    if cfg.min_effective_epochs <= 0:
        raise ValueError("min_effective_epochs 必须大于 0")

    device, backend = _resolve_device(cfg.device)
    if cfg.require_accelerator and backend == "cpu":
        raise RuntimeError("未检测到可用加速后端（CUDA/MPS/DirectML），训练被中止")
    _emit(f"[train] backend={backend} device={device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    foreach = cfg.optimizer_foreach
    if foreach is None:
        foreach = backend != "directml"
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("没有可训练参数，请检查模型冻结逻辑")

    full_steps_per_epoch = max(len(train_loader), 1)
    effective_steps_per_epoch = (
        min(full_steps_per_epoch, cfg.max_steps_per_epoch)
        if cfg.max_steps_per_epoch is not None and cfg.max_steps_per_epoch > 0
        else full_steps_per_epoch
    )
    total_effective_epochs = (effective_steps_per_epoch * cfg.epochs) / full_steps_per_epoch
    if total_effective_epochs < cfg.min_effective_epochs:
        msg = (
            f"训练覆盖率过低: effective_epochs={total_effective_epochs:.3f} "
            f"(epochs={cfg.epochs}, full_steps_per_epoch={full_steps_per_epoch}, "
            f"effective_steps_per_epoch={effective_steps_per_epoch})，"
            f"建议 >= {cfg.min_effective_epochs:.3f}"
        )
        if cfg.enforce_min_effective_epochs:
            raise RuntimeError(msg)
        _emit(f"[train][warn] {msg}")
    opt_name = cfg.optimizer.lower().strip()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            foreach=foreach,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            foreach=foreach,
        )
    else:
        raise ValueError("optimizer 仅支持 'adamw' 或 'adam'")
    best_acc = 0.0
    best_top5 = 0.0
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        current_lr = _compute_epoch_lr(cfg, epoch)
        _set_optimizer_lr(optimizer, current_lr)
        _emit(f"[train] epoch={epoch}/{cfg.epochs} start")
        _emit(f"[train] lr={current_lr:.7f}")
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
            grad_clip_norm=cfg.grad_clip_norm,
        )
        try:
            val_acc = evaluate_accuracy(model, val_loader, device)
        except RuntimeError as exc:
            if "version_counter" not in str(exc):
                raise
            _emit("[train] 验证阶段在当前后端失败，回退到 CPU 验证")
            model.to(torch.device("cpu"))
            val_acc = evaluate_accuracy(model, val_loader, torch.device("cpu"))
            val_top5 = evaluate_topk_accuracy(model, val_loader, torch.device("cpu"), k=5)
            model.to(device)
        else:
            val_top5 = evaluate_topk_accuracy(model, val_loader, device, k=5)
        _emit(
            f"[train] epoch={epoch}/{cfg.epochs} done "
            f"train_loss={train_loss:.4f} val_top1={val_acc:.4f} val_top5={val_top5:.4f} lr={current_lr:.7f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_top5 = val_top5
            best_loss = min(best_loss, train_loss)
            best_epoch = epoch
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_top1": val_acc,
                    "val_top5": val_top5,
                    "epoch": epoch,
                    "lr": current_lr,
                    "optimizer": cfg.optimizer,
                },
                path,
            )

    return {
        "best_val_top1": best_acc,
        "best_val_top5": best_top5,
        "best_train_loss": best_loss,
        "best_epoch": float(best_epoch),
        "backend": backend,
    }


def train_classifier_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage1_save_path: str | Path,
    stage1_config: TrainConfig,
    stage2_save_path: str | Path | None = None,
    stage2_config: TrainConfig | None = None,
    unfreeze_fn: Callable[[nn.Module], None] | None = None,
) -> dict[str, Any]:
    _emit("[train] two-stage training: stage1(frozen) start")
    stage1_metrics = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=stage1_save_path,
        config=stage1_config,
    )
    _emit("[train] two-stage training: stage1 done")

    if unfreeze_fn is not None:
        unfreeze_fn(model)
    _emit("[train] two-stage training: stage2(unfrozen) start")
    stage2_cfg = stage2_config or TrainConfig(
        epochs=1,
        lr=1e-5,
        device=stage1_config.device,
        log_every_n_steps=stage1_config.log_every_n_steps,
        max_steps_per_epoch=stage1_config.max_steps_per_epoch,
        use_amp=stage1_config.use_amp,
        require_accelerator=stage1_config.require_accelerator,
        optimizer_foreach=stage1_config.optimizer_foreach,
    )
    stage2_path = Path(stage2_save_path) if stage2_save_path else Path(stage1_save_path).with_name("food101_resnet50_stage2.pt")
    stage2_metrics = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=stage2_path,
        config=stage2_cfg,
    )
    _emit("[train] two-stage training: stage2 done")
    return {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "stage1_checkpoint": str(Path(stage1_save_path)),
        "stage2_checkpoint": str(stage2_path),
    }
