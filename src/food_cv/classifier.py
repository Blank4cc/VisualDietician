from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from .data_pipeline import build_eval_transform
from .schemas import ClassificationPrediction


DEFAULT_FOOD101_LABELS: list[str] = [f"class_{idx}" for idx in range(101)]


def build_resnet50_classifier(num_classes: int = 101, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def unfreeze_last_two_blocks(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("layer3.") or name.startswith("layer4.") or name.startswith("fc."):
            param.requires_grad = True


class FoodClassifier:
    def __init__(
        self,
        model: nn.Module,
        labels: Sequence[str] | None = None,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.labels = list(labels) if labels else DEFAULT_FOOD101_LABELS
        self.transform = build_eval_transform(image_size)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"模型文件不存在: {ckpt}")
        state = torch.load(ckpt, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            raise ValueError("不支持的 checkpoint 格式")
        self.model.eval()

    @torch.inference_mode()
    def predict_topk(self, image_path: str | Path, topk: int = 3) -> list[ClassificationPrediction]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        if topk <= 0:
            raise ValueError("topk 必须大于 0")
        image = Image.open(path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        topk = min(topk, probs.shape[1])
        values, indices = torch.topk(probs, k=topk, dim=1)
        preds: list[ClassificationPrediction] = []
        for score, idx in zip(values[0].tolist(), indices[0].tolist()):
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            preds.append(ClassificationPrediction(label=label, confidence=float(score)))
        return preds
