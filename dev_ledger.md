# Dev Ledger

## Snapshot
- Date: 2026-04-03
- Project: VisualDietician
- Objective: Food-101 classification + portion estimation + USDA nutrition fusion
- Current Main Entry: `main.ipynb`

## Environment
- OS: Windows
- Python Env: `VisualDietician` (conda)
- Core Packages:
  - torch 2.3.1
  - torchvision 0.18.1
  - ultralytics 8.3.0
  - pandas 2.2.3
  - Pillow 10.4.0
  - opencv-python 4.10.0.84
  - matplotlib 3.9.2
- Accelerator Strategy:
  - Windows: DirectML (`torch-directml`)
  - Other: CUDA/MPS auto fallback

## Completed
- Built modular codebase under `src/food_cv`:
  - `data_pipeline.py`
  - `classifier.py`
  - `portion_estimator.py`
  - `nutrition_engine.py`
  - `pipeline.py`
  - `training.py`
- Added progress logs in training:
  - step progress
  - average loss
  - speed
  - ETA
- Added validation metrics:
  - Top-1 and Top-5 accuracy in epoch logs and checkpoint metadata
- Added two-stage fine-tuning pipeline:
  - Stage 1: frozen backbone training
  - Stage 2: unfreeze last blocks and fine-tune
- Added accelerator guard:
  - `require_accelerator=True` fails fast if backend falls back to CPU
- Added validation fallback path:
  - if DirectML validation triggers `version_counter` runtime error, automatically validate on CPU and resume
- Replaced eval context from `inference_mode` to `no_grad` for backend compatibility
- Enhanced nutrition retrieval:
  - merged USDA Foundation + SR Legacy lookup
  - added token-overlap candidate search
  - expanded Food-101 alias mapping and fallback nutrition profiles
- Added quick evaluation utilities:
  - sampled test-set Top-1/Top-5/avg_loss reporting after training
  - end-to-end nutrition nonzero-calorie hit-rate reporting
  - optional JSON-driven end-to-end MAE/MAPE evaluation for calories and total weight
  - auto template generator for `eval_samples.json` to bootstrap ground-truth annotation
- Reduced DirectML optimizer CPU-fallback noise/risk by using `Adam(..., foreach=False)` in training config
- Added root `.gitignore` and dataset/model artifact exclusions

## In Progress
- Running stable smoke-training configuration:
  - batch_size=16
  - num_workers=0
  - stage1 epochs=1 / max_steps=60
  - stage2 epochs=1 / max_steps=60
- Backend target for local Windows AMD setup:
  - `backend=directml`

## Issues Seen
- Early stage had `ModuleNotFoundError: torch` due missing dependencies.
- Training appeared “stuck” because original loop had no per-step telemetry.
- In some runs notebook still used legacy `device="cuda"` setting, causing CPU fallback on non-CUDA systems.
- Encountered `AttributeError: torch.library.register_fake` from torchvision, indicating torch/torchvision version mismatch in active env.
- Importing `food_cv.training` triggered package-level eager import chain (`__init__ -> pipeline -> classifier`) too early.
- Validation on DirectML raised `RuntimeError: Cannot set version_counter for inference tensor` in batch_norm.

## Next Actions
- Verify current run prints `backend=directml`.
- Verify startup prints `[env] torch=2.3.1 torchvision=0.18.1`.
- Verify validation stage no longer crashes; fallback message should appear only when DirectML hits known backend bug.
- Expand two-stage fine-tuning:
  - Stage 1: freeze backbone
  - Stage 2: unfreeze last blocks
- Add Food-101 label to USDA mapping table coverage expansion (long tail classes).
- Add end-to-end evaluation metrics:
  - prepare `eval_samples.json` ground-truth file for stable MAE/MAPE tracking
  - verify two-stage training output contains both `stage1` and `stage2` metrics in notebook

## Decision Log
- Keep Food-101 and USDA datasets out of git tracking.
- Keep notebook as orchestrator; core logic remains in importable modules.
- Enforce accelerator-required mode during local training to prevent accidental CPU runs.
- Make package init lazy for heavy objects to avoid eager dependency import failures.
- Force-check torch/torchvision compatibility at notebook startup; auto-reinstall mismatched stack and require kernel restart.
