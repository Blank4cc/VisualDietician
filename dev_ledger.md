# Dev Ledger

## Snapshot
- Date: 2026-04-03
- Project: VisualDietician
- Objective: Food-101 classification + portion estimation + USDA nutrition fusion
- Current Main Entry: `main.ipynb`
- Chosen Strategy: `Scheme A` (class-level static USDA mapping)

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
  - USDA CSV-based auto-fill for `eval_samples.json` GT calories/weight
- Reduced DirectML optimizer CPU-fallback noise/risk by using `Adam(..., foreach=False)` in training config
- Added root `.gitignore` and dataset/model artifact exclusions
- Main notebook aligned to Scheme A default path:
  - no mandatory dependency on `eval_samples.json` for normal training/inference run
  - `eval_samples.json` retained as optional extended E2E evaluation asset
- Data split hygiene improved:
  - validation set is now split from Food-101 train set (deterministic seed)
  - test set remains independent Food-101 test split

## In Progress
- Running stable smoke-training configuration:
  - batch_size=16
  - num_workers=0
  - stage1 epochs=1 / max_steps=60
  - stage2 epochs=1 / max_steps=60
  - val_ratio=0.1 (from train split)
- Backend target for local Windows AMD setup:
  - `backend=directml`

## Latest Run (2026-04-03)
- Pipeline version: `solo-stage2-hardfix-v4`
- Device backend: `directml`
- Two-stage training finished successfully:
  - Stage 1: top1=0.0145, top5=0.0766, loss=4.6231
  - Stage 2: top1=0.0253, top5=0.1020, loss=4.5869
- Checkpoints generated:
  - `models/food101_resnet50_stage1.pt`
  - `models/food101_resnet50_stage2.pt`
- Quick sampled test metrics:
  - top1=0.0930
  - top5=0.2250
  - avg_loss=4.4772
- Demo end-to-end output:
  - nutrition nonzero-calorie hit-rate = 1.0
  - total calories/protein/fat/carbs are all non-zero
- Current Scheme A reference metrics:
  - stage2 validation: top1≈0.024, top5≈0.105
  - sampled test: top1≈0.034, top5≈0.162
  - optional E2E MAE/MAPE is for trend tracking only unless GT is measured.
- Evaluation dataset bootstrap:
  - `eval_samples.json` expanded to 10 samples using `food-101/images/*` paths for immediate batch testing.
  - initial prior-based GT values populated for calories and total weight to enable non-zero MAPE baseline.

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
- Align acceptance to Scheme A:
  - primary: classification (Top-1/Top-5) + nutrition nonzero hit-rate
  - optional: E2E MAE/MAPE trend (only authoritative when GT is measured)

## Next Sprint Plan
- P0: improve Scheme A production quality.
  - Acceptance: sampled top1 > 0.15, top5 > 0.35, nonzero_calorie_rate >= 0.9.
- P0(optional): replace auto-filled GT in `eval_samples.json` with measured/label-based real GT and rerun MAE/MAPE.
  - Acceptance: `sample_count >= 10`, all GT entries marked as measured.
- P1: improve classification quality with longer stage2 training.
  - Plan: stage1(1 epoch/60 steps) + stage2(3-5 epochs/full val).
  - Acceptance: sampled top1 > 0.15, top5 > 0.35.
- P1: extend Food-101 -> USDA mapping for long-tail classes.
  - Acceptance: `nonzero_calorie_rate >= 0.9` on 20-image test set.
- P2: add portion/calorie error report export.
  - Acceptance: one JSON summary + one notebook table for report insertion.

## Decision Log
- Keep Food-101 and USDA datasets out of git tracking.
- Keep notebook as orchestrator; core logic remains in importable modules.
- Enforce accelerator-required mode during local training to prevent accidental CPU runs.
- Make package init lazy for heavy objects to avoid eager dependency import failures.
- Force-check torch/torchvision compatibility at notebook startup; auto-reinstall mismatched stack and require kernel restart.
- Adopt Scheme A as primary project route: classify first, then static Food-101→USDA mapping with per-100g nutrition and estimated weight scaling.
