# Dev Ledger

## Snapshot
- Date: 2026-04-20
- Project: VisualDietician
- Objective: Food-101 classification + portion estimation + USDA nutrition fusion
- Main Entry: `main.ipynb`
- Active Strategy: `Scheme A` (classification-first + USDA mapping)
- Current Status: **Major recovery completed**, pipeline promoted from "unstable baseline" to "deliverable-grade baseline"

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
- Accelerator:
  - Windows: DirectML (`torch-directml`)
  - Others: CUDA/MPS auto fallback

## Architecture Notes
- End-to-end flow:
  - `FoodClassifier` outputs top-k class logits/probabilities
  - `PortionEstimator` estimates weight from detections (now fault-tolerant)
  - `USDANutritionEngine` maps class -> nutrition per 100g -> scaled totals
  - `MealPredictor` fuses all outputs and returns `top3/items/total/meta`
- Design hardening (this update):
  - confidence gating for nutrition output
  - stratified test sampling for fair batch metrics
  - training coverage guard to block pseudo-training runs
  - robust fallback when YOLO/USDA calls fail

## Completed (2026-04-20 Hardening)
- `evaluation.py`
  - Added robust test sampling in `get_food101_test_images(...)`:
    - `shuffle`, `seed`, `stratified`, `per_class_limit`
  - Added confidence-aware metrics in `run_scheme_a_batch_test(...)`:
    - `pred_top1_confidence` (CSV/JSON output)
    - `low_confidence_rate`
    - `trusted_nonzero_total_calorie_rate`
    - `trusted_avg_total_calories`
- `pipeline.py`
  - `MealPredictor` now auto-loads labels from `models/class_names.json` when labels are not passed explicitly
  - Added `confidence_threshold` + `block_nutrition_when_low_confidence`
  - Added structured `meta` output with trust flags
  - Added defensive exception handling around portion and nutrition stages
- `portion_estimator.py`
  - YOLO lazy import + initialization fallback
  - Prediction exception fallback to deterministic default item
  - Added `single_item_mode=True` default
  - Added bbox area sanity filters (`min_bbox_area_cm2`, `max_bbox_area_cm2`)
- `training.py`
  - Added anti-pseudo-training guard:
    - `min_effective_epochs`
    - `enforce_min_effective_epochs` (default `True`)
  - Runtime now warns/raises if effective training coverage is too low
- `main.ipynb`
  - Deliverable profile changed to realistic budget:
    - stage1: 3 epochs, uncapped steps
    - stage2: 5 epochs, uncapped steps
  - Batch test switched to stratified sampling:
    - `limit=505`, `stratified=True`, `per_class_limit=5`, `seed=42`
  - Batch eval now passes confidence threshold to stats function
- `visualization.py`
  - Added report-oriented plotting module (`src/food_cv/visualization.py`)
  - Added merged dashboard figure export:
    - KPI + confidence distribution + calorie distribution + hardest classes in one figure
  - Added confidence-threshold trade-off figure:
    - retained coverage vs retained top1 accuracy
  - Added optional class-detail figure for appendix/defense backup
  - Added one-shot export API: `export_report_figures(...)`

## Latest Run (2026-04-20)
- Pipeline version: `schemeA-deliverable-fast-v2`
- Data split:
  - train=68175
  - val=7575
  - test=25250
  - labels=101
- Training profile:
  - stage1=(epochs=3, steps=None)
  - stage2=(epochs=5, steps=None)
- Backend: `directml`
- Quick sampled test metrics:
  - top1=0.78671875
  - top5=0.946875
  - avg_loss=0.7416134685743601
- Scheme A batch test (`outputs/scheme_a_test_results.*`):
  - image_count=505.0
  - top1_acc=0.7801980198019802
  - top3_acc=0.906930693069307
  - nonzero_total_calorie_rate=0.8336633663366336
  - nonzero_item_calorie_rate=0.7194719471947195
  - avg_total_calories=243.11550143681617
  - low_confidence_rate=0.22376237623762377
  - trusted_nonzero_total_calorie_rate=0.8367346938775511
  - trusted_avg_total_calories=242.04746869828222
- Artifacts confirmed:
  - `models/food101_resnet50_stage1.pt`
  - `models/food101_resnet50_stage2.pt`
  - `models/class_names.json`
  - `outputs/scheme_a_test_results.json`
  - `outputs/scheme_a_test_results.csv`

## Problem Analysis (Closed)
- Root cause 1 (critical): training coverage was previously far below 1 effective epoch, causing near-random classifier behavior.
- Root cause 2 (critical): batch evaluation sampling was non-stratified and heavily class-biased.
- Root cause 3 (high): nutrition output was not confidence-gated, so low-quality classification could still produce plausible nutrition totals.
- Root cause 4 (high): portion estimation error propagation under YOLO instability.
- Resolution state: **closed with code-level fixes and validated metrics**.

## Open Risks
- About 22.4% predictions are still below confidence threshold (hard classes remain).
- Some USDA mappings still return zero for long-tail labels; nutritional completeness not yet perfect.
- Demo outputs can appear inconsistent if notebook kernel keeps stale predictor instances (must rebuild predictor after reload).

## Next Actions (Priority)
- P0
  - Add class-level error analysis report (per-class top1/top3, confusion hotspots).
  - Export low-confidence and wrong-prediction image lists for targeted retraining.
- P1
  - Improve hard classes with targeted augmentation and extended stage2 epochs.
  - Expand Food-101 -> USDA alias/fallback table for long-tail classes.
- P2
  - Add regression test script for:
    - stratified sampling behavior
    - confidence gating behavior
    - training effective-epoch guard

## Acceptance Baseline (Updated)
- Classification:
  - sampled top1 >= 0.75 (achieved: 0.7802)
  - sampled top3 >= 0.90 (achieved: 0.9069)
- Nutrition robustness:
  - nonzero_total_calorie_rate >= 0.80 (achieved: 0.8337)
  - trusted metrics must be reported (achieved)
- Process robustness:
  - training coverage guard enabled by default (achieved)
  - stratified evaluation path used by default notebook flow (achieved)

## Decision Log
- Keep notebook as orchestration layer; core logic stays in importable modules under `src/food_cv`.
- Keep dataset/model artifacts out of git.
- Enforce accelerator-required mode in local training runs.
- Use confidence-aware nutrition output as default safety policy.
- Use stratified sampling for batch test KPIs to prevent class-order bias.
