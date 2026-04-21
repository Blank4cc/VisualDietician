# EE5811 Project 6: Food Recognition and Calorie Estimation
## AI-Readable As-Built Workflow (Updated to Current Implementation)

---

## PROJECT OVERVIEW

| Field | Detail |
|---|---|
| Course | EE5811 Topics in Computer Vision |
| Project Codename | VisualDietician |
| Objective | Food-101 classification + portion estimation + USDA nutrition fusion |
| Main Entry | `main.ipynb` |
| Core Source | `src/food_cv/` |
| Strategy | Scheme A (classification-first + static USDA mapping) |
| Status | Deliverable-grade baseline completed |

---

## CURRENT TECH STACK (IMPLEMENTED)

- Python, PyTorch, torchvision
- Ultralytics YOLOv8 (pretrained model inference)
- pandas, Pillow, OpenCV, matplotlib
- Food-101 dataset
- USDA FoodData Central CSV (Foundation + optional SR Legacy)
- Accelerator fallback: DirectML (Windows), CUDA/MPS, CPU fallback

Pinned package versions are tracked in `requirements.txt`.

---

## MODULE ARCHITECTURE (AS BUILT)

```
Input Image
    │
    ├── Module A: Food Classification (`classifier.py`)
    │     - ResNet-50 transfer learning (ImageNet pretrained)
    │     - Top-3 predictions with confidence
    │
    ├── Module B: Portion Estimation (`portion_estimator.py`)
    │     - YOLOv8n detection (lazy-loaded; fault-tolerant)
    │     - 2D bbox area -> estimated weight via density lookup
    │     - Default fallback weight when detection is unavailable
    │
    ├── Module C: Nutrition Engine (`nutrition_engine.py`)
    │     - Food-101 label -> USDA lookup (alias + token overlap + fuzzy)
    │     - Per-100g nutrient extraction and weight scaling
    │     - Nutrition fallback profile for unresolved labels
    │
    └── Module D: Pipeline Fusion (`pipeline.py`)
          - `MealPredictor.predict_meal(image_path)`
          - Confidence gating (`confidence_threshold=0.35`)
          - Optional nutrition block on low-confidence predictions
          - Structured output: `top3_classification`, `items`, `total`, `meta`
```

---

## SUPPORTING MODULES

| Module | Implemented Functions |
|---|---|
| `data_pipeline.py` | Food-101 dataloaders, train/eval transforms, train/val split |
| `training.py` | One-stage and two-stage training, effective-epoch guard, backend fallback |
| `evaluation.py` | Classification metrics, stratified Food-101 sampling, Scheme A batch test, JSON-based E2E evaluation |
| `visualization.py` | Report dashboard, confidence trade-off plot, optional class detail plot |

---

## EXECUTION FLOW (CURRENT NOTEBOOK FLOW)

1. Build paths and datasets via `ProjectPaths` and `Food101DataModule`.
2. Train classifier in two stages (frozen backbone -> unfreeze last blocks) or reuse existing checkpoint.
3. Run test evaluation (Top-1/Top-5/loss).
4. Build `MealPredictor` with checkpoint + USDA CSV.
5. Run Scheme A batch test on stratified Food-101 samples.
6. Export report figures from batch CSV.

---

## IMPLEMENTED TRAINING PROFILE

- Stage 1: 3 epochs, LR `1e-4`
- Stage 2: 5 epochs, LR `3e-5`
- Guardrail: minimum effective epoch coverage check enabled by default
- Checkpoints:
  - `models/food101_resnet50_stage1.pt`
  - `models/food101_resnet50_stage2.pt`
  - `models/class_names.json`

---

## LATEST CONFIRMED KPI SNAPSHOT

From current project records (`dev_ledger.md` and `outputs/scheme_a_test_results.*`):

- Classification (sampled):
  - Top-1: `0.7802`
  - Top-3: `0.9069`
  - Top-5 (quick eval): `0.9469`
- Nutrition robustness:
  - Non-zero total calorie rate: `0.8337`
  - Trusted non-zero total calorie rate: `0.8367`
- Reliability:
  - Low-confidence rate (`conf < 0.35`): `0.2238`

---

## OUTPUT ARTIFACTS (CURRENT REPO)

- `outputs/scheme_a_test_results.csv`
- `outputs/scheme_a_test_results.json`
- `outputs/report_figures/report_dashboard.png`
- `outputs/report_figures/report_confidence_tradeoff.png`

---

## KNOWN LIMITATIONS (CURRENT)

| # | Limitation | Current Mitigation |
|---|---|---|
| 1 | 2D area does not capture true 3D volume for portion estimation | Default weight fallback + bounded area filters + disclaimer in reporting |
| 2 | Label ambiguity for visually similar classes | Top-3 output + confidence-aware trust policy |
| 3 | Food-101 to USDA mismatch for long-tail classes | Alias table + token/fuzzy matching + fallback nutrition profiles |
| 4 | Detection/runtime instability in some environments | YOLO lazy initialization + exception-safe fallback path |

---

## DELIVERABLE CHECKLIST (AS BUILT)

- [x] Unified notebook entry (`main.ipynb`)
- [x] Modular source package (`src/food_cv`)
- [x] Scheme A batch evaluation export (CSV + JSON)
- [x] Report-ready figures for 4-page paper
- [ ] Final CVPR-style report PDF
- [ ] Final presentation slides

---
*Updated to match repository implementation state.*
