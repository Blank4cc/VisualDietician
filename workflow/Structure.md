# Project Structure and Constraint Analysis (As Built)

---

## 1. Goal Decomposition

The current implementation is organized around four independent but complete deliverables:

| Deliverable | Current Output | Current Acceptance Signal |
|---|---|---|
| Food Classifier | Top-3 labels + confidence per image | Sampled Food-101 Top-1 around `0.78`, Top-3 around `0.91` |
| Portion Estimator | Estimated item weight (grams) | Stable fallback behavior under detector failure |
| Nutrition Engine | Calories/protein/fat/carbs from class + weight | Non-zero calorie rate and trusted-rate tracked in batch evaluation |
| Report Assets | CSV/JSON metrics + dashboard figures | Report figures exported in `outputs/report_figures/` |

Scheme A remains the active strategy: classify first, then map category to USDA nutrition values per 100g, then scale by estimated weight.

---

## 2. Real Constraints and Mitigations

| Constraint | Implementation Reality | Mitigation Already Implemented |
|---|---|---|
| 2D image cannot recover true 3D food volume | Portion estimation uses bbox area approximation | Area sanity bounds + capped weight + deterministic fallback |
| Low-confidence predictions can generate misleading nutrition totals | Classifier confidence varies across hard classes | Confidence gating in pipeline (`confidence_threshold=0.35`) |
| Food-101 names do not always match USDA descriptions | Long-tail label mismatch exists | Alias mapping + token overlap + fuzzy match + nutrition fallback profile |
| YOLO runtime can fail in some setups | Model loading/inference may be unavailable | Lazy YOLO initialization and exception-safe fallback path |

---

## 3. Implemented Module Graph

| Module | Responsibility | Input | Output | Upstream Dependency |
|---|---|---|---|---|
| `data_pipeline.py` | Food-101 dataloaders and transforms | Dataset root | Train/val/test loaders | None |
| `classifier.py` | ResNet-50 classification inference | Image path | Top-k predictions | Data transforms |
| `portion_estimator.py` | YOLO-based weight estimation with fallbacks | Image path + candidate label(s) | Portion estimation list | Class labels |
| `nutrition_engine.py` | USDA CSV lookup and nutrient scaling | Label + weight | Nutrition breakdown | USDA CSV files |
| `pipeline.py` | End-to-end fusion and trust policy | Image path | `top3_classification/items/total/meta` | Classifier + portion + nutrition |
| `evaluation.py` | Metrics and batch evaluation utilities | Predictor + test images | KPI dictionary + optional CSV/JSON | Pipeline |
| `visualization.py` | Report-oriented chart export | Scheme A CSV | Dashboard and trade-off figures | Evaluation output |

---

## 4. Current End-to-End Data Flow

```
image_path
   |
   +--> classify with ResNet-50 --> top3 labels + confidences
   |
   +--> if top1 confidence < threshold and block enabled:
   |       return zero-nutrition output with trust metadata
   |
   +--> estimate portion (YOLO or fallback default weight)
   |
   +--> map label to USDA nutrition per 100g
   |
   +--> scale nutrients by estimated grams
   |
   +--> return items + totals + trust metadata
```

---

## 5. Current Milestone State

- Environment and dependency pinning completed.
- Data pipeline module completed.
- Two-stage classifier training pipeline completed.
- Portion estimation module with fallback completed.
- Scheme A nutrition integration completed.
- Batch evaluation and report-figure export completed.
- Final CVPR-style report writing remains open.

---

## 6. Evidence Files in Repository

- Main notebook: `main.ipynb`
- Core package: `src/food_cv/`
- Batch outputs:
  - `outputs/scheme_a_test_results.csv`
  - `outputs/scheme_a_test_results.json`
- Report figures:
  - `outputs/report_figures/report_dashboard.png`
  - `outputs/report_figures/report_confidence_tradeoff.png`
- Engineering log:
  - `dev_ledger.md`

---

## 7. Recommended Final Validation Before Submission

1. Re-run Scheme A batch evaluation with fixed seed and stratified sampling.
2. Regenerate report figures from latest CSV.
3. Validate notebook output consistency after kernel restart.
4. Freeze final KPI table and copy into the CVPR report.

---
*This file is updated to reflect actual implementation status rather than initial planning assumptions.*

---

## 8. ASCII Architecture Diagram (As Built)

```text
                                +----------------------+
                                |      main.ipynb      |
                                |  (orchestration)     |
                                +----------+-----------+
                                           |
                                           v
                         +-----------------+-----------------+
                         | src/food_cv/config.py             |
                         | ProjectPaths (data/model paths)   |
                         +-----------------+-----------------+
                                           |
                                           v
                    +----------------------+----------------------+
                    | src/food_cv/data_pipeline.py              |
                    | Food101DataModule + transforms            |
                    +----------------------+----------------------+
                                           |
                         train/val/test loaders + class names
                                           |
                                           v
      +------------------------------------+------------------------------------+
      | src/food_cv/training.py                                                 |
      | train_classifier_two_stage (stage1 frozen -> stage2 partial unfreeze)  |
      +------------------------------------+------------------------------------+
                                           |
                                           v
                 +-------------------------+-------------------------+
                 | models/*.pt + class_names.json                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                   +-----------------------+-----------------------+
                   | src/food_cv/pipeline.py                       |
                   | MealPredictor.predict_meal(image_path)        |
                   +-----------------------+-----------------------+
                                           |
          +--------------------------------+---------------------------------+
          |                                |                                 |
          v                                v                                 v
+---------------------------+  +----------------------------+  +-------------------------------+
| classifier.py             |  | portion_estimator.py       |  | nutrition_engine.py           |
| ResNet-50 Top-3           |  | YOLOv8 / fallback weight   |  | USDA CSV mapping + fallback   |
| + confidence              |  | estimate from bbox area     |  | per-100g -> scaled nutrients  |
+-------------+-------------+  +--------------+-------------+  +---------------+---------------+
              |                               |                                |
              +---------------+---------------+--------------------------------+
                              |
                              v
                 +------------+-----------------------------------+
                 | Inference output dict                          |
                 | top3_classification / items / total / meta     |
                 +------------+-----------------------------------+
                              |
                              v
          +-------------------+-------------------+---------------------------+
          |                                       |                           |
          v                                       v                           v
+---------------------------+      +----------------------------+  +-----------------------------+
| evaluation.py             |      | visualization.py           |  | outputs/                    |
| batch metrics + CSV/JSON  | ---> | report figures export      |  | scheme_a_test_results.*     |
| stratified sampling       |      | dashboard + tradeoff plots |  | report_figures/*.png        |
+---------------------------+      +----------------------------+  +-----------------------------+
```
