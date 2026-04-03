# 🍱 EE5811 Project 6: Food Recognition & Calorie Estimation
## AI-Readable Project Workflow

---

## PROJECT OVERVIEW

| Field | Detail |
|---|---|
| Course | EE5811 Topics in Computer Vision |
| Topic | Food Image Recognition + Calorie Estimation |
| Stack | Python · PyTorch · OpenCV · YOLO · Food-101 · pandas |
| Team Size | 2–3 persons recommended |
| Duration | ~8 weeks |

---

## MODULE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│             [Single food photo / meal photo]                        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Data & Preprocessing                                     │
│  • Resize → 224×224 (ResNet input norm)                             │
│  • Normalize (ImageNet mean/std)                                    │
│  • Augmentation: flip, crop, color jitter (training only)           │
│  • Reference object detection (coin/hand for scale)                 │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────────────────┐
│  MODULE 2:      │     │  MODULE 3:                  │
│  Food           │     │  Portion / Size Detection   │
│  Classification │     │                             │
│                 │     │  • YOLOv8 (fine-tuned)      │
│  • ResNet-50    │     │    detect food bounding box │
│    pretrained   │     │  • Reference object scale   │
│    fine-tuned   │     │    → pixel-to-cm mapping    │
│    on Food-101  │     │  • Estimate 2D area → mass  │
│                 │     │    (density lookup table)   │
│  OUTPUT:        │     │                             │
│  Top-3 food     │     │  OUTPUT:                    │
│  categories +   │     │  Estimated weight (grams)   │
│  confidence     │     │  per detected food item     │
└────────┬────────┘     └────────────┬────────────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 4: Nutrition Calculation Engine                             │
│                                                                     │
│  food_category + estimated_weight                                   │
│       │                                                             │
│       ├─→ USDA FoodData Central CSV (offline, 本地文件)               │
│       │   food.csv + food_nutrient.csv + nutrient.csv               │
│       │   启动时一次性 pd.read_csv() 加载到内存                         │
│       │   OR hardcoded fallback dict (30种常见食物)                   │
│       │                                                             │
│       └─→ calories = (cal_per_100g / 100) × weight_g                │
│           + protein_g, fat_g, carb_g                                │
│                                                                     │
│  OUTPUT: Nutritional breakdown dict per food item                   │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 5: Presentation / Report Layer                              │
│                                                                     │
│  • Annotated image: bounding boxes + label + calorie overlay        │
│  • Pie chart: calorie breakdown by food item                        │
│  • Macro table: protein / fat / carb / total calories              │
│  • Jupyter Notebook demo (course submission requirement)            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## EXECUTION PHASES

### PHASE 0 — Environment Setup (Day 1–2)
**Can be done in parallel by all members.**

- [ ] Setup Google Colab environment + GPU runtime
- [ ] Install: `torch torchvision ultralytics pandas Pillow`
- [ ] Download USDA CSV from fdc.nal.usda.gov/download-datasets
      → Foundation Foods (~29MB) + SR Legacy (~54MB) 解压到 data/usda/
- [ ] Download Food-101 dataset (~5GB, ~101k images)
- [ ] Register USDA FoodData Central API key (free)
- [ ] Split team roles: Member A → Classification, Member B → Detection, Member C → Integration

---

### PHASE 1 — Data Pipeline (Day 3–5)
**Dependency: Phase 0 complete**

```
[Food-101 Dataset]
      │
      ├─ train/val/test split (75/15/10)
      ├─ DataLoader with augmentation
      └─ Baseline: random guess accuracy = 1/101 ≈ 0.99%
         Target: fine-tuned accuracy ≥ 70%
```

- [ ] Build PyTorch Dataset class for Food-101
- [ ] Implement preprocessing pipeline
- [ ] Verify data loading with sample visualization

---

### PHASE 2A — Food Classification (Day 6–12) [PARALLEL with 2B]
**Dependency: Phase 1**
**Owner: Member A**

```
ResNet-50 (pretrained ImageNet)
  │
  ├─ Freeze backbone layers (first run)
  ├─ Replace final FC layer: 2048 → 101 classes
  ├─ Train with:
  │    optimizer = Adam(lr=1e-4)
  │    loss = CrossEntropyLoss
  │    epochs = 10–20
  │    batch_size = 32
  └─ Fine-tune (unfreeze last 2 blocks, lr=1e-5)

Evaluation:
  Top-1 Accuracy | Top-5 Accuracy | Confusion Matrix
```

- [ ] Implement transfer learning script
- [ ] Run training on Colab GPU (~2–4 hours)
- [ ] Save best model checkpoint
- [ ] Evaluate on test set, visualize top errors

---

### PHASE 2B — Portion Detection (Day 6–12) [PARALLEL with 2A]
**Dependency: Phase 1**
**Owner: Member B**

```
YOLOv8n (nano, fast)
  │
  ├─ Use pretrained COCO weights (no fine-tuning needed for demo)
  ├─ Detect food regions → bounding box (x, y, w, h)
  ├─ Reference object: standard coin (2.5cm diameter)
  │    pixel_per_cm = coin_pixel_width / 2.5
  ├─ Estimate food area = (bbox_w / ppc) × (bbox_h / ppc) cm²
  └─ Lookup density table → weight estimate (grams)

KNOWN LIMITATION: 2D area ≠ 3D volume → add ±30% error disclaimer
```

- [ ] Integrate YOLOv8 inference pipeline
- [ ] Build density lookup table (top 20 Food-101 categories)
- [ ] Validate against known portion sizes
- [ ] Handle "no reference object" fallback (use default portion size)

---

### PHASE 3 — Integration (Day 13–16)
**Dependency: Phase 2A AND Phase 2B both complete**
**Owner: Member C (with A & B review)**

```
Input Image
    │
    ├──[Module 2]──→ food_label (str), confidence (float)
    ├──[Module 3]──→ weight_estimate (float, grams)
    │
    └──[Module 4]──→ query nutrition DB
                     → pd.DataFrame 本地查询 (food_nutrient.csv)
                     → fuzzy match food_name → fdc_id → nutrients
                     → {calories, protein, fat, carbs}
                     → render annotated output image
```

- [ ] Write `predict_meal(image_path)` unified function
- [ ] Connect classification output → nutrition lookup
- [ ] Connect detection output → weight multiplier
- [ ] Handle multi-item meals (loop over YOLO detections)

---

### PHASE 4 — Evaluation & Report (Day 17–21)
**Dependency: Phase 3**

| Evaluation Dimension | Method |
|---|---|
| Classification accuracy | Top-1 / Top-5 on Food-101 test set |
| Portion estimation error | Compare vs. manual weighing (5–10 test images) |
| End-to-end calorie error | Compare vs. nutrition label ground truth |
| Speed | Inference time per image (ms) |

- [ ] Run full evaluation suite
- [ ] Write 4-page CVPR-format report
- [ ] Prepare 3-min TED-style presentation demo
- [ ] Submit: `.ipynb` notebook + report PDF

---

## DEPENDENCY GRAPH

```
Phase 0 (Setup)
    │
    └──→ Phase 1 (Data Pipeline)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
Phase 2A              Phase 2B
(Classification)      (Detection)
    │                    │
    └─────────┬──────────┘
              ▼
         Phase 3 (Integration)
              │
              ▼
         Phase 4 (Eval + Report)
```

**Parallelizable:** Phase 2A ‖ Phase 2B
**Hard Dependencies:** 0→1→{2A,2B}→3→4

---

## POTENTIAL BLOCKERS & MITIGATIONS

| # | Blocker | Mitigation |
|---|---|---|
| 1 | **Portion estimation without depth** — 2D image can't give true 3D volume | Use standard portion size as fallback; add confidence interval in output |
| 2 | **Food-101 fine-tuning overfitting** — small compute budget | Use transfer learning (freeze backbone), data augmentation, early stopping |
| 3 | **Mixed-dish ambiguity** — "fried rice" vs "egg fried rice" look identical | Report Top-3 predictions; let user confirm via simple UI toggle |
| 4 | **Food-101 类名与 USDA 描述不匹配** — 如 "edamame" vs "Edamame, frozen, prepared" | 
    预处理时构建一个 Food-101→USDA 关键词映射表（约 30 条手写规则即可覆盖大部分） |

---

## DELIVERABLES CHECKLIST

- [ ] `food_classifier.ipynb` — training + evaluation notebook
- [ ] `portion_detector.ipynb` — YOLO detection demo
- [ ] `calorie_estimator.ipynb` — end-to-end unified demo
- [ ] `report.pdf` — 4-page CVPR format
- [ ] `slides.pptx` — 3-min TED presentation

---
*Generated for EE5811 Project Planning | CityU HK*
