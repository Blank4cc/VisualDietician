# VisualDietician (Course Project )

This repository is prepared for course project source-code submission.\
Its goal is to provide a deliverable baseline for food recognition and calorie estimation (Scheme A).

## 1. Project Overview

- Input a food image and output Top-3 class predictions with confidence scores
- Estimate portion size (grams) from detection boxes and density priors
- Compute calories/protein/fat/carbohydrates through USDA FoodData Central mapping
- Export batch evaluation outputs (CSV/JSON) and report figures (dashboard/trade-off)

## 2. Code Structure (Submission-Oriented)

```text
CV_course_project/
├─ main.ipynb                      # Main entry (training/inference/evaluation/visualization)
├─ src/food_cv/
│  ├─ config.py                    # Path configuration
│  ├─ data_pipeline.py             # Food-101 data loading and preprocessing
│  ├─ classifier.py                # ResNet-50 classifier
│  ├─ training.py                  # Two-stage training logic
│  ├─ portion_estimator.py         # YOLO portion estimation and fallback policy
│  ├─ nutrition_engine.py          # USDA mapping and nutrition computation
│  ├─ pipeline.py                  # Unified inference interface: predict_meal()
│  ├─ evaluation.py                # Metric computation and batch export
│  └─ visualization.py             # Report figure export
├─ outputs/
│  ├─ scheme_a_test_results.csv
│  ├─ scheme_a_test_results.json
│  └─ report_figures/
├─ requirements.txt                # Pinned dependencies
└─ dev_ledger.md                   # Engineering log
```

## 3. Why Food-101 Is Not Included

**The Food-101 dataset is large (\~5GB, \~101k images)**, so it is not included in this source-code submission package.

This is a standard submission practice: code, configuration, and representative outputs are committed, while large raw datasets are provided via official download links.

Food-101 official dataset page:\
<https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>

## 4. External Data Dependencies

- Food-101 (classification dataset): see the link above
- USDA FoodData Central (nutrition dataset download):\
  <https://fdc.nal.usda.gov/download-datasets/>

> Note: USDA archives may also be large and are commonly excluded from source-code submissions. Download locally and place them in your project data directory.

## 5. Quick Start

1. Install dependencies
   - `pip install -r requirements.txt`
2. Prepare datasets
   - Download Food-101 to a local data directory
   - Download USDA CSV archives (Foundation; optionally SR Legacy)
3. Run the project
   - Open and execute `main.ipynb`

## 6. Submission Notes

- Included in this repository: source code, documentation, sample evaluation outputs
- Not included: raw Food-101 dataset (excluded due to size), and report figures(included in the report pdf)
- For full reproducibility, download datasets from the links above before running the notebook

