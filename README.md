# Trustworthy Ordinal Grading of Diabetic Retinopathy Under Label Noise and Domain Shift

**Patient-Level Leakage Audits · Ordinal Classification · Uncertainty Quantification · Conformal Risk Control**

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-blue)](https://www.kaggle.com/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-6f42c1)](https://en.wikipedia.org/wiki/Computer_vision)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-orange)](#scope-and-experimental-status)

**Repository:** <https://github.com/hkoul2000/trustworthy-diabetic-retinopathy>

This project investigates how to evaluate diabetic retinopathy (DR) grading models more responsibly when retinal images are correlated within patients, labels may be noisy, predictions may be miscalibrated, and deployment data may differ from the training distribution. The notebook combines patient-level splitting and leakage audits with ordinal classification, uncertainty analysis, conformal-style control of missed referable cases, and exploratory label-noise experiments.

> **Research status and interpretation:** This repository currently contains the Jupyter notebook. The results below are taken from saved notebook outputs; they are not a claim of clinical validation. External-domain evaluation was not completed in the saved run because the APTOS dataset was not attached. The project is a research prototype, not a medical device or diagnostic system.

---

## Table of contents

- [1. Project overview](#1-project-overview)
- [2. Problem statement](#2-problem-statement)
- [3. Research questions](#3-research-questions)
- [4. Dataset](#4-dataset)
- [5. Repository contents](#5-repository-contents)
- [6. Methodology and pipeline](#6-methodology-and-pipeline)
- [7. Patient-level leakage audit](#7-patient-level-leakage-audit)
- [8. Image quality, caching, and near-duplicate audit](#8-image-quality-caching-and-near-duplicate-audit)
- [9. Ordinal classification and backbones](#9-ordinal-classification-and-backbones)
- [10. Uncertainty and calibration](#10-uncertainty-and-calibration)
- [11. Conformal risk control for referable DR](#11-conformal-risk-control-for-referable-dr)
- [12. Label-noise experiments](#12-label-noise-experiments)
- [13. Domain shift and external evaluation](#13-domain-shift-and-external-evaluation)
- [14. Verified results](#14-verified-results)
- [15. Scope and experimental status](#15-scope-and-experimental-status)
- [16. Running the notebook on Kaggle](#16-running-the-notebook-on-kaggle)
- [17. Configuration and reproducibility](#17-configuration-and-reproducibility)
- [18. Outputs and model export](#18-outputs-and-model-export)
- [19. Limitations and threats to validity](#19-limitations-and-threats-to-validity)
- [20. Responsible use and ethics](#20-responsible-use-and-ethics)
- [21. Future work](#21-future-work)
- [22. References and links](#22-references-and-links)
- [23. Author](#23-author)

---

## 1. Project overview

Diabetic retinopathy grading is an ordinal image-classification task: the five grades have a meaningful severity order, and errors between adjacent grades are not equivalent to errors across the full scale. A trustworthy system also needs to communicate uncertainty and be evaluated on data that are independent at the patient level.

This notebook explores the following components:

1. **Patient-level data splitting:** extract patient and eye identifiers from image filenames and keep each patient's eyes within a single partition.
2. **Leakage audit:** compare patient-level and image-level splits, then measure the proportion of image-level test records whose patient also appears in training.
3. **Near-duplicate and label-conflict audit:** use a 256-bit difference hash, thumbnail comparison, and duplicate clustering to flag possible duplicates and conflicting labels.
4. **Ordinal model comparison:** compare cross-entropy, scalar regression, and CORN ordinal heads across selected image backbones.
5. **Uncertainty and calibration:** compare temperature scaling, deep ensembles, MC-dropout, and an evidential/Dirichlet method.
6. **Conformal-style risk control:** construct contiguous ordinal prediction sets and evaluate missed referable cases against a target risk level.
7. **Label-noise investigation:** generate out-of-fold predictions, use Cleanlab-based issue detection, and compare dropping, relabeling, and soft-label variants.
8. **Interpretability and export:** generate qualitative Grad-CAM overlays and provide code to export the selected checkpoint and an inference helper when the required saved run is available.

The notebook is designed to run in Kaggle with GPU acceleration and persistent output folders. Some phases depend on additional datasets or prior output artifacts and are not demonstrated as completed in the saved notebook.

## 2. Problem statement

A high image-level score can be misleading if highly related images from the same patient appear in both training and testing. Fundus images from a patient's left and right eyes are not independent observations. In addition, the dataset is imbalanced, labels may be imperfect, confidence can be miscalibrated, and a model calibrated on one dataset may not preserve its behavior after a domain shift.

This project studies the evaluation protocol as much as the model itself. It asks whether patient-aware evaluation, ordinal prediction, uncertainty estimation, and risk-aware prediction sets provide a more informative picture of model reliability than accuracy alone.

### Intended research scope

- Five-class grading of retinal fundus images.
- Patient-grouped model evaluation and a deliberately image-level leakage comparison.
- Risk analysis for **referable DR**, defined in this notebook as grades **2–4** (`Moderate`, `Severe`, `Proliferate_DR`).
- Empirical evaluation of calibration, uncertainty, referral rate, and prediction-set size.
- Exploratory label-noise and domain-shift studies.

The experiments are research analyses and do not establish clinical effectiveness or deployment readiness.

## 3. Research questions

| ID | Research question | Notebook approach |
|---|---|---|
| RQ1 | How does an image-level split differ from a patient-level split? | Compare split strategies across three seeds and measure patient overlap. |
| RQ2 | Can label-issue detection and noise-aware training improve grading? | Five-fold out-of-fold probabilities, Cleanlab-based issue detection, and drop/relabel/soft-label variants. |
| RQ3 | How do backbone, ordinal head, preprocessing, and uncertainty method affect evaluation? | Successive-halving model selection, preprocessing ablation, and uncertainty comparisons. |
| RQ4 | Can contiguous ordinal prediction sets reduce missed referable cases at a chosen risk level? | Calibrate a set-mass threshold using referable calibration examples and evaluate empirical false-negative rate. |
| RQ5 | Does source-domain calibration remain reliable under domain shift? | Planned APTOS 2019 evaluation and repeated target recalibration experiments; not completed in the saved run because APTOS was unavailable. |

These are the notebook's research questions, not a claim that each question has been conclusively answered.

## 4. Dataset

### Primary dataset: Diabetic Retinopathy 2015 Data Colored Resized

- **Kaggle dataset title:** Diabetic Retinopathy 2015 Data Colored Resized
- **Kaggle author/uploader:** Sovit Ranjan Rath
- **Dataset page:** <https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized>
- **Underlying source:** EyePACS / the 2015 Diabetic Retinopathy Detection challenge
- **Released image resolution:** 224 × 224 pixels
- **Task:** Five-grade diabetic retinopathy classification
- **Notebook's observed initial inventory:** 35,126 images and 17,563 parsed patients, with two eyes per patient.

The Kaggle release contains resized images organized into five severity folders. This is a **224 × 224 resized release**, not the original full-resolution image collection. Upscaling these images would not restore lost retinal detail, so the notebook keeps both its source cache and model input at 224 × 224.

### Severity classes

| Label | Folder/class name | Meaning in the dataset |
|---:|---|---|
| 0 | `No_DR` | No diabetic retinopathy |
| 1 | `Mild` | Mild diabetic retinopathy |
| 2 | `Moderate` | Moderate diabetic retinopathy |
| 3 | `Severe` | Severe diabetic retinopathy |
| 4 | `Proliferate_DR` | Proliferative diabetic retinopathy |

### Initial class distribution recorded by the notebook

These are the notebook's **pre-deduplication** counts, before the near-duplicate filtering step.

| Class | Images | Share |
|---|---:|---:|
| `No_DR` | 25,810 | 73.48% |
| `Mild` | 2,443 | 6.95% |
| `Moderate` | 5,292 | 15.07% |
| `Severe` | 873 | 2.49% |
| `Proliferate_DR` | 708 | 2.02% |
| **Total** | **35,126** | **100%** |

The distribution is highly imbalanced: `No_DR` is the majority class, while severe and proliferative cases are much less common. Overall accuracy can therefore conceal poor recall on clinically important minority classes.

### Accessing the dataset on Kaggle

1. Open the [dataset page](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized).
2. Sign in to Kaggle and accept any dataset terms or access requirements shown there.
3. Create or open the project notebook.
4. Use **Add Input** and attach the dataset.
5. Confirm that the attached input contains the five expected class folders, including `No_DR`.

The notebook searches `/kaggle/input` for a directory containing `No_DR`. If automatic discovery fails, set `CFG["DATA_ROOT"]` to the correct dataset directory.

### Optional external datasets

| Dataset | Intended role | Status in the saved run |
|---|---|---|
| [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | Minimum viable external-domain evaluation | **Not run:** notebook output states that APTOS was not found. |
| Messidor-2 | Optional external evaluation | Optional path/configuration exists; no completed result is shown. |
| IDRiD | Optional external evaluation and possible lesion-mask localization analysis | Optional path/configuration exists; no completed external evaluation is shown. Lesion masks were not configured. |

Do not interpret these optional datasets as part of the primary training set. Any future external evaluation should document label mapping, preprocessing, patient/eye grouping where available, licensing, and whether any target-domain labels were used for recalibration.

**Dataset attribution:** Cite the Kaggle dataset page and the original 2015 challenge. The Kaggle dataset page lists its own license information; review the current terms on the source page before redistributing any data. The raw dataset is not included in this GitHub repository.

## 5. Repository contents

The project folder currently contains the notebook and this README. The notebook generates additional artifacts under its working directory when executed; those generated artifacts are not currently part of the repository unless added separately.

```text
trustworthy-diabetic-retinopathy/
├── README.md
└── diabetic-retinopathy2015-data-coloured - 1.ipynb
```

The notebook's generated Kaggle working structure includes folders such as:

```text
/kaggle/working/
├── cache/       # uint8 image cache, hashes, thumbnails, optional CLAHE cache
├── ckpt/        # model checkpoints
├── runs/        # run metadata and predictions
├── results/     # CSV tables, figures, and evaluation outputs
└── final_model_export/  # created by the export cell if a selected run is available
```

These are **runtime output locations**, not folders currently committed to GitHub. In particular, model weights, cached images, and result CSVs are not included in the current repository.

## 6. Methodology and pipeline

The notebook is organized as a staged experimental pipeline:

1. **Environment setup and configuration** — import dependencies, detect CPU/GPU, set seeds and experiment flags, and prepare output folders.
2. **Dataset inventory** — discover class folders, enumerate images, and parse patient/eye IDs from filenames.
3. **Patient and class audit** — calculate class counts, left/right agreement, and images per patient.
4. **Caching and near-duplicate audit** — cache RGB images as `uint8`, compute difference hashes and thumbnails, cluster candidate duplicates, and remove duplicate records from the retained dataset.
5. **Split construction** — create patient-grouped train/validation/test splits and a separate image-level split for the leakage experiment.
6. **Backbone/head search** — run a first-stage model comparison, retain a smaller set of candidates, and train a second-stage comparison; test preprocessing variants.
7. **Uncertainty and conformal experiments** — compare prediction uncertainty methods and evaluate ordinal prediction sets for referable risk.
8. **Label-noise experiments** — create patient-grouped out-of-fold predictions and compare baseline and noise-aware training variants.
9. **External evaluation** — attempt APTOS discovery and run external comparisons only when a valid external dataset is attached.
10. **Qualitative explainability and final evaluation** — generate Grad-CAM overlays and save separate OOF and held-out-test confusion matrices and classification reports.
11. **Optional export** — serialize the selected model and metadata if a selected run/checkpoint is available.

## 7. Patient-level leakage audit

### Patient and eye parsing

The notebook extracts patient and eye identifiers from filenames using a pattern equivalent to:

```text
10_left  -> patient 10, eye left
10_right -> patient 10, eye right
```

Both eyes share the same patient prefix. The notebook's saved audit found:

- 35,126 initial images.
- 17,563 parsed patients.
- 17,563 patients with both eyes represented.
- Quadratic weighted kappa between left/right eye grades: **0.855**.
- Left and right eyes had identical grades for **87.2%** of patient pairs.
- Every parsed patient in the initial inventory had two images.

This high within-patient similarity motivates grouping both eyes into the same split.

### Patient-level split

After duplicate filtering, the notebook constructs a stratified split based on each patient's **worst grade**:

- 70% training
- 10% validation
- 20% held-out test

The split is seeded and keeps patient groups separate. In the saved run, the retained dataset contained 32,874 images after the duplicate-removal step; the split sizes were 23,014 training images, 3,303 validation images, and 6,557 test images.

### Image-level split as a diagnostic comparison

The notebook also creates an image-level split, used only to study leakage. It does not respect patient grouping.

The saved audit reported:

- **67.3%** of image-level test images had the same patient represented in training.
- **0%** patient overlap was reported for the patient-level split.

Importantly, the saved three-seed QWK comparison did **not** show an increase in mean QWK for the image-level split: mean QWK was 0.6426 for the image-level split and 0.6476 for the patient-level split. The reported image-minus-patient difference was −0.0050, with a seed-0 cluster-bootstrap 95% interval of [−0.0347, +0.0411].

Therefore, this run demonstrates substantial patient overlap under image-level splitting, but **does not establish that image-level splitting inflated QWK in this particular experiment**. The confidence interval is compatible with differences in either direction. This is an important negative/uncertain result and should not be rewritten as proof of metric inflation.

## 8. Image quality, caching, and near-duplicate audit

### Image caching

The notebook:

- Reads images with OpenCV and converts BGR to RGB.
- Keeps images at the dataset's 224 × 224 resolution.
- Stores the raw image cache as `uint8` arrays.
- Creates 256-bit difference hashes and 32 × 32 thumbnails for near-duplicate checks.
- Can create a CLAHE-preprocessed cache as an ablation.

The notebook output reports that the raw cache required approximately **5.29 GB**. This is a runtime storage requirement and should be considered before reproducing the full run.

### Near-duplicate procedure

The implemented audit uses a 256-bit difference hash, candidate buckets, Hamming distance (threshold 6), and thumbnail mean absolute difference as a verification step. The saved output reports:

- **2,252** images dropped as near-duplicates.
- **167** duplicate clusters with conflicting labels.

Conflicting labels in a near-duplicate cluster are evidence of possible label inconsistency, but do not automatically prove which label is correct. The audit is heuristic: hash thresholds and candidate-bucket limits can produce false positives and false negatives. Any publication should include a sensitivity check for the duplicate thresholds and, ideally, manual review of a sample of flagged pairs.

## 9. Ordinal classification and backbones

### Ordinal heads implemented in the notebook

| Head | Output form | Description |
|---|---|---|
| Cross-entropy (`ce`) | Five class logits | Standard five-class baseline; treats output classes categorically. |
| Scalar regression (`reg`) | One scalar | Predicts a severity score; the implementation maps it to class probabilities using a distance-based distribution. |
| CORN (`corn`) | Four ordinal logits | Ordinal threshold formulation for ordered grades. |
| Evidential (`evid`) | Five evidence outputs | Uses a Dirichlet-style probability representation for the uncertainty experiments. |

The saved successive-halving grid explicitly compares `ce`, `reg`, and `corn`. The evidential head is used in the uncertainty/conformal phase. The presence of a head in the code does not imply that it is the best choice for every metric.

### Backbones

| Backbone | Notebook identifier | Status |
|---|---|---|
| EfficientNet-B0 | `efficientnet_b0` | Trained in the saved run; used for leakage and OOF/noise baselines. |
| EfficientNet-B3 | `efficientnet_b3` | Trained in the saved run. |
| ConvNeXt-Tiny | `convnext_tiny` | Trained in the saved run; the validation-QWK-selected configuration was ConvNeXt-Tiny + regression head + raw preprocessing. |
| DINOv2-Small | `vit_small_patch14_dinov2.lvd142m` | Attempted in the first grid; saved validation QWK values were near zero in this run. This should be reported as a failed/unsuccessful configuration, not as a competitive result. |
| RETFound | Optional checkpoint | Only enabled if a compatible checkpoint is supplied through `CFG["RETFOUND_CKPT"]`; no completed RETFound result is shown. |

### Preprocessing and augmentation

- RGB conversion and 224 × 224 input.
- ImageNet normalization (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`) in the export metadata.
- Raw versus CLAHE preprocessing comparison.
- GPU-side random rotation, horizontal flip, scale, brightness, and contrast augmentation.
- A configurable sampling power to partially compensate for class imbalance.

The selected final configuration in the saved run used **raw preprocessing**, not CLAHE.

## 10. Uncertainty and calibration

The notebook implements or configures the following uncertainty approaches:

- **Temperature scaling:** post-hoc calibration of predicted probabilities.
- **Deep ensemble:** combination of multiple trained model members; default ensemble count is five.
- **MC-dropout:** repeated stochastic forward passes; configured for 20 passes in the full run. The notebook applies dropout to the prediction head only, so this is not full-network Bayesian inference.
- **Evidential/Dirichlet method:** models class evidence and is compared using predictive and risk-related metrics.

The notebook calculates metrics including:

- **ECE** — expected calibration error.
- **NLL** — negative log-likelihood.
- **Brier score** — probability quality under a squared-error scoring rule.
- **AURC** — area under the risk–coverage curve.
- **QWK** — quadratic weighted kappa for ordinal agreement.
- **AUROC for referable DR** — discrimination between grades 0–1 and grades 2–4.
- Sensitivity at selected specificity operating points.

These metrics measure different properties. AUROC is not accuracy, and good discrimination does not imply good calibration. ECE also depends on binning and sample size, so it should not be interpreted as a complete calibration guarantee.

## 11. Conformal risk control for referable DR

### Target outcome

The notebook defines a referable case as a ground-truth grade of **2 or higher**:

\[
Y \geq 2 \quad \Longleftrightarrow \quad
\{\text{Moderate}, \text{Severe}, \text{Proliferate\_DR}\}.
\]

For each image, the method creates a contiguous set of ordinal grades. It starts at the modal grade and expands toward neighboring grades until the set reaches a probability-mass threshold \(q\).

The risk event is a missed referable case: the predicted set contains no grade \(\geq 2\) even though the true grade is \(\geq 2\). Calibration uses referable examples to estimate a threshold, and the notebook evaluates false-negative rate (FNR), referral rate, and average set size.

### Interpretation and assumptions

The intended risk target is conditional on referable cases and relies on calibration/deployment exchangeability assumptions. Dataset shift can invalidate those assumptions. A finite-sample calibration procedure should not be described as a universal clinical guarantee, and empirical test-set results are not proof of deployment-time risk control.

The notebook reports a 200-split validity experiment in code and a conformal table for the saved run. The displayed conformal table identifies **Evidential (Dirichlet)** as the main method by validation NLL. At \(\alpha = 0.05\), the saved table reports:

| Method/configuration | Empirical FNR | Referral rate | Average set size | Within target in saved evaluation |
|---|---:|---:|---:|---|
| Evidential/Dirichlet, \(\alpha=0.05\) | 0.0414 | 0.6713 | 2.7182 | Yes |

The associated displayed interval for the empirical FNR was [0.0297, 0.0547] over 1,279 referable test examples. This is an empirical result from this saved run, not evidence that the method will meet the target under arbitrary domain shift. The selected model for the final QWK evaluation is a different configuration (ConvNeXt-Tiny + regression), so these conformal results should not be attributed to that selected model without a corresponding run.

## 12. Label-noise experiments

The notebook produces five-fold patient-grouped out-of-fold (OOF) probabilities, uses Cleanlab-based issue detection, and compares baseline training with variants that drop flagged examples, relabel flagged examples, or use soft labels.

The saved noise-comparison table reports the following test metrics against the original (potentially noisy) labels:

| Variant | Training examples | QWK | Referable AUROC | Sensitivity at 90% specificity | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 23,014 | 0.6499 | 0.8585 | 0.6654 | 0.0420 | 0.7278 |
| Drop flagged | 19,148 | 0.6038 | 0.8588 | 0.6544 | 0.1556 | 1.2603 |
| Relabel | 23,014 | 0.6110 | 0.8508 | 0.6294 | 0.1182 | 0.9504 |
| Soft labels | 23,014 | 0.5874 | 0.8700 | 0.6747 | 0.0240 | 0.7459 |

In this saved comparison, the noise-aware variants did not improve QWK over the baseline on the original held-out labels. Soft labels had the best ECE among these four variants but lower QWK; this is a trade-off, not a general win. The notebook itself notes that an external, cleaner evaluation set would be a fairer test of whether noise-aware training helps. The planned APTOS comparison was not available in the saved run.

## 13. Domain shift and external evaluation

The notebook includes code for loading APTOS 2019 and optional Messidor-2/IDRiD data, preprocessing external fundus images, evaluating model probabilities, and repeating target-domain recalibration experiments with approximately 200 target examples.

**External validation status: not completed in the saved notebook.** The saved output explicitly states: `APTOS not found. Add the aptos2019-blindness-detection dataset as input.` Therefore, no APTOS, Messidor-2, or IDRiD performance numbers should be reported from this run.

When external evaluation is completed, the report should distinguish:

1. Source model evaluated directly on external images without recalibration.
2. Target-only calibration using the designated target calibration subset.
3. Pooled source-plus-target calibration, if used.
4. Performance on the remaining target test subset.
5. Repeated random calibration/test partitions and uncertainty intervals.

Target recalibration uses target labels, so it is a form of adaptation and must be clearly distinguished from zero-shot external validation. The notebook also notes that APTOS preprocessing differs from the EyePACS resized release, so observed differences may reflect both population/domain differences and preprocessing differences.

## 14. Verified results

The tables below reproduce values printed in the saved notebook. They should be interpreted in the context of this notebook's split design, labels, and training setup. The dataset has substantial class imbalance and likely label noise.

### 14.1 Final selected model on the held-out patient-level test set

The notebook selected the configuration using validation QWK. The saved selected configuration was **ConvNeXt-Tiny + regression head + raw preprocessing** (`g2_convnext_t_reg_raw`).

| Metric | Held-out test result |
|---|---:|
| Number of test images | 6,557 |
| Accuracy | 0.7540 |
| Macro-F1 | 0.4487 |
| Quadratic weighted kappa (QWK) | 0.7407 |
| AUROC for referable DR | 0.9035 |
| Sensitivity at 90% specificity | 0.7701 |
| Sensitivity at 95% specificity | 0.7037 |
| Expected calibration error (ECE) | 0.1779 |
| Negative log-likelihood (NLL) | 0.8668 |
| Brier score | 0.4632 |
| AURC | 0.0169 |
| QWK at 80% coverage | 0.6022 |

The model has useful ordinal agreement and referable-disease discrimination in this split, but its ECE is high enough to warrant caution about probability reliability. Minority-class recall is uneven. These results are not clinical validation and should not be summarized as “90% accuracy”; the 0.9035 figure is **AUROC**, not accuracy.

### 14.2 Five-fold patient-grouped out-of-fold evaluation

The notebook also reports an independent OOF summary for **EfficientNet-B0 + cross-entropy**. It is a separate evaluation from the selected ConvNeXt-Tiny held-out test result above.

| Metric | Five-fold OOF result |
|---|---:|
| Number of OOF predictions | 23,014 |
| Accuracy | 0.6861 |
| Macro-F1 | 0.4388 |
| Quadratic weighted kappa (QWK) | 0.6245 |

Do not combine this OOF result with the held-out test result into a single headline score: they use different evaluation protocols and model configurations.

### 14.3 Validation model comparison

The second-stage model comparison selected configurations by validation QWK. The notebook's final held-out-test table contains these test results for the compared models:

| Backbone | Head | Validation QWK | Test QWK | Referable AUROC |
|---|---|---:|---:|---:|
| ConvNeXt-Tiny | Regression | 0.7510 | 0.7407 | 0.9035 |
| ConvNeXt-Tiny | CORN | 0.7286 | 0.7259 | 0.9099 |
| ConvNeXt-Tiny | Cross-entropy | 0.7156 | 0.7198 | 0.9048 |
| EfficientNet-B3 | Regression | 0.7052 | 0.6918 | 0.8701 |
| EfficientNet-B0 | Regression | 0.6912 | 0.6848 | 0.8520 |

The validation score is used for model selection; the test score is for reporting and comparison. The best QWK model is not necessarily the best AUROC or calibration model.

### 14.4 Leakage experiment

| Split | Mean QWK across three seeds | Standard deviation |
|---|---:|---:|
| Image-level | 0.6426 | 0.0122 |
| Patient-level | 0.6476 | 0.0121 |

Although 67.3% of image-level test images shared a patient with training data, the image-level mean QWK was lower in this run. The estimated mean difference was −0.0050, with a seed-0 cluster-bootstrap 95% interval of [−0.0347, +0.0411]. The experiment demonstrates leakage exposure but does not demonstrate positive QWK inflation.

### 14.5 What is not yet established

- APTOS/Messidor-2/IDRiD external-domain performance.
- That conformal risk control will maintain its target under domain shift.
- That noise-aware training improves generalization on a cleaner external set.
- Clinical usefulness, safety, or deployment readiness.
- Novelty relative to all prior research.

## 15. Scope and experimental status

| Component | Evidence in saved notebook | Status |
|---|---|---|
| Dataset inventory and class counts | Printed counts and class distribution | Executed |
| Patient/eye parsing and agreement | 17,563 patients; left/right QWK 0.855 | Executed |
| Near-duplicate audit | 2,252 dropped; 167 conflicting-label clusters | Executed |
| Patient-level split | 0% patient overlap reported | Executed |
| Image-level leakage comparison | Three seeds; mean metrics and overlap printed | Executed |
| Backbone/head search | Validation and held-out-test results printed | Executed for saved configurations |
| Uncertainty comparison | Saved conformal/uncertainty tables | Partially executed; interpret each table in context |
| Conformal risk evaluation | FNR/referral/set-size results printed | Executed empirically on this split |
| OOF label-noise variants | Metrics printed for baseline/drop/relabel/soft labels | Executed; variants did not improve QWK here |
| APTOS external evaluation | “APTOS not found” output | Not executed |
| Messidor-2 / IDRiD external evaluation | Optional configuration only; no saved external metrics | Not verified as executed |
| Grad-CAM | Qualitative overlays produced | Executed qualitatively |
| Lesion pointing game | No lesion masks configured | Not executed |
| Final model export | Export cell has no saved execution output | Code present; export artifact not verified |

## 16. Running the notebook on Kaggle

The most faithful reproduction environment is Kaggle Notebooks, because the code uses `/kaggle/input` and `/kaggle/working` paths and can download pretrained `timm` weights.

### Step 1: Open or import the notebook

1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Upload or import `diabetic-retinopathy2015-data-coloured - 1.ipynb`.
3. Attach the [Diabetic Retinopathy 2015 Data Colored Resized dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized) using **Add Input**.

### Step 2: Enable a GPU

Select a GPU accelerator in notebook settings. The saved run recorded a **Tesla T4** GPU and `timm 1.0.29`, but Kaggle hardware availability may differ. If pretrained weights must be downloaded, enable internet for the initial run or provide the required weights as a Kaggle input dataset.

### Step 3: Start with a debug run

The configuration cell defines:

```python
CFG["DEBUG"] = False
CFG["DEBUG_PATIENTS"] = 800
```

For a smoke test, change `CFG["DEBUG"]` to `True`. Debug mode reduces the number of epochs and several experiment sizes; **debug results are not publishable final results**. Once setup and data loading work, return to `False` for the intended experiment.

### Step 4: Run cells in order

Run the notebook from the first setup cell through the desired phases. The primary dataset must be attached before the audit cell. The notebook uses folder names beginning with `No`, `Mild`, `Moder`, `Sev`, and `Prolif` to identify the five classes.

### Step 5: Manage memory and session time

The notebook defaults to:

```python
CFG["IMG_SIZE"] = 224
CFG["BS"] = 64
CFG["PRED_BS"] = 256
CFG["SESSION_LIMIT_H"] = 11.5
```

The raw cache alone was approximately 5.29 GB in the saved run. If GPU memory runs out, reduce `BS` (for example, 32, 16, or 8) and reduce `PRED_BS` as needed. Smaller batches may be slower. Ensure the Kaggle working-storage limit is sufficient for the cache, CLAHE cache, checkpoints, run files, and results.

### Step 6: Optional phases

- For APTOS evaluation, attach the `aptos2019-blindness-detection` dataset as an input. The notebook's saved run did not have it attached.
- Messidor-2 requires `CFG["MESSIDOR2"]` to be configured with image directory, CSV path, ID column, label column, and extension.
- IDRiD evaluation requires `CFG["IDRID_ROOT"]`; the lesion pointing game also requires a matching binary-mask directory configured via `CFG["IDRID_MASK_DIR"]`.
- RETFound requires an available checkpoint path in `CFG["RETFOUND_CKPT"]`.

Do not enable optional phases without verifying their dataset paths and label mappings.

## 17. Configuration and reproducibility

The principal settings in the notebook include:

| Setting | Default | Purpose |
|---|---:|---|
| `SEED` | 42 | Main random seed |
| `IMG_SIZE` | 224 | Model input resolution |
| `BS` | 64 | Training batch size |
| `PRED_BS` | 256 | Prediction/evaluation batch size |
| `N_SEEDS` | 3 | Seeds for leakage study/key models |
| `EPOCHS_LEAK` | 8 | Leakage experiment epochs |
| `EPOCHS_RUNG1` | 3 | First successive-halving stage |
| `RUNG2_KEEP` | 5 | Number of configurations retained for stage two |
| `EPOCHS_GRID` | 10 | Second-stage/final recipe epochs |
| `EPOCHS_OOF` | 5 | OOF training epochs |
| `EPOCHS_NOISE` | 8 | Noise-variant training epochs |
| `ENS_N` | 5 | Ensemble member count |
| `MC_T` | 20 | MC-dropout passes |
| `ALPHA_MAIN` | 0.05 | Main conformal target level |
| `ALPHAS` | `[0.02, 0.05, 0.10, 0.15, 0.20]` | Risk levels evaluated |
| `RECAL_N` | 200 | Target examples used for each recalibration draw |
| `N_BOOT` | 300 | Cluster-bootstrap repetitions |
| `SESSION_LIMIT_H` | 11.5 | Stop before the notebook session limit |

The code uses Python, NumPy, and PyTorch random seeding. GPU kernels and library versions may still introduce nondeterminism; reproducibility should include saved configuration, package versions, split IDs, and checkpoint metadata.

### Resuming interrupted Kaggle runs

The notebook stores working artifacts under `/kaggle/working/cache`, `/kaggle/working/ckpt`, `/kaggle/working/runs`, and `/kaggle/working/results`. Its setup cell can restore these folders from a previously saved working-output root attached as an input dataset.

1. Save a notebook version with the relevant `/kaggle/working` artifacts preserved.
2. Attach the saved output as an input dataset to a later session.
3. Let the setup cell attempt automatic restoration.
4. If automatic detection does not find the right root, set `CFG["RESUME_ROOT"]` to the correct directory.
5. Verify that the expected checkpoint and run metadata were restored before continuing.

The saved notebook output reported a total of 6.2 GPU-hours across the listed completed phases. This is the ledger from that run, not a guaranteed runtime estimate for future sessions.

## 18. Outputs and model export

When the relevant cells execute, the notebook writes CSV tables and figures to `/kaggle/working/results`, including:

- `phase1_leak_runs.csv`
- `phase2_rung1.csv`
- `phase2_rung2.csv`
- `phase2_preproc.csv`
- `phase3_uncertainty.csv`
- `phase4_conformal.csv`
- `phase4_validity_200_splits.csv`
- `phase5_noise_test.csv`
- `gpu_ledger_by_phase.csv`
- `final_evaluation/` confusion matrices, normalized confusion matrices, and classification reports

The export cell is designed to create `/kaggle/working/final_model_export/` with a model checkpoint, configuration metadata, an inference script, and a README when a validation-selected run is available. **The export cell has not produced a verified saved output in the notebook provided to this repository**, so do not assume that model weights or the export bundle are currently available.

If you publish result artifacts later, include provenance (configuration, split, model tag, package versions, and whether the metric is validation, OOF, held-out test, or external). Do not upload the raw image dataset or large cache files to GitHub.

## 19. Limitations and threats to validity

1. **Reduced image resolution:** the source images in this Kaggle release are 224 × 224. Fine retinal details lost during resizing cannot be recovered by upscaling.
2. **Patient grouping depends on filenames:** patient IDs are parsed from filename prefixes. If a different copy of the dataset uses another naming convention, the notebook warns that the RQ1 analysis may be invalid.
3. **Duplicate detection is heuristic:** perceptual hashes and thumbnail similarity can miss duplicates or flag legitimate similar images. Conflicting labels require review rather than automatic truth assignment.
4. **Label noise:** EyePACS labels may contain errors. Metrics against noisy labels may underestimate or distort model quality.
5. **Class imbalance:** severe and proliferative cases are relatively rare, and minority-class recall remains uneven.
6. **Calibration:** the selected QWK model's ECE is 0.1779 in the saved test table. Reliable ranking does not guarantee reliable probabilities.
7. **Leakage experiment result:** substantial overlap was measured in the image-level split, but mean QWK was not higher than patient-level QWK in this run. More repetitions and robust patient-cluster uncertainty analysis are needed.
8. **Domain shift:** APTOS was not attached, so external generalization is not established. Any future external result must account for differences in image preprocessing, camera characteristics, patient populations, and grading distributions.
9. **Conformal assumptions:** risk-control claims depend on calibration design and exchangeability. They may fail after a domain shift and do not replace prospective clinical validation.
10. **Single-run model selection:** the notebook notes that the grid uses one seed per configuration, with multiple seeds only for selected configurations.
11. **Interpretability:** Grad-CAM overlays are qualitative in the saved run; no lesion masks were configured, so a lesion pointing-game score was not established.
12. **Reproducibility artifacts:** the repository currently does not include dependency lockfiles, split manifests, trained checkpoints, or generated result tables.
13. **Clinical limitations:** no prospective clinical study, workflow evaluation, regulatory approval, or external clinical validation is established by this repository.

## 20. Responsible use and ethics

This repository is for **research and education only**.

- It is not intended to diagnose, screen, triage, or treat patients.
- Do not use the model output as a substitute for assessment by a qualified eye-care professional.
- Do not deploy it in clinical workflows without independent validation, appropriate regulatory review, safety assessment, and prospective evaluation.
- Do not infer that a model is clinically safe because it has a high AUROC or QWK.
- Respect the dataset's current license and terms, and do not redistribute the original images through this repository.
- Any research publication should disclose split design, patient grouping, class distribution, label limitations, calibration assumptions, and all external-validation gaps.

## 21. Future work

Priorities for strengthening the research evidence include:

1. Re-run the full pipeline from a clean Kaggle session and save all outputs with a configuration manifest.
2. Repeat model comparisons with multiple seeds and report uncertainty intervals for the main metrics.
3. Investigate the duplicate-detection threshold and manually review representative duplicate/conflicting-label clusters.
4. Complete the APTOS external evaluation with a documented label mapping and strict separation of target calibration and target test data.
5. Evaluate whether noise-aware training helps on an external dataset rather than relying only on noisy source labels.
6. Report class-wise sensitivity/recall and precision with confidence intervals, especially for grades 2–4.
7. Evaluate calibration across subgroups and operating points, and report coverage–risk trade-offs for conformal sets.
8. Validate conformal risk behavior under deliberate and real domain shifts.
9. Add IDRiD lesion masks and quantify Grad-CAM localization if a valid mask mapping is available.
10. Export and test a model bundle from a saved run, then add inference smoke tests and dependency specifications.
11. Conduct a systematic literature review before making any claim that the combined protocol is novel.

## 22. References and links

1. **Primary dataset:** Sovit Ranjan Rath, *Diabetic Retinopathy 2015 Data Colored Resized*. Kaggle.  
   <https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized>
2. **Original challenge:** *Diabetic Retinopathy Detection*, Kaggle / EyePACS.  
   <https://www.kaggle.com/c/diabetic-retinopathy-detection>
3. **APTOS 2019 Blindness Detection:** Kaggle competition page.  
   <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
4. **PyTorch:** Deep learning framework.  
   <https://pytorch.org/>
5. **timm:** PyTorch Image Models.  
   <https://github.com/huggingface/pytorch-image-models>
6. **Cleanlab:** Tools for learning with label issues.  
   <https://github.com/cleanlab/cleanlab>
7. **scikit-learn:** Machine-learning utilities and metrics.  
   <https://scikit-learn.org/stable/>

These links provide dataset and software context. A publication-quality manuscript should add primary research citations for ordinal classification, CORN, calibration, conformal prediction/risk control, label-noise estimation, and domain generalization after those sources have been independently checked. The project does not claim that its combination of methods is novel solely because the methods appear together.

## 23. Author

**Hemant Koul**  
M.Tech student in Artificial Intelligence and Data Science

- **GitHub repository:** <https://github.com/hkoul2000/trustworthy-diabetic-retinopathy>
- **GitHub profile:** <https://github.com/hkoul2000>

---

*Last documentation review: 2026-10-22. Metrics and status statements above reflect the saved outputs in the notebook version reviewed for this README. Re-run and re-audit the notebook before using these numbers in a paper, presentation, or application.*
