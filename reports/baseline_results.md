# Baseline Flood Probability Modeling


## 1. Objective

The objective of Phase 4 was to develop and compare baseline machine learning models for predicting annual flood-inundation presence across Bihar's 0.25-degree rainfall grid.

The target variable is `flood_label`, where 1 indicates that a grid cell intersects the annual flood-inundation extent and 0 indicates no recorded intersection in the available annual dataset.

## 2. Dataset

- Study area: Bihar
- Spatial grid cells: 301
- Labeled grid-year observations: 1,806
- Labeled years: 2015, 2016, 2017, 2018, 2019 and 2021
- Training period for the temporal experiments: 2015–2018
- Temporal validation year: 2019
- Temporal test year: 2021

The 2020 flood label is unavailable in the primary annual dataset. Missing flood labels were not treated as non-flood observations.

## 3. Models Implemented

1. Logistic Regression
2. Decision Tree
3. Spatial-only Decision Tree
4. Rainfall-only Decision Tree
5. Spatial + Rainfall Decision Tree
6. Spatial generalization Decision Tree
7. Random Forest
8. Gradient Boosting
9. River-feature experiment using Gradient Boosting

## 4. Main Spatial-Generalization Results

The spatial-generalization experiment used GroupShuffleSplit with a 20% region holdout and random state 42. It contained 240 training regions and 61 test regions, with zero overlapping region IDs.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.822 | 0.640 | 0.946 | 0.764 | 0.873 | 0.649 |
| Random Forest | 0.817 | 0.643 | 0.892 | 0.747 | 0.910 | 0.805 |
| Gradient Boosting | 0.842 | 0.693 | 0.856 | 0.766 | 0.934 | 0.850 |

Gradient Boosting achieved the highest accuracy, F1, ROC-AUC and PR-AUC among these three spatial-holdout experiments. Decision Tree achieved the highest recall.

## 5. Feature Analysis

The Decision Tree feature-importance analysis assigned approximately 97.95% combined importance to latitude and longitude.

The Gradient Boosting analysis assigned 92.93% combined importance to latitude and longitude, while rainfall features accounted for 7.07%.

A spatial-only Decision Tree achieved a test F1 of approximately 0.866 in the temporal split. The rainfall-only Decision Tree achieved a test F1 of approximately 0.469.

These findings indicate that the current annual flood labels have a strong spatial structure. Feature importance describes the fitted model's use of features; it does not establish causation or prove that rainfall is physically unimportant.

## 6. River-Feature Experiment

The river-feature experiment compared Gradient Boosting with spatial and rainfall features against the same model with additional annual river features.

The reported metrics were identical:

- Accuracy: 0.842
- Precision: 0.693
- Recall: 0.856
- F1: 0.766
- ROC-AUC: 0.934
- PR-AUC: 0.850

River-feature availability was limited. The mean river-observation coverage was zero from 2015 through 2019 and approximately 12.70% in 2021. Approximately 94.10% of training rows and 92.62% of test rows had at least one river feature missing.

The current experiment therefore does not establish that river conditions are unimportant. More complete and temporally aligned river observations are needed for a meaningful assessment.

## 7. Limitations

- The target represents annual flood-inundation presence, not the timing of individual flood events.
- The labeled dataset has only six available years.
- The spatial holdout prevents identical region IDs from appearing in both train and test, but nearby grid cells can still be split across the two sets.
- The temporal split reuses geographic cells across years, so it evaluates a different generalization setting from the spatial holdout.
- River observations are sparse and not available across the full labeled period.
- The current results do not establish calibrated flood probabilities or operational warning performance.
- The rainfall grid resolution is 0.25 degrees; predictions should not be presented as building-level or 30-metre-resolution predictions.

## 8. Conclusion

Gradient Boosting is the leading candidate from the tested spatial-generalization experiments. The Decision Tree remains useful when high recall is prioritized.

The models are experimental baselines, not yet validated operational flood forecasts. Future work should improve temporal target design, evaluate spatial cross-validation, assess probability calibration and threshold selection, and obtain better-aligned river and flood observations before final model selection.


## 9. Phase 5 — Spatial Cross-Validation and Threshold Analysis

Five-fold GroupKFold cross-validation was conducted using `region_id` as the grouping variable. Each fold held out distinct region IDs, with zero region overlap between training and validation sets.

### Threshold Comparison

| Threshold | Mean F1 | Mean Precision | Mean Recall |
|---:|---:|---:|---:|
| 0.30 | 0.8293 | 0.7469 | 0.9351 |
| 0.40 | 0.8356 | 0.7757 | 0.9068 |
| 0.50 | 0.8336 | 0.8127 | 0.8564 |
| 0.60 | 0.8219 | 0.8589 | 0.7881 |
| 0.70 | 0.7908 | 0.9171 | 0.6959 |

### Cross-Validation Summary

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy at 0.50 | 0.8728 | 0.0241 |
| ROC-AUC | 0.9489 | 0.0103 |
| PR-AUC | 0.9165 | 0.0199 |
| F1 at 0.50 | 0.8336 | 0.0379 |

### Findings and Decision

- Threshold 0.40 achieved the highest mean F1 (0.8356) in the exploratory threshold comparison.
- Threshold 0.30 achieved the highest mean recall (0.9351), while threshold 0.70 achieved the highest mean precision (0.9171).
- The mean F1 difference between thresholds 0.40 and 0.50 was small.
- **Decision:** Retain 0.50 as the default threshold for now. Treat 0.40 as a candidate requiring separate validation rather than selecting it as the final threshold based on the same cross-validation results.

### Limitations

The folds hold out region IDs but do not enforce a geographic-distance buffer, so nearby grid cells may still appear in different folds. The supervised labels cover only 2015–2019 and 2021; missing years must not be interpreted as non-flood observations. These results evaluate experimental classification performance and do not establish calibrated probabilities or operational flood-warning performance.