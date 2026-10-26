
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline

from src.models.prepare_data import (
    BASE_FEATURES,
    TARGET,
    load_supervised_data,
)


DATA_PATH = Path("data/processed/flood_supervised_grid_year_2015_2021.csv")
REPORT_PATH = Path("reports/baseline_results.md")

THRESHOLDS = np.arange(0.30, 0.71, 0.05)


def classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def probability_metrics(y_true, probabilities):
    """Evaluate probability quality independently of a classification threshold."""
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(
            y_true,
            np.column_stack([1 - probabilities, probabilities]),
            labels=[0, 1],
        ),
    }


def evaluate_temporal_calibration():
    df = load_supervised_data(DATA_PATH)

    if "year" not in df.columns:
        raise ValueError("The dataset must contain a 'year' column.")

    train = df[df["year"].between(2015, 2018)].copy()
    validation = df[df["year"] == 2019].copy()
    test = df[df["year"] == 2021].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Temporal split is incomplete. Expected training years "
            "2015–2018, validation year 2019, and test year 2021."
        )

    for name, split in [
        ("training", train),
        ("validation", validation),
        ("test", test),
    ]:
        if split[TARGET].nunique() != 2:
            raise ValueError(
                f"The {name} split must contain both target classes."
            )

    X_train = train[BASE_FEATURES]
    y_train = train[TARGET].astype(int)

    X_validation = validation[BASE_FEATURES]
    y_validation = validation[TARGET].astype(int)

    X_test = test[BASE_FEATURES]
    y_test = test[TARGET].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    print("=== TEMPORAL SPLIT ===")
    print(f"Training rows (2015–2018): {len(train):,}")
    print(f"Validation rows (2019): {len(validation):,}")
    print(f"Test rows (2021): {len(test):,}")

    # Fit only on the training period.
    model.fit(X_train, y_train)

    # Select the threshold using validation data only.
    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    threshold_results = pd.DataFrame([
        classification_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ])

    print("\n=== VALIDATION THRESHOLD COMPARISON (2019) ===")
    print(threshold_results.to_string(index=False, float_format="%.4f"))

    # Tie-break: prefer the threshold closest to 0.50.
    best_f1 = threshold_results["f1"].max()
    tied = threshold_results[
        np.isclose(threshold_results["f1"], best_f1)
    ]
    selected_threshold = float(
        tied.iloc[
            (tied["threshold"] - 0.50).abs().argmin()
        ]["threshold"]
    )

    print(f"\nSelected threshold from 2019 validation: {selected_threshold:.2f}")

    # Evaluate the selected threshold once on the held-out 2021 test year.
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_classification = classification_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )
    test_probability = probability_metrics(y_test, test_probabilities)

    print("\n=== HELD-OUT TEMPORAL TEST (2021) ===")
    print(pd.DataFrame([test_classification]).to_string(
        index=False, float_format="%.4f"
    ))

    print("\n=== TEST PROBABILITY METRICS (2021) ===")
    print(pd.DataFrame([test_probability]).to_string(
        index=False, float_format="%.4f"
    ))

    # Calibration curve is descriptive; it does not itself calibrate the model.
    observed_fraction, mean_predicted_probability = calibration_curve(
        y_test,
        test_probabilities,
        n_bins=5,
        strategy="quantile",
    )

    calibration_results = pd.DataFrame({
        "mean_predicted_probability": mean_predicted_probability,
        "observed_flood_fraction": observed_fraction,
    })

    print("\n=== TEST CALIBRATION CURVE (2021) ===")
    print(calibration_results.to_string(
        index=False, float_format="%.4f"
    ))

    return {
        "model": model,
        "threshold_results": threshold_results,
        "selected_threshold": selected_threshold,
        "test_classification": test_classification,
        "test_probability": test_probability,
        "calibration_curve": calibration_results,
    }


if __name__ == "__main__":
    evaluate_temporal_calibration()
