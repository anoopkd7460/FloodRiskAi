
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.models.prepare_data import load_supervised_data
from src.models.gradient_boosting_baseline import FEATURES


THRESHOLDS=[0.30,0.40,0.50,0.60,0.70]


def evaluate(model,X,y,fold: int) -> dict:
    probabilities=model.predict_proba(X)[:,1]

    results={
        "fold":fold,
        "accuracy_at_0.50":accuracy_score(
            y,(probabilities>=0.50).astype(int)
        ),
        "roc_auc":roc_auc_score(y,probabilities),
        "pr_auc":average_precision_score(y,probabilities)
    }

    for threshold in THRESHOLDS:
        predictions=(probabilities>=threshold).astype(int)
        suffix=f"{threshold:.2f}"

        results[f"f1_{suffix}"]=f1_score(
            y,predictions,zero_division=0
        )
        results[f"precision_{suffix}"]=precision_score(
            y,predictions,zero_division=0
        )
        results[f"recall_{suffix}"]=recall_score(
            y,predictions,zero_division=0
        )

    return results


def run_spatial_cross_validation():
    data_path="data/processed/flood_supervised_grid_year_2015_2021.csv"
    df=load_supervised_data(data_path)

    groups=df["region_id"]
    X=df[FEATURES]
    y=df["flood_label"].astype(int)

    if y.nunique()!=2:
        raise ValueError("The target must contain both classes.")

    splitter=GroupKFold(n_splits=5)
    results=[]

    print("=== 5-FOLD REGION-GROUPED CROSS-VALIDATION ===")

    for fold,(train_idx,val_idx) in enumerate(
        splitter.split(X,y,groups),start=1
    ):
        X_train=X.iloc[train_idx]
        X_val=X.iloc[val_idx]
        y_train=y.iloc[train_idx]
        y_val=y.iloc[val_idx]

        train_regions=set(groups.iloc[train_idx])
        val_regions=set(groups.iloc[val_idx])
        overlap=len(train_regions & val_regions)

        if overlap:
            raise ValueError(
                f"Fold {fold} has {overlap} overlapping regions."
            )

        if y_train.nunique()<2 or y_val.nunique()<2:
            raise ValueError(
                f"Fold {fold} does not contain both target classes."
            )

        model=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("classifier",GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=10,
                random_state=42
            ))
        ])

        model.fit(X_train,y_train)
        fold_results=evaluate(model,X_val,y_val,fold)
        results.append(fold_results)

        print(
            f"Fold {fold}: train regions={len(train_regions)}, "
            f"validation regions={len(val_regions)}, "
            f"overlap={overlap}"
        )

    result_df=pd.DataFrame(results)

    print("\n=== FOLD RESULTS ===")
    print(result_df.to_string(index=False))

    metric_columns=[
        "accuracy_at_0.50",
        "roc_auc",
        "pr_auc"
    ]

    for threshold in THRESHOLDS:
        suffix=f"{threshold:.2f}"
        metric_columns.extend([
            f"f1_{suffix}",
            f"precision_{suffix}",
            f"recall_{suffix}"
        ])

    summary=result_df[metric_columns].agg(["mean","std"]).T

    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(summary.to_string())

    f1_columns=[f"f1_{threshold:.2f}" for threshold in THRESHOLDS]
    mean_f1=result_df[f1_columns].mean()

    print("\n=== MEAN F1 BY THRESHOLD ===")
    print(mean_f1.to_string())

    best_threshold=float(mean_f1.idxmax().replace("f1_",""))
    print(f"\nBest mean-F1 threshold across folds: {best_threshold:.2f}")
    print(
        "Treat this as exploratory: select the final threshold using "
        "a separate validation procedure, not the final test set."
    )

    output_path=Path("reports/phase5_spatial_cv_results.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result_df.to_csv(output_path,index=False)

    print(f"\nUpdated existing report: {output_path}")

    return result_df,summary


if __name__=="__main__":
    run_spatial_cross_validation()
