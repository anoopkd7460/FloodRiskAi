
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.models.prepare_data import load_supervised_data
from src.models.gradient_boosting_baseline import FEATURES as BASE_FEATURES

RIVER_FEATURES=[
    "river_observation_days",
    "river_observation_rate",
    "annual_mean_river_level",
    "annual_max_river_level",
    "annual_mean_discharge",
    "annual_max_discharge",
    "max_river_level_change",
    "max_river_level_3d_change",
    "max_river_level_7d_change",
    "max_discharge_change",
    "max_discharge_3d_change",
    "max_discharge_7d_change"
]

def evaluate(model,X,y,name: str) -> dict:
    predictions=model.predict(X)
    probabilities=model.predict_proba(X)[:,1]
    return {
        "dataset":name,
        "accuracy":accuracy_score(y,predictions),
        "precision":precision_score(y,predictions,zero_division=0),
        "recall":recall_score(y,predictions,zero_division=0),
        "f1":f1_score(y,predictions,zero_division=0),
        "roc_auc":roc_auc_score(y,probabilities),
        "pr_auc":average_precision_score(y,probabilities)
    }

def run_experiment():
    df=load_supervised_data(Path("data/processed/flood_supervised_grid_year_2015_2021.csv"))
    features=BASE_FEATURES+RIVER_FEATURES
    missing=[column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
    train_indices,test_indices=next(
        splitter.split(df,df["flood_label"],groups=df["region_id"])
    )
    train=df.iloc[train_indices].copy()
    test=df.iloc[test_indices].copy()

    y_train=train["flood_label"].astype(int)
    y_test=test["flood_label"].astype(int)

    X_train_base=train[BASE_FEATURES]
    X_test_base=test[BASE_FEATURES]
    X_train_river=train[features]
    X_test_river=test[features]

    print("=== SPATIAL HOLDOUT ===")
    print(f"Train regions: {train['region_id'].nunique()}")
    print(f"Test regions: {test['region_id'].nunique()}")
    print(f"Overlapping regions: {len(set(train['region_id']) & set(test['region_id']))}")
    print(f"Training rows with any river feature missing: {X_train_river[RIVER_FEATURES].isna().any(axis=1).mean():.2%}")
    print(f"Test rows with any river feature missing: {X_test_river[RIVER_FEATURES].isna().any(axis=1).mean():.2%}")

    def build_model():
        return Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("classifier",GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=10,
                random_state=42
            ))
        ])

    baseline=build_model()
    river_model=build_model()

    print("\nTraining baseline: Spatial + Rainfall...")
    baseline.fit(X_train_base,y_train)
    print("Training model: Spatial + Rainfall + River...")
    river_model.fit(X_train_river,y_train)

    results=[
        evaluate(baseline,X_test_base,y_test,"spatial_rainfall"),
        evaluate(river_model,X_test_river,y_test,"spatial_rainfall_river")
    ]

    result_df=pd.DataFrame(results)
    print("\n=== RIVER FEATURE COMPARISON: SPATIAL TEST ===")
    print(result_df.to_string(index=False))

    importance=pd.DataFrame({
        "feature":features,
        "importance":river_model.named_steps["classifier"].feature_importances_
    }).sort_values("importance",ascending=False)
    print("\n=== TOP 12 FEATURE IMPORTANCES ===")
    print(importance.head(12).to_string(index=False))

    return baseline,river_model,result_df,importance

if __name__=="__main__":
    run_experiment()
