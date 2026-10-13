from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.models.prepare_data import load_supervised_data

FEATURES=[
    "latitude",
    "longitude",
    "annual_rainfall_total",
    "annual_rainfall_mean",
    "annual_rainfall_max",
    "annual_rainfall_3d_max",
    "annual_rainfall_7d_max",
    "annual_rainfall_14d_max",
    "monsoon_rainfall_total",
    "monsoon_rainfall_mean",
    "monsoon_rainfall_max",
    "monsoon_rainfall_3d_max",
    "monsoon_rainfall_7d_max",
    "monsoon_rainfall_14d_max",
    "heavy_rain_days",
    "very_heavy_rain_days",
    "extreme_rain_days",
    "rainy_days"
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


def train_spatial_generalization():
    df=load_supervised_data(Path("data/processed/flood_supervised_grid_year_2015_2021.csv"))

    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
    train_indices,test_indices=next(splitter.split(df,df["flood_label"],groups=df["region_id"]))

    train=df.iloc[train_indices].copy()
    test=df.iloc[test_indices].copy()

    X_train=train[FEATURES]
    y_train=train["flood_label"].astype(int)
    X_test=test[FEATURES]
    y_test=test["flood_label"].astype(int)

    print("=== SPATIAL GENERALIZATION SPLIT ===")
    print(f"Train samples: {len(train):,}")
    print(f"Test samples: {len(test):,}")
    print(f"Train regions: {train['region_id'].nunique():,}")
    print(f"Test regions: {test['region_id'].nunique():,}")
    print(f"Overlapping regions: {len(set(train['region_id']) & set(test['region_id']))}")

    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("classifier",DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("\nTraining Spatial Generalization Decision Tree...")
    model.fit(X_train,y_train)

    results=[
        evaluate(model,X_train,y_train,"train"),
        evaluate(model,X_test,y_test,"spatial_test")
    ]

    print("\n=== SPATIAL GENERALIZATION RESULTS ===")
    print(pd.DataFrame(results).to_string(index=False))

if __name__=="__main__":
    train_spatial_generalization()