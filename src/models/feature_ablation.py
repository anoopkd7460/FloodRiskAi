
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.models.prepare_data import load_supervised_data
from src.models.gradient_boosting_baseline import FEATURES

SPATIAL_FEATURES=["latitude","longitude"]
ENVIRONMENTAL_FEATURES=[feature for feature in FEATURES if feature not in SPATIAL_FEATURES]

def evaluate(model,X,y,fold: int,feature_set: str) -> dict:
    predictions=model.predict(X)
    probabilities=model.predict_proba(X)[:,1]
    return {
        "feature_set":feature_set,
        "fold":fold,
        "accuracy":accuracy_score(y,predictions),
        "precision":precision_score(y,predictions,zero_division=0),
        "recall":recall_score(y,predictions,zero_division=0),
        "f1":f1_score(y,predictions,zero_division=0),
        "roc_auc":roc_auc_score(y,probabilities),
        "pr_auc":average_precision_score(y,probabilities)
    }

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

def run_feature_ablation():
    df=load_supervised_data("data/processed/flood_supervised_grid_year_2015_2021.csv")
    y=df["flood_label"].astype(int)
    groups=df["region_id"]
    feature_sets={
        "spatial_rainfall":FEATURES,
        "rainfall_only":ENVIRONMENTAL_FEATURES
    }
    splitter=GroupKFold(n_splits=5)
    results=[]
    print("=== FEATURE ABLATION: 5-FOLD SPATIAL CV ===")
    for fold,(train_idx,val_idx) in enumerate(splitter.split(df,y,groups),start=1):
        train_regions=set(groups.iloc[train_idx])
        val_regions=set(groups.iloc[val_idx])
        overlap=len(train_regions & val_regions)
        if overlap:
            raise ValueError(f"Fold {fold} has {overlap} overlapping regions.")
        y_train=y.iloc[train_idx]
        y_val=y.iloc[val_idx]
        if y_train.nunique()<2 or y_val.nunique()<2:
            raise ValueError(f"Fold {fold} does not contain both target classes.")
        print(f"Fold {fold}: train regions={len(train_regions)}, validation regions={len(val_regions)}, overlap={overlap}")
        for name,features in feature_sets.items():
            X_train=df.iloc[train_idx][features]
            X_val=df.iloc[val_idx][features]
            model=build_model()
            model.fit(X_train,y_train)
            results.append(evaluate(model,X_val,y_val,fold,name))
    result_df=pd.DataFrame(results)
    print("\n=== FOLD RESULTS ===")
    print(result_df.to_string(index=False))
    metrics=["accuracy","precision","recall","f1","roc_auc","pr_auc"]
    summary=result_df.groupby("feature_set")[metrics].agg(["mean","std"])
    print("\n=== SUMMARY: MEAN AND STANDARD DEVIATION ===")
    print(summary.round(4).to_string())
    output_path=Path("reports/phase5_feature_ablation_results.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result_df.to_csv(output_path,index=False)
    print(f"\nSaved results: {output_path}")
    return result_df,summary

if __name__=="__main__":
    run_feature_ablation()
