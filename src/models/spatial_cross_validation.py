
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.models.prepare_data import load_supervised_data
from src.models.gradient_boosting_baseline import FEATURES

def evaluate(model,X,y,fold: int) -> dict:
    predictions=model.predict(X)
    probabilities=model.predict_proba(X)[:,1]
    return {
        "fold":fold,
        "accuracy":accuracy_score(y,predictions),
        "precision":precision_score(y,predictions,zero_division=0),
        "recall":recall_score(y,predictions,zero_division=0),
        "f1":f1_score(y,predictions,zero_division=0),
        "roc_auc":roc_auc_score(y,probabilities),
        "pr_auc":average_precision_score(y,probabilities)
    }

def run_spatial_cross_validation():
    df=load_supervised_data("data/processed/flood_supervised_grid_year_2015_2021.csv")
    groups=df["region_id"]
    X=df[FEATURES]
    y=df["flood_label"].astype(int)
    splitter=GroupKFold(n_splits=5)
    results=[]
    print("=== 5-FOLD SPATIAL CROSS-VALIDATION ===")
    for fold,(train_idx,val_idx) in enumerate(splitter.split(X,y,groups),start=1):
        X_train,X_val=X.iloc[train_idx],X.iloc[val_idx]
        y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
        train_regions=set(groups.iloc[train_idx])
        val_regions=set(groups.iloc[val_idx])
        overlap=len(train_regions & val_regions)
        if overlap:
            raise ValueError(f"Fold {fold} has {overlap} overlapping regions.")
        if y_train.nunique()<2 or y_val.nunique()<2:
            raise ValueError(f"Fold {fold} does not contain both target classes.")
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
        results.append(evaluate(model,X_val,y_val,fold))
        print(f"Fold {fold}: train regions={len(train_regions)}, validation regions={len(val_regions)}, overlap={overlap}")
    result_df=pd.DataFrame(results)
    print("\n=== FOLD RESULTS ===")
    print(result_df.to_string(index=False))
    metrics=["accuracy","precision","recall","f1","roc_auc","pr_auc"]
    summary=result_df[metrics].agg(["mean","std"]).T
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(summary.to_string())
    output_path="reports/phase5_spatial_cv_results.csv"
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    result_df.to_csv(output_path,index=False)
    print(f"\nSaved fold results: {output_path}")
    return result_df,summary

if __name__=="__main__":
    run_spatial_cross_validation()
