
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.models.prepare_data import load_supervised_data
from src.models.gradient_boosting_baseline import FEATURES

def analyze_feature_importance():
    df=load_supervised_data(Path("data/processed/flood_supervised_grid_year_2015_2021.csv"))
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
    train_indices,test_indices=next(splitter.split(df,df["flood_label"],groups=df["region_id"]))
    train=df.iloc[train_indices].copy()
    X_train=train[FEATURES]
    y_train=train["flood_label"].astype(int)
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
    print("Training Gradient Boosting for feature importance...")
    model.fit(X_train,y_train)
    classifier=model.named_steps["classifier"]
    importance=pd.DataFrame({
        "feature":FEATURES,
        "importance":classifier.feature_importances_
    }).sort_values("importance",ascending=False)
    print("\n=== GRADIENT BOOSTING FEATURE IMPORTANCE ===")
    print(importance.to_string(index=False))
    print(f"\nTop 2 features combined importance: {importance.head(2)['importance'].sum():.4f}")
    print(f"All rainfall features combined importance: {importance[~importance['feature'].isin(['latitude','longitude'])]['importance'].sum():.4f}")

if __name__=="__main__":
    analyze_feature_importance()
