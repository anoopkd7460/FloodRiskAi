from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.prepare_data import load_supervised_data,create_temporal_datasets

def analyze_feature_importance():
    data_path=Path("data/processed/flood_supervised_grid_year_2015_2021.csv")
    df=load_supervised_data(data_path)
    X_train,y_train,_,_,_,_=create_temporal_datasets(df)

    imputer=SimpleImputer(strategy="median")
    X_train_imputed=imputer.fit_transform(X_train)

    model = DecisionTreeClassifier(
        max_depth=5,min_samples_leaf=10,class_weight="balanced",random_state=42
    )
    model.fit(X_train_imputed,y_train)

    importance=pd.DataFrame({
        "feature":X_train.columns,
        "importance":model.feature_importances_
    }).sort_values("importance",ascending=False)

    print("=== DECISION TREE FEATURE IMPORTANCE ===")
    print(importance.to_string(index=False))


if __name__=="__main__":
    analyze_feature_importance()