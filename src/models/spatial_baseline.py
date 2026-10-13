from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.prepare_data import load_supervised_data,create_temporal_datasets

SPATIAL_FEATURES=["latitude","longitude"]

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


def train_spatial_baseline():
    df=load_supervised_data(Path("data/processed/flood_supervised_grid_year_2015_2021.csv"))

    train=df[df["year"].between(2015,2018)].copy()
    validation=df[df["year"]==2019].copy()
    test=df[df["year"]==2021].copy()

    X_train=train[SPATIAL_FEATURES]
    y_train=train["flood_label"].astype(int)
    X_validation=validation[SPATIAL_FEATURES]
    y_validation=validation["flood_label"].astype(int)
    X_test=test[SPATIAL_FEATURES]
    y_test=test["flood_label"].astype(int)

    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("classifier",DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,class_weight="balanced",random_state=42))
    ])

    print("Training Spatial-Only Decision Tree...")
    model.fit(X_train,y_train)

    results =[
        evaluate(model,X_train,y_train,"trian"),
        evaluate(model,X_validation,y_validation,"validation"),
        evaluate(model,X_test,y_test,"test")
    ]

    
    print("\n=== SPATIAL-ONLY RESULTS ===")
    print(pd.DataFrame(results).to_string(index=False))

if __name__=="__main__":
    train_spatial_baseline()