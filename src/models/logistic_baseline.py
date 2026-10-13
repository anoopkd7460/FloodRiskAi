from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import load_supervised_data,create_temporal_datasets

def evaluate_model(model,X,y,name: str) -> dict:
    predictions=model.predict(X)
    probabilities=model.predict_proba(X)[:,1]
    tn,fp,fn,tp=confusion_matrix(y,predictions).ravel()
    metrics={
        "dataset":name,
        "accuracy":accuracy_score(y,predictions),
        "precision":precision_score(y,predictions,zero_division=0),
        "recall":recall_score(y,predictions,zero_division=0),
        "f1":f1_score(y,predictions,zero_division=0),
        "roc_auc":roc_auc_score(y,probabilities),
        "pr_auc":average_precision_score(y,probabilities),
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp
    }
    return metrics

def train_logistic_baseline():
    data_path=Path("data/processed/flood_supervised_grid_year_2015_2021.csv")
    df=load_supervised_data(data_path)
    X_train,y_train,X_validation,y_validation,X_test,y_test=create_temporal_datasets(df)

    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
        ("classifier",LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training Logistic Regression...")
    model.fit(X_train,y_train)

    results=[
        evaluate_model(model,X_train,y_train,"train"),
        evaluate_model(model,X_validation,y_validation,"validation"),
        evaluate_model(model,X_test,y_test,"test")
    ]

    result_df=pd.DataFrame(results)

    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(result_df.to_string(index=False))

    return model,result_df

if __name__=="__main__":
    train_logistic_baseline()