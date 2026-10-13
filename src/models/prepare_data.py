from pathlib import Path
import pandas as pd

BASE_FEATURES=[
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

TARGET="flood_label"

def load_supervised_data(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Supervised dataset not found: {path}")
    df=pd.read_csv(path)
    missing=[column for column in BASE_FEATURES+[TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def create_temporal_datasets(df: pd.DataFrame):
    train=df[df["year"].between(2015,2018)].copy()
    validation=df[df["year"]==2019].copy()
    test=df[df["year"]==2021].copy()

    X_train=train[BASE_FEATURES]
    y_train=train[TARGET].astype(int)

    X_validation=validation[BASE_FEATURES]
    y_validation=validation[TARGET].astype(int)

    X_test=test[BASE_FEATURES]
    y_test=test[TARGET].astype(int)

    print("=== DATASET SHAPES ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test: {X_test.shape}")

    print("\n=== TARGET DISTRIBUTION ===")
    print(f"Train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Validation: {y_validation.value_counts().sort_index().to_dict()}")
    print(f"Test: {y_test.value_counts().sort_index().to_dict()}")

    return X_train,y_train,X_validation,y_validation,X_test,y_test

if __name__=="__main__":
    df=load_supervised_data(
        "data/processed/flood_supervised_grid_year_2015_2021.csv"
    )
    create_temporal_datasets(df)