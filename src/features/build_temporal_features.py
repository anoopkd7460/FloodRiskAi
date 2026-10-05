from pathlib import Path
import pandas as pd
from src.features.temporal_features import create_temporal_features

def build_temporal_dataset(input_path: str | Path,output_path: str | Path) -> pd.DataFrame:
    input_path=Path(input_path)
    output_path=Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Rainfall file not found: {input_path}")
    print("Loading rainfall data...")
    df=pd.read_csv(input_path,parse_dates=["date"])
    print(f"Input rows: {len(df):,}")
    print("Creating temporal rainfall features...")
    result=create_temporal_features(df)
    feature_columns=[
        "rainfall_1d","rainfall_3d","rainfall_7d","rainfall_14d",
        "rainfall_3d_mean","rainfall_7d_mean","rainfall_7d_max",
        "rainfall_lag_1d","rainfall_lag_2d","rainfall_lag_3d",
        "rainfall_lag_7d","rainfall_lag_14d"
    ]
    print("\nFeature summary:")
    print(result[feature_columns].describe().to_string())
    print("\nMissing values:")
    print(result[feature_columns].isna().sum().to_string())
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)
    print(f"\nSaved: {output_path}")
    print(f"Output rows: {len(result):,}")
    print(f"Output columns: {len(result.columns):,}")
    return result

if __name__=="__main__":
    build_temporal_dataset(
        "data/interim/rainfall_bihar_2015_2025.csv",
        "data/interim/rainfall_temporal_bihar_2015_2025.csv"
    )