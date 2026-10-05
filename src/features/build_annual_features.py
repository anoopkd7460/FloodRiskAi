from pathlib import Path
import pandas as pd
from src.features.aggregation import aggregate_rainfall_dataset

def build_annual_dataset(input_path: str | Path,output_path: str | Path) -> pd.DataFrame:
    input_path=Path(input_path)
    output_path=Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Temporal rainfall file not found: {input_path}")
    print("Loading temporal rainfall data...")
    df=pd.read_csv(input_path,parse_dates=["date"])
    print(f"Input rows: {len(df):,}")
    print("Creating annual rainfall features...")
    result=aggregate_rainfall_dataset(df)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(f"Grid cells: {result['region_id'].nunique():,}")
    print(f"Years: {sorted(result['year'].unique())}")
    print("\nFeature summary:")
    print(result.describe().T.to_string())
    return result

if __name__=="__main__":
    build_annual_dataset(
        "data/interim/rainfall_temporal_bihar_2015_2025.csv",
        "data/interim/rainfall_annual_bihar_2015_2025.csv"
    )