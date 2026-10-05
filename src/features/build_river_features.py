from pathlib import Path
import pandas as pd
from src.features.river_aggregation import aggregate_river_features

def build_river_dataset(input_path: str | Path,output_path: str | Path) -> pd.DataFrame:
    input_path=Path(input_path)
    output_path=Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Integrated dataset not found: {input_path}")
    print("Loading integrated data...")
    df=pd.read_csv(input_path,parse_dates=["date"])
    print(f"Input rows: {len(df):,}")
    print("Creating annual river features...")
    result=aggregate_river_features(df)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(f"Grid cells: {result['region_id'].nunique():,}")
    print(f"Years: {sorted(result['year'].unique())}")
    print("\nMissing values:")
    print(result.isna().sum().to_string())
    return result

if __name__=="__main__":
    build_river_dataset(
        "data/interim/integrated_bihar_2015_2025.csv",
        "data/interim/river_annual_bihar_2015_2025.csv"
    )