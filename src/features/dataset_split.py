from pathlib import Path
import pandas as pd

def create_temporal_split(
    input_path: str | Path
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    input_path=Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Supervised dataset not found: {input_path}")

    df=pd.read_csv(input_path)
    train=df[df["year"].between(2015,2018)].copy()
    validation=df[df["year"]==2019].copy()
    test=df[df["year"]==2021].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more temporal splits are empty.")

    print("=== TEMPORAL SPLIT ===")
    print(f"Train: {len(train):,} samples | Years: {sorted(train['year'].unique())}")
    print(f"Validation: {len(validation):,} samples | Years: {sorted(validation['year'].unique())}")
    print(f"Test: {len(test):,} samples | Years: {sorted(test['year'].unique())}")

    print("\n=== CLASS DISTRIBUTION ===")
    print(f"Train flood rate: {train['flood_label'].mean()*100:.2f}%")
    print(f"Validation flood rate: {validation['flood_label'].mean()*100:.2f}%")
    print(f"Test flood rate: {test['flood_label'].mean()*100:.2f}%")

    return train,validation,test

if __name__=="__main__":
    create_temporal_split(
        "data/processed/flood_supervised_grid_year_2015_2021.csv"
    )