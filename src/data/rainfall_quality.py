from pathlib import Path
import pandas as pd

def generate_report(path: str | Path) -> None:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    print("=" * 70)
    print("FLOODRISKAI RAINFALL QUALITY REPORT")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Grid cells: {df['region_id'].nunique():,}")
    print(f"Missing values: {df.isna().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Minimum rainfall: {df['rainfall'].min():.2f}")
    print(f"Maximum rainfall: {df['rainfall'].max():.2f}")
    print(f"Mean rainfall: {df['rainfall'].mean():.2f}")
    print(f"Median rainfall: {df['rainfall'].median():.2f}")
    print("=" * 70)
    print("\nYEARLY RECORDS")
    print(df.groupby(df["date"].dt.year).size())
    print("\nYEARLY MEAN RAINFALL")
    print(df.groupby(df["date"].dt.year)["rainfall"].mean().round(2))
    print("\nMISSING VALUES BY COLUMN")
    print(df.isna().sum())

if __name__ == "__main__":
    generate_report(
        "data/interim/rainfall_bihar_2015_2025.csv"
    )