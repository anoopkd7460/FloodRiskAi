from pathlib import Path
import pandas as pd

def load_integrated_data(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Integrated dataset not found: {path}")
    df=pd.read_csv(path,parse_dates=["date"])
    return df

def basic_summary(df: pd.DataFrame) -> None:
    print("\n=== BASIC SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Grid cells: {df['region_id'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print("\nColumns:")
    print(df.columns.tolist())

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    result=pd.DataFrame({
        "missing_count":df.isna().sum(),
        "missing_pct":df.isna().mean()*100
    })
    result=result.sort_values("missing_pct",ascending=False)
    print("\n=== MISSING VALUES ===")
    print(result.to_string())
    return result

def flood_summary(df: pd.DataFrame) -> pd.DataFrame:
    labeled=df[df["flood_label"].notna()].copy()
    labeled["year"]=labeled["date"].dt.year
    yearly=labeled.groupby("year").agg(
        records=("flood_label","size"),
        flooded=("flood_label","sum"),
        flood_rate=("flood_label","mean")
    ).reset_index()
    yearly["flood_rate_pct"]=yearly["flood_rate"]*100
    print("\n=== FLOOD LABEL SUMMARY ===")
    print(f"Total labeled records: {len(labeled):,}")
    print(f"Flooded records: {int(labeled['flood_label'].sum()):,}")
    print(f"Overall flood rate: {labeled['flood_label'].mean()*100:.2f}%")
    print("\nYear-wise:")
    print(yearly.to_string(index=False))
    return yearly

def rainfall_summary(df: pd.DataFrame) -> None:
    print("\n=== RAINFALL SUMMARY ===")
    print(df["rainfall"].describe().to_string())
    yearly=df.groupby(df["date"].dt.year)["rainfall"].agg(
        mean="mean",
        max="max",
        total="sum"
    ).reset_index()
    print("\nYear-wise rainfall:")
    print(yearly.to_string(index=False))

def river_summary(df: pd.DataFrame) -> None:
    print("\n=== RIVER SUMMARY ===")
    if "river_observation_available" in df.columns:
        yearly=df.groupby(df["date"].dt.year)["river_observation_available"].mean().reset_index()
        yearly["coverage_pct"]=yearly["river_observation_available"]*100
        print(yearly[["date","coverage_pct"]].to_string(index=False))
    print(f"River-level observations: {df['river_level'].notna().sum():,}")
    print(f"Discharge observations: {df['discharge_m3s'].notna().sum():,}")

def spatial_flood_summary(df: pd.DataFrame) -> pd.DataFrame:
    labeled=df[df["flood_label"].notna()].copy()
    result=labeled.groupby("region_id").agg(
        latitude=("latitude","first"),
        longitude=("longitude","first"),
        flood_count=("flood_label","sum"),
        years_observed=("flood_label","count"),
        flood_rate=("flood_label","mean")
    ).reset_index()
    result=result.sort_values("flood_rate",ascending=False)
    print("\n=== SPATIAL FLOOD SUMMARY ===")
    print(result.head(20).to_string(index=False))
    return result

def run_eda(path: str | Path) -> None:
    df=load_integrated_data(path)
    basic_summary(df)
    missing_summary(df)
    flood_summary(df)
    rainfall_summary(df)
    river_summary(df)
    spatial_flood_summary(df)

if __name__=="__main__":
    run_eda("data/interim/integrated_bihar_2015_2025.csv")