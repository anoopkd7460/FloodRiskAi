
from pathlib import Path
import pandas as pd

def run_target_quality_audit():
    path=Path("data/processed/flood_supervised_grid_year_2015_2021.csv")
    if not path.exists():
        raise FileNotFoundError(f"Supervised dataset not found: {path}")
    df=pd.read_csv(path)
    required={"region_id","year","flood_label"}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df=df.dropna(subset=["region_id","year","flood_label"]).copy()
    df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
    df["flood_label"]=pd.to_numeric(df["flood_label"],errors="raise").astype(int)
    if not df["flood_label"].isin([0,1]).all():
        raise ValueError("Flood labels must be binary.")
    if df.duplicated(["region_id","year"]).any():
        raise ValueError("Duplicate region-year records found.")
    print("=== LABEL COVERAGE BY YEAR ===")
    print(df.groupby("year").agg(
        grid_cells=("region_id","nunique"),
        flooded_cells=("flood_label","sum"),
        flood_rate=("flood_label","mean")
    ).to_string())
    region_summary=df.groupby("region_id").agg(
        labeled_years=("flood_label","count"),
        flooded_years=("flood_label","sum"),
        flood_rate=("flood_label","mean")
    )
    region_summary["persistent_flood_label"]=(region_summary["flood_rate"]==1).astype(int)
    region_summary["never_flood_label"]=(region_summary["flood_rate"]==0).astype(int)
    print("\n=== LABEL CONSISTENCY ACROSS AVAILABLE YEARS ===")
    print(f"Grid cells: {len(region_summary)}")
    print(f"Always labeled flooded: {region_summary['persistent_flood_label'].sum()}")
    print(f"Never labeled flooded: {region_summary['never_flood_label'].sum()}")
    print(f"Changing labels: {((region_summary['flood_rate']>0)&(region_summary['flood_rate']<1)).sum()}")
    print(f"Mean per-region flood rate: {region_summary['flood_rate'].mean():.4f}")
    print("\n=== REGION SUMMARY ===")
    print(region_summary.describe().round(4).to_string())
    output_path=Path("reports/phase5_target_quality_by_region.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    region_summary.to_csv(output_path)
    print(f"\nSaved region audit: {output_path}")
    return region_summary

if __name__=="__main__":
    run_target_quality_audit()
