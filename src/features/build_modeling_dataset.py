from pathlib import Path
import pandas as pd

def build_modeling_dataset(
    rainfall_path: str | Path,
    river_path: str | Path,
    flood_path: str | Path,
    output_path: str | Path
) -> pd.DataFrame:
    rainfall_path=Path(rainfall_path)
    river_path=Path(river_path)
    flood_path=Path(flood_path)
    output_path=Path(output_path)

    for path,label in [
        (rainfall_path,"Rainfall"),
        (river_path,"River"),
        (flood_path,"Flood")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    print("Loading annual rainfall features...")
    rainfall=pd.read_csv(rainfall_path)

    print("Loading annual river features...")
    river=pd.read_csv(river_path)

    print("Loading flood labels...")
    flood=pd.read_csv(flood_path)

    flood["year"]=pd.to_numeric(flood["year"],errors="coerce")
    flood["flood_label"]=pd.to_numeric(flood["flood_label"],errors="coerce")
    flood=flood.dropna(subset=["region_id","year","flood_label"])
    flood["year"]=flood["year"].astype(int)

    print("Merging rainfall and river features...")
    result=rainfall.merge(
        river,
        on=["region_id","year"],
        how="left",
        validate="one_to_one"
    )

    print("Adding flood labels...")
    result=result.merge(
        flood[["region_id","year","flood_label"]],
        on=["region_id","year"],
        how="left",
        validate="one_to_one"
    )

    result=result.sort_values(["region_id","year"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(f"Grid cells: {result['region_id'].nunique():,}")
    print(f"Years: {sorted(result['year'].unique())}")

    print("\nFlood label coverage:")
    print(result.groupby("year")["flood_label"].agg(
        records="count",
        flooded="sum"
    ).to_string())

    print("\nMissing flood labels:")
    print(f"{result['flood_label'].isna().sum():,}")

    print("\nDuplicate grid-year records:")
    print(result.duplicated(["region_id","year"]).sum())

    return result

if __name__=="__main__":
    build_modeling_dataset(
        "data/interim/rainfall_annual_bihar_2015_2025.csv",
        "data/interim/river_annual_bihar_2015_2025.csv",
        "data/interim/flood_labels_bihar_2015_2021.csv",
        "data/interim/modeling_bihar_grid_year_2015_2025.csv"
    )