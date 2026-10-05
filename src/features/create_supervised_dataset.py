from pathlib import Path
import pandas as pd

def create_supervised_dataset(input_path: str | Path,output_path: str | Path) -> pd.DataFrame:
    input_path=Path(input_path)
    output_path=Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {input_path}")

    df=pd.read_csv(input_path)
    labeled=df[df["flood_label"].notna()].copy()
    labeled["flood_label"]=labeled["flood_label"].astype(int)
    labeled=labeled.sort_values(["year","region_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True,exist_ok=True)
    labeled.to_csv(output_path,index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(labeled):,}")
    print(f"Columns: {len(labeled.columns):,}")
    print(f"Grid cells: {labeled['region_id'].nunique():,}")
    print(f"Years: {sorted(labeled['year'].unique())}")
    print("\nClass distribution:")
    print(labeled["flood_label"].value_counts().sort_index().to_string())
    print("\nYear distribution:")
    print(labeled.groupby("year")["flood_label"].agg(
        samples="count",
        flooded="sum",
        flood_rate="mean"
    ).to_string())

    return labeled

if __name__=="__main__":
    create_supervised_dataset(
        "data/interim/modeling_bihar_grid_year_2015_2025.csv",
        "data/processed/flood_supervised_grid_year_2015_2021.csv"
    )