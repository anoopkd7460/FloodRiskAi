from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["region_id","date","flood_label"]

def load_flood_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flood file not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing flood columns: {missing}")
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    df["flood_label"] = pd.to_numeric(df["flood_label"],errors="coerce")
    df = df.dropna(subset=["region_id","date","flood_label"])
    df["flood_label"] = df["flood_label"].astype(int)
    invalid = ~df["flood_label"].isin([0,1])
    if invalid.any():
        raise ValueError("Flood label must contain only 0 or 1.")
    return df.reset_index(drop=True)

def load_flood_extent(path: str | Path):
    import geopandas as gpd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flood extent file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Flood extent dataset is empty.")
    return gdf