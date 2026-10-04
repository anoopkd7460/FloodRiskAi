from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

def prepare_date(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    return df

def create_grid_station_mapping(
    rainfall: pd.DataFrame,
    river: pd.DataFrame
) -> pd.DataFrame:
    rainfall_grid=rainfall[
        ["region_id","latitude","longitude"]
    ].drop_duplicates("region_id").reset_index(drop=True)

    stations=river[
        ["station_id","latitude","longitude"]
    ].dropna().drop_duplicates("station_id").reset_index(drop=True)

    if rainfall_grid.empty:
        raise ValueError("Rainfall grid is empty.")

    if stations.empty:
        raise ValueError("River station data is empty.")

    grid_coordinates=np.radians(
        rainfall_grid[["latitude","longitude"]].to_numpy()
    )

    station_coordinates=np.radians(
        stations[["latitude","longitude"]].to_numpy()
    )

    tree=BallTree(station_coordinates,metric="haversine")
    distances,indices=tree.query(grid_coordinates,k=1)

    rainfall_grid["station_id"]=stations.iloc[
        indices[:,0]
    ]["station_id"].to_numpy()

    rainfall_grid["river_station_distance_km"]=(
        distances[:,0]*6371.0088
    )

    return rainfall_grid[
        [
            "region_id",
            "station_id",
            "river_station_distance_km"
        ]
    ]

def merge_rainfall_river(
    rainfall: pd.DataFrame,
    river: pd.DataFrame
) -> pd.DataFrame:
    rainfall=prepare_date(rainfall)
    river=prepare_date(river)

    mapping=create_grid_station_mapping(rainfall,river)

    rainfall=rainfall.merge(
        mapping,
        on="region_id",
        how="left",
        validate="many_to_one"
    )

    river_columns=[
        "station_id",
        "date",
        "river_level",
        "discharge_m3s",
        "river_level_change",
        "river_level_velocity",
        "river_level_3d_change",
        "river_level_7d_change",
        "river_level_3d_mean",
        "river_level_7d_mean",
        "river_level_7d_max",
        "discharge_change",
        "discharge_3d_change",
        "discharge_7d_change",
        "discharge_7d_mean",
        "discharge_7d_max"
    ]

    river=river[river_columns].drop_duplicates(
        ["station_id","date"]
    )

    result=rainfall.merge(
        river,
        on=["station_id","date"],
        how="left",
        validate="many_to_one"
    )

    result["river_observation_available"]=(
        result["river_level"].notna().astype(int)
    )
    
    return result

def merge_flood_labels(
    environmental: pd.DataFrame,
    flood: pd.DataFrame
) -> pd.DataFrame:
    environmental=prepare_date(environmental)
    flood=flood.copy()

    flood["year"]=pd.to_numeric(
        flood["year"],
        errors="coerce"
    )

    flood=flood.dropna(
        subset=["region_id","year","flood_label"]
    )

    flood["year"]=flood["year"].astype(int)

    environmental["year"]=environmental["date"].dt.year

    result=environmental.merge(
        flood[
            [
                "region_id",
                "year",
                "flood_label"
            ]
        ],
        on=["region_id","year"],
        how="left",
        validate="many_to_one"
    )

    return result

def create_integrated_dataset(
    rainfall_path: str | Path,
    river_path: str | Path,
    flood_path: str | Path,
    output_path: str | Path
) -> pd.DataFrame:
    rainfall_path=Path(rainfall_path)
    river_path=Path(river_path)
    flood_path=Path(flood_path)
    output_path=Path(output_path)

    if not rainfall_path.exists():
        raise FileNotFoundError(
            f"Rainfall file not found: {rainfall_path}"
        )

    if not river_path.exists():
        raise FileNotFoundError(
            f"River file not found: {river_path}"
        )

    if not flood_path.exists():
        raise FileNotFoundError(
            f"Flood file not found: {flood_path}"
        )

    print("Loading rainfall...")
    rainfall=pd.read_csv(rainfall_path)

    print("Loading river data...")
    river=pd.read_csv(river_path)

    print("Loading flood labels...")
    flood=pd.read_csv(flood_path)

    print("Integrating rainfall and river data...")
    environmental=merge_rainfall_river(
        rainfall,
        river
    )

    print("Adding flood labels...")
    integrated=merge_flood_labels(
        environmental,
        flood
    )

    integrated=integrated.sort_values(
        ["region_id","date"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    integrated.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(integrated):,}")
    print(f"Columns: {len(integrated.columns):,}")
    print(f"Grid cells: {integrated['region_id'].nunique():,}")
    print(
        f"Date range: "
        f"{integrated['date'].min()} → "
        f"{integrated['date'].max()}"
    )

    return integrated

if __name__=="__main__":
    create_integrated_dataset(
        "data/interim/rainfall_bihar_2015_2025.csv",
        "data/interim/river_bihar_2020_2025.csv",
        "data/interim/flood_labels_bihar_2015_2021.csv",
        "data/interim/integrated_bihar_2015_2025.csv"
    )