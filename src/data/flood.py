from pathlib import Path
import json
import pandas as pd
from shapely.geometry import box,shape
from shapely.strtree import STRtree

REQUIRED_COLUMNS=["region_id","date","flood_label"]

def load_flood_csv(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flood file not found: {path}")
    df=pd.read_csv(path)
    missing=[column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing flood columns: {missing}")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df["flood_label"]=pd.to_numeric(df["flood_label"],errors="coerce")
    df=df.dropna(subset=["region_id","date","flood_label"])
    df["flood_label"]=df["flood_label"].astype(int)
    if not df["flood_label"].isin([0,1]).all():
        raise ValueError("Flood label must contain only 0 or 1.")
    return df.reset_index(drop=True)

def load_flood_extent(path: str | Path):
    import geopandas as gpd
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flood extent file not found: {path}")
    gdf=gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Flood extent dataset is empty.")
    return gdf

def load_rainfall_grid(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rainfall file not found: {path}")
    df=pd.read_csv(path,usecols=["region_id","latitude","longitude"])
    df=df.drop_duplicates("region_id")
    return df.reset_index(drop=True)

def create_grid_cells(grid: pd.DataFrame,cell_size: float = 0.25) -> list:
    half=cell_size/2
    return [
        box(
            row.longitude-half,
            row.latitude-half,
            row.longitude+half,
            row.latitude+half
        )
        for row in grid.itertuples()
    ]

def create_flood_labels(
    flood_path: str | Path,
    rainfall_path: str | Path,
    output_path: str | Path,
    start_year: int = 2015,
    end_year: int = 2021
) -> pd.DataFrame:
    flood_path=Path(flood_path)
    rainfall_path=Path(rainfall_path)
    output_path=Path(output_path)

    if not flood_path.exists():
        raise FileNotFoundError(f"Flood extent file not found: {flood_path}")

    grid=load_rainfall_grid(rainfall_path)
    grid_geometries=create_grid_cells(grid)
    tree=STRtree(grid_geometries)

    years=set(range(start_year,end_year+1))
    available_years=set()
    flooded=set()
    total_features=0

    with open(flood_path,encoding="utf-8") as file:
        for line in file:
            feature=json.loads(line)
            properties=feature.get("properties",{})
            year_value=properties.get("year")

            if year_value is None:
                continue

            year=int(year_value)

            if year not in years:
                continue

            available_years.add(year)
            geometry=feature.get("geometry")

            if geometry is None:
                continue

            flood_geometry=shape(geometry)
            if flood_geometry.is_empty:
                continue

            total_features+=1

            matches=tree.query(flood_geometry,predicate="intersects")

            for index in matches:
                flooded.add((year,int(index)))

    rows=[]

    for year in sorted(available_years):
        for index,row in grid.iterrows():
            label=int((year,index) in flooded)
            rows.append({
                "region_id":row["region_id"],
                "year":year,
                "flood_label":label
            })

    result=pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)

    print(f"Flood features processed: {total_features:,}")
    print(f"Available years: {sorted(available_years)}")
    print(f"Grid cells: {grid['region_id'].nunique():,}")
    print(f"Label records: {len(result):,}")
    print(f"Flooded records: {result['flood_label'].sum():,}")
    print(f"Saved: {output_path}")

    return result

if __name__=="__main__":
    create_flood_labels(
        "data/raw/flood/NDEM_BR_Yearly_Aggregate_Flood_Innundation_1998_to_2021.geojsonl",
        "data/interim/rainfall_bihar_2015_2025.csv",
        "data/interim/flood_labels_bihar_2015_2021.csv"
    )