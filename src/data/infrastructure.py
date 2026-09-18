from pathlib import Path
import geopandas as gpd

def load_infrastructure(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Infrastructure file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Infrastructure dataset is empty.")
    return gdf

def count_features_by_region(
    infrastructure: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    region_column: str = "region_id"
) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        infrastructure,
        regions[[region_column,"geometry"]],
        how="left",
        predicate="within"
    )
    counts = joined.groupby(region_column).size().reset_index(name="feature_count")
    return regions.merge(counts,on=region_column,how="left").fillna({"feature_count":0})