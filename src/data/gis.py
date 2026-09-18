from pathlib import Path
import geopandas as gpd

def load_vector_data(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GIS file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"GIS dataset is empty: {path}")
    return gdf

def reproject(gdf: gpd.GeoDataFrame,crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input GIS dataset has no CRS.")
    return gdf.to_crs(crs)

def calculate_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf.copy()
    result["centroid"] = result.geometry.centroid
    return result