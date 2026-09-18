from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN = 24.0
LAT_MAX = 27.5
LON_MIN = 83.0
LON_MAX = 88.5

def get_rainfall_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    files = sorted(input_dir.glob("RF25_ind*_rfp25.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF rainfall files found in {input_dir}")
    return files

def load_netcdf(path: str | Path) -> xr.Dataset:
    path = Path(path)
    try:
        return xr.open_dataset(path,engine="netcdf4")
    except Exception as exc:
        raise RuntimeError(f"Unable to open NetCDF file {path.name}: {exc}") from exc

def validate_dataset_structure(dataset: xr.Dataset) -> None:
    required_dimensions = {"TIME","LATITUDE","LONGITUDE"}
    required_variables = {"RAINFALL"}
    missing_dimensions = required_dimensions - set(dataset.dims)
    missing_variables = required_variables - set(dataset.data_vars)
    if missing_dimensions:
        raise ValueError(f"Missing dimensions: {missing_dimensions}")
    if missing_variables:
        raise ValueError(f"Missing variables: {missing_variables}")

def extract_bihar_data(dataset: xr.Dataset) -> pd.DataFrame:
    subset = dataset.sel(
        LATITUDE=slice(LAT_MIN,LAT_MAX),
        LONGITUDE=slice(LON_MIN,LON_MAX)
    )
    df = subset["RAINFALL"].to_dataframe(name="rainfall").reset_index()
    df = df.rename(columns={
        "TIME":"date",
        "LATITUDE":"latitude",
        "LONGITUDE":"longitude"
    })
    return df

def clean_rainfall_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"],errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"],errors="coerce")
    df["rainfall"] = pd.to_numeric(df["rainfall"],errors="coerce")
    df = df.dropna(subset=["date","latitude","longitude","rainfall"])
    df = df.replace([np.inf,-np.inf],np.nan)
    df = df.dropna(subset=["rainfall"])
    df["rainfall"] = df["rainfall"].clip(lower=0)
    df = df[
        df["latitude"].between(LAT_MIN,LAT_MAX) &
        df["longitude"].between(LON_MIN,LON_MAX)
    ]
    return df.reset_index(drop=True)

def create_region_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["region_id"] = (
        "BHR_" +
        df["latitude"].map(lambda x: f"{x:.2f}") +
        "_" +
        df["longitude"].map(lambda x: f"{x:.2f}")
    )
    return df

def process_file(path: Path) -> pd.DataFrame:
    print(f"Processing: {path.name}")
    dataset = load_netcdf(path)
    try:
        validate_dataset_structure(dataset)
        df = extract_bihar_data(dataset)
    finally:
        dataset.close()
    df = clean_rainfall_data(df)
    df = create_region_id(df)
    print(f"Records: {len(df):,}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Grid cells: {df['region_id'].nunique():,}")
    return df

def process_all_files(input_dir: str | Path) -> pd.DataFrame:
    files = get_rainfall_files(input_dir)
    frames = []
    for path in files:
        try:
            frames.append(process_file(path))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if not frames:
        raise RuntimeError("No rainfall files could be processed.")
    df = pd.concat(frames,ignore_index=True)
    df = df.sort_values(["region_id","date"])
    df = df.drop_duplicates(
        subset=["region_id","date"],
        keep="first"
    )
    return df.reset_index(drop=True)

def save_rainfall_data(df: pd.DataFrame,path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)
    print(f"Saved: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    input_dir = Path("data/raw/rainfall")
    output_file = Path("data/interim/rainfall_bihar_2015_2025.csv")
    rainfall = process_all_files(input_dir)
    save_rainfall_data(rainfall,output_file)