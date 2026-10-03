from pathlib import Path
import numpy as np
import pandas as pd

DISCHARGE_COLUMNS = [
    "Station_District",
    "Station_Block",
    "Site",
    "River",
    "Basin",
    "Zero of Gauge (Mts)",
    "Latitude",
    "Longitude",
    "Catchment (Sq KM)",
    "Discharge (M3/s)",
    "Date"
]

WATER_LEVEL_COLUMNS = [
    "Station_District",
    "Station_Block",
    "Site",
    "River",
    "Basin",
    "Zero of Gauge (Mts)",
    "Latitude",
    "Longitude",
    "Catchment (Sq KM)",
    "Discharge (M3/s)",
    "Date"
]


def _validate_columns(df: pd.DataFrame,required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _clean_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"],dayfirst=True,errors="coerce")

    numeric_columns = [
        "Zero of Gauge (Mts)",
        "Latitude",
        "Longitude",
        "Catchment (Sq KM)"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column],errors="coerce")

    df = df.dropna(subset=["Site","Date","Latitude","Longitude"])
    df = df[
        df["Latitude"].between(-90,90) &
        df["Longitude"].between(-180,180)
    ]

    return df


def load_discharge_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Discharge file not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df,DISCHARGE_COLUMNS)
    df = _clean_common_columns(df)

    df["discharge_m3s"] = pd.to_numeric(
        df["Discharge (M3/s)"],
        errors="coerce"
    )

    df = df.dropna(subset=["discharge_m3s"])
    df["discharge_m3s"] = df["discharge_m3s"].clip(lower=0)

    df = df.rename(
        columns={
            "Station_District":"station_district",
            "Station_Block":"station_block",
            "Site":"site",
            "River":"river",
            "Basin":"basin",
            "Zero of Gauge (Mts)":"zero_of_gauge_m",
            "Latitude":"latitude",
            "Longitude":"longitude",
            "Catchment (Sq KM)":"catchment_sq_km",
            "Date":"date"
        }
    )

    return df[
        [
            "station_district",
            "station_block",
            "site",
            "river",
            "basin",
            "zero_of_gauge_m",
            "latitude",
            "longitude",
            "catchment_sq_km",
            "date",
            "discharge_m3s"
        ]
    ].reset_index(drop=True)


def load_water_level_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Water-level file not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df,WATER_LEVEL_COLUMNS)
    df = _clean_common_columns(df)

    df["river_level"] = pd.to_numeric(
        df["Discharge (M3/s)"],
        errors="coerce"
    )

    df = df.dropna(subset=["river_level"])
    df = df[df["river_level"] >= 0]

    df = df.rename(
        columns={
            "Station_District":"station_district",
            "Station_Block":"station_block",
            "Site":"site",
            "River":"river",
            "Basin":"basin",
            "Zero of Gauge (Mts)":"zero_of_gauge_m",
            "Latitude":"latitude",
            "Longitude":"longitude",
            "Catchment (Sq KM)":"catchment_sq_km",
            "Date":"date"
        }
    )

    return df[
        [
            "station_district",
            "station_block",
            "site",
            "river",
            "basin",
            "zero_of_gauge_m",
            "latitude",
            "longitude",
            "catchment_sq_km",
            "date",
            "river_level"
        ]
    ].reset_index(drop=True)


def create_station_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["station_id"] = (
        df["site"].astype(str).str.strip() +
        "_" +
        df["latitude"].round(6).astype(str) +
        "_" +
        df["longitude"].round(6).astype(str)
    )

    return df


def create_river_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id","date"])

    grouped_level = df.groupby("station_id")["river_level"]
    grouped_discharge = df.groupby("station_id")["discharge_m3s"]

    df["river_level_change"] = grouped_level.diff()
    df["river_level_velocity"] = df["river_level_change"]
    df["river_level_3d_change"] = grouped_level.diff(3)
    df["river_level_7d_change"] = grouped_level.diff(7)

    df["river_level_3d_mean"] = grouped_level.transform(
        lambda x: x.rolling(3,min_periods=1).mean()
    )

    df["river_level_7d_mean"] = grouped_level.transform(
        lambda x: x.rolling(7,min_periods=1).mean()
    )

    df["river_level_7d_max"] = grouped_level.transform(
        lambda x: x.rolling(7,min_periods=1).max()
    )

    df["discharge_change"] = grouped_discharge.diff()
    df["discharge_3d_change"] = grouped_discharge.diff(3)
    df["discharge_7d_change"] = grouped_discharge.diff(7)

    df["discharge_7d_mean"] = grouped_discharge.transform(
        lambda x: x.rolling(7,min_periods=1).mean()
    )

    df["discharge_7d_max"] = grouped_discharge.transform(
        lambda x: x.rolling(7,min_periods=1).max()
    )

    return df.reset_index(drop=True)


def process_river_data(
    discharge_path: str | Path,
    water_level_path: str | Path,
    output_path: str | Path
) -> pd.DataFrame:
    print("Loading discharge data...")
    discharge = load_discharge_data(discharge_path)
    print(f"Discharge records: {len(discharge):,}")

    print("Loading water-level data...")
    water_level = load_water_level_data(water_level_path)
    print(f"Water-level records: {len(water_level):,}")

    print("Creating station IDs...")
    discharge = create_station_id(discharge)
    water_level = create_station_id(water_level)

    metadata_columns = [
        "station_id",
        "station_district",
        "station_block",
        "site",
        "river",
        "basin",
        "zero_of_gauge_m",
        "latitude",
        "longitude",
        "catchment_sq_km"
    ]

    metadata = pd.concat(
        [
            water_level[metadata_columns],
            discharge[metadata_columns]
        ],
        ignore_index=True
    )

    metadata = metadata.drop_duplicates(subset=["station_id"])

    print("Merging water level and discharge...")

    df = water_level[
        [
            "station_id",
            "date",
            "river_level"
        ]
    ].merge(
        discharge[
            [
                "station_id",
                "date",
                "discharge_m3s"
            ]
        ],
        on=["station_id","date"],
        how="outer"
    )

    print("Adding station metadata...")

    df = df.merge(
        metadata,
        on="station_id",
        how="left",
        validate="many_to_one"
    )

    df = df[
        [
            "station_id",
            "station_district",
            "station_block",
            "site",
            "river",
            "basin",
            "zero_of_gauge_m",
            "latitude",
            "longitude",
            "catchment_sq_km",
            "date",
            "river_level",
            "discharge_m3s"
        ]
    ]

    df = df.sort_values(["station_id","date"])
    df = df.drop_duplicates(
        subset=["station_id","date"],
        keep="first"
    )

    print("Creating river temporal features...")
    df = create_river_features(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_path,index=False)

    print(f"Saved: {output_path}")
    print(f"Final records: {len(df):,}")
    print(f"Stations: {df['station_id'].nunique():,}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    return df


if __name__ == "__main__":
    process_river_data(
        "data/raw/river/bihar_sw_daily_discharge_manual.csv",
        "data/raw/river/bihar_sw_daily_water_level_manual.csv",
        "data/interim/river_bihar_2020_2025.csv"
    )