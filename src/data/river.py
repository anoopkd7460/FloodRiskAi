from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["station_id","date","river_level"]

def load_river_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"River file not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing river columns: {missing}")
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    df["river_level"] = pd.to_numeric(df["river_level"],errors="coerce")
    df = df.dropna(subset=["station_id","date","river_level"])
    return df.reset_index(drop=True)

def create_river_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id","date"])
    grouped = df.groupby("station_id")["river_level"]
    df["river_level_change"] = grouped.diff()
    df["river_level_velocity"] = df["river_level_change"]
    df["river_level_3d_change"] = grouped.diff(3)
    df["river_level_7d_change"] = grouped.diff(7)
    df["river_level_3d_mean"] = grouped.transform(lambda x: x.rolling(3,min_periods=1).mean())
    df["river_level_7d_mean"] = grouped.transform(lambda x: x.rolling(7,min_periods=1).mean())
    df["river_level_7d_max"] = grouped.transform(lambda x: x.rolling(7,min_periods=1).max())
    return df.reset_index(drop=True)