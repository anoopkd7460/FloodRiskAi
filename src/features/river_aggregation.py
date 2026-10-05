import pandas as pd

RIVER_COLUMNS=[
    "river_level",
    "discharge_m3s",
    "river_level_change",
    "river_level_3d_change",
    "river_level_7d_change",
    "discharge_change",
    "discharge_3d_change",
    "discharge_7d_change"
]

def aggregate_river_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])
    df["year"]=df["date"].dt.year
    result=df.groupby(["region_id","year"],as_index=False).agg(
        river_observation_days=("river_observation_available","sum"),
        river_observation_rate=("river_observation_available","mean"),
        annual_mean_river_level=("river_level","mean"),
        annual_max_river_level=("river_level","max"),
        annual_mean_discharge=("discharge_m3s","mean"),
        annual_max_discharge=("discharge_m3s","max"),
        max_river_level_change=("river_level_change","max"),
        max_river_level_3d_change=("river_level_3d_change","max"),
        max_river_level_7d_change=("river_level_7d_change","max"),
        max_discharge_change=("discharge_change","max"),
        max_discharge_3d_change=("discharge_3d_change","max"),
        max_discharge_7d_change=("discharge_7d_change","max")
    )
    return result