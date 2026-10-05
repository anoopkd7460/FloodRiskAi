import pandas as pd

MONSOON_MONTHS=[6,7,8,9]

def add_temporal_labels(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])
    df["year"]=df["date"].dt.year
    df["month"]=df["date"].dt.month
    df["is_monsoon"]=df["month"].isin(MONSOON_MONTHS).astype(int)
    return df

def aggregate_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    df=add_temporal_labels(df)
    df["monsoon_rainfall"]=df["rainfall"].where(df["is_monsoon"]==1)
    df["monsoon_rainfall_3d"]=df["rainfall_3d"].where(df["is_monsoon"]==1)
    df["monsoon_rainfall_7d"]=df["rainfall_7d"].where(df["is_monsoon"]==1)
    df["monsoon_rainfall_14d"]=df["rainfall_14d"].where(df["is_monsoon"]==1)
    df["monsoon_rainfall_7d_max"]=df["rainfall_7d_max"].where(df["is_monsoon"]==1)
    result=df.groupby(["region_id","year"],as_index=False).agg(
        latitude=("latitude","first"),
        longitude=("longitude","first"),
        annual_rainfall_total=("rainfall","sum"),
        annual_rainfall_mean=("rainfall","mean"),
        annual_rainfall_max=("rainfall","max"),
        annual_rainfall_3d_max=("rainfall_3d","max"),
        annual_rainfall_7d_max=("rainfall_7d","max"),
        annual_rainfall_14d_max=("rainfall_14d","max"),
        monsoon_rainfall_total=("monsoon_rainfall","sum"),
        monsoon_rainfall_mean=("monsoon_rainfall","mean"),
        monsoon_rainfall_max=("monsoon_rainfall","max"),
        monsoon_rainfall_3d_max=("monsoon_rainfall_3d","max"),
        monsoon_rainfall_7d_max=("monsoon_rainfall_7d","max"),
        monsoon_rainfall_14d_max=("monsoon_rainfall_14d","max")
    )
    return result

def create_rainfall_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df=add_temporal_labels(df)
    result=df.groupby(["region_id","year"],as_index=False).agg(
        heavy_rain_days=("rainfall",lambda x:(x>=20).sum()),
        very_heavy_rain_days=("rainfall",lambda x:(x>=50).sum()),
        extreme_rain_days=("rainfall",lambda x:(x>=100).sum()),
        rainy_days=("rainfall",lambda x:(x>0).sum())
    )
    return result

def aggregate_rainfall_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rainfall=aggregate_rainfall_features(df)
    events=create_rainfall_event_features(df)
    result=rainfall.merge(events,on=["region_id","year"],how="left",validate="one_to_one")
    return result