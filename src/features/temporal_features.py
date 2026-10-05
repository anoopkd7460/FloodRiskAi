import pandas as pd

def create_rainfall_features(df: pd.DataFrame,rainfall_column: str = "rainfall") -> pd.DataFrame:
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])
    df=df.sort_values(["region_id","date"])
    grouped=df.groupby("region_id")[rainfall_column]
    df["rainfall_1d"]=df[rainfall_column]
    df["rainfall_3d"]=grouped.transform(lambda x:x.rolling(3,min_periods=1).sum())
    df["rainfall_7d"]=grouped.transform(lambda x:x.rolling(7,min_periods=1).sum())
    df["rainfall_14d"]=grouped.transform(lambda x:x.rolling(14,min_periods=1).sum())
    df["rainfall_3d_mean"]=grouped.transform(lambda x:x.rolling(3,min_periods=1).mean())
    df["rainfall_7d_mean"]=grouped.transform(lambda x:x.rolling(7,min_periods=1).mean())
    df["rainfall_7d_max"]=grouped.transform(lambda x:x.rolling(7,min_periods=1).max())
    return df.reset_index(drop=True)

def create_rainfall_lag_features(df: pd.DataFrame,rainfall_column: str = "rainfall") -> pd.DataFrame:
    df=df.copy()
    df=df.sort_values(["region_id","date"])
    grouped=df.groupby("region_id")[rainfall_column]
    for lag in [1,2,3,7,14]:
        df[f"rainfall_lag_{lag}d"]=grouped.shift(lag)
    return df.reset_index(drop=True)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])
    df=df.sort_values(["region_id","date"])
    df=create_rainfall_features(df)
    df=create_rainfall_lag_features(df)
    return df.reset_index(drop=True)