import pandas as pd

def prepare_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    return df

def merge_rainfall_flood(
    rainfall: pd.DataFrame,
    flood: pd.DataFrame
) -> pd.DataFrame:
    rainfall = prepare_date(rainfall)
    flood = prepare_date(flood)
    return rainfall.merge(
        flood,
        on=["region_id","date"],
        how="left"
    )

def merge_environmental_data(
    rainfall: pd.DataFrame,
    river: pd.DataFrame,
    flood: pd.DataFrame
) -> pd.DataFrame:
    rainfall = prepare_date(rainfall)
    river = prepare_date(river)
    flood = prepare_date(flood)
    df = rainfall.merge(
        river,
        on="date",
        how="left"
    )
    df = df.merge(
        flood,
        on=["region_id","date"],
        how="left"
    )
    return df