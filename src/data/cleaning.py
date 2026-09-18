import pandas as pd

def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean().values * 100).round(2),
        "unique_values": [df[column].nunique() for column in df.columns]
    })

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["latitude"].between(-90,90)]
    df = df[df["longitude"].between(-180,180)]
    return df.reset_index(drop=True)

def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)