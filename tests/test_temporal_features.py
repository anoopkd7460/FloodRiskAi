import pandas as pd
from src.features.temporal_features import create_rainfall_features,create_rainfall_lag_features

def test_create_rainfall_features():
    df=pd.DataFrame({
        "region_id":["A","A","A","A"],
        "date":pd.date_range("2024-01-01",periods=4),
        "rainfall":[1.0,2.0,3.0,4.0]
    })
    result=create_rainfall_features(df)
    assert result.loc[0,"rainfall_1d"]==1.0
    assert result.loc[2,"rainfall_3d"]==6.0
    assert result.loc[3,"rainfall_3d"]==9.0
    assert result.loc[3,"rainfall_7d"]==10.0

def test_create_rainfall_lag_features():
    df=pd.DataFrame({
        "region_id":["A","A","A"],
        "date":pd.date_range("2024-01-01",periods=3),
        "rainfall":[1.0,2.0,3.0]
    })
    result=create_rainfall_lag_features(df)
    assert pd.isna(result.loc[0,"rainfall_lag_1d"])
    assert result.loc[1,"rainfall_lag_1d"]==1.0
    assert result.loc[2,"rainfall_lag_1d"]==2.0
    assert result.loc[2,"rainfall_lag_2d"]==1.0