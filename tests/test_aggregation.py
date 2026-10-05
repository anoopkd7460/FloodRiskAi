import pandas as pd
from src.features.aggregation import aggregate_rainfall_dataset

def test_aggregate_rainfall_dataset():
    df=pd.DataFrame({
        "region_id":["A"]*4,
        "date":pd.to_datetime([
            "2024-06-01",
            "2024-07-01",
            "2024-10-01",
            "2024-11-01"
        ]),
        "latitude":[25.0]*4,
        "longitude":[85.0]*4,
        "rainfall":[10.0,60.0,120.0,0.0],
        "rainfall_3d":[10.0,70.0,180.0,120.0],
        "rainfall_7d":[10.0,70.0,190.0,120.0],
        "rainfall_14d":[10.0,70.0,190.0,120.0],
        "rainfall_7d_max":[10.0,70.0,190.0,190.0]
    })
    result=aggregate_rainfall_dataset(df)
    assert len(result)==1
    assert result.loc[0,"annual_rainfall_total"]==190.0
    assert result.loc[0,"monsoon_rainfall_total"]==70.0
    assert result.loc[0,"heavy_rain_days"]==2
    assert result.loc[0,"very_heavy_rain_days"]==2
    assert result.loc[0,"extreme_rain_days"]==1