import pandas as pd
from src.features.river_aggregation import aggregate_river_features

def test_aggregate_river_features():
    df=pd.DataFrame({
        "region_id":["A"]*3,
        "date":pd.date_range("2024-01-01",periods=3),
        "river_observation_available":[1,1,0],
        "river_level":[10.0,15.0,None],
        "discharge_m3s":[100.0,150.0,None],
        "river_level_change":[1.0,5.0,None],
        "river_level_3d_change":[1.0,5.0,None],
        "river_level_7d_change":[1.0,5.0,None],
        "discharge_change":[10.0,50.0,None],
        "discharge_3d_change":[10.0,50.0,None],
        "discharge_7d_change":[10.0,50.0,None]
    })
    result=aggregate_river_features(df)
    assert len(result)==1
    assert result.loc[0,"river_observation_days"]==2
    assert result.loc[0,"river_observation_rate"]==(2/3)
    assert result.loc[0,"annual_mean_river_level"]==12.5
    assert result.loc[0,"annual_max_river_level"]==15.0
    assert result.loc[0,"annual_mean_discharge"]==125.0
    assert result.loc[0,"annual_max_discharge"]==150.0