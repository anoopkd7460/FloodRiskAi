import pandas as pd

from src.data.integration import merge_rainfall_river
from src.data.integration import (
    create_grid_station_mapping,
    merge_flood_labels
)

def test_create_grid_station_mapping():
    rainfall=pd.DataFrame({
        "region_id":["A","B"],
        "latitude":[25.00,25.50],
        "longitude":[85.00,85.50]
    })

    river=pd.DataFrame({
        "station_id":["S1","S2"],
        "latitude":[25.01,25.49],
        "longitude":[85.01,85.49]
    })

    result=create_grid_station_mapping(
        rainfall,
        river
    )

    assert len(result)==2
    assert set(result["station_id"])=={"S1","S2"}
    assert (result["river_station_distance_km"]>=0).all()

def test_merge_flood_labels():
    environmental=pd.DataFrame({
        "region_id":["A","A","B"],
        "date":pd.to_datetime([
            "2015-07-01",
            "2016-07-01",
            "2015-07-01"
        ])
    })

    flood=pd.DataFrame({
        "region_id":["A","A","B"],
        "year":[2015,2016,2015],
        "flood_label":[1,0,1]
    })

    result=merge_flood_labels(
        environmental,
        flood
    )

    assert result["flood_label"].tolist()==[1,0,1]

def test_merge_rainfall_river_observation_flag():
    rainfall=pd.DataFrame({
        "region_id":["A"],
        "date":pd.to_datetime(["2024-01-01"]),
        "latitude":[25.00],
        "longitude":[85.00]
    })

    river=pd.DataFrame({
        "station_id":["S1"],
        "latitude":[25.01],
        "longitude":[85.01],
        "date":pd.to_datetime(["2024-01-01"]),
        "river_level":[10.0],
        "discharge_m3s":[100.0],
        "river_level_change":[1.0],
        "river_level_velocity":[1.0],
        "river_level_3d_change":[2.0],
        "river_level_7d_change":[3.0],
        "river_level_3d_mean":[9.0],
        "river_level_7d_mean":[8.0],
        "river_level_7d_max":[12.0],
        "discharge_change":[10.0],
        "discharge_3d_change":[20.0],
        "discharge_7d_change":[30.0],
        "discharge_7d_mean":[90.0],
        "discharge_7d_max":[120.0]
    })

    result=merge_rainfall_river(rainfall,river)

    assert result.loc[0,"river_observation_available"]==1