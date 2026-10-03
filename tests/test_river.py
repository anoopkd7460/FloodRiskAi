import pandas as pd

from src.data.river import (
    create_station_id,
    create_river_features
)


def test_create_station_id():
    df = pd.DataFrame({
        "site": ["Patna"],
        "latitude": [25.5941],
        "longitude": [85.1376]
    })

    result = create_station_id(df)

    assert result.loc[0, "station_id"] == "Patna_25.5941_85.1376"


def test_create_river_features():
    df = pd.DataFrame({
        "station_id": ["A", "A", "A"],
        "date": pd.date_range(
            "2024-01-01",
            periods=3
        ),
        "river_level": [10.0, 12.0, 15.0],
        "discharge_m3s": [100.0, 120.0, 150.0]
    })

    result = create_river_features(df)

    assert result.loc[1, "river_level_change"] == 2.0
    assert result.loc[2, "river_level_change"] == 3.0
    assert result.loc[1, "discharge_change"] == 20.0
    assert result.loc[2, "discharge_change"] == 30.0