import pandas as pd
from src.data.rainfall import (
    clean_rainfall_data,
    create_region_id
)

def test_clean_rainfall_data():
    df = pd.DataFrame({
        "date":["2020-01-01","2020-01-02"],
        "latitude":[25.5,25.75],
        "longitude":[85.0,85.25],
        "rainfall":[10.5,20.0]
    })
    result = clean_rainfall_data(df)
    assert len(result) == 2
    assert result["rainfall"].min() >= 0

def test_create_region_id():
    df = pd.DataFrame({
        "date":pd.to_datetime(["2020-01-01"]),
        "latitude":[25.5],
        "longitude":[85.0],
        "rainfall":[10.5]
    })
    result = create_region_id(df)
    assert result.loc[0,"region_id"] == "BHR_25.50_85.00"