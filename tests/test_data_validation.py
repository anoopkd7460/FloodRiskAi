import pandas as pd
from src.data.validation import validate_dataset

def test_valid_dataset():
    df = pd.DataFrame({
        "region_id":["BHR_001"],
        "date":["2024-07-15"],
        "latitude":[25.6],
        "longitude":[85.1]
    })
    validate_dataset(df)

def test_invalid_latitude():
    df = pd.DataFrame({
        "region_id":["BHR_001"],
        "date":["2024-07-15"],
        "latitude":[100],
        "longitude":[85.1]
    })
    try:
        validate_dataset(df)
        assert False
    except ValueError:
        assert True