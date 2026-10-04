import pandas as pd
from src.data.flood import create_grid_cells

def test_create_grid_cells():
    df=pd.DataFrame({
        "region_id":["BHR_25.50_85.00"],
        "latitude":[25.50],
        "longitude":[85.00]
    })

    result=create_grid_cells(df)

    assert len(result)==1
    assert result[0].bounds==(84.875,25.375,85.125,25.625)


def test_create_grid_cells_count():
    df=pd.DataFrame({
        "region_id":["A","B","C"],
        "latitude":[25.0,25.25,25.50],
        "longitude":[85.0,85.25,85.50]
    })

    result=create_grid_cells(df)

    assert len(result)==3