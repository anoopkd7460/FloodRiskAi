import pandas as pd


def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a basic data-quality report."""

    report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ),
        "unique_values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    return report