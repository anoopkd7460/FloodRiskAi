import pandas as pd


REQUIRED_COLUMNS = [
    "region_id",
    "date",
    "latitude",
    "longitude",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the minimum schema required by FloodRiskAI."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_coordinates(df: pd.DataFrame) -> None:
    """Validate latitude and longitude ranges."""

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values found.")

    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values found.")


def validate_dates(df: pd.DataFrame) -> None:
    """Validate date column."""

    converted = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if converted.isna().any():
        raise ValueError("Invalid dates found.")


def validate_dataset(df: pd.DataFrame) -> None:
    """Run all validation checks."""

    validate_schema(df)
    validate_coordinates(df)
    validate_dates(df)

    print("Dataset validation passed.")