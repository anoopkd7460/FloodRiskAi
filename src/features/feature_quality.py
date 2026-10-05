from pathlib import Path
import pandas as pd

def analyze_feature_quality(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {path}")

    df=pd.read_csv(path)

    print("=== DATASET SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    labeled=df[df["flood_label"].notna()].copy()

    print("\n=== SUPERVISED DATASET ===")
    print(f"Labeled rows: {len(labeled):,}")
    print(f"Positive labels: {int(labeled['flood_label'].sum()):,}")
    print(f"Negative labels: {int((labeled['flood_label']==0).sum()):,}")

    metadata_columns=[
        "region_id",
        "year",
        "latitude",
        "longitude",
        "flood_label"
    ]

    feature_columns=[
        column for column in df.columns
        if column not in metadata_columns
    ]

    result=pd.DataFrame({
        "dtype":df[feature_columns].dtypes.astype(str),
        "missing_count":df[feature_columns].isna().sum(),
        "missing_pct":df[feature_columns].isna().mean()*100,
        "unique_values":df[feature_columns].nunique()
    })

    result=result.sort_values("missing_pct",ascending=False)

    print("\n=== FEATURE QUALITY ===")
    print(result.to_string())

    print("\n=== CORRELATION WITH FLOOD LABEL ===")
    correlations=labeled[feature_columns].corrwith(labeled["flood_label"])
    correlations=correlations.sort_values(key=abs,ascending=False)
    print(correlations.to_string())

    return result

if __name__=="__main__":
    analyze_feature_quality(
        "data/interim/modeling_bihar_grid_year_2015_2025.csv"
    )