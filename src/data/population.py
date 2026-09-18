from pathlib import Path
import pandas as pd

def load_population_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Population file not found: {path}")
    df = pd.read_csv(path)
    return df

def validate_population_data(df: pd.DataFrame) -> None:
    required = ["latitude","longitude","population"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing population columns: {missing}")