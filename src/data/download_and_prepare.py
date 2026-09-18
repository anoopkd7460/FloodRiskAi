from pathlib import Path
from rainfall import load_rainfall_csv,filter_bihar,save_rainfall

RAW_DIR = Path("data/raw/rainfall")
INTERIM_DIR = Path("data/interim")

def main():
    input_file = RAW_DIR / "imd_rainfall.csv"
    output_file = INTERIM_DIR / "rainfall_bihar.csv"
    rainfall = load_rainfall_csv(input_file)
    print(f"Raw rainfall records: {len(rainfall)}")
    rainfall = filter_bihar(rainfall)
    print(f"Bihar rainfall records: {len(rainfall)}")
    save_rainfall(rainfall,output_file)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()