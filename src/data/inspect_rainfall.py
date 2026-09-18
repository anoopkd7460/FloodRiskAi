from pathlib import Path
import xarray as xr

DATA_DIR = Path("data/raw/rainfall")

def inspect_file(path: Path) -> None:
    print("=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)
    try:
        dataset = xr.open_dataset(path)
        print("\nDATASET:")
        print(dataset)
        print("\nDIMENSIONS:")
        for name,size in dataset.sizes.items():
            print(f"{name}: {size}")
        print("\nCOORDINATES:")
        for name in dataset.coords:
            print(name)
        print("\nDATA VARIABLES:")
        for name in dataset.data_vars:
            print(name)
        print("\nGLOBAL ATTRIBUTES:")
        for key,value in dataset.attrs.items():
            print(f"{key}: {value}")
        dataset.close()
    except Exception as exc:
        print(f"ERROR: {exc}")

def main():
    files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file()
    ]
    if not files:
        print("No rainfall files found.")
        return
    for path in files:
        inspect_file(path)

if __name__ == "__main__":
    main()