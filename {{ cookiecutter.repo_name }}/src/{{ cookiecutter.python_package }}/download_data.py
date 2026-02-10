"""Script to download WANDS dataset for retrieval assessment."""

from __future__ import annotations


import urllib.request
from pathlib import Path


# WANDS dataset URLs (from Wayfair)
WANDS_BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
WANDS_FILES = {
    "products.csv": f"{WANDS_BASE_URL}/product.csv",
    "query.csv": f"{WANDS_BASE_URL}/query.csv",
    "labels.csv": f"{WANDS_BASE_URL}/label.csv",
}


def download_wands_dataset(data_dir: str = "data/01_raw") -> None:
    """Download WANDS dataset files.

    Args:
        data_dir: Directory to save the files
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading WANDS dataset...")

    for filename, url in WANDS_FILES.items():
        target_path = data_path / filename

        if target_path.exists():
            print(f"  [OK] {filename} already exists, skipping")
            continue

        print(f"  -> Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  [OK] {filename} downloaded successfully")
        except Exception as e:
            print(f"  [ERROR] Error downloading {filename}: {e}")

    print("\nDataset download complete!")
    print(f"Files are in: {data_path.absolute()}")


def main() -> None:
    """Main entry point."""
    # Find project root
    current_dir = Path(__file__).parent
    if current_dir.name == "{{ cookiecutter.python_package }}":
        project_root = current_dir.parent.parent
    else:
        project_root = Path.cwd()

    data_dir = project_root / "data" / "01_raw"
    download_wands_dataset(str(data_dir))


if __name__ == "__main__":
    main()
