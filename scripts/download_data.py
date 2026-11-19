"""Download the UCI voice dataset and check that the spiral images are in place.

The Kaggle spiral set needs a Kaggle login, so it is downloaded by hand
(see data/README.md).
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qpd.data import SPIRAL_DIR, UCI_URL, VOICE_PATH, list_spirals  # noqa: E402


def main():
    if VOICE_PATH.exists():
        print(f"Voice data already at {VOICE_PATH}")
    else:
        VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, VOICE_PATH)
        print(f"Saved {VOICE_PATH}")

    try:
        table = list_spirals(SPIRAL_DIR)
        counts = table["label"].value_counts().to_dict()
        print(f"Spiral images found: {len(table)} (healthy {counts.get(0, 0)}, parkinson {counts.get(1, 0)})")
    except FileNotFoundError as e:
        print(e)
        print("Download the Kaggle 'Parkinson's Drawings' set and copy its spiral/ folder into data/spiral/.")


if __name__ == "__main__":
    main()
