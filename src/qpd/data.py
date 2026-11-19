"""Loading the voice and spiral datasets, and pairing them into fused samples.

Voice: UCI Parkinsons (Little et al., 2007), 195 recordings from 31 people.
Spiral: Kaggle "Parkinson's Drawings" (Zham et al., 2017), spiral subset.

The two datasets come from different people, so fused samples are built by
pairing on the label. Each voice subject is given its own spiral images, so a
spiral never appears in both the training and the test fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
VOICE_PATH = DATA_DIR / "voice" / "parkinsons.data"
SPIRAL_DIR = DATA_DIR / "spiral"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def subject_id(name: str) -> str:
    """phon_R01_S01_1 -> phon_R01_S01 (one person, several recordings)."""
    return name.rsplit("_", 1)[0]


def load_voice(path: Path = VOICE_PATH) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path)
    groups = df["name"].map(subject_id).to_numpy()
    y = df["status"].to_numpy(dtype=int)
    X = df.drop(columns=["name", "status"])
    return Dataset(X.to_numpy(dtype=float), y, groups, list(X.columns))


def list_spirals(root: Path = SPIRAL_DIR) -> pd.DataFrame:
    """Every spiral image under root, labelled by its folder name (healthy / parkinson)."""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = {q.lower() for q in p.parts}
        if "parkinson" in parts:
            label = 1
        elif "healthy" in parts:
            label = 0
        else:
            continue
        rows.append({"path": str(p), "label": label})
    if not rows:
        raise FileNotFoundError(
            f"No spiral images under {root}. Expected folders like "
            "spiral/training/healthy and spiral/training/parkinson (see data/README.md)."
        )
    return pd.DataFrame(rows)


def load_spiral_features(root: Path = SPIRAL_DIR) -> Dataset:
    from .features import hog_features

    table = list_spirals(root)
    X = np.stack([hog_features(p) for p in table["path"]])
    y = table["label"].to_numpy(dtype=int)
    groups = np.array([f"img{i}" for i in range(len(y))])
    return Dataset(X, y, groups, [f"hog_{i}" for i in range(X.shape[1])])


def pair_by_label(voice: Dataset, spiral: Dataset, seed: int = 0) -> Dataset:
    """Build fused samples: each voice recording gets a spiral with the same label.

    Spiral images are split into disjoint pools, one per voice subject, so that
    grouping by voice subject also keeps every spiral on one side of a split.
    """
    rng = np.random.default_rng(seed)
    voice_X, spiral_X = voice.X, spiral.X
    spiral_for_row = np.empty(len(voice.y), dtype=int)

    for label in (0, 1):
        subjects = np.unique(voice.groups[voice.y == label])
        images = rng.permutation(np.flatnonzero(spiral.y == label))
        if len(images) == 0:
            raise ValueError(f"No spiral images with label {label}")
        pools = np.array_split(images, len(subjects)) if len(images) >= len(subjects) else None
        for k, subj in enumerate(rng.permutation(subjects)):
            rows = np.flatnonzero(voice.groups == subj)
            # Fewer images than subjects: reuse is unavoidable, so draw from all of them.
            pool = pools[k] if pools is not None else images
            spiral_for_row[rows] = rng.choice(pool, size=len(rows), replace=len(pool) < len(rows))

    X = np.hstack([voice_X, spiral_X[spiral_for_row]])
    names = voice.feature_names + spiral.feature_names
    return Dataset(X, voice.y.copy(), voice.groups.copy(), names)


def load_fused(seed: int = 0) -> tuple[Dataset, int]:
    """Fused raw features and the number of voice columns at the front.

    Scaling, voice feature selection and spiral PCA happen later inside each
    training fold (features.FusionReducer), so nothing leaks from the test fold.
    """
    voice = load_voice()
    spiral = load_spiral_features()
    return pair_by_label(voice, spiral, seed=seed), voice.X.shape[1]
