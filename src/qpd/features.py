"""Feature extraction and in-fold reduction to a few numbers per signal."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

IMG_SIZE = (200, 200)


def hog_features(path: str) -> np.ndarray:
    """HOG (edge-direction histogram) of a spiral image, ink made bright."""
    from skimage import color, feature, io, transform

    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    img = transform.resize(img, IMG_SIZE, anti_aliasing=True)
    img = 1.0 - img
    return feature.hog(img, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2))


def angle_scaler() -> MinMaxScaler:
    """Map each feature to [0, pi] so it can be loaded as a qubit rotation."""
    return MinMaxScaler(feature_range=(0.0, np.pi), clip=True)


def voice_reducer(k: int = 4, seed: int = 0):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=seed), k=k),
        angle_scaler(),
    )


def spiral_reducer(k: int = 4, seed: int = 0):
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed), angle_scaler())


class FusionReducer(BaseEstimator, TransformerMixin):
    """Reduce [voice columns | spiral columns] to n_voice + n_spiral angles.

    Fit on the training fold only; the voice part keeps its top features, the
    spiral part is compressed with PCA.
    """

    def __init__(self, n_voice_cols: int = 22, n_voice: int = 4, n_spiral: int = 4, seed: int = 0):
        self.n_voice_cols = n_voice_cols
        self.n_voice = n_voice
        self.n_spiral = n_spiral
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.voice_ = voice_reducer(self.n_voice, self.seed).fit(X[:, : self.n_voice_cols], y)
        self.spiral_ = spiral_reducer(self.n_spiral, self.seed).fit(X[:, self.n_voice_cols :], y)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack(
            [
                self.voice_.transform(X[:, : self.n_voice_cols]),
                self.spiral_.transform(X[:, self.n_voice_cols :]),
            ]
        )

    def feature_names(self, voice_names: list[str]) -> list[str]:
        """Readable names for the reduced columns, for SHAP plots."""
        mask = self.voice_.named_steps["selectkbest"].get_support()
        chosen = [n for n, keep in zip(voice_names, mask) if keep]
        return chosen + [f"spiral_PC{i + 1}" for i in range(self.n_spiral)]
