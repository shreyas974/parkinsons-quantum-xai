import numpy as np
import pytest

from qpd.data import Dataset, pair_by_label, subject_id
from qpd.features import FusionReducer


def fake_voice(n_subjects=8, recs=6, n_feat=22, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"S{i:02d}" for i in range(n_subjects)], recs)
    y = np.repeat([i % 2 for i in range(n_subjects)], recs)
    X = rng.normal(size=(len(y), n_feat)) + y[:, None]
    return Dataset(X, y, groups, [f"v{i}" for i in range(n_feat)])


def fake_spiral(n=24, n_feat=50, seed=1):
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    return Dataset(rng.normal(size=(n, n_feat)) + y[:, None], y, np.array([f"img{i}" for i in range(n)]),
                   [f"h{i}" for i in range(n_feat)])


def test_subject_id():
    assert subject_id("phon_R01_S01_1") == "phon_R01_S01"


def test_pairing_keeps_labels_and_spirals_per_subject():
    voice, spiral = fake_voice(), fake_spiral()
    fused = pair_by_label(voice, spiral, seed=0)
    assert fused.X.shape == (len(voice.y), voice.X.shape[1] + spiral.X.shape[1])
    spiral_part = fused.X[:, voice.X.shape[1]:]
    # Each spiral row must come from a spiral with the same label.
    for row, label in zip(spiral_part, fused.y):
        match = np.flatnonzero((spiral.X == row).all(1))
        assert spiral.y[match[0]] == label
    # No spiral is shared between two subjects.
    owner = {}
    for row, g in zip(spiral_part, fused.groups):
        key = row.tobytes()
        assert owner.setdefault(key, g) == g


def test_fusion_reducer_gives_angles():
    voice, spiral = fake_voice(), fake_spiral()
    fused = pair_by_label(voice, spiral)
    red = FusionReducer(n_voice_cols=22).fit(fused.X, fused.y)
    Z = red.transform(fused.X)
    assert Z.shape == (len(fused.y), 8)
    assert Z.min() >= 0 and Z.max() <= np.pi + 1e-9
    assert len(red.feature_names(voice.feature_names)) == 8


def test_vqc_trains_on_tiny_data():
    pytest.importorskip("pennylane")
    from qpd.quantum import VQC

    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(24, 4))
    y = (X[:, 0] > np.pi / 2).astype(int)
    model = VQC(n_layers=1, epochs=3, batch_size=8).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (24, 2)
    assert np.allclose(p.sum(1), 1)
