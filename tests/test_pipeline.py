import numpy as np
import pytest

from qpd.data import Dataset, pair_by_label, subject_id
from qpd.experiments import (
    TrainingGroupKFold, average_probabilities, pair_outer_folds, person_folds,
)
from qpd.features import FusionReducer, angle_scaler, spiral_reducer, voice_reducer


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



def test_grouped_splits_never_share_people():
    voice = fake_voice(n_subjects=12)
    for train, test in person_folds(voice.X, voice.y, voice.groups, seed=0):
        assert set(voice.groups[train]).isdisjoint(voice.groups[test])
        assert set(voice.y[train]) == set(voice.y[test]) == {0, 1}
        for inner_train, validation in TrainingGroupKFold(3).split(
            voice.X[train], voice.y[train], voice.groups[train],
        ):
            assert set(voice.groups[train][inner_train]).isdisjoint(voice.groups[train][validation])


def test_pairing_rejects_insufficient_disjoint_images():
    with pytest.raises(ValueError, match="disjoint subject pools"):
        pair_by_label(fake_voice(), fake_spiral(n=6))


def test_fusion_keeps_whole_spiral_groups_in_outer_folds():
    voice, spiral = fake_voice(n_subjects=12), fake_spiral(n=60)
    spiral.groups = np.repeat([f"artist{i}" for i in range(30)], 2)
    folds = person_folds(voice.X, voice.y, voice.groups, seed=0)
    paired = pair_outer_folds(voice, spiral, folds, seed=0)
    lookup = {row.tobytes(): group for row, group in zip(spiral.X, spiral.groups)}
    owners = np.array([lookup[row.tobytes()] for row in paired.X[:, voice.X.shape[1]:]])
    for train, test in folds:
        assert set(owners[train]).isdisjoint(owners[test])

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


def test_quantum_models_survive_save_and_load():
    pytest.importorskip("pennylane")
    import pickle

    from qpd.quantum import VQC, QuantumKernelSVM

    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, size=(16, 3))
    y = np.array([0, 1] * 8)
    for model in (VQC(n_layers=1, epochs=2, batch_size=8).fit(X, y), QuantumKernelSVM().fit(X, y)):
        loaded = pickle.loads(pickle.dumps(model))
        assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize("kind", ["angle", "voice", "spiral", "fusion"])
def test_all_reducers_clip_held_out_values(kind):
    voice, spiral = fake_voice(), fake_spiral()
    if kind in {"angle", "voice"}:
        X, y = voice.X, voice.y
        reducer = angle_scaler() if kind == "angle" else voice_reducer()
    elif kind == "spiral":
        X, y, reducer = spiral.X, spiral.y, spiral_reducer()
    else:
        fused = pair_by_label(voice, spiral)
        X, y, reducer = fused.X, fused.y, FusionReducer()
    reducer.fit(X, y)
    held_out = reducer.transform(X[:6] * 100)
    assert np.isfinite(held_out).all()
    assert np.all((0 <= held_out) & (held_out <= np.pi))


def test_late_fusion_averages_valid_probabilities():
    voice = np.array([0.0, 0.2, 1.0])
    spiral = np.array([0.0, 0.4, 1.0])
    fused = average_probabilities(voice, spiral)
    np.testing.assert_allclose(fused, [0.0, 0.3, 1.0])
    assert np.all((0 <= fused) & (fused <= 1))


def test_bundle_has_web_contract_and_survives_serialization(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    from qpd.bundle import MODEL_NAMES, build_bundle

    voice = fake_voice()
    fused = pair_by_label(voice, fake_spiral())
    reducer = FusionReducer().fit(fused.X, fused.y)
    reduced = reducer.transform(fused.X)
    model = LogisticRegression().fit(reduced, fused.y)
    models = dict.fromkeys(MODEL_NAMES, model)
    metrics = dict.fromkeys(MODEL_NAMES, 0.5)
    bundle = build_bundle(reducer, models, voice.feature_names, reduced, metrics)
    assert set(bundle) == {"reducer", "models", "feature_names", "voice_columns", "background", "metrics"}
    assert set(bundle["models"]) == set(bundle["metrics"]) == {"vqc", "qsvm", "rf", "lr"}
    assert len(bundle["feature_names"]) == 8
    assert len(bundle["voice_columns"]) == 22
    assert bundle["background"].shape == (10, 8)
    path = tmp_path / "test.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored["reducer"].transform(fused.X), reduced)
    np.testing.assert_array_equal(restored["models"]["lr"].predict_proba(reduced), model.predict_proba(reduced))
