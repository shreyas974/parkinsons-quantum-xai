"""Run every model on every input setting with person-wise cross-validation.

Example:
    python -m qpd.experiments --models lr,rf,vqc --inputs voice --seeds 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .baselines import get_baselines
from .data import ROOT, Dataset, load_spiral_features, load_voice, pair_by_label
from .features import FusionReducer, spiral_reducer, voice_reducer
from .quantum import VQC, QuantumKernelSVM

RESULTS = ROOT / "results" / "results.csv"
CLASSICAL_GRIDS = {
    "lr": {"model__C": [0.1, 1.0, 10.0]},
    "svm": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
    "rf": {"model__max_depth": [None, 5], "model__min_samples_leaf": [1, 3]},
    "mlp": {"model__alpha": [0.001, 0.01], "model__hidden_layer_sizes": [(16,), (32, 16)]},
    "xgb": {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.1]},
}
CLASSICAL_GRIDS["rf_full"] = CLASSICAL_GRIDS["rf"]


def get_model(name: str, seed: int, device: str = "default.qubit", learning_rate: float = 0.05,
              epochs: int = 60):
    quantum = {
        "vqc": {},
        "vqc_noent": {"entangle": False},
        "vqc_noise": {"noise": 0.01},
        **{f"vqc_l{depth}": {"n_layers": depth} for depth in range(1, 5)},
    }
    if name in quantum:
        return VQC(seed=seed, device=device, lr=learning_rate, epochs=epochs, **quantum[name])
    if name == "qsvm":
        return QuantumKernelSVM(seed=seed, device=device)
    baselines = get_baselines(seed)
    baselines["rf_full"] = baselines["rf"]
    if name not in baselines:
        options = sorted([*baselines, *quantum, "qsvm"])
        raise ValueError(f"Unknown model {name!r}. Options: {options}")
    return baselines[name]


def make_input(kind: str, seed: int, n_splits: int = 5):
    """Data, groups and the in-fold reducer for one input setting."""
    if kind == "voice":
        d = load_voice()
        return d.X, d.y, d.groups, voice_reducer(4, seed), StratifiedGroupKFold
    if kind == "spiral":
        d = load_spiral_features()
        return d.X, d.y, d.groups, spiral_reducer(4, seed), StratifiedGroupKFold
    if kind == "fusion":
        voice = load_voice()
        voice_folds = person_folds(voice.X, voice.y, voice.groups, seed, n_splits)
        d = pair_outer_folds(voice, load_spiral_features(), voice_folds, seed)
        reducer = FusionReducer(n_voice_cols=voice.X.shape[1], seed=seed)
        return d.X, d.y, d.groups, reducer, StratifiedGroupKFold
    raise ValueError(f"Unknown input {kind!r}: use voice, spiral or fusion")


def _has_both_classes(y, splits):
    return all(len(np.unique(y[part])) == 2 for split in splits for part in split)


def person_folds(X, y, groups, seed, n_splits=5):
    rng = np.random.default_rng(seed)
    candidate_seeds = [seed, *rng.integers(0, 2**32 - 1, size=127)]
    for split_seed in candidate_seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(split_seed))
        splits = list(cv.split(X, y, groups))
        if _has_both_classes(y, splits):
            return splits
    raise ValueError("Could not find person-wise outer folds containing both classes")


def _subset(dataset, index):
    return Dataset(dataset.X[index], dataset.y[index], dataset.groups[index], dataset.feature_names)


def pair_outer_folds(voice, spiral, voice_folds, seed):
    """Allocate whole spiral filename groups to voice folds before pairing."""
    voice_tests = [test for _, test in voice_folds]
    people = np.array([
        [len(np.unique(voice.groups[test][voice.y[test] == label])) for label in (0, 1)]
        for test in voice_tests
    ])
    rng = np.random.default_rng(seed)
    candidate_seeds = [seed, *rng.integers(0, 2**32 - 1, size=127)]
    for split_seed in candidate_seeds:
        cv = StratifiedGroupKFold(len(voice_folds), shuffle=True, random_state=int(split_seed))
        spiral_tests = [test for _, test in cv.split(spiral.X, spiral.y, spiral.groups)]
        counts = np.array([np.bincount(spiral.y[test], minlength=2) for test in spiral_tests])
        feasible = (counts[None, :, :] >= people[:, None, :]).all(axis=2)
        _, assignment = linear_sum_assignment(~feasible)
        if feasible[np.arange(len(voice_tests)), assignment].all():
            break
    else:
        raise ValueError("Not enough spiral images in disjoint filename groups for the voice folds")

    X = np.empty((len(voice.y), voice.X.shape[1] + spiral.X.shape[1]), dtype=float)
    for rows, spiral_fold in zip(voice_tests, assignment):
        paired = pair_by_label(_subset(voice, rows), _subset(spiral, spiral_tests[spiral_fold]), seed)
        X[rows] = paired.X
    return Dataset(X, voice.y, voice.groups, voice.feature_names + spiral.feature_names)


class TrainingGroupKFold(GroupKFold):
    """Retry group ordering when an inner split omits a class."""

    def split(self, X, y=None, groups=None):
        _, codes = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(0)
        for _ in range(128):
            splits = list(super().split(X, y, codes))
            if _has_both_classes(y, splits):
                yield from splits
                return
            codes = rng.permutation(codes.max() + 1)[codes]
        raise ValueError("Could not find person-wise inner folds containing both classes")


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1),
        "specificity": recall_score(y, pred, pos_label=0),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, p),
    }


def subsample_people(indices, y, groups, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    if fraction == 1:
        return indices
    population = np.unique(groups[indices])
    count = max(2, int(np.ceil(len(population) * fraction)))
    rng = np.random.default_rng(seed)
    for _ in range(128):
        selected = rng.choice(population, size=count, replace=False)
        sample = indices[np.isin(groups[indices], selected)]
        if len(np.unique(y[sample])) == 2:
            return sample
    raise ValueError("Could not select a training subset with both classes")


def run_one(
    model_name: str, kind: str, seed: int, n_splits: int = 5, *,
    device: str = "default.qubit", learning_rate: float = 0.05,
    train_fraction: float = 1.0, epochs: int = 60, loss_dir: Path | None = None,
) -> list[dict]:
    X, y, groups, reducer, _ = make_input(kind, seed, n_splits)
    if model_name == "rf_full":
        if kind != "voice":
            raise ValueError("rf_full is the full-feature voice reference")
        reducer = "passthrough"

    split = person_folds(X, y, groups, seed, n_splits)
    tuning_groups = groups
    if kind == "fusion" and model_name in CLASSICAL_GRIDS:
        if n_splits < 4:
            raise ValueError("Tuned fusion requires at least four outer folds")
        # Each outer partition contains disjoint voice subjects and spiral filename groups.
        tuning_groups = np.empty(len(groups), dtype=int)
        for fold, (_, test) in enumerate(split):
            tuning_groups[test] = fold

    rows = []
    for fold, (tr, te) in enumerate(split):
        tr = subsample_people(tr, y, groups, train_fraction, seed + fold)
        transform = clone(reducer) if reducer != "passthrough" else reducer
        pipe = Pipeline([("reduce", transform), ("model", get_model(model_name, seed, device, learning_rate, epochs))])
        t0 = time.perf_counter()
        if model_name in CLASSICAL_GRIDS:
            fitted = GridSearchCV(
                pipe, CLASSICAL_GRIDS[model_name], cv=TrainingGroupKFold(n_splits=3),
                scoring="balanced_accuracy", error_score="raise",
            ).fit(X[tr], y[tr], groups=tuning_groups[tr])
        else:
            fitted = pipe.fit(X[tr], y[tr])
        train_time = time.perf_counter() - t0
        p = fitted.predict_proba(X[te])[:, 1]
        if loss_dir is not None and model_name.startswith("vqc"):
            loss_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{model_name}-{kind}-seed{seed}-fold{fold}-{device}-lr{learning_rate}-n{train_fraction}"
            if epochs != 60:
                tag += f"-epochs{epochs}"
            losses = fitted.named_steps["model"].loss_curve_
            np.savetxt(loss_dir / f"{tag}.csv", np.column_stack([np.arange(1, len(losses) + 1), losses]),
                       delimiter=",", header="epoch,loss", comments="")
        rows.append({"model": model_name, "input": kind, "seed": seed, "fold": fold,
                     "train_time_s": train_time, **scores(y[te], p)})
        print(f"{model_name:10s} {kind:7s} seed={seed} fold={fold} "
              f"bal_acc={rows[-1]['balanced_accuracy']:.3f} auc={rows[-1]['auc']:.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="lr,svm,rf,mlp,xgb,vqc,qsvm")
    ap.add_argument("--inputs", default="voice,spiral,fusion")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", choices=["default.qubit", "lightning.qubit"], default="default.qubit")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--loss-dir", type=Path)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for kind in args.inputs.split(","):
            for model in args.models.split(","):
                rows = pd.DataFrame(run_one(
                    model, kind, seed, args.folds, device=args.device, learning_rate=args.learning_rate,
                    train_fraction=args.train_fraction, epochs=args.epochs, loss_dir=args.loss_dir,
                ))
                rows.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)

    summary = pd.read_csv(args.out).groupby(["input", "model"])[
        ["balanced_accuracy", "sensitivity", "specificity", "f1", "auc", "train_time_s"]
    ].agg(["mean", "std"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
