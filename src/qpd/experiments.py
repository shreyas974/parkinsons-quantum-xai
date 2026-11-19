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
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .baselines import get_baselines
from .data import ROOT, load_fused, load_spiral_features, load_voice
from .features import FusionReducer, spiral_reducer, voice_reducer
from .quantum import VQC, QuantumKernelSVM

RESULTS = ROOT / "results" / "results.csv"


def get_model(name: str, seed: int):
    quantum = {
        "vqc": lambda: VQC(seed=seed),
        "vqc_noent": lambda: VQC(entangle=False, seed=seed),
        "qsvm": lambda: QuantumKernelSVM(seed=seed),
    }
    if name in quantum:
        return quantum[name]()
    baselines = get_baselines(seed)
    if name not in baselines:
        raise ValueError(f"Unknown model {name!r}. Options: {sorted(baselines) + sorted(quantum)}")
    return baselines[name]


def make_input(kind: str, seed: int):
    """Data, groups and the in-fold reducer for one input setting."""
    if kind == "voice":
        d = load_voice()
        return d.X, d.y, d.groups, voice_reducer(4, seed), StratifiedGroupKFold
    if kind == "spiral":
        d = load_spiral_features()
        return d.X, d.y, None, spiral_reducer(4, seed), StratifiedKFold
    if kind == "fusion":
        d, n_voice_cols = load_fused(seed)
        return d.X, d.y, d.groups, FusionReducer(n_voice_cols=n_voice_cols, seed=seed), StratifiedGroupKFold
    raise ValueError(f"Unknown input {kind!r}: use voice, spiral or fusion")


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1),
        "specificity": recall_score(y, pred, pos_label=0),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, p),
    }


def run_one(model_name: str, kind: str, seed: int, n_splits: int = 5) -> list[dict]:
    X, y, groups, reducer, splitter = make_input(kind, seed)
    cv = splitter(n_splits=n_splits, shuffle=True, random_state=seed)
    split = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    rows = []
    for fold, (tr, te) in enumerate(split):
        pipe = Pipeline([("reduce", clone(reducer)), ("model", get_model(model_name, seed))])
        t0 = time.perf_counter()
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        rows.append({"model": model_name, "input": kind, "seed": seed, "fold": fold,
                     "train_time_s": time.perf_counter() - t0, **scores(y[te], p)})
        print(f"{model_name:10s} {kind:7s} seed={seed} fold={fold} "
              f"bal_acc={rows[-1]['balanced_accuracy']:.3f} auc={rows[-1]['auc']:.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="lr,svm,rf,mlp,xgb,vqc,qsvm")
    ap.add_argument("--inputs", default="voice,spiral,fusion")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for kind in args.inputs.split(","):
            for model in args.models.split(","):
                rows = pd.DataFrame(run_one(model, kind, seed, args.folds))
                rows.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)

    summary = pd.read_csv(args.out).groupby(["input", "model"])[
        ["balanced_accuracy", "sensitivity", "specificity", "f1", "auc", "train_time_s"]
    ].agg(["mean", "std"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
