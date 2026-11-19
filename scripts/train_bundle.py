"""Fit, save and verify the fused model bundle used by the web app."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qpd.bundle import MODEL_NAMES, build_bundle  # noqa: E402
from qpd.data import ROOT, VOICE_PATH, load_voice  # noqa: E402
from qpd.experiments import RESULTS, fit_fold_model, fold_membership, get_model, make_input, person_folds  # noqa: E402

OUT = ROOT / "results" / "model_bundle.joblib"
VERIFY = """
import sys
import joblib
import numpy as np

bundle = joblib.load(sys.argv[1])
with np.load(sys.argv[2]) as expected:
    reduced = bundle["reducer"].transform(expected["raw"])
    np.testing.assert_array_equal(reduced, expected["reduced"])
    for name, model in bundle["models"].items():
        probabilities = model.predict_proba(reduced)
        np.testing.assert_array_equal(probabilities, expected[name])
        print(f"{name}: exact probabilities after reload on {len(probabilities)} rows")
"""


def fusion_metrics():
    if not RESULTS.exists():
        raise FileNotFoundError("Run the fusion experiments before building the model bundle")
    results = pd.read_csv(RESULTS)
    if "experiment" in results.columns:
        results = results[results["experiment"] == "E3"]
    means = results[results["input"] == "fusion"].groupby("model")["balanced_accuracy"].mean()
    return means.reindex(MODEL_NAMES).to_dict()


def save_verified(bundle, raw, reduced):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT.parent) as temporary:
        candidate = Path(temporary) / OUT.name
        probabilities = {name: model.predict_proba(reduced) for name, model in bundle["models"].items()}
        expected = Path(temporary) / "expected.npz"
        np.savez(expected, raw=raw, reduced=reduced, **probabilities)
        joblib.dump(bundle, candidate)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run([sys.executable, "-c", VERIFY, str(candidate), str(expected)], cwd=ROOT, env=env, check=True)
        candidate.replace(OUT)


def main(seed: int = 0):
    metrics = fusion_metrics()
    X, y, groups, reducer, _ = make_input("fusion", seed)
    folds = person_folds(X, y, groups, seed)
    tuning_groups = fold_membership(folds, len(y))
    print("Tuning RF and LR on grouped full-data splits", flush=True)
    rf = fit_fold_model("rf", reducer, X, y, tuning_groups, seed).best_estimator_
    lr = fit_fold_model("lr", reducer, X, y, tuning_groups, seed).best_estimator_
    reducer = rf.named_steps["reduce"]
    Z = reducer.transform(X)
    np.testing.assert_array_equal(lr.named_steps["reduce"].transform(X), Z)

    models = {"rf": rf.named_steps["model"], "lr": lr.named_steps["model"]}
    for name in ("vqc", "qsvm"):
        print(f"Fitting {name}", flush=True)
        models[name] = get_model(name, seed).fit(Z, y)

    voice_columns = load_voice().feature_names
    bundle = build_bundle(reducer, models, voice_columns, Z, metrics)
    save_verified(bundle, X, Z)
    sample = pd.read_csv(VOICE_PATH, usecols=voice_columns, nrows=1, dtype=str)
    sample[voice_columns].to_csv(OUT.parent / "sample_voice_row.csv", index=False)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
