"""Fit the final models on all fused data and save results/model_bundle.joblib for the web app.

The bundle is trained on all the data, so its scores are not results; the
cross-validated scores in results/results.csv are.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from qpd.data import ROOT, load_fused, load_voice  # noqa: E402
from qpd.experiments import RESULTS, get_model  # noqa: E402
from qpd.features import FusionReducer  # noqa: E402

MODELS = ["vqc", "qsvm", "rf", "lr"]
OUT = ROOT / "results" / "model_bundle.joblib"


def main(seed: int = 0):
    fused, n_voice_cols = load_fused(seed)
    voice_names = load_voice().feature_names
    reducer = FusionReducer(n_voice_cols=n_voice_cols, seed=seed).fit(fused.X, fused.y)
    Z = reducer.transform(fused.X)

    models = {}
    for m in MODELS:
        print(f"Fitting {m}...")
        models[m] = get_model(m, seed).fit(Z, fused.y)

    metrics = {}
    if RESULTS.exists():
        r = pd.read_csv(RESULTS)
        metrics = r[r["input"] == "fusion"].groupby("model")["balanced_accuracy"].mean().round(3).to_dict()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "reducer": reducer,
            "models": models,
            "feature_names": reducer.feature_names(voice_names),
            "voice_columns": voice_names,
            "background": shap.kmeans(Z, 10).data,
            "metrics": metrics,
        },
        OUT,
    )
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
