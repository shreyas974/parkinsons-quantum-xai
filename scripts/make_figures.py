"""Explain RF and VQC predictions on a common held-out fusion fold."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402

from qpd.data import ROOT, load_voice  # noqa: E402
from qpd.experiments import fit_fold_model, fold_membership, get_model, make_input, person_folds  # noqa: E402
from qpd.explain import explain, modality_share, rank_agreement, save_figures  # noqa: E402

OUT = ROOT / "figures" / "raw"
RESULTS = ROOT / "results"
MARKERS = ("PPE", "spread1", "spread2", "RPDE", "jitter", "shimmer", "NHR")


def main(seed: int = 0, n_explain: int = 30):
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X, y, groups, reducer, _ = make_input("fusion", seed)
    splits = person_folds(X, y, groups, seed)
    train, test = splits[0]
    if len(test) < n_explain:
        raise ValueError("The selected held-out fold has fewer rows than requested for SHAP")
    membership = fold_membership(splits, len(y))
    rf = fit_fold_model("rf", reducer, X[train], y[train], membership[train], seed).best_estimator_
    fitted_reducer = rf.named_steps["reduce"]
    Z_train = fitted_reducer.transform(X[train])
    Z_test = fitted_reducer.transform(X[test])
    names = fitted_reducer.feature_names(load_voice().feature_names)
    rows = np.random.default_rng(seed).choice(len(test), size=n_explain, replace=False)

    explanation = {}
    report = {"seed": seed, "fold": 0, "n_explained": n_explain, "held_out_rows": len(test), "models": {}}
    roc, roc_ax = plt.subplots()
    for model_name in ("rf", "vqc"):
        print(f"Fitting or explaining {model_name} on the shared split", flush=True)
        model = rf.named_steps["model"] if model_name == "rf" else get_model("vqc", seed).fit(Z_train, y[train])
        display = RocCurveDisplay.from_predictions(y[test], model.predict_proba(Z_test)[:, 1],
                                                   name=model_name.upper(), ax=roc_ax)
        np.random.seed(seed)
        explained = explain(model, Z_train, Z_test[rows], names)
        explanation[model_name] = explained
        save_figures(explained, OUT, model_name)

        voice, spiral = modality_share(explained, fitted_reducer.n_voice)
        importance = np.abs(explained.values).mean(axis=0)
        voice_features = sorted(zip(names[:fitted_reducer.n_voice], importance), key=lambda pair: -pair[1])
        report["models"][model_name] = {
            "roc_auc": float(display.roc_auc),
            "voice_share": voice,
            "spiral_share": spiral,
            "voice_importance": {name: float(value) for name, value in voice_features},
            "markers_present": [
                name for name, _ in voice_features if any(marker.lower() in name.lower() for marker in MARKERS)
            ],
        }
        if model_name == "vqc":
            loss, loss_ax = plt.subplots()
            loss_ax.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_)
            loss_ax.set(xlabel="Epoch", ylabel="Training binary cross-entropy")
            loss.savefig(OUT / "loss.png", dpi=200, bbox_inches="tight")
            circuit, _ = model.draw()
            circuit.savefig(OUT / "vqc_circuit.png", dpi=200, bbox_inches="tight")
            plt.close(loss)
            plt.close(circuit)
    roc.savefig(OUT / "roc.png", dpi=200, bbox_inches="tight")
    plt.close(roc)

    report["spearman"] = rank_agreement(explanation["vqc"], explanation["rf"])
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (RESULTS / "shap_metrics.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Figures in {OUT}")


if __name__ == "__main__":
    main()
