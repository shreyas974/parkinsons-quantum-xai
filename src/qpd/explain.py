"""SHAP explanations that work the same way for classical and quantum models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.stats import spearmanr


def explain(model, X_train, X_explain, feature_names, n_background: int = 10, nsamples: int = 200):
    """SHAP values for P(Parkinson's). X_* are already-reduced features.

    KernelExplainer calls the model many times; for quantum models keep
    X_explain to 20-40 rows.
    """
    background = shap.kmeans(X_train, min(n_background, len(X_train)))
    explainer = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], background)
    values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    return shap.Explanation(values=values, base_values=np.full(len(X_explain), explainer.expected_value),
                            data=X_explain, feature_names=feature_names)


def save_figures(expl: shap.Explanation, out_dir: Path, tag: str) -> list[Path]:
    """Beeswarm, mean |SHAP| bar and one waterfall (figures F1-F3)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, draw in [
        ("beeswarm", lambda: shap.plots.beeswarm(expl, show=False)),
        ("bar", lambda: shap.plots.bar(expl, show=False)),
        ("waterfall", lambda: shap.plots.waterfall(expl[0], show=False)),
    ]:
        plt.figure()
        draw()
        p = out_dir / f"{tag}_{kind}.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close("all")
        paths.append(p)
    return paths


def rank_agreement(a: shap.Explanation, b: shap.Explanation) -> float:
    """Spearman correlation of mean |SHAP| between two models (e.g. VQC vs RF)."""
    return float(spearmanr(np.abs(a.values).mean(0), np.abs(b.values).mean(0)).statistic)


def modality_share(expl: shap.Explanation, n_voice: int = 4) -> tuple[float, float]:
    """Share of total |SHAP| from voice vs spiral columns (figure F4)."""
    mag = np.abs(expl.values).sum(0)
    voice, spiral = mag[:n_voice].sum(), mag[n_voice:].sum()
    total = voice + spiral
    return float(voice / total), float(spiral / total)
