"""Raw explainability figures for the paper (F1-F3, F6, F7) into figures/raw/.

Holds out 25% of people, fits Random Forest and the VQC on the rest, explains
30 held-out samples with SHAP, and prints the agreement and voice/spiral share.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from qpd.data import ROOT, load_fused, load_voice  # noqa: E402
from qpd.experiments import get_model  # noqa: E402
from qpd.explain import explain, modality_share, rank_agreement, save_figures  # noqa: E402
from qpd.features import FusionReducer  # noqa: E402

OUT = ROOT / "figures" / "raw"
MODELS = ["rf", "vqc"]


def main(seed: int = 0, n_explain: int = 30):
    OUT.mkdir(parents=True, exist_ok=True)
    fused, n_voice_cols = load_fused(seed)
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=seed).split(fused.X, fused.y, fused.groups))
    reducer = FusionReducer(n_voice_cols=n_voice_cols, seed=seed).fit(fused.X[tr], fused.y[tr])
    Ztr, Zte = reducer.transform(fused.X[tr]), reducer.transform(fused.X[te])
    names = reducer.feature_names(load_voice().feature_names)

    expl = {}
    roc_ax = plt.figure().gca()
    for m in MODELS:
        print(f"Fitting and explaining {m}...")
        model = get_model(m, seed).fit(Ztr, fused.y[tr])
        RocCurveDisplay.from_predictions(fused.y[te], model.predict_proba(Zte)[:, 1], name=m, ax=roc_ax)
        expl[m] = explain(model, Ztr, Zte[:n_explain], names)
        save_figures(expl[m], OUT, m)
        voice, spiral = modality_share(expl[m])
        print(f"{m}: voice share {voice:.2f}, spiral share {spiral:.2f}")
        if m == "vqc":
            plt.figure()
            plt.plot(model.loss_curve_)
            plt.xlabel("epoch")
            plt.ylabel("training loss")
            plt.savefig(OUT / "loss.png", dpi=200, bbox_inches="tight")
            fig, _ = model.draw()
            fig.savefig(OUT / "vqc_circuit.png", dpi=200, bbox_inches="tight")
    roc_ax.figure.savefig(OUT / "roc.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    print(f"SHAP rank agreement, VQC vs RF (Spearman): {rank_agreement(expl['vqc'], expl['rf']):.2f}")
    print(f"Figures in {OUT}")


if __name__ == "__main__":
    main()
