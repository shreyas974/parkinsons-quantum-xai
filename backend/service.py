"""Turns uploaded voice features and a spiral image into an explained prediction.

The trained models come from results/model_bundle.joblib, produced by
scripts/train_bundle.py (Shreyas's side). Its keys are the contract between us:
reducer, models, feature_names, voice_columns, background, metrics.
Without the file the service runs in demo mode: inputs are still checked, but
the output is a fixed example and is labelled as such.
"""

from __future__ import annotations

import csv
import io
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
# The qpd package must be the same code that built the bundle. QPD_SRC points at another checkout's src/
# (for example the research branch) until both branches are merged; otherwise this repo's src/ is used.
QPD_SRC = Path(os.environ.get("QPD_SRC", ROOT / "src"))
if str(QPD_SRC) not in sys.path:
    sys.path.insert(0, str(QPD_SRC))

BUNDLE_PATH = Path(os.environ.get("QPD_BUNDLE", ROOT / "results" / "model_bundle.joblib"))
RESULTS_PATH = Path(os.environ.get("QPD_RESULTS", ROOT / "results" / "results.csv"))
SAMPLES_DIR = Path(__file__).with_name("samples")
SHAP_NSAMPLES = int(os.environ.get("QPD_SHAP_NSAMPLES", "200"))

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 6000

UCI_VOICE_COLUMNS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

MODEL_INFO = {
    "vqc": ("Variational quantum classifier", "quantum"),
    "vqc_noent": ("VQC without entanglement", "quantum"),
    "qsvm": ("Quantum kernel SVM", "quantum"),
    "rf": ("Random Forest", "classical"),
    "rf_full": ("Random Forest, all 22 voice features", "classical"),
    "lr": ("Logistic Regression", "classical"),
    "svm": ("SVM (RBF kernel)", "classical"),
    "mlp": ("Small neural network", "classical"),
    "xgb": ("XGBoost", "classical"),
    "late_fusion": ("Late fusion (voice + spiral models)", "classical"),
}
DEMO_MODELS = ["vqc", "qsvm", "rf", "lr"]

SAMPLES = [
    ("parkinsons_s01", "voice_parkinsons_S01.csv", "Recording phon_R01_S01_1",
     "Real UCI recording from a person with Parkinson's"),
    ("healthy_s07", "voice_healthy_S07.csv", "Recording phon_R01_S07_1",
     "Real UCI recording from a healthy control"),
]


class InputError(ValueError):
    """A problem with what the user sent. The message is shown to them as is."""


def model_info(model_id: str) -> tuple[str, str]:
    if model_id in MODEL_INFO:
        return MODEL_INFO[model_id]
    if model_id.startswith("vqc"):
        return (f"Variational quantum classifier ({model_id})", "quantum")
    return (model_id, "classical")


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------- input checks ----------

def voice_from_mapping(values: dict, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in values or str(values[c]).strip() == ""]
    if missing:
        shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        raise InputError(f"The voice data is missing {len(missing)} value(s): {shown}.")
    out = []
    for c in columns:
        f = _finite(values[c])
        if f is None:
            raise InputError(f"The voice value for {c} must be a number (got {values[c]!r}).")
        out.append(f)
    return np.array(out, dtype=float)


def voice_from_csv(data: bytes, columns: list[str]) -> np.ndarray:
    if not data.strip():
        raise InputError("The voice file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        raise InputError("The voice file is not a readable CSV. Use a header row with the 22 feature names.")
    if len(df) == 0:
        raise InputError("The voice file has a header but no row of values.")
    return voice_from_mapping(df.iloc[0].to_dict(), columns)


def load_image(data: bytes) -> Image.Image:
    """Decode an uploaded spiral, flattening any transparency onto white paper."""
    if not data:
        raise InputError("The spiral image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("The spiral image is larger than 8 MB. Use a smaller photo or scan.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise InputError("The spiral file is not a PNG or JPG image.")
    if img.format not in ("PNG", "JPEG"):
        raise InputError("The spiral must be a PNG or JPG image.")
    if max(img.size) > MAX_IMAGE_SIDE or min(img.size) < 32:
        raise InputError("The spiral image must be between 32 and 6000 pixels on each side.")
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        paper = Image.new("RGB", rgba.size, (255, 255, 255))
        paper.paste(rgba, mask=rgba.getchannel("A"))
        return paper
    return img.convert("RGB")


def spiral_features(img: Image.Image) -> np.ndarray:
    from qpd.features import hog_features  # same feature code as training

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f, format="PNG")
        f.flush()
        return np.asarray(hog_features(f.name), dtype=float)


# ---------- the predictor ----------

class Predictor:
    def __init__(self, bundle_path: Path = BUNDLE_PATH, bundle: dict | None = None):
        self.bundle_path = Path(bundle_path)
        self.bundle: dict | None = None
        self.reason = ""
        self.broken: dict[str, str] = {}
        self._lock = threading.Lock()  # one heavy prediction at a time; simulators are not thread-safe
        if bundle is not None:
            self._accept(bundle)
        else:
            self.reload()

    def _accept(self, bundle: dict):
        missing = {"reducer", "models", "feature_names", "voice_columns", "background"} - set(bundle)
        if missing:
            raise ValueError(f"the bundle is missing {', '.join(sorted(missing))}")
        # Run every model once on the stored background rows: a model whose code no longer matches the
        # qpd package (for example an older checkout) is hidden here instead of failing mid-request.
        background = np.asarray(bundle["background"], dtype=float)
        self.broken = {}
        for name, model in bundle["models"].items():
            try:
                p = np.asarray(model.predict_proba(background[:2]), dtype=float)
                if p.shape != (2, 2) or not np.all(np.isfinite(p)):
                    raise ValueError(f"unexpected output shape {p.shape}")
            except Exception as e:
                self.broken[name] = f"{type(e).__name__}: {e}"
        if len(self.broken) == len(bundle["models"]):
            raise ValueError("no model in the bundle could run: " + "; ".join(f"{k} ({v})" for k, v in self.broken.items()))
        self.bundle = bundle
        self.reason = (
            "Unavailable with the installed qpd code: " + ", ".join(sorted(self.broken)) + ". Set QPD_SRC to the "
            "checkout that built the bundle." if self.broken else ""
        )

    def reload(self) -> None:
        if not self.bundle_path.exists():
            self.bundle = None
            self.reason = (f"No trained model yet ({self.bundle_path.name} is missing), "
                           "so results are a fixed example.")
            return
        try:
            import joblib

            self._accept(joblib.load(self.bundle_path))
        except Exception as e:  # a broken or incompatible file must not take the site down
            self.bundle = None
            self.reason = f"The model file could not be loaded ({type(e).__name__}: {e}), so results are a fixed example."

    @property
    def demo(self) -> bool:
        return self.bundle is None

    @property
    def voice_columns(self) -> list[str]:
        return list(self.bundle["voice_columns"]) if self.bundle else list(UCI_VOICE_COLUMNS)

    def model_ids(self) -> list[str]:
        if not self.bundle:
            return list(DEMO_MODELS)
        # Quantum models first, in the order the research compares them.
        order = {m: i for i, m in enumerate(["vqc", "qsvm", "rf", "lr"])}
        ids = [m for m in self.bundle["models"] if m not in self.broken]
        return sorted(ids, key=lambda m: order.get(m, len(order)))

    def models(self) -> list[dict]:
        metrics = (self.bundle or {}).get("metrics") or {}
        out = []
        for m in self.model_ids():
            label, kind = model_info(m)
            out.append({"id": m, "label": label, "kind": kind, "balanced_accuracy": _finite(metrics.get(m))})
        return out

    def predict(self, voice: np.ndarray, image: Image.Image, model_id: str) -> dict:
        t0 = time.perf_counter()
        if model_id in self.broken:
            raise InputError(f"The {model_info(model_id)[0]} can't run with the installed code. Choose another model.")
        if model_id not in self.model_ids():
            raise InputError(f"Unknown model {model_id!r}. Choose one of: {', '.join(self.model_ids())}.")
        if self.demo:
            return demo_result(model_id, t0, self.reason)

        b = self.bundle
        with self._lock:
            raw = np.concatenate([voice, spiral_features(image)])[None, :]
            x = np.asarray(b["reducer"].transform(raw), dtype=float)
            model = b["models"][model_id]
            p = float(model.predict_proba(x)[0, 1])
            values, base = shap_for_one(model, np.asarray(b["background"], dtype=float), x)

        names = list(b["feature_names"])
        n_voice = int(getattr(b["reducer"], "n_voice", 4))
        contributions = [
            {"feature": n, "signal": "voice" if i < n_voice else "spiral", "value": float(x[0, i]),
             "shap": float(values[i])}
            for i, n in enumerate(names)
        ]
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        mag = np.abs(values)
        total = float(mag.sum())
        voice_share = float(mag[:n_voice].sum() / total) if total > 0 else 0.0
        label, kind = model_info(model_id)
        return {
            "demo": False, "model": model_id, "model_label": label, "model_kind": kind,
            "probability": p, "label": "higher" if p >= 0.5 else "lower", "base_value": base,
            "contributions": contributions, "angles": [float(np.clip(a, 0, np.pi)) for a in x[0]],
            "feature_names": names, "voice_share": voice_share,
            "spiral_share": (1.0 - voice_share) if total > 0 else 0.0,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "note": "This model was trained on the full research data, so this score is not a tested result. "
                    "Tested performance is in the research results.",
        }


def shap_for_one(model, background: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """SHAP values of P(Parkinson's) for one row, the same KernelExplainer setup as the paper figures."""
    import shap

    explainer = shap.KernelExplainer(lambda X: model.predict_proba(np.asarray(X))[:, 1], background)
    values = np.asarray(explainer.shap_values(x, nsamples=SHAP_NSAMPLES, silent=True), dtype=float)
    return values.reshape(-1)[: x.shape[1]], float(np.ravel(explainer.expected_value)[0])


def demo_result(model_id: str, t0: float, reason: str) -> dict:
    names = ["PPE", "spread1", "MDVP:Fo(Hz)", "HNR", "spiral_PC1", "spiral_PC2", "spiral_PC3", "spiral_PC4"]
    angles = [2.30, 2.05, 0.95, 1.10, 2.10, 1.35, 0.70, 1.60]
    shaps = [0.14, 0.10, -0.03, -0.04, 0.08, 0.03, -0.02, 0.02]
    contributions = sorted(
        ({"feature": n, "signal": "voice" if i < 4 else "spiral", "value": a, "shap": s}
         for i, (n, a, s) in enumerate(zip(names, angles, shaps))),
        key=lambda c: abs(c["shap"]), reverse=True,
    )
    label, kind = model_info(model_id)
    voice = sum(abs(s) for s in shaps[:4]) / sum(abs(s) for s in shaps)
    return {
        "demo": True, "model": model_id, "model_label": label, "model_kind": kind,
        "probability": 0.5 + sum(shaps), "label": "higher", "base_value": 0.5,
        "contributions": contributions, "angles": angles, "feature_names": names,
        "voice_share": voice, "spiral_share": 1 - voice,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "note": "Example output. " + reason,
    }


# ---------- samples and results ----------

def voice_samples() -> list[dict]:
    out = []
    for sid, fn, label, desc in SAMPLES:
        path = SAMPLES_DIR / fn
        if not path.exists():
            continue
        row = next(csv.DictReader(path.open()))
        values = {k: float(v) for k, v in row.items() if k in UCI_VOICE_COLUMNS}
        out.append({"id": sid, "label": label, "description": desc, "values": values})
    return out


def results_summary(path: Path = RESULTS_PATH) -> dict:
    if not path.exists():
        return {"available": False, "rows": [],
                "note": "No experiment results yet. They appear here once results/results.csv exists."}
    df = pd.read_csv(path)
    if df.empty or not {"input", "model"} <= set(df.columns):
        return {"available": False, "rows": [], "note": "results.csv has no rows to summarise."}
    rows = []
    # The research file tags each row with its experiment (E1-E5 main runs, A1-A4 ablations). Ablations reuse
    # model names with other settings, so they are left out here; the research infographic covers them.
    if "experiment" in df.columns:
        df = df[df["experiment"].astype(str).str.match(r"^E\d+$")]
        if df.empty:
            return {"available": False, "rows": [], "note": "results.csv has no main-experiment (E1-E5) rows."}
    keys = ["experiment", "input", "model"] if "experiment" in df.columns else ["input", "model"]
    for key, d in df.groupby(keys):
        exp, inp, model = key if len(keys) == 3 else ("", *key)
        label, kind = model_info(str(model))
        row = {"experiment": str(exp), "input": str(inp), "model": str(model), "label": label, "kind": kind,
               "folds": int(len(d)), "seeds": int(d["seed"].nunique()) if "seed" in d else None}
        for m in ["balanced_accuracy", "sensitivity", "specificity", "f1", "auc", "train_time_s"]:
            row[m] = _finite(d[m].mean()) if m in d else None
        row["balanced_accuracy_std"] = _finite(d["balanced_accuracy"].std()) if "balanced_accuracy" in d else None
        rows.append(row)
    rows.sort(key=lambda r: (r["experiment"], r["input"], -(r["balanced_accuracy"] or 0)))
    return {"available": True, "rows": rows,
            "note": "Cross-validated by person. Mean ± SD over every seed and fold in results.csv."}
