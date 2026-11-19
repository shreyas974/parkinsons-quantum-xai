"""Serialization contract for the fused model bundle."""

import numpy as np
import shap

MODEL_NAMES = ("vqc", "qsvm", "rf", "lr")


def build_bundle(reducer, models, voice_columns, reduced_training, metrics):
    names = reducer.feature_names(voice_columns)
    if len(voice_columns) != 22 or len(names) != 8 or reduced_training.shape[1] != 8:
        raise ValueError("Expected 22 original voice columns and 8 reduced features")
    if set(models) != set(MODEL_NAMES) or set(metrics) != set(MODEL_NAMES):
        raise ValueError("Expected a fitted model and a fusion balanced accuracy for each bundle model")
    if not np.all(np.isfinite(list(metrics.values()))) or any(not 0 <= value <= 1 for value in metrics.values()):
        raise ValueError("Balanced accuracy must be finite and within [0, 1]")
    return {
        "reducer": reducer,
        "models": models,
        "feature_names": names,
        "voice_columns": voice_columns,
        "background": shap.kmeans(reduced_training, min(10, len(reduced_training))).data,
        "metrics": metrics,
    }
