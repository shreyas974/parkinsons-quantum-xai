"""Classical baseline classifiers, all given the same reduced features."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_baselines(seed: int = 0) -> dict:
    models = {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=seed
        )
    except ImportError:
        pass
    return models
