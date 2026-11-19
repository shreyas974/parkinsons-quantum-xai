"""API tests. They need no trained model and no dataset: demo mode plus a small stand-in bundle."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend import main, service


def spiral_png(transparent: bool = False) -> bytes:
    img = Image.new("RGBA" if transparent else "RGB", (240, 240), (0, 0, 0, 0) if transparent else "white")
    d = ImageDraw.Draw(img)
    pts = [(120 + 3.2 * t * np.cos(t), 120 + 3.2 * t * np.sin(t)) for t in np.linspace(0, 30, 400)]
    d.line(pts, fill=(20, 20, 60, 255), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "predictor", service.Predictor(bundle_path=tmp_path / "missing.joblib"))
    return TestClient(main.app)


def sample_json() -> str:
    return json.dumps(service.voice_samples()[0]["values"])


def test_health_reports_demo_mode(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["demo"] is True and "missing" in body["reason"]
    assert len(body["voice_columns"]) == 22


def test_samples_are_two_real_rows(client):
    samples = client.get("/api/samples").json()
    assert [s["id"] for s in samples] == ["parkinsons_s01", "healthy_s07"]
    assert all(len(s["values"]) == 22 for s in samples)


def test_models_list(client):
    ids = [m["id"] for m in client.get("/api/models").json()]
    assert "vqc" in ids and "rf" in ids


def test_predict_demo_with_sample_and_drawn_spiral(client):
    r = client.post("/api/predict", data={"model": "vqc", "voice_json": sample_json()},
                    files={"spiral": ("spiral.png", spiral_png(transparent=True), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["demo"] is True and body["note"].startswith("Example output")
    assert 0 <= body["probability"] <= 1
    assert len(body["angles"]) == 8 and len(body["contributions"]) == 8


def test_predict_accepts_csv_upload(client):
    csv_bytes = (service.SAMPLES_DIR / "voice_healthy_S07.csv").read_bytes()
    r = client.post("/api/predict", data={"model": "rf"},
                    files={"voice_csv": ("v.csv", csv_bytes, "text/csv"),
                           "spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 200, r.text


def test_missing_columns_named(client):
    values = json.loads(sample_json())
    del values["PPE"]
    r = client.post("/api/predict", data={"voice_json": json.dumps(values)},
                    files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 422 and "PPE" in r.json()["detail"]


def test_non_numeric_value_rejected(client):
    values = json.loads(sample_json())
    values["HNR"] = "loud"
    r = client.post("/api/predict", data={"voice_json": json.dumps(values)},
                    files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 422 and "HNR" in r.json()["detail"]


def test_bad_image_rejected(client):
    r = client.post("/api/predict", data={"voice_json": sample_json()},
                    files={"spiral": ("s.txt", b"not an image", "text/plain")})
    assert r.status_code == 422 and "PNG or JPG" in r.json()["detail"]


def test_voice_required(client):
    r = client.post("/api/predict", files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 422 and "voice" in r.json()["detail"].lower()


def test_unknown_model_rejected(client):
    r = client.post("/api/predict", data={"voice_json": sample_json(), "model": "nope"},
                    files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 422 and "Unknown model" in r.json()["detail"]


class _Reducer:
    n_voice = 4

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] > 22, "voice + HOG columns expected"
        return np.tile(np.linspace(0.2, 2.9, 8), (len(X), 1))


class _Model:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = 1 / (1 + np.exp(-(X[:, 0] - 1.0)))
        return np.column_stack([1 - p, p])


def test_predict_with_a_bundle(monkeypatch):
    pytest.importorskip("skimage")
    pytest.importorskip("shap")
    bundle = {"reducer": _Reducer(), "models": {"lr": _Model()},
              "feature_names": [f"f{i}" for i in range(8)], "voice_columns": service.UCI_VOICE_COLUMNS,
              "background": np.random.default_rng(0).uniform(0, np.pi, (10, 8)), "metrics": {"lr": 0.81}}
    monkeypatch.setattr(main, "predictor", service.Predictor(bundle=bundle))
    monkeypatch.setattr(service, "SHAP_NSAMPLES", 64)
    c = TestClient(main.app)
    assert c.get("/api/health").json()["demo"] is False
    assert c.get("/api/models").json()[0]["balanced_accuracy"] == 0.81
    r = c.post("/api/predict", data={"model": "lr", "voice_json": sample_json()},
               files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["demo"] is False
    assert abs(body["voice_share"] + body["spiral_share"] - 1) < 1e-6
    assert body["contributions"][0]["feature"] == "f0"  # the only feature the stand-in model uses


class _Broken:
    def predict_proba(self, X):
        raise AttributeError("'VQC' object has no attribute 'circuit_'")


def test_models_that_cannot_run_are_hidden_with_a_reason(monkeypatch):
    bundle = {"reducer": _Reducer(), "models": {"vqc": _Broken(), "lr": _Model()},
              "feature_names": [f"f{i}" for i in range(8)], "voice_columns": service.UCI_VOICE_COLUMNS,
              "background": np.random.default_rng(0).uniform(0, np.pi, (10, 8)), "metrics": {}}
    p = service.Predictor(bundle=bundle)
    assert p.model_ids() == ["lr"] and "vqc" in p.reason
    monkeypatch.setattr(main, "predictor", p)
    r = TestClient(main.app).post("/api/predict", data={"model": "vqc", "voice_json": sample_json()},
                                  files={"spiral": ("s.png", spiral_png(), "image/png")})
    assert r.status_code == 422 and "can't run" in r.json()["detail"]


def test_results_summary_keeps_main_experiments_apart(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "model,input,seed,fold,train_time_s,balanced_accuracy,sensitivity,specificity,f1,auc,experiment\n"
        "vqc,fusion,0,0,30,0.60,0.9,0.3,0.8,0.7,E3\n"
        "vqc,fusion,0,1,30,0.50,0.9,0.1,0.8,0.6,E3\n"
        "vqc,fusion,0,0,30,0.99,0.9,0.9,0.9,0.9,A2\n"
    )
    s = service.results_summary(csv_path)
    assert [(r["experiment"], r["model"], r["folds"]) for r in s["rows"]] == [("E3", "vqc", 2)]
    assert abs(s["rows"][0]["balanced_accuracy"] - 0.55) < 1e-9


def test_results_summary(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "model,input,seed,fold,train_time_s,balanced_accuracy,sensitivity,specificity,f1,auc\n"
        "rf,voice,0,0,1.0,0.80,0.9,0.7,0.85,0.88\n"
        "rf,voice,0,1,1.2,0.70,0.8,0.6,0.80,0.81\n"
        "vqc,voice,0,0,30,0.75,0.8,0.7,0.80,0.83\n"
    )
    monkeypatch.setattr(service, "RESULTS_PATH", csv_path)
    s = service.results_summary(csv_path)
    assert s["available"] is True
    rf = next(r for r in s["rows"] if r["model"] == "rf")
    assert rf["folds"] == 2 and abs(rf["balanced_accuracy"] - 0.75) < 1e-9
