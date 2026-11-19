# Explainable Hybrid Quantum-Classical Learning for Parkinson's Prediction

Predicts Parkinson's disease from **voice features** and **hand-drawn spirals**, using a small simulated quantum circuit (PennyLane) alongside classical models, and explains every prediction with **SHAP**. A web app shows the result.

> Research prototype. Not a medical device and not a diagnosis.

## Developers

| Who | Owns | Branch |
| --- | --- | --- |
| Shreyas | Data, classical ML, quantum models, experiments, SHAP (`src/qpd/`, `scripts/`, `tests/`) | `shreyas/data-ml-quantum` |
| Thanusri V | Web app and its technology choice (`app/`, `docs/webapp-tech.md`) | `thanusri/web-app` |

Design (Nayeem) and documentation (Altaf) work outside this repo. See [CONTRIBUTING.md](CONTRIBUTING.md) for how we branch and merge.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
```

The spiral images need a Kaggle login; see [data/README.md](data/README.md).

## Run the research pipeline

The tests use synthetic data. Install the dependencies above, then run them with `python -m pytest -q`. Before running experiments, check that the downloader reports both modalities. The UCI file contains 32 distinct identifiers parsed from `name`, including 8 healthy IDs. Its dataset description lists 31 participants.


```bash
# One quick check: classical baselines on voice only
python -m qpd.experiments --models lr,rf --inputs voice --seeds 0 --out results/quick.csv

# Everything (slow: the quantum models are simulated)
nohup python -u scripts/run_full_experiments.py --jobs 4 > results/run.log 2>&1 &

# E4 late fusion using the seed-0 voice and spiral winners
python -m qpd.experiments --models late_fusion --inputs fusion --late-voice-model mlp --late-spiral-model rf --seeds 0 --out results/late_check.csv

# Explain RF and VQC, then train and verify the shared model bundle
python scripts/make_figures.py
python scripts/train_bundle.py

# Web app
streamlit run app/streamlit_app.py
```

The full runner writes `results/results.csv`, `results/summary.csv`, and `results/wilcoxon.csv` after validating all runs. Use `--out` for individual checks and `--loss-dir` to save VQC training loss by epoch. Choose the VQC depth with `vqc_l1` through `vqc_l4`. Use `vqc_noent` to omit CNOTs and `vqc_noise` to apply depolarizing noise on each wire after each layer with `default.mixed`. Set `--train-fraction` to subsample complete training groups. The full runner stores validated shards in `results/full_run/` and reuses them when resumed. Choose a fresh `--shard-dir` to recompute after changing model or data settings.

The figure script writes the raw PNGs to `figures/raw/` and the held-out attribution statistics to `results/shap_metrics.json`. The bundle script uses the main early-fusion scores for its metrics and writes `results/model_bundle.joblib` and `results/sample_voice_row.csv`.

## Layout

```
src/qpd/
  data.py         load voice + spirals, person IDs, label pairing
  features.py     HOG for spirals, in-fold scaling / selection / PCA to [0, pi]
  baselines.py    LR, SVM, Random Forest, MLP, XGBoost
  quantum.py      VQC and quantum kernel SVM (scikit-learn interface)
  experiments.py  person-wise cross-validation, metrics, results CSV
  explain.py      SHAP figures, rank agreement, voice vs spiral share
  bundle.py       web model bundle fields and background samples
app/
  streamlit_app.py  web app starter
scripts/
  download_data.py
  run_full_experiments.py
  make_figures.py
  train_bundle.py
tests/
docs/
  webapp-tech.md    web app technology options
```

## Rules that keep results honest

- Group voice folds by the subject ID in `name`. Group spiral-only folds by the repeated V-number in the image filename.
- Voice recordings and spiral drawings come from different people. Fusion pairs them using their known labels, with a disjoint pool of images per voice subject. Fusion experiments keep each spiral filename prefix within one outer fold and tune on those disjoint partitions. These scores describe an experiment with label-matched inputs.
- Tune classical hyperparameters on grouped splits inside each training fold. `rf_full` uses all voice features as the E5 reference.
- Fit scaling, feature selection and PCA **inside** each training fold.
- Report balanced accuracy, sensitivity, specificity, F1 and AUC.
- Results come from a simulator; we do not claim quantum advantage.
