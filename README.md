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

## Run

```bash
# One quick check: classical baselines on voice only
python -m qpd.experiments --models lr,rf --inputs voice --seeds 0

# Everything (slow: the quantum models are simulated)
python -m qpd.experiments --models lr,svm,rf,mlp,xgb,vqc,qsvm --inputs voice,spiral,fusion --seeds 0,1,2

# E4 late fusion using the seed-0 voice and spiral winners
python -m qpd.experiments --models late_fusion --inputs fusion --late-voice-model mlp --late-spiral-model rf --seeds 0

# Web app
streamlit run app/streamlit_app.py
```

Results are appended to `results/results.csv`. Use `--out` for separate ablation runs and `--loss-dir` to save VQC training loss by epoch. Choose the VQC depth with `vqc_l1` through `vqc_l4`. Use `vqc_noent` to omit CNOTs and `vqc_noise` to apply depolarizing noise on each wire after each layer with `default.mixed`. Set `--train-fraction` to subsample complete training subjects.

## Layout

```
src/qpd/
  data.py         load voice + spirals, person IDs, label pairing
  features.py     HOG for spirals, in-fold scaling / selection / PCA to [0, pi]
  baselines.py    LR, SVM, Random Forest, MLP, XGBoost
  quantum.py      VQC and quantum kernel SVM (scikit-learn interface)
  experiments.py  person-wise cross-validation, metrics, results CSV
  explain.py      SHAP figures, rank agreement, voice vs spiral share
app/
  streamlit_app.py  web app starter
scripts/
  download_data.py
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
