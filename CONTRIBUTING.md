# Working together

## Branches

- `main` is always runnable. Nobody pushes to it directly.
- Each developer works on their own branch:
  - Shreyas: `shreyas/data-ml-quantum`
  - Thanusri V: `thanusri/web-app`
- Merge into `main` through a pull request at a sync point, after checking that `pytest` passes and the app starts.

```bash
git checkout shreyas/data-ml-quantum   # or thanusri/web-app
git pull
# work, then
git add <files>
git commit -m "Short description of the change"
git push
```

To pick up the other developer's merged work: `git checkout main && git pull && git checkout <your-branch> && git merge main`.

## Who edits what

| Path | Owner |
| --- | --- |
| `src/qpd/`, `scripts/`, `tests/` | Shreyas |
| `app/`, `docs/webapp-tech.md` | Thanusri V |
| `README.md`, `requirements.txt` | Either, by agreement |

Editing someone else's files? Ask first, or open a pull request for them to review.

## The interface between us

The app only needs these calls from `qpd`, and they keep their shape:

- `results/model_bundle.joblib`: a dict with the fitted `reducer`, `models` (name to fitted model) and `feature_names`.
- `model.predict_proba(X)[:, 1]`: the probability of Parkinson's.
- `qpd.explain.explain(model, X_train, X_explain, feature_names)`: a SHAP explanation for the waterfall chart.

## Data

Never commit data files or trained models. `data/` and `results/` are ignored except their README and `.gitkeep`.
