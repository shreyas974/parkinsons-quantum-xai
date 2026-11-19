# Data

Data files are not committed. Put them here locally.

## Voice: UCI Parkinsons

195 recordings from 31 people (23 with Parkinson's), 22 voice features. CC BY 4.0.

```bash
python scripts/download_data.py
```

This saves `data/voice/parkinsons.data`.

## Spiral: Kaggle "Parkinson's Drawings"

102 spiral images (Zham et al., 2017). Needs a Kaggle login, so download it by hand and copy the `spiral` folder here:

```
data/spiral/training/healthy/*.png
data/spiral/training/parkinson/*.png
data/spiral/testing/healthy/*.png
data/spiral/testing/parkinson/*.png
```

Run `python scripts/download_data.py` again to check the images are found.

## Important

- Each person has about 6 voice recordings. Always split by person (`groups`), never by row.
- Voice and spiral data come from different people. `qpd.data.pair_by_label` pairs them by label and keeps each spiral on one side of a split.
