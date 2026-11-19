# Sample voice rows

Two real rows from the UCI Parkinsons dataset (Little et al., 2007), licensed CC BY 4.0:
https://archive.ics.uci.edu/dataset/174/parkinsons

| File | Recording | Recorded from |
| --- | --- | --- |
| `voice_parkinsons_S01.csv` | `phon_R01_S01_1` | a person with Parkinson's |
| `voice_healthy_S07.csv` | `phon_R01_S07_1` | a healthy control |

The `status` column is left out on purpose: the app never sees the true label.
These files are also the format people can upload: a header with the 22 feature
names and one row of values. A `name` column is optional and ignored.
