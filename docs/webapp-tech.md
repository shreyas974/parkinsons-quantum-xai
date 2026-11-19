# Web app: technology choice

Owner: Thanusri V. Fill in the decision at the bottom once you've compared the options.

## What the app must do

1. Take a voice-feature CSV (22 UCI features) and a spiral image.
2. Run the saved pipeline from `results/model_bundle.joblib` (Python: scikit-learn and PennyLane).
3. Show the Parkinson's probability, a SHAP waterfall chart and a clear "not a diagnosis" note.
4. Work on a laptop for the demo; a public link is a bonus.

Because the models are Python, the backend has to run Python. The question is how much of a separate frontend we want.

## Options to compare

| Option | Frontend | Backend | Good at | Watch out for |
| --- | --- | --- | --- | --- |
| Streamlit | Built in | Same Python file | Fastest to build; project brief names it; SHAP plots via `st.pyplot` | Limited layout control |
| Gradio | Built in | Same Python file | Very quick ML demos; free hosting on Hugging Face Spaces | Less flexible for multi-step screens |
| FastAPI + React (or Next.js) | Separate JS app | FastAPI REST API | Full design control; production-style | Two codebases; much more time |
| Flask + HTML templates | Simple HTML/CSS | Flask | Familiar, lightweight | Charts and uploads take more hand-work |

Hosting ideas: Streamlit Community Cloud, Hugging Face Spaces, Render.

## Questions to answer

- Can it show a matplotlib SHAP figure without extra work?
- How long does a quantum model prediction take, and does the UI need a spinner?
- Can Nayeem's design (colours, fonts, layout) be applied?
- Can we deploy it for free for the paper demo?

## Decision

_To be filled in by Thanusri:_ chosen stack, why, and what we give up.
