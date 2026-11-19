"""Web app prototype (owner: Thanusri V).

Starter screen only. Run with:  streamlit run app/streamlit_app.py

The real prediction plugs in once results/model_bundle.joblib exists. Until then
the app runs in demo mode and says so.
"""

from pathlib import Path

import streamlit as st

BUNDLE = Path(__file__).resolve().parents[1] / "results" / "model_bundle.joblib"

st.set_page_config(page_title="Parkinson's screening prototype", layout="wide")
st.title("Parkinson's screening prototype")
st.warning("Research prototype. It is not a medical device and does not give a diagnosis.")

left, right = st.columns(2)
with left:
    st.subheader("1. Voice features")
    voice_csv = st.file_uploader("One-row CSV with the 22 UCI voice features", type="csv")
with right:
    st.subheader("2. Spiral drawing")
    spiral = st.file_uploader("Photo or scan of a hand-drawn spiral", type=["png", "jpg", "jpeg"])
    if spiral:
        st.image(spiral, width=260)

model_name = st.selectbox("Model", ["VQC (quantum)", "Quantum kernel SVM", "Random Forest"])

if st.button("Predict", type="primary", disabled=not (voice_csv and spiral)):
    if not BUNDLE.exists():
        st.info("Demo mode: no trained model yet (results/model_bundle.joblib is missing).")
    else:
        st.error("Prediction wiring is not built yet. See docs/webapp-tech.md.")
