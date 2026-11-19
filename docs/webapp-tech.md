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

**FastAPI backend + React frontend with a Three.js 3D layer.** Streamlit (`app/streamlit_app.py`) stays as the fallback.

| Layer | Choice | Why |
| --- | --- | --- |
| API | FastAPI, Uvicorn, Pydantic | Python, so it loads the same `qpd` code and model bundle; file uploads and typed responses built in; docs at `/docs` |
| Frontend | React 18, TypeScript, Vite | Fast to build, types catch API mistakes, splits the 3D code into a separate download |
| 3D | three.js, React Three Fiber, drei, postprocessing (bloom) | The standard stack behind most award-winning 3D sites; declarative scenes in React |
| Motion | GSAP ScrollTrigger, Lenis | Pinned scroll story, reveals and smooth wheel scrolling, all switched off for reduced motion |
| Charts | Hand-written SVG | SHAP waterfall and results bars drawn to scale, values printed on every bar |

What we give up: two codebases (Python and TypeScript) and a larger download than Streamlit (the 3D code is about 236 KB compressed, loaded after the text).

## How the site is built

One fixed 3D canvas sits behind normal, readable HTML. As you scroll, the camera flies through the scenes:

1. **Hero:** a gold tremor spiral draws itself and follows the pointer.
2. **Two signals:** a steady spiral and voice wave beside shaky ones (jitter shifts timing, shimmer changes height).
3. **Eight qubits (pinned for two screens):** scrolling runs the circuit. Arrows start at |0⟩, tip to each feature's RY angle (encode), turn while the CNOT ring glows (entangle), then only the measured qubit stays bright (measure). After a prediction, the arrows use the real encoded angles.
4. **Screening tool:** voice sample or CSV, spiral drawn on a pad / uploaded / synthetic example, model choice, then the score ring, SHAP waterfall and voice vs spiral share.
5. **Results:** cross-validated scores from `results/results.csv`, once Shreyas's experiments exist.

Colour follows a 60-30-10 split with four colours: navy `#0A1828` ground, turquoise `#178582` / `#3CC2BB` structure and quantum parts, gold `#E3BD72` for calls to action and pushes towards Parkinson's, off-white text.

Real-world practices applied: text first and 3D loaded later; pixel ratio capped and bloom turned off when the frame rate drops; a lighter scene on phones and low-core machines; no React state updates inside the render loop; a static fallback without WebGL; reduced motion gives a still scene, no pin and no smooth scroll; 44–48 px touch targets; 3 px focus outline; motion at 150 / 250 / 400 ms with one easing curve.

References read for this: [Awwwards 3D websites](https://www.awwwards.com/websites/3d/), [Utsubo, best Three.js websites 2026](https://www.utsubo.com/blog/best-threejs-websites-2026), [Hostinger, website colour schemes](https://www.hostinger.com/in/tutorials/website-color-schemes/), [claudedesignskills](https://github.com/freshtechbro/claudedesignskills) (three.js, R3F, GSAP and modern-web-design skills), [3D Web Experience Architect](https://mcpmarket.com/tools/skills/3d-web-experience-architect-5), [MindStudio, animated 3D websites](https://www.mindstudio.ai/blog/animated-3d-websites-claude-code-ai-video-generation).

## How to run

```bash
# Backend (from the repo root)
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

```bash
# Frontend, in a second terminal (opens http://localhost:5173, forwards /api to port 8000)
cd frontend && npm install && npm run dev
```

For one server: `cd frontend && npm run build`, then the backend also serves the site at http://localhost:8000.

Without `results/model_bundle.joblib` the site runs in **demo mode**: inputs are checked, and the result is a fixed example clearly labelled as such. When Shreyas's `scripts/train_bundle.py` writes the bundle, restart the backend (or `POST /api/reload`) and the real models are used. The bundle must be built with the same `qpd` code the backend imports, so merge Shreyas's branch into `main` before the final demo.

Tests: `python -m pytest backend/tests -q` and `cd frontend && npm test`. GitHub Actions runs both (`.github/workflows/webapp.yml`).

## Research results section

Section 04 shows the verified findings from the research branch (`shreyas/data-ml-quantum`, infographic commit `2079598`) as HTML, so they read on phones and with screen readers. The numbers live in one file, `frontend/src/lib/research.ts`. The infographic is `frontend/public/research/research-results.png`, identical to `docs/research-results.png` at that commit, plus a 1400 px WebP preview made from it. Section 05 ("Explore the runs") appears only when the server has `results/results.csv`; it lists E1–E5 and leaves the A1–A4 ablations to the infographic.

## Running with the real model bundle

The bundle, datasets and raw results are local, ignored artifacts. The backend finds them through environment variables, and the `qpd` code it imports must be the code that built the bundle:

| Variable | Default | Points to |
| --- | --- | --- |
| `QPD_BUNDLE` | `results/model_bundle.joblib` | The exported bundle |
| `QPD_RESULTS` | `results/results.csv` | Fold-level results for section 05 |
| `QPD_SRC` | `src/` in this checkout | The `src/` folder of the checkout that built the bundle |
| `QPD_SHAP_NSAMPLES` | `200` | SHAP samples per explanation (lower is faster) |

Until the research branch is merged, this branch's own `src/qpd` is older: with it, Random Forest and Logistic Regression load, but the VQC and quantum-kernel SVM cannot rebuild their circuits. The backend now tests every model when it loads the bundle, hides any that fail, and says why in `/api/health`. Example with the research code checked out next to this one:

```bash
QPD_SRC=../parkinsons-quantum-xai/src \
QPD_BUNDLE=../parkinsons-quantum-xai/results/model_bundle.joblib \
QPD_RESULTS=../parkinsons-quantum-xai/results/results.csv \
uvicorn backend.main:app --port 8000
```

Predictions from the bundle come from models trained on all the research data, so the site labels them as not tested results and links to section 04.
