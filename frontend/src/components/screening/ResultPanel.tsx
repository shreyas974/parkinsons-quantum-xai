import { formatPct } from "../../lib/features";
import type { PredictResponse } from "../../lib/types";
import { ShapWaterfall } from "./ShapWaterfall";

function Ring({ p, higher }: { p: number; higher: boolean }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  return (
    <svg className="ring" viewBox="0 0 140 140" role="img" aria-label={`Score ${formatPct(p)}`}>
      <circle cx="70" cy="70" r={r} className="ring-track" />
      <circle cx="70" cy="70" r={r} className={`ring-value ${higher ? "up" : "down"}`} strokeDasharray={`${c * p} ${c}`} transform="rotate(-90 70 70)" />
      <line x1="70" y1="8" x2="70" y2="22" className="ring-mid" transform="rotate(180 70 70)" />
      <text x="70" y="78" textAnchor="middle" className="ring-text">{formatPct(p)}</text>
    </svg>
  );
}

interface Props {
  status: "idle" | "running" | "done" | "error";
  result: PredictResponse | null;
  error: string;
  elapsed: number;
  onRetry: () => void;
}

export function ResultPanel({ status, result, error, elapsed, onRetry }: Props) {
  if (status === "running") {
    return (
      <div className="result glass is-running" aria-live="polite" aria-busy="true">
        <div className="spinner" aria-hidden="true" />
        <h3>{elapsed < 2 ? "Encoding 8 features into qubits…" : elapsed < 6 ? "Running the circuit…" : "Explaining the score with SHAP…"}</h3>
        <p className="muted">{elapsed}s · quantum models are simulated, so this can take a little while.</p>
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="result glass" role="alert">
        <h3>That didn’t work</h3>
        <p>{error}</p>
        <button type="button" className="btn btn-ghost btn-small" onClick={onRetry}>Try again</button>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="result glass is-empty" aria-live="polite">
        <h3>Your result appears here</h3>
        <p className="muted">Choose a voice recording, draw or upload a spiral, pick a model, then press Predict.</p>
      </div>
    );
  }

  const higher = result.label === "higher";
  return (
    <div className="result glass" aria-live="polite">
      {result.demo && (
        <p className="demo-banner" role="status">
          <b>Example output.</b> {result.note.replace(/^Example output\.\s*/, "")}
        </p>
      )}
      <div className="score-row">
        <Ring p={result.probability} higher={higher} />
        <div>
          <span className={`pill ${higher ? "up" : "down"}`}>{higher ? "Higher likelihood" : "Lower likelihood"}</span>
          <p className="score-text">
            The {result.model_label.toLowerCase()} scored these inputs at <b>{formatPct(result.probability)}</b>,{" "}
            {higher ? "above" : "below"} the 50% line.
          </p>
          <p className="muted small">
            {result.model_kind === "quantum" ? "Quantum" : "Classical"} model · {(result.elapsed_ms / 1000).toFixed(1)} s
          </p>
        </div>
      </div>

      <h4>Why: how each feature pushed the score</h4>
      <ShapWaterfall base={result.base_value} items={result.contributions} />

      <h4>Where the decision came from</h4>
      <div className="share" role="img" aria-label={`Voice ${formatPct(result.voice_share)}, spiral ${formatPct(result.spiral_share)}`}>
        <span className="share-voice" style={{ flexGrow: Math.max(0.001, result.voice_share) }}>Voice {formatPct(result.voice_share)}</span>
        <span className="share-spiral" style={{ flexGrow: Math.max(0.001, result.spiral_share) }}>Spiral {formatPct(result.spiral_share)}</span>
      </div>

      {!result.demo && result.note && (
        <p className="fine">
          {result.note} <a href="#research">See the research results</a>.
        </p>
      )}
      <a className="btn btn-ghost btn-small qubit-link" href="#quantum">See your 8 qubits in 3D ↑</a>
      <p className="disclaimer">Research prototype. Not a medical device and not a diagnosis. Talk to a doctor about any health concern.</p>
    </div>
  );
}
