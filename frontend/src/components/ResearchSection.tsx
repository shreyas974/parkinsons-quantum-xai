import { useState } from "react";
import { RESEARCH } from "../lib/research";

const R = RESEARCH;

function FusionChart() {
  const ROW = 46;
  const LEFT = 250;
  const W = 760;
  const H = R.fusion.length * ROW + 52;
  const X = (v: number) => LEFT + v * (W - LEFT - 130);
  return (
    <svg
      className="bars research-bars"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Balanced accuracy, mean and standard deviation: ${R.fusion.map((f) => `${f.label} ${f.mean.toFixed(3)} ± ${f.sd.toFixed(3)}`).join("; ")}.`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={4} y2={H - 30} className={t === 0.5 ? "chance" : "grid"} />
          <text x={X(t)} y={H - 10} textAnchor="middle" className="tick">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={X(0.5) + 5} y={14} className="chance-label">coin flip</text>
      {R.fusion.map((f, i) => {
        const y = 22 + i * ROW;
        const hi = Math.min(1, f.mean + f.sd);
        return (
          <g key={f.id}>
            <text x={LEFT - 12} y={y + 8} textAnchor="end" className="label">{f.label}</text>
            <text x={LEFT - 12} y={y + 26} textAnchor="end" className="tag">{f.kind} · {f.experiment}</text>
            <rect x={X(0)} y={y} width={X(f.mean) - X(0)} height={22} rx={4} className={f.kind} />
            <line x1={X(f.mean - f.sd)} x2={X(hi)} y1={y + 11} y2={y + 11} className="sd" />
            <line x1={X(f.mean - f.sd)} x2={X(f.mean - f.sd)} y1={y + 5} y2={y + 17} className="sd" />
            <line x1={X(hi)} x2={X(hi)} y1={y + 5} y2={y + 17} className="sd" />
            <text x={X(hi) + 8} y={y + 16} className="val">{f.mean.toFixed(3)} ± {f.sd.toFixed(3)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ShareBar({ model, voice, spiral }: { model: string; voice: number; spiral: number }) {
  return (
    <div className="share-row">
      <span className="share-model">{model}</span>
      <div className="share" role="img" aria-label={`${model}: voice ${voice}%, spiral ${spiral}% of absolute SHAP attribution`}>
        <span className="share-voice" style={{ flexGrow: voice }}>Voice {voice}%</span>
        <span className="share-spiral" style={{ flexGrow: spiral }}>Spiral {spiral}%</span>
      </div>
    </div>
  );
}

/** Verified findings from the research branch, with the full infographic as a zoomable preview. */
export function ResearchSection() {
  const [open, setOpen] = useState(false);
  return (
    <section className="research" id="research" data-anchor="4" aria-labelledby="research-h">
      <div className="section-head" data-reveal>
        <p className="eyebrow">04 · Research results</p>
        <h2 id="research-h">What the experiments found</h2>
        <p className="lede">
          Seven classical and quantum models, each scored on people it never saw in training: {R.design},{" "}
          {R.foldResults} fold results in all. Balanced accuracy of 0.50 is a coin flip.
        </p>
      </div>

      <div className="research-grid">
        <article className="glass research-card wide" data-reveal>
          <h3>Fusing voice and spiral: balanced accuracy</h3>
          <p className="muted small">Mean ± standard deviation over 15 runs per model.</p>
          <FusionChart />
          <ul className="fusion-list">
            {R.fusion.map((f) => (
              <li key={f.id}>
                <span className="fl-name">{f.label}</span>
                <span className="fl-val">{f.mean.toFixed(3)} ± {f.sd.toFixed(3)}</span>
                <span className="fl-meta">
                  <span className={f.kind === "quantum" ? "quantum-kind" : undefined}>{f.kind}</span> · {f.experiment}
                </span>
              </li>
            ))}
          </ul>
        </article>

        <div className="research-side" data-reveal-stagger>
          <article className="glass research-card" data-tilt>
            <p className="kicker">Quantum vs classical</p>
            <p className="stat">p = {R.wilcoxon.p}</p>
            <p>
              The {R.wilcoxon.quantum} and the {R.wilcoxon.classical} were statistically indistinguishable in a
              paired Wilcoxon test across {R.wilcoxon.pairs} matched seed/fold pairs. The test is exploratory.
            </p>
          </article>
          <article className="glass research-card" data-tilt>
            <p className="kicker">Do the explanations agree?</p>
            <p className="stat">ρ = {R.agreement.spearman}</p>
            <p>
              Spearman rank agreement between {R.agreement.a} and {R.agreement.b} SHAP feature rankings, on{" "}
              {R.agreement.examples} shared held-out examples: moderate agreement.
            </p>
          </article>
        </div>

        <article className="glass research-card wide" data-reveal>
          <h3>Where the decisions come from</h3>
          <p className="muted small">Share of absolute SHAP attribution, on the same held-out fold.</p>
          <div className="share-list">
            {R.shares.map((s) => (
              <ShareBar key={s.model} {...s} />
            ))}
          </div>
          <p>
            Voice carries most of the decision for both models. <b>{R.topVoiceFeatures[0]}</b> (pitch irregularity) and{" "}
            <b>{R.topVoiceFeatures[1]}</b> (pitch spread) were the top voice features for both on that fold, in line with
            known voice markers of Parkinson’s.
          </p>
        </article>
      </div>

      <figure className="glass infographic" data-reveal>
        <figcaption>
          <div>
            <h3>The full results board</h3>
            <p className="muted small">
              Every experiment (E1–E5) and ablation (A1–A4), the complete metrics matrix, simulator timing, and the raw
              SHAP, ROC, loss and circuit figures.
            </p>
          </div>
          <a className="btn btn-ghost btn-small" href={R.image.full} target="_blank" rel="noopener">
            Open full resolution ({R.image.width} × {R.image.height} PNG, {R.image.sizeMb} MB)
          </a>
        </figcaption>
        <div className={`infographic-frame${open ? " is-open" : ""}`} id="infographic-frame">
          <img
            src={R.image.preview}
            srcSet={`${R.image.preview} 1400w, ${R.image.full} ${R.image.width}w`}
            sizes="(max-width: 1240px) 100vw, 1180px"
            width={1400}
            height={3360}
            loading="lazy"
            decoding="async"
            alt="Research results infographic. Bar charts of balanced accuracy for voice (E1), spiral (E2) and early-fusion (E3) models; reference runs for late fusion (E4) and a random forest on all voice features (E5); line charts for the entanglement, circuit depth, depolarizing noise and training-group ablations (A1 to A4); simulator timing; SHAP attribution by modality and by voice feature; a complete matrix of mean and standard deviation for balanced accuracy, sensitivity, specificity, F1, ROC-AUC and training time; and SHAP beeswarm, importance and waterfall plots, ROC curves, the VQC training loss and the circuit diagram. The key numbers are written out on this page."
          />
        </div>
        <button type="button" className="btn btn-ghost btn-small infographic-toggle" aria-expanded={open} aria-controls="infographic-frame" onClick={() => setOpen((o) => !o)}>
          {open ? "Show less" : "Show the whole board"}
        </button>
      </figure>

      <article className="glass research-card limits" data-reveal>
        <h3>Limits of this study</h3>
        <ul>
          <li>
            The voice file has {R.data.recordings} recordings and {R.data.subjectIds} parsed subject IDs, including{" "}
            {R.data.healthyIds} healthy IDs.
          </li>
          <li>The spiral set has {R.data.spirals} drawings.</li>
          <li>
            The voice recordings and spiral drawings came from different people. They were paired using their known
            labels for the experiment.
          </li>
          <li>The quantum models ran on simulators, not quantum hardware.</li>
          <li>
            <b>This is research only.</b> Clinical use is outside the study’s scope.
          </li>
        </ul>
        <p className="fine">
          Source: research branch {R.branch}, commit {R.commit}. {R.foldResults} fold results, {R.summaryRows} summary
          rows; the research test suite ({R.tests} tests) passed.
        </p>
      </article>
    </section>
  );
}
