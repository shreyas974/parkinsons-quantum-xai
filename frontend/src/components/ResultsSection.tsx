import { useEffect, useMemo, useState } from "react";
import { getResults } from "../lib/api";
import type { ResultRow, ResultsSummary } from "../lib/types";

const INPUT_LABEL: Record<string, string> = { voice: "Voice only", spiral: "Spiral only", fusion: "Voice + spiral" };
const fmt = (v: number | null, d = 2) => (v === null ? "—" : v.toFixed(d));

function Chart({ rows }: { rows: ResultRow[] }) {
  const ROW = 30;
  const LEFT = 230;
  const W = 680;
  const H = rows.length * ROW + 46;
  const X = (v: number) => LEFT + v * (W - LEFT - 30);
  return (
    <svg className="bars" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Balanced accuracy per model, cross-validated by person">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={4} y2={H - 30} className={t === 0.5 ? "chance" : "grid"} />
          <text x={X(t)} y={H - 12} textAnchor="middle" className="tick">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={X(0.5) + 4} y={14} className="chance-label">chance</text>
      {rows.map((r, i) => {
        const y = 10 + i * ROW;
        const v = r.balanced_accuracy ?? 0;
        const sd = r.balanced_accuracy_std ?? 0;
        return (
          <g key={r.model}>
            <text x={LEFT - 12} y={y + 14} textAnchor="end" className="label">
              {r.label}
              <tspan dx={6} className="tag">{r.kind}</tspan>
            </text>
            <rect x={X(0)} y={y} width={X(v) - X(0)} height={18} rx={4} className={r.kind} />
            {sd > 0 && <line x1={X(Math.max(0, v - sd))} x2={X(Math.min(1, v + sd))} y1={y + 9} y2={y + 9} className="sd" />}
            <text x={X(Math.min(1, v + sd)) + 6} y={y + 14} className="val">{v.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function ResultsSection() {
  const [data, setData] = useState<ResultsSummary | null>(null);
  const [failed, setFailed] = useState(false);
  const inputs = useMemo(() => Array.from(new Set((data?.rows ?? []).map((r) => r.input))), [data]);
  const [input, setInput] = useState("fusion");

  useEffect(() => {
    getResults().then(setData).catch(() => setFailed(true));
  }, []);
  useEffect(() => {
    if (inputs.length && !inputs.includes(input)) setInput(inputs[0]);
  }, [inputs, input]);

  const rows = (data?.rows ?? []).filter((r) => r.input === input);

  return (
    <section className="results" id="results" data-anchor="4" aria-labelledby="results-h">
      <div className="section-head" data-reveal>
        <p className="eyebrow">04 · Results</p>
        <h2 id="results-h">How the models compare</h2>
        <p className="lede">Balanced accuracy, tested on people the models never saw in training. 0.50 is a coin flip.</p>
      </div>
      <div className="glass results-card" data-reveal>
        {failed && <p className="muted">Results load from the prediction service, which isn’t reachable right now.</p>}
        {data && !data.available && (
          <div className="empty">
            <h3>No experiment results yet</h3>
            <p className="muted">{data.note}</p>
          </div>
        )}
        {data?.available && (
          <>
            <div className="segmented" role="radiogroup" aria-label="Input">
              {inputs.map((i) => (
                <button key={i} type="button" role="radio" aria-checked={i === input} onClick={() => setInput(i)}>
                  {INPUT_LABEL[i] ?? i}
                </button>
              ))}
            </div>
            <Chart rows={rows} />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th><th>Balanced acc.</th><th>Sensitivity</th><th>Specificity</th><th>F1</th><th>AUC</th><th>Train time</th><th>Folds</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.model}>
                      <td>{r.label} <span className={`tag ${r.kind}`}>{r.kind}</span></td>
                      <td>{fmt(r.balanced_accuracy)} ± {fmt(r.balanced_accuracy_std)}</td>
                      <td>{fmt(r.sensitivity)}</td>
                      <td>{fmt(r.specificity)}</td>
                      <td>{fmt(r.f1)}</td>
                      <td>{fmt(r.auc)}</td>
                      <td>{r.train_time_s === null ? "—" : `${r.train_time_s.toFixed(1)} s`}</td>
                      <td>{r.folds}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="fine">{data.note}</p>
          </>
        )}
      </div>
    </section>
  );
}
