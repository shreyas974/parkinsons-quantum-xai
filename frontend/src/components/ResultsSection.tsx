import { useEffect, useMemo, useState } from "react";
import { getResults } from "../lib/api";
import type { ResultRow, ResultsSummary } from "../lib/types";

const EXPERIMENT_LABEL: Record<string, string> = {
  E1: "E1 · Voice",
  E2: "E2 · Spiral",
  E3: "E3 · Early fusion",
  E4: "E4 · Late fusion",
  E5: "E5 · All voice features",
};
const INPUT_LABEL: Record<string, string> = { voice: "Voice only", spiral: "Spiral only", fusion: "Voice + spiral" };
const fmt = (v: number | null, d = 3) => (v === null ? "—" : v.toFixed(d));
const groupOf = (r: ResultRow) => r.experiment || r.input;
const groupLabel = (g: string) => EXPERIMENT_LABEL[g] ?? INPUT_LABEL[g] ?? g;

function Chart({ rows }: { rows: ResultRow[] }) {
  const ROW = 30;
  const LEFT = 250;
  const W = 680;
  const H = rows.length * ROW + 46;
  const X = (v: number) => LEFT + v * (W - LEFT - 70);
  return (
    <svg className="bars" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Balanced accuracy per model, cross-validated by person">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={4} y2={H - 30} className={t === 0.5 ? "chance" : "grid"} />
          <text x={X(t)} y={H - 12} textAnchor="middle" className="tick">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={X(0.5) + 4} y={14} className="chance-label">coin flip</text>
      {rows.map((r, i) => {
        const y = 10 + i * ROW;
        const v = r.balanced_accuracy ?? 0;
        const sd = r.balanced_accuracy_std ?? 0;
        return (
          <g key={`${r.input}-${r.model}`}>
            <text x={LEFT - 12} y={y + 14} textAnchor="end" className="label">
              {r.label}
              <tspan dx={6} className="tag">{r.kind}</tspan>
            </text>
            <rect x={X(0)} y={y} width={X(v) - X(0)} height={18} rx={4} className={r.kind} />
            {sd > 0 && <line x1={X(Math.max(0, v - sd))} x2={X(Math.min(1, v + sd))} y1={y + 9} y2={y + 9} className="sd" />}
            <text x={X(Math.min(1, v + sd)) + 6} y={y + 14} className="val">{v.toFixed(3)}</text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Every main experiment from results/results.csv, when the server has that file (a local research artifact).
 * The public site has no such file, so the section stays hidden there; the research section covers it.
 */
export function ResultsSection() {
  const [data, setData] = useState<ResultsSummary | null>(null);
  const groups = useMemo(() => Array.from(new Set((data?.rows ?? []).map(groupOf))), [data]);
  const [group, setGroup] = useState("E3");

  useEffect(() => {
    getResults().then(setData).catch(() => setData(null));
  }, []);
  useEffect(() => {
    if (groups.length && !groups.includes(group)) setGroup(groups.includes("E3") ? "E3" : groups[0]);
  }, [groups, group]);

  if (!data?.available) return null;
  const rows = data.rows.filter((r) => groupOf(r) === group);

  return (
    <section className="results" id="results" aria-labelledby="results-h">
      <div className="section-head">
        <p className="eyebrow">05 · Explore the runs</p>
        <h2 id="results-h">Every main experiment, model by model</h2>
        <p className="lede">Read live from this server’s results file. Ablations are on the results board above.</p>
      </div>
      <div className="glass results-card">
        <div className="segmented" role="radiogroup" aria-label="Experiment">
          {groups.map((g) => (
            <button key={g} type="button" role="radio" aria-checked={g === group} onClick={() => setGroup(g)}>
              {groupLabel(g)}
            </button>
          ))}
        </div>
        <Chart rows={rows} />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Balanced acc.</th><th>Sensitivity</th><th>Specificity</th><th>F1</th><th>AUC</th><th>Train time</th><th>Runs</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.input}-${r.model}`}>
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
      </div>
    </section>
  );
}
